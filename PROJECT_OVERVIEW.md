# DuplexAudio 项目说明

这个项目用于把现有文本 SFT 数据转换成“全双工对话状态管理”训练数据。输出数据包含玩家 query 的 TTS 音频、拼接后的高斯噪声区域、以及按固定 chunk 对齐的 timeline 标签。

固定控制标签及五类 timeline 的唯一规范见
[`DUPLEX_LABEL_PROTOCOL.md`](DUPLEX_LABEL_PROTOCOL.md)。

Roleplay v3 的 Qwen3-TTS、真实 RAMC backchannel 和完整执行命令见
[`ROLEPLAY_V3_PIPELINE.md`](ROLEPLAY_V3_PIPELINE.md)。

当前主要覆盖五类场景：

- `normal_qa`：正常问答。玩家说话期间输出 `<FD_D_WAIT>`，说完后输出 `<FD_A_ANSWER>`、回复 token、`<EOR>`，再回到 `<FD_IDLE>`。如果原始数据是多轮，完整轮次之间会在 `<EOR>` 后插入 1-3 秒随机底噪 `<FD_IDLE>`。
- `incomplete_query`：玩家 query 被切成前后两段。前半句最后一个语音 chunk 输出一次 `<FD_F_WAIT>`，中间 0.5-2 秒无语音区间输出 `<FD_IDLE>`，后半句恢复 `<FD_D_WAIT>`。完整 query 后才输出 `<FD_A_ANSWER>`。
- `incomplete_query_clarification`：把一轮“玩家半句 query + AI 追问/确认回复”插入原对话。半句末尾输出一次 `<FD_F_WAIT>`，随后 3-5 秒随机底噪输出 `<FD_IDLE>`，再触发 `<FD_A_ANSWER>` 和澄清回复。
- `player_interrupts_ai`：只使用原始数据里真正 `history` 非空的多轮 case。后一轮玩家 query 打断前一轮 AI 未说完的回复，玩家语音首 chunk 输出 `<FD_G_INTERRUPT>`，其余语音输出 `<FD_D_WAIT>`；旧回复不输出 `<EOR>`。
- `player_backchannel`：AI 回复中玩家插入真实短反馈音频。首 chunk 输出 `<FD_G_INTERRUPT>`，其余语音输出 `<FD_D_WAIT>`；AI 继续原回复前依次输出 `<FD_H_CONTINUE>`、`<FD_A_ANSWER>`。

`sysprompt` 里的对话历史只作为上下文，不参与训练，也不能用于构造 interrupt。清洗后会整理成：

```text
【对话历史】
玩家：...
吉莉：...
```

## 目录

```text
scripts/00_select_duplex_turns.py          选择并清洗原始文本数据
scripts/01_make_scenario_candidate_pools.py 生成 normal / incomplete / interrupt 候选池
scripts/02_make_turn_tts_tasks.py          生成玩家 query 的 TTS 任务
scripts/03_run_tts.py                      调 CosyVoice 生成玩家语音
scripts/04_format_duplex_manifest.py       拼音频、加高斯噪声、生成 timeline manifest
scripts/05_validate_duplex_manifest.py     校验 manifest 和音频长度是否对齐
scripts/06_mix_duplex_manifest.py          后续用于混合不同场景比例
scripts/07_extract_esd_samples.py          从 ESD parquet 抽参考音频
scripts/08_build_esd_voice_bank.py         构建 ESD voice bank
scripts/09_export_incomplete_clarification_requests.py 导出半句 query 追问回复的 LLM 请求列表
scripts/10_build_incomplete_clarification_scenarios.py 用 LLM 回复生成 clarification 场景 index
scripts/16_extract_magicdata_backchannels.py 提取真实高频 backchannel
scripts/17_select_roleplay_scenario_mix.py  在 TTS 前按比例分配场景
scripts/18_concat_duplex_manifests.py       无二次抽样地拼接最终 manifest
scripts/19_upgrade_manifest_with_backchannel.py 给旧成品增加 backchannel
scripts/20_upgrade_manifest_label_protocol.py  把旧成品升级到 fd_control_v1

configs/esd_voice_tts.yaml                 TTS 参考音频配置
tokenizers/qwen3_8b/tokenizer.json         Qwen3-8B tokenizer
tools/duplex_viewer.py                     本地可视化查看器
```

## OIT 环境准备

建议代码放在：

```bash
/data/haifengjia/Projects/DuplexAudio
```

需要另外准备的数据和模型：

```text
原始 cgame 文本数据 json
CosyVoice repo
CosyVoice 模型
ESD reference voice bank 或 ESD parquet 原始文件
```

CosyVoice 路径按 OIT 机器实际情况传参，例如：

```bash
--cosyvoice_repo /data/haifengjia/models/CosyVoice
--model_dir /data/haifengjia/models/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
```

`04_format_duplex_manifest.py` 默认使用 `silero-vad` 替换 TTS 首尾静音为高斯噪声。如果环境里没有，需要在 `cosyvoice` 环境中安装：

```bash
pip install silero-vad
```

## 基本流程

第一步，筛选并清洗原始文本数据：

```bash
python scripts/00_select_duplex_turns.py \
  --input /path/to/source.json \
  --out outputs/selected_turns/selected.jsonl \
  --stats_out outputs/selected_turns/stats.json
```

第二步，生成场景候选池：

```bash
python scripts/01_make_scenario_candidate_pools.py \
  --input outputs/selected_turns/selected.jsonl \
  --out_dir outputs/scenario_candidates \
  --limit_each 0 \
  --seed 20260717 \
  --interrupt_pair_mode same_row_previous
```

第三步，给每类场景生成 TTS 任务：

```bash
python scripts/02_make_turn_tts_tasks.py \
  --input outputs/scenario_candidates/normal_qa_candidates.jsonl \
  --out_dir outputs/pipeline_normal
```

第四步，跑 CosyVoice：

```bash
conda activate cosyvoice

python scripts/03_run_tts.py \
  --tasks outputs/pipeline_normal/tts_tasks.jsonl \
  --results outputs/pipeline_normal/tts_results.jsonl \
  --cosyvoice_repo /data/haifengjia/models/CosyVoice \
  --model_dir /data/haifengjia/models/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B
```

第五步，格式化 manifest：

```bash
python scripts/04_format_duplex_manifest.py \
  --index outputs/pipeline_normal/scenario_index.jsonl \
  --out outputs/pipeline_normal/manifest.jsonl \
  --sample_rate 24000 \
  --chunk_ms 180 \
  --tokenizer_json tokenizers/qwen3_8b/tokenizer.json \
  --vad_mode silero
```

可选：生成 incomplete query clarification 的 LLM 请求列表，然后用填好的回复构建场景 index：

```bash
python scripts/09_export_incomplete_clarification_requests.py \
  --input outputs/pipeline_normal/scenario_index.jsonl \
  --out outputs/incomplete_clarification/llm_requests.jsonl \
  --limit 1000 \
  --seed 20260722

python scripts/10_build_incomplete_clarification_scenarios.py \
  --input outputs/incomplete_clarification/llm_requests_filled.jsonl \
  --out outputs/incomplete_clarification/candidates.jsonl
```

之后对 `outputs/incomplete_clarification/candidates.jsonl` 继续跑 `02_make_turn_tts_tasks.py`、TTS 和 `04_format_duplex_manifest.py`。

第六步，校验：

```bash
python scripts/05_validate_duplex_manifest.py \
  --manifest outputs/pipeline_normal/manifest.jsonl
```

## 数据策略

目前对 cgame 数据的默认策略：

- 去掉 category 为 `以游戏为中心-基于状态的闲聊-决策问答` 的数据。
- 去掉 answer 或 history answer 中含中文括号表情的数据。
- `normal` 和 `incomplete` 可以使用所有筛选后的可用 rows。
- 原始多轮样本会在完整轮次之间插入 1-3 秒随机底噪 `<FD_IDLE>`。
- `incomplete_query` 的 split query 内部、`interrupt` 的 base answer prefix 到 interrupt query 中间，不套用完整轮间 idle 规则。
- `incomplete_query_clarification` 由半句 query 和 LLM 生成的追问回复组成，可以插入单轮前或多轮中间。
- `interrupt` 只能使用原始 `history` 非空、转换后 `turns >= 2` 的 rows。
- `sysprompt` 里的历史只作为 system 上下文，不参与训练，不用于 interrupt。

计划的最终比例：

```text
normal      70%
incomplete  15%
interrupt   15%
```

如果要求“一条源数据只用一次”，需要在最终混合阶段保证 `source_id` 不重复。

## 可视化

本地查看 manifest：

```bash
python tools/duplex_viewer.py --host 0.0.0.0 --port 8765
```

页面会展示音频、timeline、标签分布、波形、Mel 图、sysprompt 和样本元信息。

