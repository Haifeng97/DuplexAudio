# CF-Duplex 多轮/状态 QA 数据新链路

本文件记录当前正在整理的新链路。旧的单轮 QA 交接脚本已移动到 `scripts/legacy/`。

## 当前脚本

```text
scripts/00_select_duplex_turns.py
  从 2G 闲聊/状态 JSON 中筛选可用样本，输出 sysprompt + turns。

scripts/01_make_scenario_candidate_pools.py
  基于 turns 输出 normal QA、玩家打断 AI、不完整 query 三类候选池。

scripts/02_make_turn_tts_tasks.py
  根据候选池为每段玩家 query 生成 TTS task 和 scenario_index。

scripts/03_run_tts.py
  执行 TTS task。支持真实 CosyVoice，也支持 --mock_tts 跑通链路。

scripts/04_format_duplex_manifest.py
  读取 scenario_index + query wav，拼 GN 和玩家语音，输出最终训练 manifest。

scripts/05_validate_duplex_manifest.py
  校验 manifest/audio/timeline 基础一致性。
```

## 已生成的中间产物

```text
outputs/duplex_turns_game_state.jsonl
  只筛 meta_info.category 以 “以游戏为中心-基于状态的闲聊” 开头的状态问答，22670 条。

outputs/duplex_turns_state.jsonl
  更宽的状态相关池，包含 meta_info.data_type 中的“状态”样本，23614 条。

outputs/scenario_candidates_state_200/
  normal_qa_candidates.jsonl
  player_interrupt_candidates.jsonl
  incomplete_query_candidates.jsonl
```

## 当前约定

```text
sysprompt
  来自 system。
  meta_info.reference_text 不追加到 sysprompt，状态数据的 system 通常已经包含完整状态。
  不 TTS，不参与训练 loss。

turns
  来自 JSON history + 当前 input/output；状态数据目前主要是当前单轮。
  每个 turn 的 question_text 需要 TTS。
  answer_text 不 TTS，只作为文本 token 监督。

玩家打断 AI
  当前只生成候选 pair。
  没有专门的 interrupt token。玩家插话时，新的玩家语音 chunk 仍标 WAIT。
  音频形式：
    GN1 + query1_audio + GN2_short + query2_audio + GN3
  Timeline 形式：
    IDLE... WAIT... ANSWER text_prefix... WAIT... ANSWER donor_text... EOR IDLE...
  其中 GN2_short 不够完整输出 base answer，因此 base answer 没有 EOR。

不完整 query
  当前生成 query_part1/query_part2 候选。
  音频形式：
    GN1 + query_part1_audio + GN_between(1-2s) + query_part2_audio + GN_answer + GN_after
  Timeline 形式：
    GN1 -> IDLE
    query_part1_audio -> WAIT
    GN_between -> WAIT
    query_part2_audio -> WAIT
    GN_answer -> ANSWER + text tokens + EOR
    GN_after -> IDLE

正常场景
  多轮时按 turn 顺序展开：
    GN1 + query1_audio + GN2 + query2_audio + GN3 ...
  每轮回复前必须有 ANSWER，回复结束后有 EOR。
  回复区域 GN 时长按 `ceil(len(answer_text) * 1.1) * chunk_ms` 估算，默认 chunk_ms=180。
```
