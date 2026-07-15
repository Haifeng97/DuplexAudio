# CF-Duplex QA 数据集交接说明

本目录用于交接“纯文本 QA -> TTS 合成用户问题语音 -> QA 训练 manifest”的完整流程。


## 1. QA 训练需要什么数据

QA 样本本质是：

```text
用户问题音频 question_audio -> 模型输出 <FD_A_ANSWER> + answer_text + <EOR>
```

每条 manifest JSONL 至少需要：

```json
{
  "id": "qa_000001",
  "task": "qa",
  "audio": "/abs/path/to/question.wav",
  "text": "这是答案文本。",
  "target_text": "这是答案文本。",
  "answer_text": "这是答案文本。",
  "question_text": "这是问题文本？",
  "text_query": "这是问题文本？",
  "asr_text": "这是问题文本？"
}
```

字段含义：

| 字段 | 必需 | 说明 |
|---|---|---|
| `id` | 是 | 样本 ID，唯一即可 |
| `task` | 建议 | 固定写 `qa`，训练时也传 `--task qa` |
| `audio` | 是 | 用户问题语音 wav，推荐 16kHz mono |
| `text` | 是 | AI 要输出的答案文本 |
| `target_text` | 建议 | 同 `text`，用于兼容训练脚本字段优先级 |
| `answer_text` | 建议 | 同 `text`，保留语义清晰性 |
| `question_text` | 建议 | 问题文本，用于审计/ASR 辅助/可视化 |
| `text_query` | 建议 | 同 `question_text`，兼容字段 |
| `asr_text` | 建议 | 同 `question_text`，如果启用 Paraformer ASR aux loss 会用到 |

训练脚本字段读取优先级：

- 答案监督文本：`target_text -> answer_text -> text -> output -> text_query`
- 问题文本：`question_text -> input -> instruction -> text_query`
- ASR 辅助文本：`asr_text -> question_text -> output -> text_query -> text`

## 2. QA 训练时序

QA 样本需要构造成：

```text
[用户问题音频 chunks]        -> <FD_D_WAIT> / 稀疏 WAIT 监督
[额外 zero/silence chunk]   -> <FD_D_WAIT>
[answer decision zero chunk]-> <FD_A_ANSWER>
[text phase chunks]    -> answer_text tokens + <EOR>
```

所以 text QA 转 语音QA 数据时，只需要合成 **用户问题音频**；答案保留文本即可，不要求答案音频。

## 3. 输入纯文本 QA 格式

推荐输入 JSONL：

```json
{"id":"qa_000001","question":"北京有哪些著名景点？","answer":"北京著名景点包括故宫、天安门、颐和园、长城等。"}
{"id":"qa_000002","question":"感冒了应该多喝水吗？","answer":"一般可以适量多喝水，并注意休息；如果症状严重或持续不缓解，应及时就医。"}
```

脚本默认会自动识别以下字段：

- 问题字段：`question`, `query`, `input`, `instruction`, `text_query`, `question_text`
- 答案字段：`answer`, `response`, `output`, `target`, `target_text`, `answer_text`, `text`

## 4. 流程总览

```bash
# 0. 路径
REPO=/data/zeynliu/repos/CF-Duplex
HANDOFF=/path/to/cf_qa_handoff
IN=/path/to/text_qa.jsonl
OUT=/data/zeynliu/work/qa_tts_v1

# 1. 生成 TTS 任务和 QA 索引
python $HANDOFF/scripts/01_make_question_tts_tasks.py \
  --input $IN \
  --out_dir $OUT \
  --ref_wav /path/to/ref.wav \
  --ref_text "参考音频对应的文本"

# 2. 调用 CosyVoice3 批量合成用户问题音频
python $HANDOFF/scripts/02_run_cosyvoice_tts.py \
  --repo $REPO \
  --tasks $OUT/tts_tasks.jsonl \
  --model_dir /path/to/CosyVoice-or-Fun-CosyVoice3 \
  --work_dir $OUT/tts_work \
  --gpus 0,1,2,3 \
  --procs_per_gpu 4 \
  --python_bin /path/to/python \
  --cosyvoice_repo /path/to/CosyVoice

# 3. format 成 QA manifest，同时完成 VAD、按 180ms 切 chunk、写 chunk 标签
python $HANDOFF/scripts/03_format_v1_qa_manifest.py \
  --index $OUT/qa_index.jsonl \
  --out $OUT/manifest_v1_qa.jsonl \
  --chunk_ms 180 \
  --vad_threshold_dbfs -40

# 4. 校验
python $HANDOFF/scripts/04_validate_v1_qa_manifest.py \
  --manifest $OUT/manifest_v1_qa.jsonl
```

## 5. 输出文件

`01_make_question_tts_tasks.py` 生成：

```text
$out_dir/tts_tasks.jsonl
$out_dir/qa_index.jsonl
$out_dir/question_wav/*.wav   # TTS 输出目标路径，第二步生成
```

`03_format_v1_qa_manifest.py` 会读取每个问题 wav，完成：

1. 读取/重采样到 16kHz mono；
2. 按 `chunk_ms=180` 切 chunk；
3. 计算每个 chunk 的 RMS dBFS；
4. 用vad判断 active；
5. 从 `first_active_chunk` 到 `last_active_chunk` 标成 `WAIT`，然后问题后会接入:`ANSWER`
6. 写入 `chunk_states`、`chunk_vad`、`vad` 审计字段。

生成：

```text
$out_dir/manifest_v1_qa.jsonl
```

## 6. 最终 manifest 样例

```json
{
  "id": "qa_000001",
  "source": "textqa_tts",
  "task": "qa",
  "audio": "/data/zeynliu/work/qa_tts_v1/question_wav/qa_000001.wav",
  "audios": ["/data/zeynliu/work/qa_tts_v1/question_wav/qa_000001.wav"],
  "text": "北京著名景点包括故宫、天安门、颐和园、长城等。",
  "target_text": "北京著名景点包括故宫、天安门、颐和园、长城等。",
  "answer_text": "北京著名景点包括故宫、天安门、颐和园、长城等。",
  "question_text": "北京有哪些著名景点？",
  "text_query": "北京有哪些著名景点？",
  "asr_text": "北京有哪些著名景点？",
  "sample_rate": 16000,
  "tts": {
    "engine": "cosyvoice",
    "question_audio": "/data/zeynliu/work/qa_tts_v1/question_wav/qa_000001.wav"
  }
}


## 7. 注意事项

1. 目前QA 只需要合成问题音频；答案不要 TTS 成音频，答案是文本监督。
2. TTS 输出最好是 `16kHz mono wav`；脚本会写 `sample_rate=16000`。
3. `question_text` 和 `asr_text` 应该等于问题文本；`text/target_text/answer_text` 应该等于答案文本。
4. 如果用 CosyVoice zero-shot，`ref_wav/ref_text` 必须匹配，否则音色/质量会不稳定。
5. 批量 TTS 断点续跑：`02_run_cosyvoice_tts.py` 会跳过已经存在且大于 1000B 的 wav。
