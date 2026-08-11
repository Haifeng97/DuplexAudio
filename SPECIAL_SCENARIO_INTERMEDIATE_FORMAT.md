# Intervene / Complete 文本中间格式规范

## 1. 目的与职责边界

本文定义两个新场景的文本中间数据格式：

- `ai_intervenes_user`：AI 检测到玩家正在输出越界内容，主动打断并制止。
- `player_complete`：玩家明确结束对话，AI 简短回应或直接保持安静。

文本生成端只负责生成 JSONL 文本数据。不要生成音频、音频路径、时长、VAD 信息、timeline、`<FD_*>` 标签或 `<EOR>`。

下游 DuplexAudio 流程负责：

1. 校验和过滤文本。
2. 生成玩家语音 TTS 任务。
3. 根据事件字段确定触发位置和重叠关系。
4. 拼接 WAV、底噪和静音区间。
5. 生成控制标签、文本 token 和最终 manifest。

## 2. 与现有中间格式的关系

继续沿用当前 `duplex_intermediate.jsonl` 的主体结构：

```text
id + sysprompt + turns + meta
```

新格式只增加：

- 顶层 `schema_version`。
- 顶层 `scenario`。
- 特殊轮次中的 `event`。
- `player_complete` 静默样本允许 `answer_text=null`。

每一行必须是一个独立、合法的 JSON 对象。实际 JSONL 文件不要使用 Markdown 代码块，不要把一个对象拆成多行。

## 3. 通用结构

```json
{
  "schema_version": "duplex_special_v1",
  "id": "globally_unique_id",
  "scenario": "ai_intervenes_user or player_complete",
  "sysprompt": "系统提示词或角色设定",
  "turns": [],
  "meta": {}
}
```

### 3.1 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `schema_version` | string | 是 | 固定为 `duplex_special_v1` |
| `id` | string | 是 | 全数据集唯一且稳定，建议仅使用 ASCII 字符 |
| `scenario` | string | 是 | `ai_intervenes_user` 或 `player_complete` |
| `sysprompt` | string | 是 | 系统提示词、角色设定或回复规则 |
| `turns` | array | 是 | 真实对话轮次，至少包含一个特殊的当前轮 |
| `meta` | object | 是 | 数据来源、语言、角色和生成器信息 |

### 3.2 通用轮次字段

```json
{
  "turn_id": 1,
  "source": "history",
  "question_text": "玩家说的话",
  "answer_text": "AI回复",
  "needs_tts": true,
  "train_answer": true,
  "question_speaker": "玩家",
  "answer_speaker": "AI"
}
```

要求：

- `turn_id` 从 `1` 开始连续递增。
- 前置真实历史轮使用 `source="history"`。
- 特殊场景轮必须是最后一轮，并使用 `source="current"`。
- `question_text` 和非空 `answer_text` 必须是可直接朗读的纯口语文本。
- 不要输出动作、表情、舞台说明、Markdown、控制 token 或说话人前缀。
- 如果某段对话需要出现在最终音频和 timeline 中，应放在 `turns`；仅作为角色规则的内容放在 `sysprompt`。

### 3.3 推荐 meta 字段

```json
{
  "dataset": "your_dataset_name",
  "split": "train",
  "language": "zh",
  "role_name": "AI角色名",
  "player_name": "玩家",
  "turn_count": 2,
  "history_turn_count": 1,
  "text_provenance": "model_generated",
  "generator_model": "模型名称",
  "generator_version": "版本或日期"
}
```

`turn_count` 必须等于 `turns` 长度；`history_turn_count` 必须等于 `source="history"` 的轮次数量。

## 4. AI 主动制止场景

### 4.1 场景定义

玩家正在输出辱骂、骚扰、仇恨、违法诱导或其他明确需要即时制止的内容。AI 在玩家尚未说完时检测到越界行为，输出 `<FD_C_INTERVENE>`，随后立即回复制止内容。

特殊轮的 `event` 格式：

```json
{
  "type": "intervene",
  "user_text_until_trigger": "从本轮开头到检测点的完整文本",
  "user_text_after_trigger": "检测点之后玩家继续说的文本",
  "violation_category": "abuse"
}
```

强制校验规则：

```text
question_text == user_text_until_trigger + user_text_after_trigger
```

- `user_text_until_trigger` 必须已经包含足以判断越界的内容。
- `user_text_after_trigger` 去除标点后必须包含至少 6 个中文字符，表示 AI 开始制止后玩家仍有明显可听语音。
- 不要把检测点放在越界语义出现之前。
- 尽量在逗号等自然语气边界切分，降低分段 TTS 的听感断裂。
- `answer_text` 必须是简短、自然、可直接朗读的制止回复。
- `train_answer=true`。

建议的 `violation_category`：

```text
abuse | harassment | hate | sexual | violence | illegal | self_harm | other
```

### 4.2 完整示例

```json
{
  "schema_version": "duplex_special_v1",
  "id": "intervene__000001",
  "scenario": "ai_intervenes_user",
  "sysprompt": "你是一个友善、克制的AI助手。遇到辱骂时应及时制止。",
  "turns": [
    {
      "turn_id": 1,
      "source": "history",
      "question_text": "今天排位打得怎么样？",
      "answer_text": "还行，最后一局配合得不错。",
      "needs_tts": true,
      "train_answer": true,
      "question_speaker": "玩家",
      "answer_speaker": "AI"
    },
    {
      "turn_id": 2,
      "source": "current",
      "question_text": "你们这群废物一个都靠不住，我早晚把你们全收拾了。",
      "answer_text": "别这样说，尊重一下别人。",
      "needs_tts": true,
      "train_answer": true,
      "question_speaker": "玩家",
      "answer_speaker": "AI",
      "event": {
        "type": "intervene",
        "user_text_until_trigger": "你们这群废物一个都靠不住",
        "user_text_after_trigger": "，我早晚把你们全收拾了。",
        "violation_category": "abuse"
      }
    }
  ],
  "meta": {
    "dataset": "your_dataset_name",
    "split": "train",
    "language": "zh",
    "role_name": "AI",
    "player_name": "玩家",
    "turn_count": 2,
    "history_turn_count": 1,
    "text_provenance": "model_generated"
  }
}
```

下游目标 timeline 语义：

```text
<FD_IDLE> × n
<FD_D_WAIT> × n                         玩家说到检测点
<FD_C_INTERVENE>                        检测到越界并决定主动制止
<FD_A_ANSWER> TEXT... <EOR>             AI开始制止；玩家剩余语音可同时存在
<FD_D_WAIT> × n                         仅当AI说完后玩家仍未说完
<FD_IDLE> × n                           玩家最终结束
```

文本生成端只提供触发点前后的文本切分，不填写绝对时间。下游分别生成两段玩家 TTS，并在 VAD trim 后以切分边界作为最早介入时间。检测后先保留 `0-2` 个 `<FD_D_WAIT>` chunk，再输出一个 `<FD_C_INTERVENE>`；下一 chunk 开始 `<FD_A_ANSWER>`。在 `180 ms` chunk 配置下，AI 回复相对触发边界延迟 `180-540 ms`。

下游还必须保证：

- `user_text_after_trigger` 的 VAD 有效语音不少于 `1.0s`，建议不少于 `1.2s`。
- `<FD_A_ANSWER>` 开始时玩家后半段音频仍在播放，否则过滤该样本或缩短反应延迟。
- 最终 manifest 记录 `trigger_time_sec`、`reaction_delay_chunks` 和 `answer_start_sec`。

## 5. 玩家收尾场景

### 5.1 场景定义

玩家明确结束当前对话。数据分成两类：

- `70%`：正常收尾，AI回复很短的确认语。
- `30%`：玩家明显不耐烦并要求AI停止，AI立即静默。

特殊轮的 `event` 格式：

```json
{
  "type": "complete",
  "completion_type": "normal_closing or force_stop",
  "response_mode": "acknowledge or silent"
}
```

### 5.2 正常收尾并简短回复

字段组合必须为：

```text
completion_type = normal_closing
response_mode = acknowledge
answer_text = 非空简短口语
train_answer = true
answer_speaker = AI角色名
```

示例：

```json
{
  "schema_version": "duplex_special_v1",
  "id": "complete__000001",
  "scenario": "player_complete",
  "sysprompt": "你是一个友善、克制的AI助手。",
  "turns": [
    {
      "turn_id": 1,
      "source": "current",
      "question_text": "好的我了解了，那就先这样吧。",
      "answer_text": "好，回头聊。",
      "needs_tts": true,
      "train_answer": true,
      "question_speaker": "玩家",
      "answer_speaker": "AI",
      "event": {
        "type": "complete",
        "completion_type": "normal_closing",
        "response_mode": "acknowledge"
      }
    }
  ],
  "meta": {
    "dataset": "your_dataset_name",
    "split": "train",
    "language": "zh",
    "turn_count": 1,
    "history_turn_count": 0,
    "text_provenance": "model_generated"
  }
}
```

下游目标 timeline：

```text
<FD_IDLE> × n
<FD_D_WAIT> × n
<FD_I_COMPLETE>
<FD_A_ANSWER> TEXT... <EOR>
<FD_IDLE> × n
```

### 5.3 强制停止并保持静默

字段组合必须为：

```text
completion_type = force_stop
response_mode = silent
answer_text = null
train_answer = false
answer_speaker = null
```

示例：

```json
{
  "schema_version": "duplex_special_v1",
  "id": "complete__000002",
  "scenario": "player_complete",
  "sysprompt": "你是一个友善、克制的AI助手。",
  "turns": [
    {
      "turn_id": 1,
      "source": "current",
      "question_text": "行了别说了，我现在不想听。",
      "answer_text": null,
      "needs_tts": true,
      "train_answer": false,
      "question_speaker": "玩家",
      "answer_speaker": null,
      "event": {
        "type": "complete",
        "completion_type": "force_stop",
        "response_mode": "silent"
      }
    }
  ],
  "meta": {
    "dataset": "your_dataset_name",
    "split": "train",
    "language": "zh",
    "turn_count": 1,
    "history_turn_count": 0,
    "text_provenance": "model_generated"
  }
}
```

下游目标 timeline：

```text
<FD_IDLE> × n
<FD_D_WAIT> × n
<FD_I_COMPLETE>
<FD_IDLE> × n
```

静默样本不得输出 `<FD_A_ANSWER>`、文本 token 或 `<EOR>`。

## 6. 文本生成质量要求

1. 只生成中文可朗读口语，不要包含动作和场景描述。
2. 同一记录中的角色、人称和语气必须与 `sysprompt` 一致。
3. AI回复应简洁，避免长篇教育、模板化免责声明和重复话术。
4. Complete 的确认回复应多样化，不要全部使用“好的”。
5. Intervene 的触发类别、玩家内容和制止表达应保持多样化。
6. Intervene 不能把完整越界话语全部放入 `user_text_until_trigger`；后半段去除标点后至少保留 6 个中文字符。
7. `id` 不得重复；禁止把同一文本仅替换标点后重复写入。
8. 多轮数据中的历史必须与当前特殊轮语义连贯。

## 7. 交付前校验清单

- 文件可以逐行执行 `json.loads`。
- 所有记录的 `schema_version` 均为 `duplex_special_v1`。
- `scenario` 只能是 `ai_intervenes_user` 或 `player_complete`。
- 每条记录恰好有一个特殊轮，且特殊轮是最后一轮。
- `turn_id` 连续，`turn_count` 和 `history_turn_count` 正确。
- Intervene 的两段文本可无损拼回 `question_text`，且后半段去除标点后至少有 6 个中文字符。
- Intervene 的 `answer_text` 非空，`train_answer=true`。
- Complete 的 `completion_type` 与 `response_mode` 组合合法。
- Silent Complete 的 `answer_text` 和 `answer_speaker` 均为 `null`。
- 整体 Complete 数据约 `70% acknowledge / 30% silent`。
- 所有需要朗读的字段不含标签、Markdown、动作描述和说话人前缀。
