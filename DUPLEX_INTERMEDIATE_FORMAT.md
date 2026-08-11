# DuplexAudio 中间数据格式规范

本文定义“文本数据生成端”交付给 DuplexAudio 流水线的中间格式。生成端只需要提供角色设定和按时间排序的真实对话；TTS、query 截断、clarification、interrupt、backchannel、音频拼接、chunk 对齐和全双工标签均由后续流水线生成。

## 1. 文件格式

- UTF-8 编码的 JSONL 文件。
- 每行一个完整 JSON object，不要使用外层 JSON array。
- 一行表示一个独立的源对话样本，可以包含一轮或多轮对话。
- `id` 必须在整个文件中唯一、稳定；重新生成同一条源数据时不要随机改变。
- 文本字段必须是字符串，不能用 `null` 代替空字符串；必填文本不得为空。

## 2. 标准结构

```json
{
  "id": "dataset_name__conversation_000001",
  "sysprompt": "请你扮演角色‘吉莉’。保持自然、简短地和玩家交流。",
  "turns": [
    {
      "turn_id": 1,
      "source": "history",
      "question_text": "你刚才去哪了？",
      "answer_text": "我去前面看了一下情况。",
      "needs_tts": true,
      "train_answer": true,
      "question_speaker": "玩家",
      "answer_speaker": "吉莉"
    },
    {
      "turn_id": 2,
      "source": "current",
      "question_text": "那边安全吗？",
      "answer_text": "暂时安全，不过我们还是得小心一点。",
      "needs_tts": true,
      "train_answer": true,
      "question_speaker": "玩家",
      "answer_speaker": "吉莉"
    }
  ],
  "meta": {
    "dataset": "dataset_name",
    "split": "train",
    "source_group_id": "dataset_name::conversation_000001",
    "language": "zh",
    "role_name": "吉莉",
    "history_turn_count": 1,
    "turn_count": 2,
    "sysprompt_ds_history_present": false,
    "text_provenance": "human_written"
  }
}
```

## 3. 必填字段

### 顶层字段

| 字段 | 类型 | 要求 |
| --- | --- | --- |
| `id` | string | 全局唯一且稳定的样本 ID。 |
| `sysprompt` | string | 角色、人设、场景或必要上下文；没有时使用空字符串。 |
| `turns` | array | 至少一轮，严格按时间从早到晚排列。 |
| `meta` | object | 至少包含 `dataset`、`split` 和 `source_group_id`。 |

不要只在顶层提供 `question_text` 和 `answer_text`。当前流水线以 `turns` 为准，并从最后一轮派生当前 query/answer。

### `turns` 中每一轮

| 字段 | 类型 | 要求 |
| --- | --- | --- |
| `question_text` | string | 玩家/user 实际说出的完整文本，不能为空。 |
| `answer_text` | string | AI 角色/assistant 实际回复的完整文本，不能为空。 |

推荐同时提供：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `turn_id` | integer | 从 1 开始递增。流水线会重新顺序编号，因此不能依赖外部跳号。 |
| `source` | string | 前面的轮次填 `history`，最后一轮填 `current`。 |
| `needs_tts` | boolean | 通常填 `true`。省略时也按 `true` 处理。 |
| `train_answer` | boolean | 通常填 `true`，表示该回复属于训练对话。 |
| `question_speaker` | string | 玩家侧说话人名，可选。 |
| `answer_speaker` | string | AI 角色名，可选。 |

## 4. 对话语义

1. `turns` 必须是真实、连续、按时间排序的对话历史，不能把无关单轮随机拼成多轮。
2. `turns[-1]` 始终是当前轮；前面的轮次都是真正的历史轮。
3. 每一轮都必须是“玩家 query -> AI reply”。不要把相邻两个玩家发言分别伪装成两轮。
4. 同一说话人的连续碎片应先合并，再构造问答轮。
5. 多轮数据会用于构造真实 interrupt，并在完整轮次边界加入 1-3 秒 `<FD_IDLE>`，所以轮次顺序不能错。
6. 单轮数据完全合法，但不能用于依赖真实上一轮回复的 same-row interrupt。

## 5. `sysprompt` 与历史

`sysprompt` 可以包含角色设定、世界状态和原数据自带的对话上下文，但必须遵守：

- `sysprompt` 中的对话只作为 system context，不参与音频和 timeline 训练。
- 如果其中包含数据集提供的历史对话，设置 `meta.sysprompt_ds_history_present=true`。
- 真正参与训练的历史必须放入 `turns`。
- 不要把同一段历史既放进 `sysprompt`，又重复放进 `turns`。
- `sysprompt` 中的历史不能用于构造 interrupt。

推荐将 sysprompt 内历史整理为可读文本，例如：

```text
请你扮演角色“吉莉”。

【对话历史】
玩家：我们从哪边走？
吉莉：先沿着围墙走。
```

## 6. 文本清洗与长度

进入中间格式前应完成以下清洗：

- 去掉 `玩家：`、`User:`、`AI:` 等说话人前缀，角色名放入 speaker 字段。
- 去掉不应该被朗读的动作、舞台说明和标注，例如 `[叹气]`、`（转身离开）`。
- 去掉首尾空白；不要保留仅由空白组成的文本。
- 保留正常标点，不要为了长度随意截断句子。
- 不要在文本中写下游控制标签或 tokenizer token。

当前候选池默认硬限制：

| 项目 | 默认值 | 超限处理 |
| --- | ---: | --- |
| 单轮 `question_text` | 最长 240 个 Python 字符 | 当前轮超限会丢整行；历史轮超限会丢该历史轮。 |
| 单轮 `answer_text` | 最长 360 个 Python 字符 | 当前轮超限会丢整行；历史轮超限会丢该历史轮。 |
| 有效轮数 | 最多 8 轮 | 只保留最近 8 轮。 |
| 非空文本 | 最少 1 个字符 | 空 query/answer 无效。 |

为了让更多样本能进入特殊场景，建议：

- 用于 incomplete query 的 `question_text` 至少 8 个字符，并且存在自然的中间截断位置。
- 用于 interrupt/backchannel 的 `answer_text` 至少 8 个字符，最好是可以自然切成前半和后半的完整回复。
- 避免大量只有“嗯”“好”“走”等极短 query；它们的 TTS 可能不足 1 秒，后续会被质量过滤。

## 7. 来源分组与去重

`meta.source_group_id` 是强制推荐字段。它表示样本来自哪一个原始对话、场景或重叠窗口。

- 同一个原始对话切出的多个样本必须共享同一个 `source_group_id`。
- 同一场景从不同角色视角导出的重复样本也应共享该 ID。
- 完全独立的原始对话使用不同 ID。
- 最终场景分配会用该字段避免同一源对话同时进入多个类别或造成重叠泄漏。

不要直接用随机 UUID 作为 `source_group_id`，除非每个 UUID 确实对应一个独立原始对话。

## 8. 可选的场景许可

默认情况下，流水线会根据文本长度和轮数自动判断可用场景。生成端如果明确知道某条数据不适合某类场景，可以添加：

```json
"selection": {
  "can_normal": true,
  "can_interrupt_base": true,
  "can_interrupt_donor": true,
  "can_incomplete_query": true
}
```

- `can_normal`：允许作为正常问答。
- `can_interrupt_base`：允许其 AI 回复作为被打断的回复。
- `can_interrupt_donor`：允许其玩家 query 作为打断内容。
- `can_incomplete_query`：允许把其 query 切成不完整 query。

没有充分理由时可以完全省略 `selection`，由流水线判断。设置为 `true` 不会绕过长度、轮数等硬检查。

## 9. 推荐的来源元数据

以下字段不会改变核心读取逻辑，但便于审计、回溯和后续定向筛选：

```json
"meta": {
  "dataset": "my_dataset",
  "split": "train",
  "source_group_id": "my_dataset::dialogue_42",
  "source_file": "part-0001.jsonl",
  "original_dialogue_id": "dialogue_42",
  "language": "zh",
  "role_name": "吉莉",
  "player_name": "玩家",
  "turn_count": 3,
  "history_turn_count": 2,
  "sysprompt_ds_history_present": false,
  "text_provenance": "human_written",
  "generator_model": "",
  "generator_version": ""
}
```

如果 query、answer 或 sysprompt 是模型生成的，必须记录生成模型和版本，不要标成 `human_written`。

## 10. 最小单轮示例

```json
{"id":"demo__000001","sysprompt":"请你扮演吉莉，与玩家自然交流。","turns":[{"turn_id":1,"source":"current","question_text":"前面还有敌人吗？","answer_text":"我暂时没看到，但我们最好贴着掩体走。","needs_tts":true,"train_answer":true,"question_speaker":"玩家","answer_speaker":"吉莉"}],"meta":{"dataset":"demo","split":"train","source_group_id":"demo::dialogue_000001","language":"zh","role_name":"吉莉","turn_count":1,"history_turn_count":0,"sysprompt_ds_history_present":false,"text_provenance":"human_written"}}
```

## 11. 不要在中间格式中生成的内容

以下均属于下游产物，输入数据里不要提前填写：

- `<FD_IDLE>`、`<FD_D_WAIT>`、`<FD_A_ANSWER>`、`<FD_F_WAIT>`、`<FD_G_INTERRUPT>`、`<FD_H_CONTINUE>`、`<FD_I_COMPLETE>`、`<FD_J_ACTIVE>`、`<EOR>`。
- `<FD_IDLE>`/`<FD_D_WAIT>` 时长、chunk 序列或 token 序列。
- incomplete query 的截断点。
- clarification 的插入位置和澄清回答。
- interrupt/backchannel 的插入点。
- TTS 音频路径、VAD 结果、底噪和最终拼接音频路径。

这些内容必须由同一版本的 DuplexAudio 流水线统一生成，避免文本端和音频端的时间对齐规则不一致。

## 12. 交付前检查

- JSONL 每一行都能被 `json.loads` 独立解析。
- `id` 无重复。
- `meta.source_group_id` 非空且分组语义正确。
- `turns` 非空，并按时间从早到晚排列。
- 每轮 `question_text`、`answer_text` 均为非空字符串。
- 最后一轮标记为 `current`，其余轮标记为 `history`。
- 没有说话人前缀、动作标注和下游控制 token。
- 多轮确实来自同一段连续对话，不是随机拼接。
- 模型生成内容带有可追踪的 provenance。
