# DuplexAudio 标签协议：fd_control_v1

本协议规定每个固定控制 token 的语义和五类训练场景的 timeline。当前每个音频 chunk 对应一个监督 token；默认 chunk 长度为 180 ms。

## 固定 Token

| Token | 类型 | 定义 |
| --- | --- | --- |
| `<FD_IDLE>` | 持续状态 | 当前没有玩家语音，模型也没有输出回复 token。 |
| `<FD_D_WAIT>` | 持续状态 | 玩家正在正常说话，模型继续听。 |
| `<FD_A_ANSWER>` | 单次事件 | AI 开始或重新开始输出回复；后面才是文本 token。 |
| `<FD_F_WAIT>` | 单次事件 | 检测到当前玩家 query 不完整；每个不完整片段只出现一次。 |
| `<FD_G_INTERRUPT>` | 单次事件 | AI 回复期间检测到玩家开始说话；只标玩家语音第一个 chunk。 |
| `<FD_H_CONTINUE>` | 单次事件 | 确认刚才的插话是 backchannel，继续原来未完成的回答。 |
| `<FD_I_COMPLETE>` | 保留 | 当前没有对应场景，不生成。 |
| text tokens | 内容 | AI 实际回复文本的 tokenizer tokens。 |
| `<EOR>` | 单次事件 | 一轮 AI 回复完整结束。 |

所有固定 token 均保留尖括号。普通文本 token 不增加 `<>`。

## 通用优先级

每个 chunk 只有一个目标 token：

1. AI 输出控制 token、文本 token 或 `<EOR>` 时，使用对应输出。
2. AI 回复期间玩家刚开口，使用 `<FD_G_INTERRUPT>`。
3. 普通玩家语音使用 `<FD_D_WAIT>`。
4. 检测到不完整 query 时，该片段最后一个有效语音 chunk 使用 `<FD_F_WAIT>`，替代原本的 `<FD_D_WAIT>`。
5. 没有玩家语音且没有其他模型输出时，使用 `<FD_IDLE>`。

## 1. 正常问答

```text
<FD_IDLE>*
<FD_D_WAIT>+
<FD_A_ANSWER>
text tokens+
<EOR>
<FD_IDLE>*
```

## 2. 不完整 Query 后继续

```text
<FD_D_WAIT>* <FD_F_WAIT>
<FD_IDLE>+                  # query 两部分之间 0.5-2 秒静音
<FD_D_WAIT>+
<FD_A_ANSWER>
text tokens+
<EOR>
```

`<FD_F_WAIT>` 只标第一部分最后一个有效语音 chunk。中间静音全部是 `<FD_IDLE>`。

## 3. 不完整 Query 后主动澄清

```text
<FD_D_WAIT>* <FD_F_WAIT>
<FD_IDLE>+                  # 3-5 秒
<FD_A_ANSWER>
clarification text tokens+
<EOR>
```

玩家随后补充完整 query 时，进入下一轮正常问答。

## 4. 真正打断

```text
<FD_A_ANSWER>
old answer text prefix
<FD_G_INTERRUPT>
<FD_D_WAIT>*
<FD_A_ANSWER>
new answer text tokens+
<EOR>
```

被打断的旧答案没有 `<EOR>`。新答案不是旧答案的 continuation。

## 5. Backchannel

```text
<FD_A_ANSWER>
answer text prefix
<FD_G_INTERRUPT>
<FD_D_WAIT>*
<FD_H_CONTINUE>
<FD_A_ANSWER>
answer text continuation
<EOR>
```

`<FD_H_CONTINUE>` 和后面的 `<FD_A_ANSWER>` 各占一个独立 chunk。前后文本属于同一轮回答，因此 prefix 后没有 `<EOR>`。

## 6. 原始多轮边界

```text
text tokens <EOR>
<FD_IDLE>+                  # 完整轮次之间随机 1-3 秒
<FD_D_WAIT>+
<FD_A_ANSWER>
...
```

特殊事件内部不插入普通轮间 IDLE；特殊轮与前后完整轮之间仍按完整轮边界规则处理。

## 场景约束

| 场景 | `F_WAIT` | `G_INTERRUPT` | `H_CONTINUE` |
| --- | ---: | ---: | ---: |
| `normal_qa` | 0 | 0 | 0 |
| `incomplete_query` | 1 | 0 | 0 |
| `incomplete_query_clarification` | 1 | 0 | 0 |
| `player_interrupts_ai` | 0 | 1 | 0 |
| `player_backchannel` | 0 | 1 | 1 |

`player_backchannel` 中必须满足：

```text
<FD_G_INTERRUPT> ... <FD_H_CONTINUE> <FD_A_ANSWER>
```

`<FD_I_COMPLETE>` 在所有当前场景中的数量均为 0。
