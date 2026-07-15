# CF-Duplex 多轮/状态 QA 数据新链路

本文件记录当前正在整理的新链路。旧的单轮 QA 交接脚本已移动到 `scripts/legacy/`。

## 当前脚本

```text
scripts/00_select_duplex_turns.py
  从 2G 闲聊/状态 JSON 中筛选可用样本，输出 sysprompt + turns。

scripts/01_make_scenario_candidate_pools.py
  基于 turns 输出 normal QA、玩家打断 AI、不完整 query 三类候选池。

scripts/02_run_cosyvoice_tts.py
  复用 CosyVoice 批量 TTS runner。后续生成 turn 级 TTS tasks 后继续使用。
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
  来自 system，可追加 meta_info.reference_text。
  不 TTS，不参与训练 loss。

turns
  来自 JSON history + 当前 input/output；状态数据目前主要是当前单轮。
  每个 turn 的 question_text 需要 TTS。
  answer_text 不 TTS，只作为文本 token 监督。

玩家打断 AI
  当前只生成候选 pair。最终 timeline 规则应参考 QA interrupt：
  base question -> base answer prefix -> donor first chunk G_INTERRUPT -> donor WAIT -> donor answer。

不完整 query
  当前只生成 partial/full 候选。partial 后停顿标 D_WAIT 还是 IDLE 仍需按训练协议确认。
```
