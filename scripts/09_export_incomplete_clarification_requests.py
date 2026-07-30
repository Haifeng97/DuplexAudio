#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


STYLE_HINTS = [
    "短促追问，像语音里自然打断后确认。",
    "轻微催促玩家把话说完，但不要不礼貌。",
    "围绕已说出的半句关键词追问，不猜完整意图。",
    "口语化确认玩家想表达什么。",
    "带一点角色性格，但不要动作描写。",
    "简短提醒玩家后半句没说完。",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def normalized_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        out = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            q = str(turn.get("question_text") or "")
            a = str(turn.get("answer_text") or "")
            if q and a:
                out.append({
                    "turn_id": len(out) + 1,
                    "source": turn.get("source", "current"),
                    "question_text": q,
                    "answer_text": a,
                    "needs_tts": True,
                    "train_answer": True,
                })
        if out:
            return out
    q = str(row.get("question_text") or row.get("full_question_text") or "")
    a = str(row.get("answer_text") or row.get("answer_text_if_complete") or "")
    if not q or not a:
        return []
    return [{
        "turn_id": 1,
        "source": "current",
        "question_text": q,
        "answer_text": a,
        "needs_tts": True,
        "train_answer": True,
    }]


def choose_insert_at(turns: List[Dict[str, Any]], rng: random.Random) -> int:
    if len(turns) <= 1:
        return 0
    return rng.randint(1, len(turns) - 1)


def choose_query_split(q: str, rng: random.Random, min_prefix_chars: int, max_prefix_chars: int, max_prefix_ratio: float) -> Optional[int]:
    q = q.strip()
    if len(q) < max(min_prefix_chars + 1, 4):
        return None
    hi = min(len(q) - 1, max_prefix_chars, max(min_prefix_chars, int(len(q) * max_prefix_ratio)))
    lo = min(min_prefix_chars, hi)
    if hi <= lo:
        return None
    for _ in range(20):
        cut = rng.randint(lo, hi)
        if q[cut - 1] not in "，。！？；、,.!?; ":
            return cut
    return rng.randint(lo, hi)


def dialogue_text(turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for turn in turns:
        lines.append(f"玩家：{turn['question_text']}")
        lines.append(f"AI：{turn['answer_text']}")
    return "\n".join(lines)


def build_request(row: Dict[str, Any], rng: random.Random, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    turns = normalized_turns(row)
    if not turns:
        return None
    insert_at = choose_insert_at(turns, rng)
    target_turn = turns[insert_at]
    full_q = target_turn["question_text"].strip()
    cut = choose_query_split(full_q, rng, args.min_prefix_chars, args.max_prefix_chars, args.max_prefix_ratio)
    if cut is None:
        return None
    partial_q = full_q[:cut].strip()
    if not partial_q:
        return None
    source_id = str(row.get("source_id") or row.get("id"))
    target_turn_id = target_turn.get("turn_id", insert_at + 1)
    req_id = f"incomplete_clarify__{source_id}__before_t{target_turn_id}__cut{cut}"
    context_turns = turns[:insert_at]
    following_turns = turns[insert_at:]
    style_hint = STYLE_HINTS[len(req_id) % len(STYLE_HINTS)] if args.deterministic_style else rng.choice(STYLE_HINTS)
    llm_prompt = (
        "你要为语音对话数据生成 AI 对玩家未说完半句话的追问回复。\n"
        "要求：\n"
        "1. 玩家 query 是半句话，AI 只能追问/催促/确认，不能直接回答完整问题。\n"
        "2. 回复要符合 sysprompt 的角色和语气。\n"
        "3. 尽量口语化、短，不超过 20 个汉字。\n"
        "4. 不要动作描写，不要括号表情，不要解释生成规则。\n"
        "5. 同批次要尽量多样化，避免都写成同一句。\n"
        f"风格提示：{style_hint}\n\n"
        f"对话历史：\n{dialogue_text(context_turns) if context_turns else '无'}\n\n"
        f"玩家未说完的 query：{partial_q}\n\n"
        "请只输出 AI 回复文本。"
    )
    messages = [{"role": "system", "content": str(row.get("sysprompt") or "") + "\n\n" + llm_prompt}]
    for turn in context_turns:
        messages.append({"role": "user", "content": turn["question_text"]})
        messages.append({"role": "assistant", "content": turn["answer_text"]})
    messages.append({"role": "user", "content": partial_q})
    return {
        "id": req_id,
        "source_id": source_id,
        "source_scenario_id": row.get("id"),
        "scenario": "incomplete_query_clarification_request",
        "sysprompt": row.get("sysprompt", ""),
        "context_turns": context_turns,
        "following_turns": following_turns,
        "insert_at": insert_at,
        "insert_before_turn_id": target_turn_id,
        "partial_question_text": partial_q,
        "full_question_text": full_q,
        "original_answer_text": target_turn.get("answer_text", ""),
        "split": {"cut_char_index": cut, "unicode_codepoint_boundary": True},
        "style_hint": style_hint,
        "llm_prompt": llm_prompt,
        "llm_messages": messages,
        "clarification_answer_text": "",
        "meta": row.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export LLM requests for incomplete-query clarification turns.")
    ap.add_argument("--input", required=True, help="normal scenario_index.jsonl or selected_turns JSONL with turns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260722)
    ap.add_argument("--min_prefix_chars", type=int, default=2)
    ap.add_argument("--max_prefix_chars", type=int, default=12)
    ap.add_argument("--max_prefix_ratio", type=float, default=0.65)
    ap.add_argument("--deterministic_style", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out_rows: List[Dict[str, Any]] = []
    skipped = 0
    for row in rows:
        req = build_request(row, rng, args)
        if req is None:
            skipped += 1
            continue
        out_rows.append(req)
        if args.limit > 0 and len(out_rows) >= args.limit:
            break
    n = write_jsonl(Path(args.out), out_rows)
    stats = {"input": args.input, "out": args.out, "written": n, "skipped_before_limit": skipped, "seed": args.seed}
    Path(args.out).with_suffix(Path(args.out).suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
