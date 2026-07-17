#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


BACKCHANNEL_TEMPLATES = [
    "嗯",
    "嗯嗯",
    "好",
    "好嘞",
    "行",
    "可以",
    "收到",
    "明白",
    "知道了",
    "对",
    "是的",
    "你继续",
    "没事你说",
    "我听着呢",
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


def first_turn(row: Dict[str, Any]) -> Dict[str, Any] | None:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        turn = turns[-1]
        if isinstance(turn, dict):
            return turn
    return None


def has_train_history(row: Dict[str, Any]) -> bool:
    turns = row.get("turns")
    return isinstance(turns, list) and len(turns) >= 2


def has_sysprompt_history(row: Dict[str, Any]) -> bool:
    meta = row.get("meta")
    if isinstance(meta, dict) and "sysprompt_ds_history_present" in meta:
        return bool(meta.get("sysprompt_ds_history_present"))
    sysprompt = str(row.get("sysprompt") or "")
    return "<｜User｜>" in sysprompt and "<｜Assistant｜>" in sysprompt


def answer_gn_chunks(answer_text: str) -> int:
    return max(1, int(len(answer_text) * 1.1 + 0.999999))


def normal_candidate(row: Dict[str, Any], chunk_ms: int) -> Dict[str, Any]:
    turn = first_turn(row) or {}
    answer_text = str(turn.get("answer_text", row.get("answer_text", "")))
    return {
        "id": row["id"],
        "scenario": "normal_qa",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "question_text": turn.get("question_text", row.get("question_text", "")),
        "answer_text": answer_text,
        "audio_plan": [
            "gn_before",
            "query_audio",
            "gn_answer_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_audio -> WAIT",
            "gn_answer_region -> ANSWER + answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "chunk_ms": chunk_ms,
            "answer_gn_chunks": answer_gn_chunks(answer_text),
            "answer_gn_duration_sec": round(answer_gn_chunks(answer_text) * chunk_ms / 1000.0, 6),
            "answer_gn_formula": "ceil(len(answer_text) * 1.1) chunks",
        },
        "meta": row.get("meta", {}),
    }


def interrupt_candidate(base: Dict[str, Any], donor: Dict[str, Any], prefix_chars: int) -> Dict[str, Any]:
    base_turn = first_turn(base) or {}
    donor_turn = first_turn(donor) or {}
    base_answer = str(base_turn.get("answer_text", ""))
    prefix_chars = max(1, min(prefix_chars, max(1, len(base_answer) - 1)))
    return {
        "id": f"interrupt__{base['id']}__{donor['id']}",
        "scenario": "player_interrupts_ai",
        "source": "candidate_pair_only_no_timeline_yet",
        "base": {
            "id": base["id"],
            "sysprompt": base.get("sysprompt", ""),
            "question_text": base_turn.get("question_text", ""),
            "answer_text": base_answer,
            "answer_prefix_text": base_answer[:prefix_chars],
            "prefix_chars": prefix_chars,
            "meta": base.get("meta", {}),
        },
        "donor": {
            "id": donor["id"],
            "sysprompt": donor.get("sysprompt", ""),
            "question_text": donor_turn.get("question_text", ""),
            "answer_text": donor_turn.get("answer_text", ""),
            "meta": donor.get("meta", {}),
        },
        "audio_plan": [
            "gn_before",
            "base_query_audio",
            "gn_base_answer_prefix_region",
            "donor_query_audio",
            "gn_donor_answer_region",
            "gn_after",
        ],
        "intended_timeline": [
            "gn_before -> IDLE",
            "base question TTS -> D_WAIT",
            "short gaussian region -> A_ANSWER + base answer prefix tokens, no EOR",
            "donor question starts while base answer is unfinished -> D_WAIT",
            "donor remaining question TTS -> D_WAIT",
            "donor answer over gaussian -> A_ANSWER + donor answer tokens + EOR",
            "gn_after -> IDLE",
        ],
    }


def same_row_interrupt_candidate(row: Dict[str, Any], prefix_chars: int) -> Dict[str, Any] | None:
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    base_turn = turns[-2]
    donor_turn = turns[-1]
    if not isinstance(base_turn, dict) or not isinstance(donor_turn, dict):
        return None
    base_answer = str(base_turn.get("answer_text", ""))
    donor_question = str(donor_turn.get("question_text", ""))
    donor_answer = str(donor_turn.get("answer_text", ""))
    if not base_answer or not donor_question or not donor_answer:
        return None
    prefix_chars = max(1, min(prefix_chars, max(1, len(base_answer) - 1)))
    return {
        "id": f"interrupt_same_row__{row['id']}__t{base_turn.get('turn_id', 'prev')}_t{donor_turn.get('turn_id', 'next')}",
        "scenario": "player_interrupts_ai",
        "source": "same_row_previous_turn_interrupted_by_next_turn",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "base": {
            "id": row["id"],
            "turn_id": base_turn.get("turn_id"),
            "sysprompt": row.get("sysprompt", ""),
            "question_text": base_turn.get("question_text", ""),
            "answer_text": base_answer,
            "answer_prefix_text": base_answer[:prefix_chars],
            "prefix_chars": prefix_chars,
            "meta": row.get("meta", {}),
        },
        "donor": {
            "id": row["id"],
            "turn_id": donor_turn.get("turn_id"),
            "sysprompt": row.get("sysprompt", ""),
            "question_text": donor_question,
            "answer_text": donor_answer,
            "meta": row.get("meta", {}),
        },
        "audio_plan": [
            "gn_before",
            "base_query_audio",
            "gn_base_answer_prefix_region",
            "donor_query_audio",
            "gn_donor_answer_region",
            "gn_after",
        ],
        "intended_timeline": [
            "gn_before -> IDLE",
            "previous turn question TTS -> WAIT",
            "short gaussian region -> ANSWER + previous answer prefix tokens, no EOR",
            "next turn question starts while previous answer is unfinished -> INTERRUPT then WAIT",
            "next turn answer over gaussian -> ANSWER + next answer tokens + EOR",
            "gn_after -> IDLE",
        ],
        "meta": row.get("meta", {}),
    }


def backchannel_candidate(row: Dict[str, Any], backchannel_text: str, prefix_chars: int, chunk_ms: int) -> Dict[str, Any]:
    turn = first_turn(row) or {}
    answer_text = str(turn.get("answer_text", ""))
    prefix_chars = max(1, min(prefix_chars, max(1, len(answer_text) - 1)))
    answer_prefix = answer_text[:prefix_chars]
    answer_remaining = answer_text[prefix_chars:]
    return {
        "id": f"backchannel__{row['id']}__p{prefix_chars}",
        "scenario": "player_backchannel",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "question_text": turn.get("question_text", ""),
        "answer_text": answer_text,
        "answer_prefix_text": answer_prefix,
        "answer_remaining_text": answer_remaining,
        "prefix_chars": prefix_chars,
        "backchannel_text": backchannel_text,
        "audio_plan": [
            "gn_before",
            "query_audio",
            "gn_answer_prefix_region",
            "backchannel_audio",
            "gn_answer_remaining_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_audio -> WAIT",
            "gn_answer_prefix_region -> ANSWER + answer prefix tokens, no EOR",
            "player backchannel audio -> WAIT",
            "gn_answer_remaining_region -> remaining answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "chunk_ms": chunk_ms,
            "answer_prefix_gn_chunks": answer_gn_chunks(answer_prefix),
            "answer_remaining_gn_chunks": answer_gn_chunks(answer_remaining),
            "answer_split_policy": "random prefix, remaining answer continues after player backchannel",
        },
        "meta": row.get("meta", {}),
    }


def choose_query_split(q: str, rng: random.Random, min_prefix_chars: int) -> int | None:
    if len(q) < max(min_prefix_chars + 2, 8):
        return None
    lo = min_prefix_chars
    hi = max(min_prefix_chars + 1, min(len(q) - 1, int(len(q) * 0.7)))
    if hi <= lo:
        return None
    return rng.randint(lo, hi)


def incomplete_candidate(row: Dict[str, Any], rng: random.Random, min_prefix_chars: int) -> Dict[str, Any] | None:
    turn = first_turn(row)
    if not turn:
        return None
    q = str(turn.get("question_text", ""))
    cut = choose_query_split(q, rng, min_prefix_chars)
    if cut is None:
        return None
    gn_between_sec = round(rng.uniform(0.5, 2.0), 3)
    part1 = q[:cut]
    part2 = q[cut:]
    return {
        "id": f"incomplete__{row['id']}__cut{cut}",
        "scenario": "incomplete_query_candidate",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "query_part1_text": part1,
        "query_part2_text": part2,
        "partial_question_text": part1,
        "full_question_text": q,
        "answer_text_if_complete": turn.get("answer_text", ""),
        "split": {
            "cut_char_index": cut,
            "unicode_codepoint_boundary": True,
        },
        "audio_plan": [
            "gn_before",
            "query_part1_audio",
            "gn_between_query_parts",
            "query_part2_audio",
            "gn_answer_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_part1_audio -> WAIT",
            "gn_between_query_parts -> WAIT",
            "query_part2_audio -> WAIT",
            "gn_answer_region -> ANSWER + answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "between_query_parts_sec": gn_between_sec,
            "between_query_parts_range_sec": [0.5, 2.0],
        },
        "meta": row.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build small candidate pools for normal QA, player interrupt, and incomplete-query scenarios.")
    ap.add_argument("--input", required=True, help="Selected turns JSONL from 00_select_duplex_turns.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_each", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--min_interrupt_answer_chars", type=int, default=8)
    ap.add_argument("--min_backchannel_answer_chars", type=int, default=8)
    ap.add_argument("--min_incomplete_prefix_chars", type=int, default=3)
    ap.add_argument("--backchannel_templates", default=",".join(BACKCHANNEL_TEMPLATES))
    ap.add_argument("--chunk_ms", type=int, default=180)
    ap.add_argument("--require_history", action="store_true")
    ap.add_argument("--require_sysprompt_history", action="store_true")
    ap.add_argument("--interrupt_pair_mode", choices=["cross_row", "same_row_previous"], default="cross_row")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    input_rows = read_jsonl(Path(args.input))
    rows = list(input_rows)
    if args.require_history:
        rows = [r for r in rows if has_train_history(r)]
    if args.require_sysprompt_history:
        rows = [r for r in rows if has_sysprompt_history(r)]
    out_dir = Path(args.out_dir)

    normal_pool = [r for r in rows if r.get("selection", {}).get("can_normal")]
    interrupt_source_rows = [r for r in rows if has_train_history(r)]
    interrupt_base_pool = [
        r for r in interrupt_source_rows
        if r.get("selection", {}).get("can_interrupt_base")
        and len(str((first_turn(r) or {}).get("answer_text", ""))) >= args.min_interrupt_answer_chars
    ]
    same_row_interrupt_pool = [
        r for r in interrupt_source_rows
        if r.get("selection", {}).get("can_interrupt_base")
        and len(str((r.get("turns") or [])[-2].get("answer_text", ""))) >= args.min_interrupt_answer_chars
    ]
    interrupt_donor_pool = [r for r in interrupt_source_rows if r.get("selection", {}).get("can_interrupt_donor")]
    backchannel_pool = [
        r for r in rows
        if r.get("selection", {}).get("can_normal")
        and len(str((first_turn(r) or {}).get("answer_text", ""))) >= args.min_backchannel_answer_chars
    ]
    incomplete_pool = [r for r in rows if r.get("selection", {}).get("can_incomplete_query")]

    rng.shuffle(normal_pool)
    rng.shuffle(interrupt_base_pool)
    rng.shuffle(same_row_interrupt_pool)
    rng.shuffle(interrupt_donor_pool)
    rng.shuffle(backchannel_pool)
    rng.shuffle(incomplete_pool)

    normal_rows = [normal_candidate(r, args.chunk_ms) for r in normal_pool[: args.limit_each]]

    interrupt_rows = []
    if args.interrupt_pair_mode == "same_row_previous":
        for row in same_row_interrupt_pool:
            turns = row.get("turns") or []
            base_answer = str(turns[-2].get("answer_text", ""))
            max_prefix = min(len(base_answer) - 1, 12)
            prefix_chars = rng.randint(1, max(1, max_prefix))
            cand = same_row_interrupt_candidate(row, prefix_chars)
            if cand is not None:
                interrupt_rows.append(cand)
            if len(interrupt_rows) >= args.limit_each:
                break
    else:
        pair_count = min(args.limit_each, len(interrupt_base_pool), len(interrupt_donor_pool))
        for i in range(pair_count):
            base = interrupt_base_pool[i]
            donor = interrupt_donor_pool[-(i + 1)]
            if base["id"] == donor["id"] and len(interrupt_donor_pool) > 1:
                donor = interrupt_donor_pool[-(i + 2)]
            base_answer = str((first_turn(base) or {}).get("answer_text", ""))
            max_prefix = min(len(base_answer) - 1, 12)
            prefix_chars = rng.randint(1, max(1, max_prefix))
            interrupt_rows.append(interrupt_candidate(base, donor, prefix_chars))

    templates = [x.strip() for x in str(args.backchannel_templates).split(",") if x.strip()]
    if not templates:
        templates = BACKCHANNEL_TEMPLATES
    backchannel_rows = []
    for row in backchannel_pool:
        answer_text = str((first_turn(row) or {}).get("answer_text", ""))
        max_prefix = min(len(answer_text) - 1, 12)
        prefix_chars = rng.randint(1, max(1, max_prefix))
        backchannel_text = templates[len(backchannel_rows) % len(templates)]
        backchannel_rows.append(backchannel_candidate(row, backchannel_text, prefix_chars, args.chunk_ms))
        if len(backchannel_rows) >= args.limit_each:
            break

    incomplete_rows = []
    for row in incomplete_pool:
        cand = incomplete_candidate(row, rng, args.min_incomplete_prefix_chars)
        if cand is not None:
            incomplete_rows.append(cand)
        if len(incomplete_rows) >= args.limit_each:
            break

    counts = {
        "input": str(Path(args.input)),
        "out_dir": str(out_dir),
        "input_rows": len(input_rows),
        "rows": len(rows),
        "require_history": bool(args.require_history),
        "require_sysprompt_history": bool(args.require_sysprompt_history),
        "interrupt_pair_mode": args.interrupt_pair_mode,
        "interrupt_requires_train_history": True,
        "interrupt_source_rows": len(interrupt_source_rows),
        "normal_pool": len(normal_pool),
        "interrupt_base_pool": len(interrupt_base_pool),
        "same_row_interrupt_pool": len(same_row_interrupt_pool),
        "interrupt_donor_pool": len(interrupt_donor_pool),
        "backchannel_pool": len(backchannel_pool),
        "incomplete_pool": len(incomplete_pool),
        "normal_written": write_jsonl(out_dir / "normal_qa_candidates.jsonl", normal_rows),
        "interrupt_written": write_jsonl(out_dir / "player_interrupt_candidates.jsonl", interrupt_rows),
        "backchannel_written": write_jsonl(out_dir / "player_backchannel_candidates.jsonl", backchannel_rows),
        "incomplete_written": write_jsonl(out_dir / "incomplete_query_candidates.jsonl", incomplete_rows),
    }
    (out_dir / "candidate_pool_stats.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
