#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


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


def normal_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    turn = first_turn(row) or {}
    return {
        "id": row["id"],
        "scenario": "normal_qa",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "question_text": turn.get("question_text", row.get("question_text", "")),
        "answer_text": turn.get("answer_text", row.get("answer_text", "")),
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
        "intended_timeline": [
            "base question TTS -> D_WAIT",
            "base answer prefix over gaussian -> A_ANSWER + prefix tokens",
            "donor first question chunk -> G_INTERRUPT",
            "donor remaining question TTS -> D_WAIT",
            "donor answer over gaussian -> A_ANSWER + donor answer tokens + EOR",
        ],
    }


def incomplete_candidate(row: Dict[str, Any], rng: random.Random, min_prefix_chars: int) -> Dict[str, Any] | None:
    turn = first_turn(row)
    if not turn:
        return None
    q = str(turn.get("question_text", ""))
    if len(q) < max(min_prefix_chars + 2, 8):
        return None
    hi = max(min_prefix_chars + 1, min(len(q) - 1, int(len(q) * 0.7)))
    if hi <= min_prefix_chars:
        return None
    cut = rng.randint(min_prefix_chars, hi)
    return {
        "id": f"incomplete__{row['id']}__cut{cut}",
        "scenario": "incomplete_query_candidate",
        "source_id": row["id"],
        "sysprompt": row.get("sysprompt", ""),
        "partial_question_text": q[:cut],
        "full_question_text": q,
        "answer_text_if_complete": turn.get("answer_text", ""),
        "label_policy_tbd": "During partial speech chunks use D_WAIT; post-partial pause label still needs protocol decision: D_WAIT vs IDLE.",
        "meta": row.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build small candidate pools for normal QA, player interrupt, and incomplete-query scenarios.")
    ap.add_argument("--input", required=True, help="Selected turns JSONL from 00_select_duplex_turns.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_each", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--min_interrupt_answer_chars", type=int, default=8)
    ap.add_argument("--min_incomplete_prefix_chars", type=int, default=3)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(Path(args.input))
    out_dir = Path(args.out_dir)

    normal_pool = [r for r in rows if r.get("selection", {}).get("can_normal")]
    interrupt_base_pool = [
        r for r in rows
        if r.get("selection", {}).get("can_interrupt_base")
        and len(str((first_turn(r) or {}).get("answer_text", ""))) >= args.min_interrupt_answer_chars
    ]
    interrupt_donor_pool = [r for r in rows if r.get("selection", {}).get("can_interrupt_donor")]
    incomplete_pool = [r for r in rows if r.get("selection", {}).get("can_incomplete_query")]

    rng.shuffle(normal_pool)
    rng.shuffle(interrupt_base_pool)
    rng.shuffle(interrupt_donor_pool)
    rng.shuffle(incomplete_pool)

    normal_rows = [normal_candidate(r) for r in normal_pool[: args.limit_each]]

    interrupt_rows = []
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
        "rows": len(rows),
        "normal_pool": len(normal_pool),
        "interrupt_base_pool": len(interrupt_base_pool),
        "interrupt_donor_pool": len(interrupt_donor_pool),
        "incomplete_pool": len(incomplete_pool),
        "normal_written": write_jsonl(out_dir / "normal_qa_candidates.jsonl", normal_rows),
        "interrupt_written": write_jsonl(out_dir / "player_interrupt_candidates.jsonl", interrupt_rows),
        "incomplete_written": write_jsonl(out_dir / "incomplete_query_candidates.jsonl", incomplete_rows),
    }
    (out_dir / "candidate_pool_stats.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
