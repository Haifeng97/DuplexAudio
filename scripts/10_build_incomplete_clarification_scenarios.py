#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
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


def clean_turn(turn: Dict[str, Any], turn_id: int) -> Dict[str, Any]:
    return {
        "turn_id": turn_id,
        "source": turn.get("source", "current"),
        "question_text": str(turn.get("question_text") or ""),
        "answer_text": str(turn.get("answer_text") or ""),
        "needs_tts": bool(turn.get("needs_tts", True)),
        "train_answer": bool(turn.get("train_answer", True)),
    }


def build_scenario(row: Dict[str, Any], answer_field: str) -> Dict[str, Any] | None:
    answer = str(row.get(answer_field) or row.get("llm_answer_text") or row.get("answer_text") or "").strip()
    if not answer:
        return None
    partial = str(row.get("partial_question_text") or "").strip()
    if not partial:
        return None
    context = row.get("context_turns") if isinstance(row.get("context_turns"), list) else []
    following = row.get("following_turns") if isinstance(row.get("following_turns"), list) else []
    turns: List[Dict[str, Any]] = []
    for turn in context:
        if isinstance(turn, dict):
            turns.append(clean_turn(turn, len(turns) + 1))
    inserted_turn_id = len(turns) + 1
    turns.append({
        "turn_id": inserted_turn_id,
        "source": "inserted_incomplete_query",
        "question_text": partial,
        "answer_text": answer,
        "needs_tts": True,
        "train_answer": True,
        "full_question_text": row.get("full_question_text", ""),
        "original_answer_text": row.get("original_answer_text", ""),
        "split": row.get("split", {}),
    })
    for turn in following:
        if isinstance(turn, dict):
            turns.append(clean_turn(turn, len(turns) + 1))
    source_id = str(row.get("source_id") or row.get("id"))
    scenario_id = str(row.get("id") or f"incomplete_clarify__{source_id}").replace("_request", "")
    if not scenario_id.startswith("incomplete_clarify__"):
        scenario_id = f"incomplete_clarify__{scenario_id}"
    return {
        "id": scenario_id,
        "scenario": "incomplete_query_clarification",
        "source_id": source_id,
        "source_request_id": row.get("id"),
        "sysprompt": row.get("sysprompt", ""),
        "turns": turns,
        "question_text": partial,
        "answer_text": answer,
        "partial_question_text": partial,
        "full_question_text": row.get("full_question_text", ""),
        "clarification_answer_text": answer,
        "inserted_turn_id": inserted_turn_id,
        "force_inter_turn_idle": True,
        "audio_plan": [
            "gn_before",
            "turn query audio + answer GN for each complete turn",
            "3-5s WAIT after the inserted partial query before its clarification answer",
            "1-3s gn_between_turns after each EOR before next turn",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "each query audio -> WAIT",
            "inserted partial query -> 3-5s WAIT before ANSWER",
            "each answer GN -> ANSWER + answer text tokens + EOR",
            "between turns -> IDLE",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "clarification_wait_range_sec": [3.0, 5.0],
            "between_turn_idle_range_sec": [1.0, 3.0],
            "force_inter_turn_idle": True,
        },
        "meta": row.get("meta", {}),
        "source_request": row,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build incomplete-query clarification scenario index rows from filled LLM request JSONL.")
    ap.add_argument("--input", required=True, help="JSONL from 09_export... with clarification answers filled")
    ap.add_argument("--out", required=True)
    ap.add_argument("--answer_field", default="clarification_answer_text")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.input))
    out_rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for row in rows:
        built = build_scenario(row, args.answer_field)
        if built is None:
            skipped.append({"id": row.get("id"), "error": "missing_partial_or_clarification_answer"})
            continue
        out_rows.append(built)
    n = write_jsonl(Path(args.out), out_rows)
    skipped_path = Path(args.out).with_suffix(Path(args.out).suffix + ".skipped.jsonl")
    if skipped:
        write_jsonl(skipped_path, skipped)
    stats = {"input": args.input, "out": args.out, "written": n, "skipped": len(skipped), "skipped_path": str(skipped_path) if skipped else ""}
    Path(args.out).with_suffix(Path(args.out).suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
