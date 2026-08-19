#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FWAIT = "<FD_F_WAIT>"


def iter_manifest(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text in {"[", "]"}:
                continue
            if text.endswith(","):
                text = text[:-1]
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def fwait_turn_id(row: dict[str, Any]) -> int:
    timeline = row.get("timeline") or []
    for item in reversed(timeline):
        if isinstance(item, dict) and item.get("label") == FWAIT:
            return int(item.get("turn_id") or 0)
    raise ValueError("timeline has no F_WAIT")


def turn_for_id(row: dict[str, Any], turn_id: int) -> tuple[int, dict[str, Any]]:
    turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    for index, turn in enumerate(turns):
        if int(turn.get("turn_id") or index + 1) == turn_id:
            return index, turn
    raise ValueError(f"turn_id={turn_id} is missing from turns")


def qa_stats_turn(row: dict[str, Any], turn_id: int) -> dict[str, Any]:
    for turn in ((row.get("stats") or {}).get("turns") or []):
        if isinstance(turn, dict) and int(turn.get("turn_id") or 0) == turn_id:
            return turn
    raise ValueError(f"turn_id={turn_id} is missing from stats.turns")


def prepare_gcp(row: dict[str, Any], chunk_sec: float) -> dict[str, Any]:
    event = row["fgh_event"]
    start_idx, end_idx = map(int, event["source_wait_span"])
    cut_idx = int(event["cut_source_idx"])
    turn_id = fwait_turn_id(row)
    _, turn = turn_for_id(row, turn_id)
    if not start_idx <= cut_idx <= end_idx:
        raise ValueError("cut_source_idx is outside source_wait_span")
    return {
        "schema_version": "external_fwait_alignment_task_v1",
        "task_id": f"gcp_pass::{row['id']}",
        "dataset": "gcp_pass",
        "source_id": row["id"],
        "turn_id": turn_id,
        "full_query": str(turn.get("question_text") or "").strip(),
        "audio_path": str(row.get("orig_audio") or ""),
        "audio_source": "orig_audio_query_segment",
        "audio_start_sec": start_idx * chunk_sec,
        "audio_end_sec": (end_idx + 1) * chunk_sec,
        "audio_duration_hint_sec": (end_idx - start_idx + 1) * chunk_sec,
        "cut_sec": (cut_idx - start_idx + 1) * chunk_sec,
        "source_wait_span": [start_idx, end_idx],
        "cut_source_idx": cut_idx,
        "chunk_sec": chunk_sec,
    }


def prepare_qa(row: dict[str, Any], chunk_sec: float) -> dict[str, Any]:
    event = row["fgh_event"]
    start_idx, end_idx = map(int, event["source_wait_span"])
    cut_idx = int(event["cut_source_idx"])
    turn_id = fwait_turn_id(row)
    turn_index, turn = turn_for_id(row, turn_id)
    stats_turn = qa_stats_turn(row, turn_id)
    question_start = float(stats_turn["question_start_sec"])
    audio_path = str(stats_turn.get("question_audio") or "")
    if not audio_path:
        paths = row.get("source_question_audios") or []
        if turn_index < len(paths):
            audio_path = str(paths[turn_index])
    if not start_idx <= cut_idx <= end_idx:
        raise ValueError("cut_source_idx is outside source_wait_span")
    cut_sec = (cut_idx + 1) * chunk_sec - question_start
    return {
        "schema_version": "external_fwait_alignment_task_v1",
        "task_id": f"qa_fd::{row['id']}",
        "dataset": "qa_fd",
        "source_id": row["id"],
        "turn_id": turn_id,
        "full_query": str(turn.get("question_text") or "").strip(),
        "audio_path": audio_path,
        "audio_source": "standalone_question_audio",
        "audio_start_sec": 0.0,
        "audio_end_sec": None,
        "audio_duration_hint_sec": float(stats_turn.get("question_audio_sec") or 0.0),
        "cut_sec": cut_sec,
        "source_wait_span": [start_idx, end_idx],
        "cut_source_idx": cut_idx,
        "question_start_sec": question_start,
        "chunk_sec": chunk_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exact-audio tasks for external F_WAIT text recovery.")
    parser.add_argument("--gcp_manifest", required=True)
    parser.add_argument("--qa_manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats", default="")
    parser.add_argument("--limit_per_dataset", type=int, default=0)
    parser.add_argument("--expected_gcp", type=int, default=0)
    parser.add_argument("--expected_qa", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    specs = [
        ("gcp_pass", Path(args.gcp_manifest), prepare_gcp, args.expected_gcp),
        ("qa_fd", Path(args.qa_manifest), prepare_qa, args.expected_qa),
    ]
    with out.open("w", encoding="utf-8") as output:
        for dataset, manifest, prepare, expected in specs:
            written = 0
            for row in iter_manifest(manifest):
                event = row.get("fgh_event") or {}
                if event.get("event") != "F_WAIT":
                    continue
                counts[f"{dataset}_fwait_seen"] += 1
                try:
                    task = prepare(row, float(row.get("chunk_ms") or 180) / 1000.0)
                    if not task["full_query"]:
                        raise ValueError("empty full_query")
                    if not task["audio_path"]:
                        raise ValueError("empty audio_path")
                    if task["cut_sec"] <= 0:
                        raise ValueError(f"invalid cut_sec={task['cut_sec']}")
                    if task["task_id"] in seen:
                        raise ValueError("duplicate task_id")
                    seen.add(task["task_id"])
                    output.write(json.dumps(task, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[f"{dataset}_written"] += 1
                    counts["written"] += 1
                    written += 1
                except Exception as exc:
                    counts[f"{dataset}_error"] += 1
                    counts["error"] += 1
                    if len(errors) < 100:
                        errors.append({"dataset": dataset, "id": str(row.get("id") or ""), "error": str(exc)})
                if args.limit_per_dataset and written >= args.limit_per_dataset:
                    break
                if expected and counts[f"{dataset}_fwait_seen"] >= expected:
                    break
            if expected and counts[f"{dataset}_fwait_seen"] != expected:
                raise RuntimeError(f"{dataset}: expected {expected} F_WAIT rows, saw {counts[f'{dataset}_fwait_seen']}")

    stats = {
        "schema_version": "external_fwait_alignment_prepare_stats_v1",
        "output": str(out.resolve()),
        "counts": dict(counts),
        "errors": errors,
    }
    stats_path = Path(args.stats) if args.stats else out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
