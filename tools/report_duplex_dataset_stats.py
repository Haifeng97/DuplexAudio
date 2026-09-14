#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_manifest(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"manifest must be LABEL=PATH: {value}")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path).resolve()


def empty_stats() -> Dict[str, Any]:
    return {
        "rows": 0,
        "audio_duration_sec_sum": 0.0,
        "audio_duration_sec_min": None,
        "audio_duration_sec_max": None,
        "turn_count_sum": 0,
        "turn_count_min": None,
        "turn_count_max": None,
        "query_count": 0,
        "query_chars_sum": 0,
        "query_chars_min": None,
        "query_chars_max": None,
        "answer_count": 0,
        "answer_chars_sum": 0,
        "answer_chars_min": None,
        "answer_chars_max": None,
        "scenario_counts": Counter(),
        "label_counts": Counter(),
        "source_namespace_counts": Counter(),
        "source_dataset_counts": Counter(),
        "parse_errors": 0,
    }


def compact_chars(value: Any) -> int:
    return sum(1 for char in str(value or "") if not char.isspace())


def update_range(stats: Dict[str, Any], prefix: str, value: float | int) -> None:
    min_key = f"{prefix}_min"
    max_key = f"{prefix}_max"
    if stats[min_key] is None or value < stats[min_key]:
        stats[min_key] = value
    if stats[max_key] is None or value > stats[max_key]:
        stats[max_key] = value


def row_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_row = row.get("source_row")
    if isinstance(source_row, dict) and isinstance(source_row.get("turns"), list):
        return [turn for turn in source_row["turns"] if isinstance(turn, dict)]
    if isinstance(row.get("turns"), list):
        return [turn for turn in row["turns"] if isinstance(turn, dict)]
    if "question_text" in row or "answer_text" in row:
        return [{
            "question_text": row.get("question_text", ""),
            "answer_text": row.get("answer_text", ""),
        }]
    return []


def row_duration_sec(row: Dict[str, Any]) -> float:
    stats = row.get("stats")
    if isinstance(stats, dict) and stats.get("duration_sec") is not None:
        return float(stats["duration_sec"])
    timeline = row.get("timeline")
    if isinstance(timeline, list):
        return len(timeline) * float(row.get("chunk_ms") or 180) / 1000.0
    return 0.0


def add_row(stats: Dict[str, Any], row: Dict[str, Any]) -> None:
    stats["rows"] += 1
    duration = row_duration_sec(row)
    stats["audio_duration_sec_sum"] += duration
    update_range(stats, "audio_duration_sec", duration)

    turns = row_turns(row)
    turn_count = len(turns)
    stats["turn_count_sum"] += turn_count
    update_range(stats, "turn_count", turn_count)
    for turn in turns:
        if "question_text" in turn:
            length = compact_chars(turn.get("question_text"))
            stats["query_count"] += 1
            stats["query_chars_sum"] += length
            update_range(stats, "query_chars", length)
        if "answer_text" in turn:
            length = compact_chars(turn.get("answer_text"))
            stats["answer_count"] += 1
            stats["answer_chars_sum"] += length
            update_range(stats, "answer_chars", length)

    scenario = str(row.get("scenario") or "unknown")
    stats["scenario_counts"][scenario] += 1
    timeline = row.get("timeline")
    if isinstance(timeline, list):
        for entry in timeline:
            if not isinstance(entry, dict):
                continue
            raw_label = str(entry.get("label") or entry.get("token_text") or "UNKNOWN")
            if raw_label == "<EOR>":
                label = raw_label
            elif entry.get("label_type") == "text" or entry.get("kind") == "text":
                label = "TEXT"
            else:
                label = raw_label
            stats["label_counts"][label] += 1

    source_row = row.get("source_row")
    if not isinstance(source_row, dict):
        source_row = {}
    namespace = source_row.get("source_namespace") or row.get("source_namespace")
    if namespace:
        stats["source_namespace_counts"][str(namespace)] += 1
    meta = source_row.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    source_dataset = meta.get("dataset") or source_row.get("dataset")
    if source_dataset:
        stats["source_dataset_counts"][str(source_dataset)] += 1


def process_chunk(task: Tuple[str, str, int, int]) -> Tuple[str, Dict[str, Any]]:
    label, path, start, end = task
    stats = empty_stats()
    with open(path, "rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        while True:
            position = handle.tell()
            if position >= end:
                break
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row is not an object")
                add_row(stats, row)
            except Exception:
                stats["parse_errors"] += 1
    return label, stats


def merge_range(target: Dict[str, Any], source: Dict[str, Any], prefix: str) -> None:
    for suffix in ("min", "max"):
        key = f"{prefix}_{suffix}"
        value = source[key]
        if value is None:
            continue
        if target[key] is None:
            target[key] = value
        elif suffix == "min":
            target[key] = min(target[key], value)
        else:
            target[key] = max(target[key], value)


def merge_stats(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key in (
        "rows",
        "audio_duration_sec_sum",
        "turn_count_sum",
        "query_count",
        "query_chars_sum",
        "answer_count",
        "answer_chars_sum",
        "parse_errors",
    ):
        target[key] += source[key]
    for prefix in ("audio_duration_sec", "turn_count", "query_chars", "answer_chars"):
        merge_range(target, source, prefix)
    for key in (
        "scenario_counts",
        "label_counts",
        "source_namespace_counts",
        "source_dataset_counts",
    ):
        target[key].update(source[key])


def chunk_tasks(manifests: Iterable[Tuple[str, Path]], chunk_bytes: int) -> List[Tuple[str, str, int, int]]:
    tasks = []
    for label, path in manifests:
        size = path.stat().st_size
        parts = max(1, math.ceil(size / chunk_bytes))
        for index in range(parts):
            start = size * index // parts
            end = size * (index + 1) // parts
            tasks.append((label, str(path), start, end))
    return tasks


def finalized(stats: Dict[str, Any]) -> Dict[str, Any]:
    rows = stats["rows"]
    query_count = stats["query_count"]
    answer_count = stats["answer_count"]
    output = dict(stats)
    output["audio_total_hours"] = round(stats["audio_duration_sec_sum"] / 3600.0, 4)
    output["audio_duration_sec_avg"] = round(stats["audio_duration_sec_sum"] / rows, 4) if rows else 0.0
    output["turn_count_avg"] = round(stats["turn_count_sum"] / rows, 4) if rows else 0.0
    output["query_chars_avg"] = round(stats["query_chars_sum"] / query_count, 4) if query_count else 0.0
    output["answer_chars_avg"] = round(stats["answer_chars_sum"] / answer_count, 4) if answer_count else 0.0
    for key in (
        "audio_duration_sec_sum",
        "scenario_counts",
        "label_counts",
        "source_namespace_counts",
        "source_dataset_counts",
    ):
        if isinstance(output[key], Counter):
            output[key] = dict(sorted(output[key].items(), key=lambda item: (-item[1], item[0])))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Report turn, text, scenario, and label statistics for large duplex manifests.")
    parser.add_argument("--manifest", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk_mb", type=int, default=512)
    args = parser.parse_args()
    manifests = [parse_manifest(value) for value in args.manifest]
    tasks = chunk_tasks(manifests, args.chunk_mb * 1024 * 1024)
    datasets = {label: empty_stats() for label, _ in manifests}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for label, partial in executor.map(process_chunk, tasks, chunksize=1):
            merge_stats(datasets[label], partial)
    overall = empty_stats()
    for stats in datasets.values():
        merge_stats(overall, stats)
    result = {
        "schema_version": "duplex_dataset_inventory_v1",
        "length_definition": "Unicode characters excluding whitespace; punctuation retained; measured per turn",
        "label_definition": "All timeline entries with label_type=text are TEXT; other entries retain their control label",
        "manifests": {label: str(path) for label, path in manifests},
        "datasets": {label: finalized(stats) for label, stats in datasets.items()},
        "combined": finalized(overall),
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output),
        "rows": result["combined"]["rows"],
        "audio_total_hours": result["combined"]["audio_total_hours"],
        "parse_errors": result["combined"]["parse_errors"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
