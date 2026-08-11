#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple


READY_STATUSES = {"ok", "cached"}


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def parse_scenario(value: str) -> Tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--scenario must be NAME=SCENARIO_INDEX.jsonl")
    return name.strip(), Path(raw_path).resolve()


def required_tasks(row: Dict[str, Any]) -> set[str]:
    assets = row.get("tts_assets")
    if not isinstance(assets, dict):
        return set()
    return {
        str(asset["task_id"])
        for asset in assets.values()
        if isinstance(asset, dict) and asset.get("task_id")
    }


def asset_quality_errors(
    row: Dict[str, Any],
    *,
    min_audio_sec: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
) -> Counter:
    errors: Counter = Counter()
    assets = row.get("tts_assets")
    if not isinstance(assets, dict):
        return errors
    for asset in assets.values():
        if not isinstance(asset, dict) or not asset.get("task_id"):
            continue
        path = Path(str(asset.get("audio") or ""))
        try:
            with wave.open(str(path), "rb") as wav:
                sample_rate = wav.getframerate()
                duration_sec = wav.getnframes() / sample_rate if sample_rate else 0.0
        except (EOFError, FileNotFoundError, OSError, wave.Error):
            errors["invalid_wav"] += 1
            continue
        if duration_sec < min_audio_sec:
            errors["audio_too_short"] += 1
        text_chars = len(str(asset.get("text") or "").strip())
        max_audio_sec = max(max_audio_floor_sec, text_chars * max_sec_per_char)
        if max_sec_per_char > 0 and duration_sec > max_audio_sec:
            errors["audio_too_long_for_text"] += 1
    return errors


def load_ready_tasks(results_dir: Path, pattern: str) -> Tuple[set[str], Counter]:
    ready: set[str] = set()
    counts: Counter = Counter()
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no result files matched {results_dir / pattern}")
    for path in paths:
        for row in read_jsonl(path):
            status = str(row.get("status") or "missing_status")
            counts[status] += 1
            task_id = str(row.get("id") or "")
            if task_id and status in READY_STATUSES:
                ready.add(task_id)
    return ready, counts


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def snapshot_scenario(
    name: str,
    source: Path,
    out_dir: Path,
    ready_tasks: set[str],
    sample_handle: Any,
    min_audio_sec: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
) -> Dict[str, Any]:
    target = out_dir / name / "scenario_index.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    input_rows = ready_rows = required_task_count = 0
    missing_tasks: Counter = Counter()
    quality_rejections: Counter = Counter()
    with target.open("w", encoding="utf-8") as output:
        for row in read_jsonl(source):
            input_rows += 1
            task_ids = required_tasks(row)
            required_task_count += len(task_ids)
            missing = task_ids - ready_tasks
            quality_errors = asset_quality_errors(
                row,
                min_audio_sec=min_audio_sec,
                max_audio_floor_sec=max_audio_floor_sec,
                max_sec_per_char=max_sec_per_char,
            ) if not missing else Counter()
            if missing or quality_errors or not task_ids:
                for task_id in missing:
                    missing_tasks[task_id] += 1
                quality_rejections.update(quality_errors)
                continue
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            sample_handle.write(json.dumps({
                "id": row.get("id"),
                "scenario": row.get("scenario"),
                "snapshot_group": name,
                "source_group_id": row.get("source_group_id"),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
            ready_rows += 1
    return {
        "source": str(source),
        "out": str(target.resolve()),
        "input_rows": input_rows,
        "ready_rows": ready_rows,
        "unready_rows": input_rows - ready_rows,
        "required_tasks": required_task_count,
        "distinct_missing_tasks": len(missing_tasks),
        "quality_rejections": dict(quality_rejections),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze scenario rows whose complete TTS asset set is present in a result snapshot."
    )
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--result_pattern", default="tts_results_*.jsonl")
    parser.add_argument("--scenario", action="append", type=parse_scenario, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_audio_sec", type=float, default=0.0)
    parser.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    parser.add_argument(
        "--max_sec_per_char",
        type=float,
        default=0.0,
        help="Reject task audio longer than max(max_audio_floor_sec, text chars * this value); 0 disables.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ready_tasks, result_counts = load_ready_tasks(results_dir, args.result_pattern)

    task_snapshot = out_dir / "ready_task_ids.jsonl"
    write_jsonl(task_snapshot, ({"id": task_id} for task_id in sorted(ready_tasks)))

    scenario_stats: Dict[str, Any] = {}
    sample_snapshot = out_dir / "ready_samples.jsonl"
    with sample_snapshot.open("w", encoding="utf-8") as sample_handle:
        for name, source in args.scenario:
            if name in scenario_stats:
                raise ValueError(f"duplicate scenario name: {name}")
            scenario_stats[name] = snapshot_scenario(
                name,
                source,
                out_dir,
                ready_tasks,
                sample_handle,
                args.min_audio_sec,
                args.max_audio_floor_sec,
                args.max_sec_per_char,
            )

    stats = {
        "results_dir": str(results_dir),
        "result_pattern": args.result_pattern,
        "result_status_counts": dict(result_counts),
        "ready_task_ids": len(ready_tasks),
        "audio_quality": {
            "min_audio_sec": args.min_audio_sec,
            "max_audio_floor_sec": args.max_audio_floor_sec,
            "max_sec_per_char": args.max_sec_per_char,
        },
        "task_snapshot": str(task_snapshot.resolve()),
        "sample_snapshot": str(sample_snapshot.resolve()),
        "scenarios": scenario_stats,
    }
    stats_path = out_dir / "snapshot_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
