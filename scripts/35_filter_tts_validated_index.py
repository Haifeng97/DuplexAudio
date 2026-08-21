#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SPECIAL_SCENARIOS = {"ai_intervenes_user", "player_complete"}
ACCEPTED_STATUSES = {"ok", "ok_short"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_statuses(paths: list[Path]) -> tuple[dict[str, str], Counter]:
    statuses: dict[str, str] = {}
    counts: Counter = Counter()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(path):
            task_id = str(row.get("task_id") or "")
            status = str(row.get("status") or "")
            if not task_id or not status:
                raise ValueError(f"invalid validation row in {path}")
            previous = statuses.get(task_id)
            if previous is not None and previous != status:
                raise ValueError(f"conflicting validation status for {task_id}: {previous} vs {status}")
            statuses[task_id] = status
            counts[status] += 1
    return statuses, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop whole scenarios when any referenced TTS task failed strict validation."
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--asr_results", action="append", required=True)
    parser.add_argument("--customized_out", required=True)
    parser.add_argument("--special_out", required=True)
    parser.add_argument("--rejected_out", required=True)
    args = parser.parse_args()

    statuses, validation_counts = load_statuses([Path(value) for value in args.asr_results])
    customized_out = Path(args.customized_out)
    special_out = Path(args.special_out)
    rejected_out = Path(args.rejected_out)
    for path in (customized_out, special_out, rejected_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter = Counter()
    with (
        customized_out.open("w", encoding="utf-8") as customized,
        special_out.open("w", encoding="utf-8") as special,
        rejected_out.open("w", encoding="utf-8") as rejected,
    ):
        for row in iter_jsonl(Path(args.index)):
            assets = row.get("tts_assets") if isinstance(row.get("tts_assets"), dict) else {}
            task_ids = sorted({
                str(asset.get("task_id") or "")
                for asset in assets.values()
                if isinstance(asset, dict) and asset.get("task_id")
            })
            failed = [
                {"task_id": task_id, "status": statuses.get(task_id, "missing_validation")}
                for task_id in task_ids
                if statuses.get(task_id) not in ACCEPTED_STATUSES
            ]
            if not task_ids:
                failed.append({"task_id": "", "status": "no_tts_assets"})
            if failed:
                write_row(rejected, {
                    "id": row.get("id"),
                    "scenario": row.get("scenario"),
                    "reason": "tts_validation_failed",
                    "failed_tasks": failed,
                })
                counts["rejected_scenarios"] += 1
                counts[f"rejected:{row.get('scenario')}"] += 1
                continue
            target = special if str(row.get("scenario") or "") in SPECIAL_SCENARIOS else customized
            write_row(target, row)
            counts["accepted_scenarios"] += 1
            counts[f"accepted:{row.get('scenario')}"] += 1

    stats = {
        "index": str(Path(args.index).resolve()),
        "validation_results": [str(Path(value).resolve()) for value in args.asr_results],
        "validation_task_counts": dict(validation_counts),
        "known_validation_tasks": len(statuses),
        "counts": dict(counts),
        "customized_out": str(customized_out.resolve()),
        "special_out": str(special_out.resolve()),
        "rejected_out": str(rejected_out.resolve()),
    }
    stats_path = rejected_out.with_suffix(rejected_out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
