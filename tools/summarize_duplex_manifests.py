#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"manifest must be LABEL=PATH: {value}")
    label, path = value.split("=", 1)
    return label, Path(path).resolve()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield row


def duration_sec(row: Dict[str, Any]) -> float:
    stats = row.get("stats") or {}
    if stats.get("duration_sec") is not None:
        return float(stats["duration_sec"])
    sample_rate = int(row.get("sample_rate") or 0)
    samples = int(stats.get("audio_samples") or 0)
    return samples / sample_rate if sample_rate else 0.0


def percentile(sorted_values: array, fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)


def duration_bin(duration: float) -> str:
    if duration < 30:
        return "lt_30s"
    if duration < 60:
        return "30s_to_1m"
    if duration < 120:
        return "1m_to_2m"
    if duration < 180:
        return "2m_to_3m"
    if duration < 240:
        return "3m_to_4m"
    if duration < 300:
        return "4m_to_5m"
    return "gte_5m"


class Aggregate:
    def __init__(self) -> None:
        self.durations = array("d")
        self.total_sec = 0.0
        self.duration_bins: Counter = Counter()
        self.turn_counts: Counter = Counter()

    def add(self, row: Dict[str, Any], duration: float) -> None:
        self.durations.append(duration)
        self.total_sec += duration
        self.duration_bins[duration_bin(duration)] += 1
        source_row = row.get("source_row") or {}
        turns = source_row.get("turns")
        if isinstance(turns, list):
            self.turn_counts[str(len(turns))] += 1
        else:
            meta = source_row.get("meta") or {}
            turn_count = meta.get("turn_count")
            self.turn_counts[str(turn_count) if turn_count is not None else "unknown"] += 1

    def summary(self) -> Dict[str, Any]:
        values = self.durations
        values = array("d", sorted(values))
        count = len(values)
        return {
            "rows": count,
            "total_hours": round(self.total_sec / 3600, 4),
            "mean_sec": round(self.total_sec / count, 3) if count else 0.0,
            "p50_sec": round(percentile(values, 0.50), 3),
            "p90_sec": round(percentile(values, 0.90), 3),
            "p95_sec": round(percentile(values, 0.95), 3),
            "p99_sec": round(percentile(values, 0.99), 3),
            "max_sec": round(float(values[-1]), 3) if values else 0.0,
            "duration_bins": dict(self.duration_bins),
            "turn_counts": dict(sorted(self.turn_counts.items(), key=lambda item: (item[0] == "unknown", item[0]))),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize large duplex manifests by dataset and scenario.")
    parser.add_argument("--manifest", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress_every", type=int, default=100000)
    args = parser.parse_args()

    overall = Aggregate()
    overall_scenarios: Dict[str, Aggregate] = defaultdict(Aggregate)
    dataset_aggregates: Dict[str, Aggregate] = {}
    dataset_scenarios: Dict[str, Dict[str, Aggregate]] = {}
    paths: Dict[str, str] = {}
    total_rows = 0

    for source_value in args.manifest:
        label, path = parse_source(source_value)
        if label in dataset_aggregates:
            raise ValueError(f"duplicate dataset label: {label}")
        paths[label] = str(path)
        dataset = Aggregate()
        scenarios: Dict[str, Aggregate] = defaultdict(Aggregate)
        for row in iter_jsonl(path):
            duration = duration_sec(row)
            scenario = str(row.get("scenario") or "")
            dataset.add(row, duration)
            scenarios[scenario].add(row, duration)
            overall.add(row, duration)
            overall_scenarios[scenario].add(row, duration)
            total_rows += 1
            if args.progress_every > 0 and total_rows % args.progress_every == 0:
                print(f"STATS {total_rows} dataset={label}", flush=True)
        dataset_aggregates[label] = dataset
        dataset_scenarios[label] = dict(scenarios)

    overall_summary = overall.summary()
    result = {
        "manifests": paths,
        "overall": {
            **overall_summary,
            "scenarios": {
                name: {
                    **aggregate.summary(),
                    "ratio": round(len(aggregate.durations) / len(overall.durations), 6),
                }
                for name, aggregate in sorted(overall_scenarios.items())
            },
        },
        "datasets": {
            label: {
                **dataset_aggregates[label].summary(),
                "scenarios": {
                    name: {
                        **aggregate.summary(),
                        "ratio": round(len(aggregate.durations) / len(dataset_aggregates[label].durations), 6),
                    }
                    for name, aggregate in sorted(dataset_scenarios[label].items())
                },
            }
            for label in dataset_aggregates
        },
    }
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(output),
        "rows": result["overall"]["rows"],
        "total_hours": result["overall"]["total_hours"],
        "scenario_counts": {
            name: stats["rows"] for name, stats in result["overall"]["scenarios"].items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
