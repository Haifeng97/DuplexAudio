#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


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


def compact_row(row: Dict[str, Any], duration: float) -> Dict[str, Any]:
    source_row = row.get("source_row") or {}
    meta = source_row.get("meta") or {}
    return {
        "id": str(row.get("id") or ""),
        "scenario": str(row.get("scenario") or ""),
        "duration_sec": round(duration, 3),
        "audio": str(row.get("audio") or ""),
        "question_text": str(row.get("question_text") or ""),
        "answer_text": str(row.get("answer_text") or ""),
        "source_dataset": str(meta.get("dataset") or source_row.get("source") or row.get("source") or ""),
        "manifest_row": row,
    }


def safe_stem(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{stem[:100]}__{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Package representative long final WAVs from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold_sec", type=float, default=240.0)
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--top_count", type=int, default=10)
    parser.add_argument("--reservoir_size", type=int, default=200)
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    if args.sample_count <= 0 or not 0 <= args.top_count <= args.sample_count:
        raise SystemExit("sample counts are invalid")
    rng = random.Random(args.seed)
    manifest = Path(args.manifest).resolve()
    bins = Counter()
    scenarios = Counter()
    threshold_scenarios = Counter()
    top_all: List[Tuple[float, int, Dict[str, Any]]] = []
    top_threshold: List[Tuple[float, int, Dict[str, Any]]] = []
    reservoir: List[Dict[str, Any]] = []
    threshold_seen = 0
    total = 0

    for sequence, row in enumerate(iter_jsonl(manifest)):
        total += 1
        duration = duration_sec(row)
        scenario = str(row.get("scenario") or "")
        scenarios[scenario] += 1
        if duration < 60:
            bins["lt_1m"] += 1
        elif duration < 120:
            bins["1m_to_2m"] += 1
        elif duration < 180:
            bins["2m_to_3m"] += 1
        elif duration < 240:
            bins["3m_to_4m"] += 1
        elif duration < 300:
            bins["4m_to_5m"] += 1
        else:
            bins["gte_5m"] += 1
        compact = compact_row(row, duration)
        heapq.heappush(top_all, (duration, sequence, compact))
        if len(top_all) > args.sample_count:
            heapq.heappop(top_all)
        if duration >= args.threshold_sec:
            threshold_seen += 1
            threshold_scenarios[scenario] += 1
            heapq.heappush(top_threshold, (duration, sequence, compact))
            if len(top_threshold) > args.top_count:
                heapq.heappop(top_threshold)
            if len(reservoir) < args.reservoir_size:
                reservoir.append(compact)
            else:
                position = rng.randrange(threshold_seen)
                if position < args.reservoir_size:
                    reservoir[position] = compact
        if args.progress_every > 0 and total % args.progress_every == 0:
            print(f"LONGSCAN {total} gte_threshold={threshold_seen}", flush=True)

    if threshold_seen:
        selected = [item[2] for item in sorted(top_threshold, reverse=True)]
        selected_ids = {row["id"] for row in selected}
        pool = [row for row in reservoir if row["id"] not in selected_ids]
        rng.shuffle(pool)
        selected.extend(pool[: max(0, args.sample_count - len(selected))])
    else:
        selected = [item[2] for item in sorted(top_all, reverse=True)]
    selected = selected[: args.sample_count]

    out_dir = Path(args.out_dir).resolve()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    packaged = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["audio"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = audio_dir / (
            f"{index:02d}__{row['duration_sec']:.2f}s__{row['scenario']}__{safe_stem(row['id'])}.wav"
        )
        shutil.copyfile(source, target)
        packaged.append({
            **row,
            "index": index,
            "packaged_audio": str(target.relative_to(out_dir)),
        })

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in packaged:
            handle.write(json.dumps(row["manifest_row"], ensure_ascii=False, separators=(",", ":")) + "\n")
    audit_rows = [{key: value for key, value in row.items() if key != "manifest_row"} for row in packaged]
    with (out_dir / "audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "manifest": str(manifest),
        "rows": total,
        "threshold_sec": args.threshold_sec,
        "duration_bins": dict(bins),
        "scenario_counts": dict(scenarios),
        "threshold_rows": threshold_seen,
        "threshold_scenarios": dict(threshold_scenarios),
        "sampled": len(packaged),
        "sample_strategy": "top_plus_reservoir" if threshold_seen else "longest_overall",
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Long final-audio listening sample",
        "",
        f"Manifest: `{manifest}`",
        f"Threshold: `{args.threshold_sec}s`",
        "",
    ]
    for row in packaged:
        lines.extend([
            f"## {row['index']}. {row['duration_sec']:.2f}s / {row['scenario']}",
            "",
            f"- Audio: `{row['packaged_audio']}`",
            f"- ID: `{row['id']}`",
            f"- Source dataset: `{row['source_dataset']}`",
            f"- Question: {row['question_text']}",
            f"- Answer: {row['answer_text']}",
            "",
        ])
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
