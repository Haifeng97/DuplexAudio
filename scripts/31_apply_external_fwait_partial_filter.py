#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FWAIT_EVENT = "F_WAIT"
VALID_JUDGMENTS = {"complete", "incomplete", "uncertain"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object at {path}:{line_no}")
            yield row


def iter_array_lines(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
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
            if not isinstance(row, dict):
                raise RuntimeError(f"expected object at {path}:{line_no}")
            yield text, row


def load_decisions(path: Path, max_partial_chars: int) -> tuple[dict[str, set[str]], Counter]:
    keep: dict[str, set[str]] = {}
    counts: Counter = Counter()
    seen: set[str] = set()
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or row.get("id") or "")
        dataset = str(row.get("dataset") or "")
        source_id = str(row.get("source_id") or "")
        if not request_id or not dataset or not source_id:
            raise ValueError("decision row has empty request_id, dataset, or source_id")
        if request_id in seen:
            raise ValueError(f"duplicate request_id: {request_id}")
        seen.add(request_id)
        judgment = str(row.get("judgment") or "").strip().lower()
        if judgment and judgment not in VALID_JUDGMENTS:
            raise ValueError(f"{request_id}: invalid judgment={judgment!r}")
        char_count = int(row.get("partial_query_char_count") or 0)
        counts[f"{dataset}:requests"] += 1
        if not judgment:
            counts[f"{dataset}:drop_unprocessed"] += 1
            continue
        counts[f"{dataset}:processed"] += 1
        if char_count > max_partial_chars:
            counts[f"{dataset}:drop_length"] += 1
            continue
        if judgment == "incomplete":
            keep.setdefault(dataset, set()).add(source_id)
            counts[f"{dataset}:keep"] += 1
        else:
            counts[f"{dataset}:drop_{judgment}"] += 1
    counts["requests"] = len(seen)
    return keep, counts


def filter_manifest(
    dataset: str,
    source: Path,
    out: Path,
    keep_ids: set[str],
    progress_every: int,
) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    counts: Counter = Counter()
    scenarios: Counter = Counter()
    found_keep: set[str] = set()
    first = True
    with tmp.open("w", encoding="utf-8") as output:
        output.write("[\n")
        for raw, row in iter_array_lines(source):
            counts["source_rows"] += 1
            source_id = str(row.get("id") or "")
            is_fwait = str((row.get("fgh_event") or {}).get("event") or "") == FWAIT_EVENT
            if is_fwait:
                counts["source_fwait"] += 1
                if source_id not in keep_ids:
                    counts["dropped_fwait"] += 1
                    continue
                found_keep.add(source_id)
                counts["retained_fwait"] += 1
            else:
                counts["retained_non_fwait"] += 1
            if not first:
                output.write(",\n")
            output.write(raw)
            first = False
            counts["written"] += 1
            scenarios[str(row.get("scenario") or "unknown")] += 1
            if progress_every > 0 and counts["source_rows"] % progress_every == 0:
                print(
                    f"{dataset}: scanned={counts['source_rows']} written={counts['written']} "
                    f"retained_fwait={counts['retained_fwait']}",
                    flush=True,
                )
        output.write("\n]\n")
    missing = keep_ids - found_keep
    if missing:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"{dataset}: {len(missing)} keep IDs missing from source: {sorted(missing)[:20]}")
    tmp.replace(out)
    return {
        "dataset": dataset,
        "source": str(source),
        "out": str(out),
        "counts": dict(counts),
        "scenario_counts": dict(scenarios),
    }


def parse_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("expected DATASET=SOURCE_MANIFEST=OUTPUT_MANIFEST")
    return parts[0], Path(parts[1]), Path(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter external F_WAIT rows using a partially filled LLM judgment file."
    )
    parser.add_argument("--filled", required=True)
    parser.add_argument("--spec", action="append", required=True, type=parse_spec)
    parser.add_argument("--max_partial_chars", type=int, default=14)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--progress_every", type=int, default=100000)
    args = parser.parse_args()

    keep, decision_counts = load_decisions(Path(args.filled), args.max_partial_chars)
    results = []
    for dataset, source, out in args.spec:
        results.append(
            filter_manifest(
                dataset,
                source,
                out,
                keep.get(dataset, set()),
                args.progress_every,
            )
        )
    stats = {
        "filled": args.filled,
        "policy": {
            "keep": "processed judgment=incomplete and partial_query_char_count <= max_partial_chars",
            "drop": "complete, uncertain, unprocessed, over-length, or absent from alignment requests",
            "max_partial_chars": args.max_partial_chars,
        },
        "decision_counts": dict(decision_counts),
        "results": results,
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
