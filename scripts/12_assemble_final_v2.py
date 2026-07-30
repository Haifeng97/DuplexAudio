#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    scenario_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
            scenario_counts[str(row.get("scenario") or "")] += 1
            source_id = ""
            source_row = row.get("source_row")
            if isinstance(source_row, dict):
                source_id = str(source_row.get("source_id") or "")
            if not source_id:
                source_id = str(row.get("source_id") or "")
            if source_id:
                source_counts[source_id] += 1
    return {
        "path": str(path),
        "n": n,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "unique_source_id": all(v == 1 for v in source_counts.values()),
        "duplicate_source_id": sum(1 for v in source_counts.values() if v > 1),
    }


def require(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def chain(paths: List[Path]) -> Iterator[Dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble final v2 manifests from reused single-turn data and newly formatted multi-turn scenes.")
    ap.add_argument("--out_dir", default="outputs/final_v2")
    ap.add_argument("--normal_single", default="outputs/final_v2_delta/reuse/normal_single_manifest.jsonl")
    ap.add_argument("--normal_multi", default="outputs/final_v2/final_normal_multi/manifest.jsonl")
    ap.add_argument("--incomplete_single", default="outputs/final_v2_delta/reuse/incomplete_single_manifest.jsonl")
    ap.add_argument("--incomplete_multi", default="outputs/final_v2/final_incomplete_multi/manifest.jsonl")
    ap.add_argument("--interrupt", default="outputs/final_v2/final_interrupt/manifest.jsonl")
    ap.add_argument("--clarification", default="outputs/final_v2/final_incomplete_clarification/manifest.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    normal_parts = [require(args.normal_single), require(args.normal_multi)]
    incomplete_parts = [require(args.incomplete_single), require(args.incomplete_multi)]
    interrupt = require(args.interrupt)
    clarification = require(args.clarification)

    stats: Dict[str, Any] = {"out_dir": str(out_dir)}
    normal_out = out_dir / "final_normal" / "manifest.jsonl"
    incomplete_out = out_dir / "final_incomplete" / "manifest.jsonl"
    all_out = out_dir / "final_all" / "manifest.jsonl"

    stats["normal"] = write_jsonl(normal_out, chain(normal_parts))
    stats["incomplete"] = write_jsonl(incomplete_out, chain(incomplete_parts))
    stats["all"] = write_jsonl(all_out, chain([normal_out, incomplete_out, interrupt, clarification]))

    stats_path = out_dir / "assemble_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
