#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Concatenate complete duplex manifests without resampling or dropping rows.")
    ap.add_argument("--manifest", action="append", required=True, help="Repeat once per input manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--allow_duplicate_id", action="store_true")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    input_counts: Dict[str, int] = {}
    for value in args.manifest:
        path = Path(value)
        current = read_jsonl(path)
        input_counts[str(path)] = len(current)
        rows.extend(current)

    ids = Counter(str(row.get("id") or "") for row in rows)
    duplicate_ids = sorted(sample_id for sample_id, count in ids.items() if sample_id and count > 1)
    if duplicate_ids and not args.allow_duplicate_id:
        raise ValueError(f"duplicate manifest ids: {duplicate_ids[:20]}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    out = Path(args.out)
    n = write_jsonl(out, rows)
    stats = {
        "out": str(out),
        "n": n,
        "inputs": input_counts,
        "scenario_counts": dict(Counter(str(row.get("scenario") or "") for row in rows)),
        "duplicate_ids": len(duplicate_ids),
        "seed": args.seed,
        "drop_policy": "none",
    }
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
