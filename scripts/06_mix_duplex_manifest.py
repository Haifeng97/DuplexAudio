#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


RATIOS = {
    "normal_qa": 0.70,
    "incomplete_query": 0.15,
    "player_interrupts_ai": 0.15,
}


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


def allocate_counts(total: int) -> Dict[str, int]:
    raw = {name: total * ratio for name, ratio in RATIOS.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(RATIOS, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def max_total(available: Dict[str, int]) -> int:
    limits = [int(available[name] / ratio) for name, ratio in RATIOS.items()]
    return max(0, min(limits))


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix duplex manifests with normal/incomplete/interrupt = 70/15/15.")
    ap.add_argument("--normal", required=True)
    ap.add_argument("--incomplete", required=True)
    ap.add_argument("--interrupt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=0, help="0 means largest total supported by all three inputs.")
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    pools = {
        "normal_qa": read_jsonl(Path(args.normal)),
        "incomplete_query": read_jsonl(Path(args.incomplete)),
        "player_interrupts_ai": read_jsonl(Path(args.interrupt)),
    }
    available = {name: len(rows) for name, rows in pools.items()}
    total = args.total if args.total > 0 else max_total(available)
    counts = allocate_counts(total)
    lacking = {name: counts[name] - available[name] for name in RATIOS if counts[name] > available[name]}
    if lacking:
        raise ValueError(f"not enough rows for requested total={total}: lacking={lacking}, available={available}")

    rng = random.Random(args.seed)
    selected: List[Dict[str, Any]] = []
    for name, count in counts.items():
        rows = list(pools[name])
        rng.shuffle(rows)
        for row in rows[:count]:
            out = dict(row)
            out["mix_source_scenario"] = name
            selected.append(out)
    rng.shuffle(selected)

    n = write_jsonl(Path(args.out), selected)
    stats = {
        "out": args.out,
        "n": n,
        "seed": args.seed,
        "requested_total": args.total,
        "counts": counts,
        "available": available,
        "ratios": RATIOS,
    }
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
