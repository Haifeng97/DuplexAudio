#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


RATIOS = {
    "normal_qa": 0.70,
    "incomplete_query": 0.15,
    "player_interrupts_ai": 0.15,
}
DEFAULT_PRIORITY = ["player_interrupts_ai", "incomplete_query", "normal_qa"]


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


def source_ids(row: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    for obj in (row, source_row):
        for key in ("source_id", "id"):
            value = obj.get(key)
            if value:
                ids.add(str(value))
    for obj in (source_row.get("base"), source_row.get("donor"), row.get("base"), row.get("donor")):
        if isinstance(obj, dict) and obj.get("id"):
            ids.add(str(obj["id"]))
    return ids or {str(row.get("id") or "")}


def select_rows(
    pools: Dict[str, List[Dict[str, Any]]],
    counts: Dict[str, int],
    priority: List[str],
    *,
    unique_source_id: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    selected_counts = {name: 0 for name in RATIOS}
    skipped_duplicate_source_id = {name: 0 for name in RATIOS}
    used_source_ids: Set[str] = set()

    for name in priority:
        need = counts.get(name, 0)
        if need <= 0:
            continue
        for row in pools[name]:
            ids = source_ids(row)
            if unique_source_id and used_source_ids.intersection(ids):
                skipped_duplicate_source_id[name] += 1
                continue
            out = dict(row)
            out["mix_source_scenario"] = name
            selected.append(out)
            selected_counts[name] += 1
            if unique_source_id:
                used_source_ids.update(ids)
            if selected_counts[name] >= need:
                break

    lacking = {name: counts[name] - selected_counts[name] for name in RATIOS if selected_counts[name] < counts[name]}
    return selected, selected_counts, lacking or skipped_duplicate_source_id


def parse_priority(value: str) -> List[str]:
    priority = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in priority if name not in RATIOS]
    if unknown:
        raise ValueError(f"unknown priority scenario(s): {unknown}")
    for name in DEFAULT_PRIORITY:
        if name not in priority:
            priority.append(name)
    return priority


def concat_rows(
    pools: Dict[str, List[Dict[str, Any]]],
    priority: List[str],
    *,
    unique_source_id: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    selected_counts = {name: 0 for name in RATIOS}
    skipped_duplicate_source_id = {name: 0 for name in RATIOS}
    used_source_ids: Set[str] = set()

    for name in priority:
        for row in pools[name]:
            ids = source_ids(row)
            if unique_source_id and used_source_ids.intersection(ids):
                skipped_duplicate_source_id[name] += 1
                continue
            out = dict(row)
            out["mix_source_scenario"] = name
            selected.append(out)
            selected_counts[name] += 1
            if unique_source_id:
                used_source_ids.update(ids)
    return selected, selected_counts, skipped_duplicate_source_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix duplex manifests with normal/incomplete/interrupt = 70/15/15.")
    ap.add_argument("--normal", required=True)
    ap.add_argument("--incomplete", required=True)
    ap.add_argument("--interrupt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=0, help="0 means largest total supported by all three inputs in ratio mode.")
    ap.add_argument("--mode", choices=["ratio", "concat"], default="ratio", help="ratio resamples to 70/15/15; concat keeps all available rows and shuffles.")
    ap.add_argument("--seed", type=int, default=20260716)
    ap.add_argument("--priority", default=",".join(DEFAULT_PRIORITY), help="Scenario selection priority for source-id de-duplication.")
    ap.add_argument("--allow_duplicate_source_id", action="store_true", help="Keep legacy behavior and allow the same source_id in multiple scenarios.")
    args = ap.parse_args()

    pools = {
        "normal_qa": read_jsonl(Path(args.normal)),
        "incomplete_query": read_jsonl(Path(args.incomplete)),
        "player_interrupts_ai": read_jsonl(Path(args.interrupt)),
    }
    available = {name: len(rows) for name, rows in pools.items()}
    priority = parse_priority(args.priority)
    unique_source_id = not args.allow_duplicate_source_id

    rng = random.Random(args.seed)
    shuffled_pools = {name: list(rows) for name, rows in pools.items()}
    for rows in shuffled_pools.values():
        rng.shuffle(rows)

    requested_total = args.total
    skipped_duplicate_source_id = {name: 0 for name in RATIOS}
    if args.mode == "concat":
        if requested_total > 0:
            raise ValueError("--total is only supported with --mode ratio")
        selected, selected_counts, skipped_duplicate_source_id = concat_rows(
            shuffled_pools,
            priority,
            unique_source_id=unique_source_id,
        )
        total = len(selected)
        counts = dict(selected_counts)
    else:
        total = requested_total if requested_total > 0 else max_total(available)
        counts = allocate_counts(total)
        selected, selected_counts, _ = select_rows(
            shuffled_pools,
            counts,
            priority,
            unique_source_id=unique_source_id,
        )
        lacking = {name: counts[name] - selected_counts[name] for name in RATIOS if selected_counts[name] < counts[name]}

        if lacking and requested_total <= 0:
            lo, hi = 0, total
            best_total = 0
            best_selected: List[Dict[str, Any]] = []
            best_counts: Dict[str, int] = {name: 0 for name in RATIOS}
            while lo <= hi:
                mid = (lo + hi) // 2
                mid_counts = allocate_counts(mid)
                mid_selected, mid_selected_counts, _ = select_rows(
                    shuffled_pools,
                    mid_counts,
                    priority,
                    unique_source_id=unique_source_id,
                )
                mid_lacking = {name: mid_counts[name] - mid_selected_counts[name] for name in RATIOS if mid_selected_counts[name] < mid_counts[name]}
                if mid_lacking:
                    hi = mid - 1
                else:
                    best_total = mid
                    best_selected = mid_selected
                    best_counts = mid_counts
                    lo = mid + 1
            total = best_total
            counts = best_counts
            selected = best_selected
            selected_counts = {name: counts.get(name, 0) for name in RATIOS}
            lacking = {}
        if lacking:
            raise ValueError(f"not enough rows for requested total={total}: lacking={lacking}, available={available}, selected={selected_counts}")

    rng.shuffle(selected)
    n = write_jsonl(Path(args.out), selected)
    stats = {
        "out": args.out,
        "n": n,
        "seed": args.seed,
        "mode": args.mode,
        "requested_total": requested_total,
        "resolved_total": total,
        "counts": counts,
        "selected_counts": selected_counts,
        "available": available,
        "ratios": RATIOS,
        "priority": priority,
        "unique_source_id": unique_source_id,
        "skipped_duplicate_source_id": skipped_duplicate_source_id,
    }
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
