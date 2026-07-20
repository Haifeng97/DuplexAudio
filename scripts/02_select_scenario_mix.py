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
    "incomplete_query_candidate": 0.15,
    "player_interrupts_ai": 0.15,
}
OUTPUT_FILES = {
    "normal_qa": "normal_qa_candidates.jsonl",
    "incomplete_query_candidate": "incomplete_query_candidates.jsonl",
    "player_interrupts_ai": "player_interrupt_candidates.jsonl",
}
DEFAULT_PRIORITY = ["player_interrupts_ai", "incomplete_query_candidate", "normal_qa"]


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


def max_total_by_available(available: Dict[str, int]) -> int:
    limits = [int(available[name] / ratio) for name, ratio in RATIOS.items()]
    return max(0, min(limits))


def source_ids(row: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = set()
    for key in ("source_id", "id"):
        value = row.get(key)
        if value:
            ids.add(str(value))
    for obj in (row.get("base"), row.get("donor")):
        if isinstance(obj, dict) and obj.get("id"):
            ids.add(str(obj["id"]))
    return ids or {str(row.get("id") or "")}


def parse_priority(value: str) -> List[str]:
    priority = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in priority if name not in RATIOS]
    if unknown:
        raise ValueError(f"unknown scenario(s) in --priority: {unknown}")
    for name in DEFAULT_PRIORITY:
        if name not in priority:
            priority.append(name)
    return priority


def select_rows(
    pools: Dict[str, List[Dict[str, Any]]],
    counts: Dict[str, int],
    priority: List[str],
    *,
    unique_source_id: bool,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], Dict[str, int]]:
    selected = {name: [] for name in RATIOS}
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
            selected[name].append(row)
            if unique_source_id:
                used_source_ids.update(ids)
            if len(selected[name]) >= need:
                break

    selected_counts = {name: len(rows) for name, rows in selected.items()}
    lacking = {name: counts[name] - selected_counts[name] for name in RATIOS if selected_counts[name] < counts[name]}
    return selected, selected_counts, lacking or skipped_duplicate_source_id


def resolve_total(
    requested_total: int,
    available: Dict[str, int],
    pools: Dict[str, List[Dict[str, Any]]],
    priority: List[str],
    unique_source_id: bool,
) -> Tuple[int, Dict[str, int], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    upper = requested_total if requested_total > 0 else max_total_by_available(available)
    if requested_total > 0:
        counts = allocate_counts(upper)
        selected, selected_counts, extra = select_rows(pools, counts, priority, unique_source_id=unique_source_id)
        lacking = {name: counts[name] - selected_counts[name] for name in RATIOS if selected_counts[name] < counts[name]}
        if lacking:
            raise ValueError(f"not enough rows for requested total={upper}: lacking={lacking}, available={available}, selected={selected_counts}")
        return upper, counts, selected, extra

    lo, hi = 0, upper
    best_total = 0
    best_counts = allocate_counts(0)
    best_selected = {name: [] for name in RATIOS}
    best_extra = {name: 0 for name in RATIOS}
    while lo <= hi:
        mid = (lo + hi) // 2
        counts = allocate_counts(mid)
        selected, selected_counts, extra = select_rows(pools, counts, priority, unique_source_id=unique_source_id)
        lacking = {name: counts[name] - selected_counts[name] for name in RATIOS if selected_counts[name] < counts[name]}
        if lacking:
            hi = mid - 1
        else:
            best_total = mid
            best_counts = counts
            best_selected = selected
            best_extra = extra
            lo = mid + 1
    return best_total, best_counts, best_selected, best_extra


def main() -> None:
    ap = argparse.ArgumentParser(description="Select final scenario candidates before TTS, with ratio and source-id de-duplication.")
    ap.add_argument("--normal", default="outputs/scenario_candidates/normal_qa_candidates.jsonl")
    ap.add_argument("--incomplete", default="outputs/scenario_candidates/incomplete_query_candidates.jsonl")
    ap.add_argument("--interrupt", default="outputs/scenario_candidates/player_interrupt_candidates.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--total", type=int, default=0, help="0 means maximum feasible total under ratios and de-duplication.")
    ap.add_argument("--seed", type=int, default=20260717)
    ap.add_argument("--priority", default=",".join(DEFAULT_PRIORITY))
    ap.add_argument("--allow_duplicate_source_id", action="store_true")
    args = ap.parse_args()

    pools = {
        "normal_qa": read_jsonl(Path(args.normal)),
        "incomplete_query_candidate": read_jsonl(Path(args.incomplete)),
        "player_interrupts_ai": read_jsonl(Path(args.interrupt)),
    }
    available = {name: len(rows) for name, rows in pools.items()}
    rng = random.Random(args.seed)
    shuffled_pools = {name: list(rows) for name, rows in pools.items()}
    for rows in shuffled_pools.values():
        rng.shuffle(rows)

    priority = parse_priority(args.priority)
    unique_source_id = not args.allow_duplicate_source_id
    total, counts, selected, extra = resolve_total(
        args.total,
        available,
        shuffled_pools,
        priority,
        unique_source_id,
    )

    out_dir = Path(args.out_dir)
    written = {name: write_jsonl(out_dir / OUTPUT_FILES[name], rows) for name, rows in selected.items()}
    stats = {
        "out_dir": str(out_dir),
        "requested_total": args.total,
        "resolved_total": total,
        "ratios": RATIOS,
        "priority": priority,
        "unique_source_id": unique_source_id,
        "available": available,
        "target_counts": counts,
        "written": written,
        "skipped_or_lacking": extra,
        "inputs": {
            "normal": args.normal,
            "incomplete": args.incomplete,
            "interrupt": args.interrupt,
        },
    }
    (out_dir / "selection_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
