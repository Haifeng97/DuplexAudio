#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


SCENARIOS = (
    "normal_qa",
    "incomplete_query_candidate",
    "player_interrupts_ai",
    "player_backchannel",
)
OUTPUT_FILES = {
    "normal_qa": "normal_qa_candidates.jsonl",
    "incomplete_query_candidate": "incomplete_query_candidates.jsonl",
    "player_interrupts_ai": "player_interrupt_candidates.jsonl",
    "player_backchannel": "player_backchannel_candidates.jsonl",
}
DEFAULT_PRIORITY = (
    "player_interrupts_ai",
    "incomplete_query_candidate",
    "player_backchannel",
    "normal_qa",
)


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


def parse_ratios(value: str) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, raw = item.partition("=")
        if not sep or name.strip() not in SCENARIOS:
            raise ValueError(f"invalid ratio item: {item!r}")
        ratio = float(raw)
        if ratio < 0:
            raise ValueError(f"ratio must be >= 0: {item!r}")
        ratios[name.strip()] = ratio
    missing = [name for name in SCENARIOS if name not in ratios]
    if missing:
        raise ValueError(f"--ratios is missing scenarios: {missing}")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"--ratios must sum to 1.0, got {total}")
    if any(ratio == 0 for ratio in ratios.values()):
        raise ValueError("all four scenario ratios must be > 0")
    return ratios


def parse_priority(value: str) -> List[str]:
    priority = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in priority if name not in SCENARIOS]
    if unknown:
        raise ValueError(f"unknown scenario(s) in --priority: {unknown}")
    for name in DEFAULT_PRIORITY:
        if name not in priority:
            priority.append(name)
    return priority


def allocate_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {name: total * ratios[name] for name in SCENARIOS}
    counts = {name: int(raw[name]) for name in SCENARIOS}
    remaining = total - sum(counts.values())
    order = sorted(SCENARIOS, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def source_groups(row: Dict[str, Any]) -> Set[str]:
    groups: Set[str] = set()
    for key in ("source_group_id", "source_id"):
        if row.get(key):
            groups.add(str(row[key]))
    values = row.get("source_group_ids")
    if isinstance(values, list):
        groups.update(str(value) for value in values if value)
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if meta.get("source_group_id"):
        groups.add(str(meta["source_group_id"]))
    for side in ("base", "donor"):
        obj = row.get(side) if isinstance(row.get(side), dict) else {}
        side_meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        if side_meta.get("source_group_id"):
            groups.add(str(side_meta["source_group_id"]))
        elif obj.get("id"):
            groups.add(str(obj["id"]))
    return groups or {str(row.get("id") or "")}


def select_rows(
    pools: Dict[str, List[Dict[str, Any]]],
    counts: Dict[str, int],
    priority: List[str],
    reserved_groups: Set[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    selected = {name: [] for name in SCENARIOS}
    used_groups: Set[str] = set(reserved_groups)
    skipped = {name: 0 for name in SCENARIOS}
    for name in priority:
        need = counts[name]
        for row in pools[name]:
            groups = source_groups(row)
            if used_groups.intersection(groups):
                skipped[name] += 1
                continue
            selected[name].append(row)
            used_groups.update(groups)
            if len(selected[name]) >= need:
                break
    return selected, skipped


def resolve_total(
    requested_total: int,
    pools: Dict[str, List[Dict[str, Any]]],
    ratios: Dict[str, float],
    priority: List[str],
    reserved_groups: Set[str],
) -> Tuple[int, Dict[str, int], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    available = {name: len(pools[name]) for name in SCENARIOS}
    upper = requested_total if requested_total > 0 else min(
        int(available[name] / ratios[name]) for name in SCENARIOS
    )
    lo = upper if requested_total > 0 else 0
    hi = upper
    best_total = -1
    best_counts: Dict[str, int] = {}
    best_selected: Dict[str, List[Dict[str, Any]]] = {}
    best_skipped: Dict[str, int] = {}
    while lo <= hi:
        total = (lo + hi) // 2
        counts = allocate_counts(total, ratios)
        selected, skipped = select_rows(pools, counts, priority, reserved_groups)
        enough = all(len(selected[name]) >= counts[name] for name in SCENARIOS)
        if enough:
            best_total = total
            best_counts = counts
            best_selected = selected
            best_skipped = skipped
            lo = total + 1
        else:
            hi = total - 1
    if best_total < 0:
        requested = allocate_counts(upper, ratios)
        raise ValueError(f"cannot satisfy total={upper}; requested={requested}, available={available}")
    if requested_total > 0 and best_total != requested_total:
        selected_counts = {name: len(best_selected.get(name, [])) for name in SCENARIOS}
        raise ValueError(
            f"cannot satisfy requested total={requested_total} after source-group dedup; "
            f"best_total={best_total}, selected={selected_counts}, available={available}"
        )
    return best_total, best_counts, best_selected, best_skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Select four roleplay duplex scenarios before TTS.")
    ap.add_argument("--normal", required=True)
    ap.add_argument("--incomplete", required=True)
    ap.add_argument("--interrupt", required=True)
    ap.add_argument("--backchannel", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--ratios",
        required=True,
        help=(
            "Explicit ratios summing to 1, for example "
            "normal_qa=0.60,incomplete_query_candidate=0.15,"
            "player_interrupts_ai=0.15,player_backchannel=0.10"
        ),
    )
    ap.add_argument("--total", type=int, default=0, help="0 selects the largest feasible total")
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--priority", default=",".join(DEFAULT_PRIORITY))
    ap.add_argument(
        "--exclude_manifest",
        action="append",
        default=[],
        help="Reserve source groups from another scenario manifest/candidate file; repeat as needed.",
    )
    args = ap.parse_args()

    ratios = parse_ratios(args.ratios)
    priority = parse_priority(args.priority)
    pools = {
        "normal_qa": read_jsonl(Path(args.normal)),
        "incomplete_query_candidate": read_jsonl(Path(args.incomplete)),
        "player_interrupts_ai": read_jsonl(Path(args.interrupt)),
        "player_backchannel": read_jsonl(Path(args.backchannel)),
    }
    rng = random.Random(args.seed)
    for rows in pools.values():
        rng.shuffle(rows)

    reserved_groups: Set[str] = set()
    excluded_rows = 0
    for value in args.exclude_manifest:
        excluded = read_jsonl(Path(value))
        excluded_rows += len(excluded)
        for row in excluded:
            reserved_groups.update(source_groups(row))

    total, counts, selected, skipped = resolve_total(
        args.total,
        pools,
        ratios,
        priority,
        reserved_groups,
    )
    out_dir = Path(args.out_dir)
    written = {
        name: write_jsonl(out_dir / OUTPUT_FILES[name], selected[name])
        for name in SCENARIOS
    }
    stats = {
        "out_dir": str(out_dir),
        "requested_total": args.total,
        "resolved_total": total,
        "ratios": ratios,
        "priority": priority,
        "available": {name: len(pools[name]) for name in SCENARIOS},
        "target_counts": counts,
        "written": written,
        "skipped_duplicate_source_group": skipped,
        "exclude_manifests": args.exclude_manifest,
        "excluded_rows": excluded_rows,
        "reserved_source_groups": len(reserved_groups),
        "seed": args.seed,
    }
    (out_dir / "selection_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
