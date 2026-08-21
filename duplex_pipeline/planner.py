from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash
from .text import effective_char_count


SCENARIOS = (
    "normal_qa",
    "player_interrupts_ai",
    "incomplete_query",
    "incomplete_query_clarification",
    "player_backchannel",
    "other",
)


def canonical_scenario(row: Dict[str, Any]) -> str:
    value = str(row.get("primary_scenario") or row.get("scenario") or "")
    aliases = {
        "incomplete_query_candidate": "incomplete_query",
        "ai_intervenes_user": "other",
        "player_complete": "other",
    }
    return aliases.get(value, value)


def largest_remainder(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {name: total * ratios[name] for name in SCENARIOS}
    counts = {name: int(raw[name]) for name in SCENARIOS}
    remaining = total - sum(counts.values())
    order = sorted(SCENARIOS, key=lambda name: (raw[name] - counts[name], name), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def count_base(paths: Iterable[str]) -> Tuple[Counter, int]:
    counts: Counter = Counter()
    total = 0
    for raw_path in paths:
        for row in iter_jsonl(Path(raw_path)):
            scenario = canonical_scenario(row)
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown base scenario {scenario!r} in {raw_path}")
            counts[scenario] += 1
            total += 1
    return counts, total


def resolve_additions(
    base: Counter,
    customized_total: int,
    special_available: int,
    ratios: Dict[str, float],
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    ratio_other = ratios["other"]
    center = round(ratio_other * (sum(base.values()) + customized_total) / (1.0 - ratio_other))
    candidates = sorted(range(max(0, center - 1000), min(special_available, center + 1000) + 1), key=lambda x: (abs(x - center), x))
    for special_count in candidates:
        target = largest_remainder(sum(base.values()) + customized_total + special_count, ratios)
        if target["other"] != base["other"] + special_count:
            continue
        additions = {name: target[name] - base[name] for name in SCENARIOS}
        if min(additions.values()) < 0:
            continue
        if sum(additions[name] for name in SCENARIOS if name != "other") != customized_total:
            continue
        if additions["other"] != special_count:
            continue
        return special_count, target, additions
    raise ValueError("cannot resolve an exact ratio allocation with all customized rows and available special rows")


def _hash_int(identifier: str, seed: int, scenario: str) -> int:
    return int(stable_hash({"id": identifier, "seed": seed, "scenario": scenario}, length=15), 16)


def _custom_eligibility(row: Dict[str, Any], min_backchannel_answer_chars: int) -> Tuple[int, int, int]:
    turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    current = turns[-1] if turns else {}
    question = str(current.get("question_text") or "")
    answer = str(current.get("answer_text") or "")
    interrupt = int(len(turns) >= 2 and effective_char_count(str(turns[-2].get("answer_text") or "")) >= 2)
    incomplete = int(effective_char_count(question) >= 4)
    backchannel = int(effective_char_count(answer) >= min_backchannel_answer_chars)
    return interrupt, incomplete, backchannel


def resolve_special_targets(special_count: int, available: Dict[str, int]) -> Dict[str, int]:
    desired_intervene = special_count // 2 + special_count % 2
    targets = {
        "ai_intervenes_user": min(desired_intervene, available.get("ai_intervenes_user", 0)),
        "player_complete": min(special_count - desired_intervene, available.get("player_complete", 0)),
    }
    remaining = special_count - sum(targets.values())
    for scenario in sorted(targets, key=lambda name: (available.get(name, 0) - targets[name], name), reverse=True):
        added = min(remaining, available.get(scenario, 0) - targets[scenario])
        targets[scenario] += added
        remaining -= added
    if remaining:
        raise ValueError(f"insufficient special rows: need {special_count}, available {sum(available.values())}")
    return targets


def build_plan(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "02_plan"
    stage_dir.mkdir(parents=True, exist_ok=True)
    normalized = run_dir / "01_normalized"
    custom_path = normalized / "customized.jsonl"
    special_path = normalized / "special.jsonl"
    if not custom_path.exists() or not special_path.exists():
        raise FileNotFoundError("normalize stage must produce customized.jsonl and special.jsonl")

    ratios = {name: float(config["ratios"][name]) for name in SCENARIOS}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1, got {sum(ratios.values())}")
    configured_base = config.get("base_counts")
    if isinstance(configured_base, dict):
        base_counts = Counter({name: int(configured_base.get(name, 0)) for name in SCENARIOS})
        base_total = sum(base_counts.values())
    else:
        base_counts, base_total = count_base(config.get("base_manifests", []))
    custom_total = sum(1 for _ in iter_jsonl(custom_path))
    special_available = sum(1 for _ in iter_jsonl(special_path))
    special_count, targets, additions = resolve_additions(base_counts, custom_total, special_available, ratios)

    db_path = stage_dir / "plan.sqlite3"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        "CREATE TABLE custom (id TEXT PRIMARY KEY,assigned TEXT,eligible_interrupt INTEGER,eligible_incomplete INTEGER,eligible_backchannel INTEGER)"
    )
    min_backchannel = int(config.get("planning", {}).get("min_backchannel_answer_chars", 8))
    for index, row in enumerate(iter_jsonl(custom_path), start=1):
        eligible = _custom_eligibility(row, min_backchannel)
        connection.execute("INSERT INTO custom VALUES (?,?,?,?,?)", (row["id"], None, *eligible))
        if index % 10000 == 0:
            connection.commit()
    connection.commit()

    seed = int(config.get("planning", {}).get("seed", 20260818))
    assignment_order = (
        ("player_interrupts_ai", "eligible_interrupt=1"),
        ("incomplete_query_clarification", "eligible_incomplete=1"),
        ("incomplete_query", "eligible_incomplete=1"),
        ("player_backchannel", "eligible_backchannel=1"),
    )
    eligibility_counts: Dict[str, int] = {}
    for scenario, condition in assignment_order:
        available = connection.execute(f"SELECT COUNT(*) FROM custom WHERE assigned IS NULL AND {condition}").fetchone()[0]
        eligibility_counts[scenario] = available
        needed = additions[scenario]
        if available < needed:
            raise ValueError(f"insufficient {scenario} candidates: need {needed}, available {available}")
        ids = [
            row[0]
            for row in connection.execute(f"SELECT id FROM custom WHERE assigned IS NULL AND {condition}")
        ]
        ids.sort(key=lambda identifier: (_hash_int(identifier, seed, scenario), identifier))
        connection.executemany("UPDATE custom SET assigned=? WHERE id=?", ((scenario, identifier) for identifier in ids[:needed]))
        connection.commit()
    remaining = connection.execute("SELECT COUNT(*) FROM custom WHERE assigned IS NULL").fetchone()[0]
    if remaining != additions["normal_qa"]:
        raise ValueError(f"normal remainder mismatch: expected {additions['normal_qa']}, got {remaining}")
    connection.execute("UPDATE custom SET assigned='normal_qa' WHERE assigned IS NULL")
    connection.commit()

    assignment_path = stage_dir / "customized_assignments.jsonl"
    assignments = {
        str(identifier): str(assigned)
        for identifier, assigned in connection.execute("SELECT id,assigned FROM custom")
    }
    with assignment_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(custom_path):
            identifier = str(row["id"])
            assigned = assignments[identifier]
            handle.write(canonical_json({
                "id": identifier,
                "primary_scenario": assigned,
                "source_version": row.get("source_version"),
                "original_id": row.get("original_id"),
                "root_source_group_id": row.get("root_source_group_id"),
            }) + "\n")
    selected_custom_path = stage_dir / "customized_selected.jsonl"
    with selected_custom_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(custom_path):
            assigned = assignments[str(row["id"])]
            row["primary_scenario"] = assigned
            handle.write(canonical_json(row) + "\n")
    del assignments
    connection.close()

    special_by_scenario: Dict[str, List[Dict[str, Any]]] = {"ai_intervenes_user": [], "player_complete": []}
    for row in iter_jsonl(special_path):
        special_by_scenario[str(row["scenario"])].append(row)
    special_targets = resolve_special_targets(
        special_count,
        {scenario: len(rows) for scenario, rows in special_by_scenario.items()},
    )
    selected_special_path = stage_dir / "special_selected.jsonl"
    with selected_special_path.open("w", encoding="utf-8") as handle:
        for scenario, rows in special_by_scenario.items():
            rows.sort(key=lambda row: (_hash_int(str(row["id"]), seed, scenario), str(row["id"])))
            need = special_targets[scenario]
            if len(rows) < need:
                raise ValueError(f"insufficient special rows for {scenario}: need {need}, available {len(rows)}")
            for row in rows[:need]:
                row["primary_scenario"] = "other"
                handle.write(canonical_json(row) + "\n")

    result = {
        "base_total": base_total,
        "base_counts": {name: base_counts[name] for name in SCENARIOS},
        "customized_total": custom_total,
        "special_available": special_available,
        "special_selected": special_count,
        "special_selected_by_scenario": special_targets,
        "target_total": sum(targets.values()),
        "target_counts": targets,
        "addition_counts": additions,
        "ratios": ratios,
        "eligibility_counts_at_assignment": eligibility_counts,
        "seed": seed,
        "outputs": {
            "customized_assignments": str(assignment_path),
            "customized_selected": str(selected_custom_path),
            "special_selected": str(selected_special_path),
        },
    }
    atomic_write_json(stage_dir / "stats.json", result)
    return result
