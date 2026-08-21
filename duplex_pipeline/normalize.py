from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io import append_jsonl, atomic_write_json, canonical_json, stable_hash
from .text import effective_char_count


def _turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]


def _validate_customized(row: Dict[str, Any], limits: Dict[str, Any]) -> Optional[str]:
    turns = _turns(row)
    if not str(row.get("id") or ""):
        return "missing_id"
    if not turns:
        return "missing_turns"
    for turn in turns:
        if not str(turn.get("question_text") or "").strip():
            return "missing_question"
        if not str(turn.get("answer_text") or "").strip():
            return "missing_answer"
        question = str(turn.get("question_text"))
        answer = str(turn.get("answer_text"))
        max_question = int(limits.get("max_question_chars", limits.get("max_question_effective_chars", 240)))
        max_answer = int(limits.get("max_answer_chars", limits.get("max_answer_effective_chars", 240)))
        if max_question > 0 and len(question) > max_question:
            return "question_too_long"
        if max_answer > 0 and len(answer) > max_answer:
            return "answer_too_long"
    return None


def _validate_special(row: Dict[str, Any], limits: Dict[str, Any]) -> Optional[str]:
    if str(row.get("scenario") or "") not in {"ai_intervenes_user", "player_complete"}:
        return "unknown_special_scenario"
    if str(row.get("schema_version") or "") != "duplex_special_v1":
        return "invalid_special_schema"
    turns = _turns(row)
    if not str(row.get("id") or "") or not turns:
        return "missing_id_or_turns"
    current = turns[-1]
    event = current.get("event") if isinstance(current.get("event"), dict) else {}
    if row["scenario"] == "ai_intervenes_user":
        if event.get("type") != "intervene":
            return "missing_intervene_event"
        if not str(event.get("user_text_until_trigger") or "").strip():
            return "missing_intervene_prefix"
        if not str(event.get("user_text_after_trigger") or "").strip():
            return "missing_intervene_suffix"
    else:
        if event.get("type") != "complete":
            return "missing_complete_event"
        mode = str(event.get("response_mode") or "")
        if mode not in {"acknowledge", "silent"}:
            return "invalid_complete_response_mode"
        if mode == "acknowledge" and not str(current.get("answer_text") or "").strip():
            return "missing_complete_answer"
    return None


def _quality(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
    turns = _turns(row)
    complete_turns = sum(
        bool(str(turn.get("question_text") or "").strip() and str(turn.get("answer_text") or "").strip())
        for turn in turns
    )
    context_chars = min(
        10000,
        sum(
            effective_char_count(str(turn.get("question_text") or ""))
            + effective_char_count(str(turn.get("answer_text") or ""))
            for turn in turns
        ),
    )
    has_prompt = int(bool(str(row.get("sysprompt") or "").strip()))
    # The hash is only a deterministic tie-break for otherwise equal duplicate rows.
    row_hash = stable_hash(row)
    return complete_turns, has_prompt, context_chars, row_hash


def _normalized_row(row: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    original_id = str(row["id"])
    namespace = str(source["namespace"])
    out = dict(row)
    out["id"] = f"{namespace}::{original_id}"
    out["original_id"] = original_id
    out["source_version"] = str(source["source_version"])
    out["source_namespace"] = namespace
    meta = dict(row.get("meta") or {})
    root_group = str(meta.get("source_group_id") or row.get("source_group_id") or original_id)
    out["root_source_group_id"] = root_group
    meta.update({
        "original_id": original_id,
        "root_source_group_id": root_group,
        "source_version": str(source["source_version"]),
        "source_namespace": namespace,
    })
    out["meta"] = meta
    return out


def normalize_sources(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage_dir = run_dir / "01_normalized"
    stage_dir.mkdir(parents=True, exist_ok=True)
    db_path = stage_dir / "normalized.sqlite3"
    rejected_path = stage_dir / "rejected.jsonl"
    if db_path.exists():
        db_path.unlink()
    if rejected_path.exists():
        rejected_path.unlink()

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """CREATE TABLE rows (
            namespace TEXT NOT NULL,
            original_id TEXT NOT NULL,
            source_version TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            quality_json TEXT NOT NULL,
            row_hash TEXT NOT NULL,
            row_json TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            PRIMARY KEY(namespace, original_id)
        )"""
    )
    stats: Dict[str, Any] = {"sources": {}, "rejected": Counter(), "duplicates": Counter()}
    limits = dict(config.get("normalization") or {})
    inputs = list(config.get("inputs", {}).get("customized", [])) + list(config.get("inputs", {}).get("special", []))
    for source in inputs:
        path = Path(str(source["path"]))
        namespace = str(source["namespace"])
        source_kind = str(source["kind"])
        source_stats = Counter()
        validator = _validate_special if source_kind == "special" else _validate_customized
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                source_stats["input"] += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    reason = "invalid_json"
                    append_jsonl(rejected_path, {"source": str(path), "line": line_no, "reason": reason, "error": str(exc)})
                    source_stats[reason] += 1
                    stats["rejected"][reason] += 1
                    continue
                if not isinstance(raw, dict):
                    reason = "not_object"
                else:
                    reason = validator(raw, limits)
                if reason:
                    append_jsonl(rejected_path, {"source": str(path), "line": line_no, "reason": reason, "id": raw.get("id") if isinstance(raw, dict) else None})
                    source_stats[reason] += 1
                    stats["rejected"][reason] += 1
                    continue
                row = _normalized_row(raw, source)
                original_id = str(raw["id"])
                quality = _quality(row)
                quality_json = canonical_json(list(quality))
                row_json = canonical_json(row)
                row_hash = stable_hash(row)
                existing = connection.execute(
                    "SELECT quality_json,row_hash,source_path,source_line FROM rows WHERE namespace=? AND original_id=?",
                    (namespace, original_id),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?)",
                        (namespace, original_id, source["source_version"], source_kind, quality_json, row_hash, row_json, str(path), line_no),
                    )
                    source_stats["accepted"] += 1
                else:
                    stats["duplicates"][namespace] += 1
                    source_stats["duplicate"] += 1
                    old_quality = tuple(json.loads(existing[0]))
                    if quality > old_quality:
                        append_jsonl(rejected_path, {
                            "source": existing[2], "line": existing[3], "reason": "duplicate_replaced",
                            "id": original_id, "replacement_source": str(path), "replacement_line": line_no,
                        })
                        connection.execute(
                            "UPDATE rows SET source_version=?,source_kind=?,quality_json=?,row_hash=?,row_json=?,source_path=?,source_line=? WHERE namespace=? AND original_id=?",
                            (source["source_version"], source_kind, quality_json, row_hash, row_json, str(path), line_no, namespace, original_id),
                        )
                        source_stats["duplicate_selected"] += 1
                    else:
                        append_jsonl(rejected_path, {"source": str(path), "line": line_no, "reason": "duplicate_not_selected", "id": original_id})
                        source_stats["duplicate_not_selected"] += 1
                if source_stats["input"] % 10000 == 0:
                    connection.commit()
        connection.commit()
        stats["sources"][namespace] = {"path": str(path), **dict(source_stats)}

    outputs: Dict[str, int] = defaultdict(int)
    handles: Dict[str, Any] = {}
    try:
        for source_kind, row_json in connection.execute(
            "SELECT source_kind,row_json FROM rows ORDER BY source_kind,namespace,original_id"
        ):
            path = stage_dir / f"{source_kind}.jsonl"
            if source_kind not in handles:
                handles[source_kind] = path.open("w", encoding="utf-8")
            handles[source_kind].write(row_json + "\n")
            outputs[source_kind] += 1
    finally:
        for handle in handles.values():
            handle.close()
        connection.close()

    if not bool(limits.get("keep_database", False)):
        db_path.unlink(missing_ok=True)

    result = {
        "run_dir": str(run_dir),
        "database": str(db_path) if db_path.exists() else "",
        "outputs": {kind: {"path": str(stage_dir / f"{kind}.jsonl"), "rows": count} for kind, count in outputs.items()},
        "sources": stats["sources"],
        "rejected_counts": dict(stats["rejected"]),
        "duplicate_counts": dict(stats["duplicates"]),
        "rejected_path": str(rejected_path),
    }
    atomic_write_json(stage_dir / "stats.json", result)
    return result
