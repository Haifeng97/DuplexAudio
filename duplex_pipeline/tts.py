from __future__ import annotations

import json
import sqlite3
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash


def valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getframerate() > 0 and wav.getnframes() > 0 and wav.getnchannels() > 0
    except (EOFError, OSError, wave.Error):
        return False


def tts_fingerprint(task: Dict[str, Any], tts_config: Dict[str, Any]) -> str:
    ref_path = Path(str(task.get("ref_wav") or "")).expanduser()
    payload = {
        "tts_text": str(task.get("text") or ""),
        "ref_wav": str(ref_path.resolve()),
        "ref_text": str(task.get("ref_text") or ""),
        "engine": str(tts_config.get("engine") or ""),
        "model": str(tts_config.get("model") or ""),
        "model_version": str(tts_config.get("model_version") or ""),
        "language": str(tts_config.get("language") or ""),
        "sample_rate": int(tts_config.get("sample_rate", 24000)),
        "generation": tts_config.get("generation") or {},
    }
    return stable_hash(payload)


def _legacy_cache(tts_config: Dict[str, Any]) -> Dict[str, str]:
    cache: Dict[str, str] = {}
    for raw_root in tts_config.get("legacy_task_roots", []):
        root = Path(str(raw_root))
        if not root.exists():
            continue
        for task_path in root.rglob("tts_tasks.jsonl"):
            for task in iter_jsonl(task_path):
                wav = Path(str(task.get("out") or ""))
                if valid_wav(wav):
                    cache.setdefault(tts_fingerprint(task, tts_config), str(wav.resolve()))
    return cache


def fingerprint_tasks(
    config: Dict[str, Any],
    tasks_path: Path,
    index_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    tts_config = dict(config["tts"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = _legacy_cache(tts_config)
    task_output = output_dir / "tts_tasks.jsonl"
    index_output = output_dir / "scenario_index.jsonl"
    wav_root = output_dir / "query_wav"
    db_path = output_dir / "fingerprints.sqlite3"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("CREATE TABLE mapping (legacy_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,out TEXT NOT NULL,fingerprint TEXT NOT NULL)")
    connection.execute("CREATE TABLE tasks (task_id TEXT PRIMARY KEY,row_json TEXT NOT NULL)")
    reused = duplicates = 0
    input_tasks = 0
    for task in iter_jsonl(tasks_path):
        input_tasks += 1
        old_id = str(task["id"])
        fingerprint = tts_fingerprint(task, tts_config)
        task_id = f"tts_{fingerprint[:32]}"
        cached_wav = cache.get(fingerprint)
        out = cached_wav or str((wav_root / fingerprint[:2] / f"{fingerprint}.wav").resolve())
        rewritten = dict(task)
        rewritten.update({
            "id": task_id,
            "legacy_task_id": old_id,
            "tts_fingerprint": fingerprint,
            "out": out,
            "tts_model": tts_config.get("model"),
            "tts_model_version": tts_config.get("model_version"),
            "tts_generation": tts_config.get("generation") or {},
        })
        try:
            connection.execute("INSERT INTO mapping VALUES (?,?,?,?)", (old_id, task_id, out, fingerprint))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate legacy TTS task id: {old_id}") from exc
        inserted = connection.execute("INSERT OR IGNORE INTO tasks VALUES (?,?)", (task_id, canonical_json(rewritten))).rowcount
        if not inserted:
            duplicates += 1
        else:
            reused += int(bool(cached_wav))
        if input_tasks % 10000 == 0:
            connection.commit()
    connection.commit()
    with task_output.open("w", encoding="utf-8") as handle:
        for row_json, in connection.execute("SELECT row_json FROM tasks ORDER BY task_id"):
            handle.write(row_json + "\n")

    rows = 0
    with index_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(index_path):
            assets = row.get("tts_assets") if isinstance(row.get("tts_assets"), dict) else {}
            for asset in assets.values():
                if not isinstance(asset, dict) or not asset.get("task_id"):
                    continue
                mapped = connection.execute(
                    "SELECT task_id,out,fingerprint FROM mapping WHERE legacy_id=?",
                    (str(asset["task_id"]),),
                ).fetchone()
                if mapped is None:
                    raise ValueError(f"index references unknown task {asset['task_id']}")
                asset["legacy_task_id"] = asset["task_id"]
                asset["task_id"], asset["audio"], asset["tts_fingerprint"] = mapped
            handle.write(canonical_json(row) + "\n")
            rows += 1
    unique_tasks = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    connection.close()
    db_path.unlink(missing_ok=True)
    result = {
        "input_tasks": str(tasks_path), "input_index": str(index_path),
        "output_tasks": str(task_output), "output_index": str(index_output),
        "input_tasks": input_tasks, "unique_tasks": unique_tasks, "duplicate_assets": duplicates,
        "legacy_cache_entries": len(cache), "legacy_reused": reused, "index_rows": rows,
        "temporary_database_retained": False,
    }
    atomic_write_json(output_dir / "fingerprint_stats.json", result)
    return result
