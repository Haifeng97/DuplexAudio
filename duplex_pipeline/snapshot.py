from __future__ import annotations

import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .io import atomic_write_json, canonical_json, iter_jsonl


READY_STATUSES = {"ok", "cached"}


def _load_ready_tasks(results_dir: Path) -> Tuple[set[str], Counter]:
    ready: set[str] = set()
    counts: Counter = Counter()
    paths = sorted(results_dir.glob("tts_results_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no TTS result files under {results_dir}")
    for path in paths:
        for row in iter_jsonl(path):
            status = str(row.get("status") or "missing_status")
            counts[status] += 1
            task_id = str(row.get("id") or "")
            if task_id and status in READY_STATUSES:
                ready.add(task_id)
    return ready, counts


def _required_assets(row: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    assets = row.get("tts_assets")
    if not isinstance(assets, dict):
        return ()
    return (
        asset
        for asset in assets.values()
        if isinstance(asset, dict) and asset.get("task_id")
    )


def _asset_error(
    asset: Dict[str, Any],
    *,
    min_audio_sec: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
    duration_guard_sec: float,
    max_audio_cap_ratio: float,
) -> str:
    audio = Path(str(asset.get("audio") or ""))
    try:
        with wave.open(str(audio), "rb") as wav:
            sample_rate = wav.getframerate()
            duration_sec = wav.getnframes() / sample_rate if sample_rate else 0.0
    except (EOFError, FileNotFoundError, OSError, wave.Error):
        return "invalid_wav"
    if duration_sec < min_audio_sec:
        return "audio_too_short"
    text_units = sum(1 for char in str(asset.get("text") or "") if not char.isspace())
    max_audio_sec = max(max_audio_floor_sec, text_units * max_sec_per_char + duration_guard_sec)
    if max_sec_per_char > 0 and duration_sec > max_audio_sec:
        return "audio_too_long_for_text"
    if max_audio_cap_ratio > 0 and duration_sec >= max_audio_sec * max_audio_cap_ratio:
        return "generation_reached_duration_cap"
    return ""


def _snapshot_index(
    source: Path,
    output: Path,
    ready: set[str],
    *,
    min_audio_sec: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
    duration_guard_sec: float,
    max_audio_cap_ratio: float,
) -> Dict[str, Any]:
    counts: Counter = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            counts["input"] += 1
            assets = list(_required_assets(row))
            if not assets:
                counts["rejected:no_tts_assets"] += 1
                continue
            if any(str(asset["task_id"]) not in ready for asset in assets):
                counts["rejected:missing_task"] += 1
                continue
            errors = Counter(
                error
                for asset in assets
                if (error := _asset_error(
                    asset,
                    min_audio_sec=min_audio_sec,
                    max_audio_floor_sec=max_audio_floor_sec,
                    max_sec_per_char=max_sec_per_char,
                    duration_guard_sec=duration_guard_sec,
                    max_audio_cap_ratio=max_audio_cap_ratio,
                ))
            )
            if errors:
                for error, count in errors.items():
                    counts[f"rejected:{error}"] += count
                continue
            handle.write(canonical_json(row) + "\n")
            counts["ready"] += 1
    return {"source": str(source), "output": str(output), "counts": dict(counts)}


def snapshot_ready_indexes(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    results_dir = run_dir / "05_tts" / "run" / "auto_results"
    fingerprinted = run_dir / "05_tts" / "fingerprinted"
    output_dir = run_dir / "05_tts" / "timed_snapshot"
    ready, result_counts = _load_ready_tasks(results_dir)
    tts = dict(config.get("tts") or {})
    generation = dict(tts.get("generation") or {})
    limits = {
        "min_audio_sec": float(tts.get("min_audio_sec", 1.0)),
        "max_audio_floor_sec": float(generation.get("max_audio_floor_sec", 10.0)),
        "max_sec_per_char": float(generation.get("max_sec_per_char", 1.2)),
        "duration_guard_sec": float(generation.get("generation_guard_sec", 5.0)),
        "max_audio_cap_ratio": float(generation.get("max_audio_cap_ratio", 0.9)),
    }
    indexes = {
        name: _snapshot_index(
            fingerprinted / f"{name}_index.jsonl",
            output_dir / f"{name}_index.jsonl",
            ready,
            **limits,
        )
        for name in ("customized", "special")
    }
    stats = {
        "results_dir": str(results_dir),
        "ready_task_ids": len(ready),
        "result_status_counts": dict(result_counts),
        "audio_quality": limits,
        "indexes": indexes,
    }
    atomic_write_json(output_dir / "snapshot_stats.json", stats)
    return stats
