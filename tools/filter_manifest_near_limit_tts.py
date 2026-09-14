#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


def iter_jsonl_lines(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line, row


def text_units(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def approximate_pcm24k_duration(path: Path) -> float | None:
    try:
        return max(0, path.stat().st_size - 44) / 48_000
    except OSError:
        return None


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            return wav.getnframes() / sample_rate if sample_rate else None
    except (EOFError, FileNotFoundError, OSError, wave.Error):
        return None


def tts_assets(row: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    source_row = row.get("source_row") or {}
    assets = source_row.get("tts_assets") or {}
    if not isinstance(assets, dict):
        return
    for key, asset in assets.items():
        if isinstance(asset, dict) and asset.get("audio"):
            yield str(key), asset


def bad_assets(
    row: Dict[str, Any],
    *,
    min_cap_ratio: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
    duration_guard_sec: float,
) -> Tuple[List[Dict[str, Any]], int]:
    bad = []
    missing = 0
    for key, asset in tts_assets(row):
        text = str(asset.get("text") or asset.get("tts_text") or "")
        units = text_units(text)
        cap = max(max_audio_floor_sec, units * max_sec_per_char + duration_guard_sec)
        audio = Path(str(asset["audio"]))
        approximate = approximate_pcm24k_duration(audio)
        if approximate is None:
            missing += 1
            continue
        if approximate / cap < max(0.0, min_cap_ratio - 0.03):
            continue
        duration = wav_duration(audio)
        if duration is None:
            missing += 1
            continue
        ratio = duration / cap if cap else 0.0
        if ratio < min_cap_ratio:
            continue
        voice = asset.get("voice") or {}
        bad.append({
            "asset_key": key,
            "task_id": str(asset.get("task_id") or ""),
            "audio": str(audio),
            "text": text,
            "text_units": units,
            "duration_sec": round(duration, 3),
            "duration_cap_sec": round(cap, 3),
            "cap_ratio": round(ratio, 6),
            "voice_id": str(voice.get("voice_id") or asset.get("voice_id") or ""),
        })
    return bad, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop manifest rows containing near-limit TTS assets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", default="")
    parser.add_argument("--stats", default="")
    parser.add_argument("--min_cap_ratio", type=float, default=0.9)
    parser.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    parser.add_argument("--max_sec_per_char", type=float, default=1.2)
    parser.add_argument("--duration_guard_sec", type=float, default=5.0)
    parser.add_argument("--progress_every", type=int, default=10000)
    args = parser.parse_args()

    if not 0 < args.min_cap_ratio <= 1:
        raise SystemExit("--min_cap_ratio must be in (0, 1]")
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    rejected = Path(args.rejected).resolve() if args.rejected else output.with_name("rejected_near_limit_tts.jsonl")
    stats_path = Path(args.stats).resolve() if args.stats else output.with_name("filter_stats.json")
    for path in (output, rejected, stats_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rejected_tmp = rejected.with_name(f".{rejected.name}.tmp-{os.getpid()}")

    counts = Counter()
    dropped_scenarios = Counter()
    dropped_voices = Counter()
    try:
        with output_tmp.open("w", encoding="utf-8") as kept_handle, rejected_tmp.open("w", encoding="utf-8") as rejected_handle:
            for original_line, row in iter_jsonl_lines(source):
                counts["input"] += 1
                bad, missing = bad_assets(
                    row,
                    min_cap_ratio=args.min_cap_ratio,
                    max_audio_floor_sec=args.max_audio_floor_sec,
                    max_sec_per_char=args.max_sec_per_char,
                    duration_guard_sec=args.duration_guard_sec,
                )
                counts["missing_audio_assets"] += missing
                if bad:
                    counts["dropped"] += 1
                    counts["dropped_assets"] += len(bad)
                    scenario = str(row.get("scenario") or "")
                    dropped_scenarios[scenario] += 1
                    dropped_voices.update(asset["voice_id"] or "missing_voice_id" for asset in bad)
                    rejected_handle.write(json.dumps({
                        "id": str(row.get("id") or ""),
                        "scenario": scenario,
                        "bad_assets": bad,
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                else:
                    kept_handle.write(original_line if original_line.endswith("\n") else original_line + "\n")
                    counts["kept"] += 1
                if args.progress_every > 0 and counts["input"] % args.progress_every == 0:
                    print(
                        f"FILTER {counts['input']} kept={counts['kept']} dropped={counts['dropped']}",
                        flush=True,
                    )
        output_tmp.replace(output)
        rejected_tmp.replace(rejected)
    except BaseException:
        output_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise

    stats = {
        "input": str(source),
        "output": str(output),
        "rejected": str(rejected),
        "thresholds": {
            "min_cap_ratio": args.min_cap_ratio,
            "max_audio_floor_sec": args.max_audio_floor_sec,
            "max_sec_per_char": args.max_sec_per_char,
            "duration_guard_sec": args.duration_guard_sec,
        },
        "counts": dict(counts),
        "dropped_scenarios": dict(dropped_scenarios),
        "dropped_voices": dict(dropped_voices.most_common()),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
