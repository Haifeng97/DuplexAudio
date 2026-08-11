#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


READY_STATUSES = {"ok", "cached"}


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def wav_info(path: Path) -> Dict[str, Any] | None:
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            return {
                "sample_rate": sample_rate,
                "frames": frames,
                "duration_sec": frames / sample_rate if sample_rate else 0.0,
                "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth(),
            }
    except (EOFError, FileNotFoundError, OSError, wave.Error):
        return None


def safe_name(task_id: str, suffix: str = ".wav") -> str:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in task_id)
    return f"{stem[:150]}__{digest}{suffix}"


def even_sample(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (float(row["duration_sec"]), str(row["id"])))
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[index] for index in indices]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def copy_short(row: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    source = Path(row["out"])
    target = out_dir / safe_name(str(row["id"]))
    shutil.copyfile(source, target)
    return {**row, "packaged_audio": str(target), "preview_type": "full_original"}


def write_long_preview(row: Dict[str, Any], out_dir: Path, segment_sec: float, gap_sec: float) -> Dict[str, Any]:
    source = Path(row["out"])
    target = out_dir / safe_name(str(row["id"]), ".preview.wav")
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        total_frames = src.getnframes()
        segment_frames = max(1, min(total_frames, int(round(segment_sec * params.framerate))))
        starts = [
            0,
            max(0, (total_frames - segment_frames) // 2),
            max(0, total_frames - segment_frames),
        ]
        starts = list(dict.fromkeys(starts))
        pieces = []
        for start in starts:
            src.setpos(start)
            pieces.append(src.readframes(segment_frames))
        silence = b"\x00" * int(round(gap_sec * params.framerate)) * params.nchannels * params.sampwidth
        with wave.open(str(target), "wb") as dst:
            dst.setparams(params)
            for index, piece in enumerate(pieces):
                if index:
                    dst.writeframesraw(silence)
                dst.writeframesraw(piece)
    segment_ranges = [
        [round(start / params.framerate, 3), round((start + segment_frames) / params.framerate, 3)]
        for start in starts
    ]
    return {
        **row,
        "packaged_audio": str(target),
        "preview_type": "head_middle_tail",
        "preview_segment_ranges_sec": segment_ranges,
        "preview_gap_sec": gap_sec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Package representative short and runaway TTS WAV files for audit.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--result_pattern", default="tts_results_*.jsonl")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--short_sec", type=float, default=1.0)
    parser.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    parser.add_argument("--max_sec_per_char", type=float, default=1.2)
    parser.add_argument("--short_samples", type=int, default=50)
    parser.add_argument("--long_preview_samples", type=int, default=50)
    parser.add_argument("--long_full_samples", type=int, default=5)
    parser.add_argument("--preview_segment_sec", type=float, default=10.0)
    parser.add_argument("--preview_gap_sec", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    short_dir = out_dir / "short_full"
    preview_dir = out_dir / "long_preview"
    full_dir = out_dir / "long_full"
    for path in (short_dir, preview_dir, full_dir):
        path.mkdir(parents=True, exist_ok=True)

    anomalies: Dict[str, Dict[str, Any]] = {}
    status_counts: Counter = Counter()
    for result_path in sorted(Path(args.results_dir).glob(args.result_pattern)):
        for row in read_jsonl(result_path):
            status = str(row.get("status") or "missing_status")
            status_counts[status] += 1
            if status not in READY_STATUSES or not row.get("id") or not row.get("out"):
                continue
            path = Path(str(row["out"]))
            info = wav_info(path)
            if info is None:
                continue
            anomalies[str(row["id"])] = {
                "id": str(row["id"]),
                "status": status,
                "out": str(path),
                **info,
            }

    for task in read_jsonl(Path(args.tasks)):
        task_id = str(task.get("id") or "")
        row = anomalies.get(task_id)
        if row is None:
            continue
        text = str(task.get("source_text") or task.get("text") or "")
        max_allowed = max(args.max_audio_floor_sec, len(text.strip()) * args.max_sec_per_char)
        reasons = []
        if float(row["duration_sec"]) < args.short_sec:
            reasons.append("audio_too_short")
        if args.max_sec_per_char > 0 and float(row["duration_sec"]) > max_allowed:
            reasons.append("audio_too_long_for_text")
        if not reasons:
            del anomalies[task_id]
            continue
        row.update({
            "reason": reasons,
            "text": text,
            "tts_text": task.get("text"),
            "text_chars": len(text.strip()),
            "max_allowed_sec": round(max_allowed, 3),
            "sample_id": task.get("sample_id"),
            "key": task.get("key"),
            "voice_id": task.get("voice_id"),
            "ref_wav": task.get("ref_wav"),
        })

    rows = sorted(anomalies.values(), key=lambda row: str(row["id"]))
    short_rows = [row for row in rows if "audio_too_short" in row["reason"]]
    long_rows = [row for row in rows if "audio_too_long_for_text" in row["reason"]]
    sampled: List[Dict[str, Any]] = []
    sampled.extend(copy_short(row, short_dir) for row in even_sample(short_rows, args.short_samples))
    sampled.extend(
        write_long_preview(row, preview_dir, args.preview_segment_sec, args.preview_gap_sec)
        for row in even_sample(long_rows, args.long_preview_samples)
    )
    full_long = sorted(long_rows, key=lambda row: (-float(row["duration_sec"]), str(row["id"])))[:args.long_full_samples]
    for row in full_long:
        source = Path(row["out"])
        target = full_dir / safe_name(str(row["id"]))
        shutil.copyfile(source, target)
        sampled.append({**row, "packaged_audio": str(target), "preview_type": "full_original"})

    write_jsonl(out_dir / "all_anomalies.jsonl", rows)
    write_jsonl(out_dir / "sample_manifest.jsonl", sampled)
    summary = {
        "tasks": str(Path(args.tasks).resolve()),
        "results_dir": str(Path(args.results_dir).resolve()),
        "status_counts": dict(status_counts),
        "thresholds": {
            "short_sec": args.short_sec,
            "max_audio_floor_sec": args.max_audio_floor_sec,
            "max_sec_per_char": args.max_sec_per_char,
        },
        "anomalies": {
            "total": len(rows),
            "short": len(short_rows),
            "long": len(long_rows),
        },
        "packaged": {
            "short_full": min(len(short_rows), args.short_samples),
            "long_preview": min(len(long_rows), args.long_preview_samples),
            "long_full": min(len(long_rows), args.long_full_samples),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "short_full contains complete WAV files shorter than the configured minimum.\n"
        "long_preview contains head/middle/tail excerpts separated by 0.25 seconds of silence.\n"
        "long_full contains a few complete runaway WAV files.\n"
        "all_anomalies.jsonl records every detected anomaly and its original path.\n"
        "sample_manifest.jsonl maps packaged files to text, duration, reason, and preview ranges.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
