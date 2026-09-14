#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


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


def text_units(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except (EOFError, FileNotFoundError, OSError, wave.Error):
        return None


def safe_stem(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return f"{stem[:90]}__{digest}"


def write_preview(source: Path, target: Path, segment_sec: float, gap_sec: float) -> List[List[float]]:
    with wave.open(str(source), "rb") as src:
        params = src.getparams()
        total_frames = src.getnframes()
        segment_frames = min(total_frames, max(1, round(segment_sec * params.framerate)))
        starts = list(dict.fromkeys([
            0,
            max(0, (total_frames - segment_frames) // 2),
            max(0, total_frames - segment_frames),
        ]))
        chunks = []
        for start in starts:
            src.setpos(start)
            chunks.append(src.readframes(segment_frames))

    silence = b"\x00" * round(gap_sec * params.framerate) * params.nchannels * params.sampwidth
    with wave.open(str(target), "wb") as dst:
        dst.setparams(params)
        for index, chunk in enumerate(chunks):
            if index:
                dst.writeframesraw(silence)
            dst.writeframesraw(chunk)
    return [
        [round(start / params.framerate, 3), round((start + segment_frames) / params.framerate, 3)]
        for start in starts
    ]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample per-turn TTS WAVs close to their duration cap.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--voice_id", default="")
    parser.add_argument("--min_cap_ratio", type=float, default=0.9)
    parser.add_argument("--samples_per_dataset", type=int, default=5)
    parser.add_argument("--candidate_rows_per_dataset", type=int, default=0, help="Stop after this many candidate rows; 0 scans fully.")
    parser.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    parser.add_argument("--max_sec_per_char", type=float, default=1.2)
    parser.add_argument("--duration_guard_sec", type=float, default=5.0)
    parser.add_argument("--full_audio_max_sec", type=float, default=90.0)
    parser.add_argument("--preview_segment_sec", type=float, default=10.0)
    parser.add_argument("--preview_gap_sec", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidate_counts: Dict[str, int] = defaultdict(int)
    missing_audio = 0

    for manifest_value in args.manifest:
        manifest = Path(manifest_value).resolve()
        dataset = manifest.parent.parent.name
        for row in read_jsonl(manifest):
            source_row = row.get("source_row") or {}
            assets = source_row.get("tts_assets") or {}
            if not isinstance(assets, dict):
                continue
            row_candidates = []
            for key, asset in assets.items():
                if not isinstance(asset, dict):
                    continue
                voice = asset.get("voice") or {}
                voice_id = str(voice.get("voice_id") or "")
                if args.voice_id and voice_id != args.voice_id:
                    continue
                text = str(asset.get("text") or asset.get("tts_text") or "")
                audio = Path(str(asset.get("audio") or ""))
                duration = wav_duration(audio)
                if duration is None:
                    missing_audio += 1
                    continue
                units = text_units(text)
                cap = max(
                    args.max_audio_floor_sec,
                    units * args.max_sec_per_char + args.duration_guard_sec,
                )
                ratio = duration / cap if cap else 0.0
                if ratio < args.min_cap_ratio:
                    continue
                row_candidates.append({
                    "dataset": dataset,
                    "manifest": str(manifest),
                    "sample_id": str(row.get("id") or ""),
                    "scenario": str(row.get("scenario") or ""),
                    "asset_key": str(key),
                    "task_id": str(asset.get("task_id") or ""),
                    "audio": str(audio),
                    "text": text,
                    "text_units": units,
                    "duration_sec": round(duration, 3),
                    "duration_cap_sec": round(cap, 3),
                    "cap_ratio": round(ratio, 6),
                    "voice_id": voice_id,
                    "ref_wav": str(voice.get("ref_wav") or ""),
                    "ref_text": str(voice.get("ref_text") or ""),
                })
            if row_candidates:
                candidate_counts[dataset] += len(row_candidates)
                candidates[dataset].append(rng.choice(row_candidates))
                if args.candidate_rows_per_dataset > 0 and len(candidates[dataset]) >= args.candidate_rows_per_dataset:
                    break

    selected = []
    for dataset in sorted(candidates):
        rows = candidates[dataset]
        rng.shuffle(rows)
        selected.extend(rows[: args.samples_per_dataset])
    rng.shuffle(selected)

    out_dir = Path(args.out_dir).resolve()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    packaged = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["audio"])
        prefix = f"{index:02d}__{row['dataset']}__ratio_{row['cap_ratio']:.3f}"
        target = audio_dir / f"{prefix}__{safe_stem(row['task_id'] or row['sample_id'])}.wav"
        if float(row["duration_sec"]) <= args.full_audio_max_sec:
            shutil.copyfile(source, target)
            preview_type = "full_original"
            ranges: List[List[float]] = []
        else:
            ranges = write_preview(source, target, args.preview_segment_sec, args.preview_gap_sec)
            preview_type = "head_middle_tail"
        packaged.append({
            **row,
            "index": index,
            "packaged_audio": str(target.relative_to(out_dir)),
            "preview_type": preview_type,
            "preview_ranges_sec": ranges,
        })

    write_jsonl(out_dir / "manifest.jsonl", packaged)
    summary = {
        "voice_id": args.voice_id,
        "min_cap_ratio": args.min_cap_ratio,
        "candidate_assets_by_dataset": dict(sorted(candidate_counts.items())),
        "candidate_rows_by_dataset": {key: len(value) for key, value in sorted(candidates.items())},
        "sampled_by_dataset": dict(sorted(
            (key, sum(row["dataset"] == key for row in packaged)) for key in candidates
        )),
        "sampled": len(packaged),
        "missing_audio": missing_audio,
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Near-limit TTS listening sample",
        "",
        f"Filter: voice_id={args.voice_id or 'ANY'}, cap_ratio >= {args.min_cap_ratio}",
        "",
        "For previews longer than the full-audio threshold, the file contains head/middle/tail excerpts.",
        "",
    ]
    for row in packaged:
        lines.extend([
            f"## {row['index']}. {row['dataset']} / {row['scenario']}",
            "",
            f"- Audio: `{row['packaged_audio']}`",
            f"- Duration: `{row['duration_sec']}s` / cap `{row['duration_cap_sec']}s` / ratio `{row['cap_ratio']:.3f}`",
            f"- Text: {row['text']}",
            f"- Sample ID: `{row['sample_id']}`",
            f"- Task ID: `{row['task_id']}`",
            f"- Preview: `{row['preview_type']}`",
            "",
        ])
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
