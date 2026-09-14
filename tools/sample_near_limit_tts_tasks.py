#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


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


def parse_source(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"task source must be LABEL=PATH: {value}")
    label, path = value.split("=", 1)
    return label, Path(path).resolve()


def text_units(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except (EOFError, FileNotFoundError, OSError, wave.Error):
        return None


def likely_duration(path: Path) -> float | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    return max(0, size - 44) / 48_000


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


def select_diverse(
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]],
    *,
    rng: random.Random,
    max_per_dataset: int,
    max_per_voice: int,
) -> List[Dict[str, Any]]:
    by_dataset: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    for (dataset, voice_id), rows in grouped.items():
        rng.shuffle(rows)
        by_dataset[dataset][voice_id] = rows

    selected = []
    for dataset in sorted(by_dataset):
        voices = list(by_dataset[dataset])
        rng.shuffle(voices)
        used = Counter()
        while len([row for row in selected if row["dataset"] == dataset]) < max_per_dataset:
            added = False
            for voice_id in voices:
                rows = by_dataset[dataset][voice_id]
                if rows and used[voice_id] < max_per_voice:
                    selected.append(rows.pop())
                    used[voice_id] += 1
                    added = True
                    if len([row for row in selected if row["dataset"] == dataset]) >= max_per_dataset:
                        break
            if not added:
                break
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample near-duration-limit WAVs directly from TTS task files.")
    parser.add_argument("--tasks", action="append", required=True, help="LABEL=PATH")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--exclude_voice_id", action="append", default=[])
    parser.add_argument("--min_cap_ratio", type=float, default=0.8)
    parser.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    parser.add_argument("--max_sec_per_char", type=float, default=1.2)
    parser.add_argument("--duration_guard_sec", type=float, default=5.0)
    parser.add_argument("--max_per_dataset", type=int, default=10)
    parser.add_argument("--max_per_voice", type=int, default=2)
    parser.add_argument("--reservoir_per_group", type=int, default=50)
    parser.add_argument("--full_audio_max_sec", type=float, default=90.0)
    parser.add_argument("--preview_segment_sec", type=float, default=10.0)
    parser.add_argument("--preview_gap_sec", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    excluded = set(args.exclude_voice_id)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    seen_per_group = Counter()
    source_counts = Counter()
    candidate_counts = Counter()
    missing_audio = 0
    seen_outputs = set()

    for source_value in args.tasks:
        dataset, task_path = parse_source(source_value)
        for task in read_jsonl(task_path):
            source_counts[dataset] += 1
            voice_id = str(task.get("voice_id") or "")
            if not voice_id or voice_id in excluded:
                continue
            audio = Path(str(task.get("out") or ""))
            audio_key = str(audio)
            if not audio_key or audio_key in seen_outputs:
                continue
            text = str(task.get("source_text") or task.get("text") or "")
            units = text_units(text)
            cap = max(args.max_audio_floor_sec, units * args.max_sec_per_char + args.duration_guard_sec)
            approximate = likely_duration(audio)
            if approximate is None:
                missing_audio += 1
                continue
            if approximate / cap < max(0.0, args.min_cap_ratio - 0.03):
                continue
            duration = wav_duration(audio)
            if duration is None:
                missing_audio += 1
                continue
            ratio = duration / cap if cap else 0.0
            if ratio < args.min_cap_ratio:
                continue
            seen_outputs.add(audio_key)
            candidate_counts[dataset] += 1
            key = (dataset, voice_id)
            seen_per_group[key] += 1
            row = {
                "dataset": dataset,
                "tasks": str(task_path),
                "sample_id": str(task.get("sample_id") or ""),
                "asset_key": str(task.get("key") or ""),
                "task_id": str(task.get("id") or ""),
                "audio": str(audio),
                "text": text,
                "text_units": units,
                "duration_sec": round(duration, 3),
                "duration_cap_sec": round(cap, 3),
                "cap_ratio": round(ratio, 6),
                "voice_id": voice_id,
                "ref_wav": str(task.get("ref_wav") or ""),
                "ref_text": str(task.get("ref_text") or ""),
            }
            reservoir = grouped[key]
            if len(reservoir) < args.reservoir_per_group:
                reservoir.append(row)
            else:
                position = rng.randrange(seen_per_group[key])
                if position < args.reservoir_per_group:
                    reservoir[position] = row

    selected = select_diverse(
        grouped,
        rng=rng,
        max_per_dataset=args.max_per_dataset,
        max_per_voice=args.max_per_voice,
    )
    selected.sort(key=lambda row: (-row["cap_ratio"], row["dataset"], row["voice_id"]))

    out_dir = Path(args.out_dir).resolve()
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    packaged = []
    for index, row in enumerate(selected, start=1):
        source = Path(row["audio"])
        prefix = f"{index:02d}__{row['dataset']}__{row['voice_id'].replace(':', '_')}__ratio_{row['cap_ratio']:.3f}"
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

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in packaged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "excluded_voice_ids": sorted(excluded),
        "min_cap_ratio": args.min_cap_ratio,
        "source_tasks": dict(source_counts),
        "candidate_tasks": dict(candidate_counts),
        "candidate_voice_groups": len(grouped),
        "sampled": len(packaged),
        "sampled_by_dataset": dict(Counter(row["dataset"] for row in packaged)),
        "sampled_by_band": {
            "ratio_gte_0.9": sum(row["cap_ratio"] >= 0.9 for row in packaged),
            "ratio_0.8_to_0.9": sum(0.8 <= row["cap_ratio"] < 0.9 for row in packaged),
        },
        "missing_audio": missing_audio,
        "seed": args.seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Other-reference near-limit TTS samples",
        "",
        f"Excluded voices: {', '.join(sorted(excluded))}",
        f"Minimum cap ratio: {args.min_cap_ratio}",
        "",
    ]
    for row in packaged:
        lines.extend([
            f"## {row['index']}. {row['dataset']} / {row['voice_id']}",
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
