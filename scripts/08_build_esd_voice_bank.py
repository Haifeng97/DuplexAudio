#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import io
import json
import random
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow.parquet as pq  # type: ignore


DEFAULT_EMOTION_COUNTS = {
    "neutral": 6,
    "happiness": 1,
    "anger": 1,
    "sadness": 1,
    "surprise": 1,
}


def maybe_fix_text(text: str) -> str:
    for encoding in ("latin1", "cp1252"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
        except Exception:
            continue
        if re.search(r"[\u4e00-\u9fff]", fixed):
            return fixed
    return text


def zh_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def parse_speakers(value: str) -> List[str]:
    speakers: List[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            speakers.extend(f"{i:04d}" for i in range(int(start), int(end) + 1))
        else:
            speakers.append(f"{int(part):04d}")
    return speakers


def parse_emotion_counts(value: str) -> Dict[str, int]:
    if not value:
        return dict(DEFAULT_EMOTION_COUNTS)
    counts: Dict[str, int] = {}
    for item in value.split(","):
        name, count = item.split(":", 1)
        counts[name.strip()] = int(count)
    return counts


def iter_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=512,
            columns=["audio", "transcript", "emotion", "speaker_id", "gender", "language"],
        ):
            yield from batch.to_pylist()


def wav_duration(audio_bytes: bytes) -> float:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def good_ref(text: str, duration: float, args: argparse.Namespace) -> bool:
    n = zh_char_count(text)
    return args.min_zh_chars <= n <= args.max_zh_chars and args.min_duration_sec <= duration <= args.max_duration_sec


def choose_evenly(rows: List[Dict[str, Any]], count: int, rng: random.Random) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) <= count:
        return list(rows)
    rows = list(rows)
    rng.shuffle(rows)
    rows.sort(key=lambda row: row["duration"])
    if count == 1:
        return [rows[len(rows) // 2]]
    selected: List[Dict[str, Any]] = []
    for i in range(count):
        idx = round(i * (len(rows) - 1) / (count - 1))
        selected.append(rows[idx])
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an ESD voice_bank.jsonl with balanced speakers and configured emotion ratios.")
    ap.add_argument("--parquet_glob", default="/data/haifengjia/datasets/esd/train-*-of-00007.parquet")
    ap.add_argument("--out_dir", default="outputs/esd_voice_bank_zh_neutral60")
    ap.add_argument("--speakers", default="1-10")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--emotion_counts", default="neutral:6,happiness:1,anger:1,sadness:1,surprise:1")
    ap.add_argument("--min_zh_chars", type=int, default=6)
    ap.add_argument("--max_zh_chars", type=int, default=30)
    ap.add_argument("--min_duration_sec", type=float, default=1.2)
    ap.add_argument("--max_duration_sec", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20260717)
    args = ap.parse_args()

    paths = sorted(Path("/").glob(args.parquet_glob.lstrip("/"))) if args.parquet_glob.startswith("/") else sorted(Path().glob(args.parquet_glob))
    if not paths:
        raise FileNotFoundError(args.parquet_glob)

    speakers = parse_speakers(args.speakers)
    emotion_counts = parse_emotion_counts(args.emotion_counts)
    needed = {(speaker, emotion): count for speaker in speakers for emotion, count in emotion_counts.items()}
    candidates: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in iter_rows(paths):
        if row.get("language") != args.language:
            continue
        key = (str(row.get("speaker_id")), str(row.get("emotion")))
        if key not in needed:
            continue
        audio = row["audio"]
        text = maybe_fix_text(str(row.get("transcript") or ""))
        duration = wav_duration(audio["bytes"])
        if not good_ref(text, duration, args):
            continue
        candidates[key].append(
            {
                "audio": audio,
                "text": text,
                "duration": duration,
                "speaker_id": key[0],
                "emotion": key[1],
                "gender": row.get("gender"),
                "language": row.get("language"),
            }
        )

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    bank_rows: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    for speaker in speakers:
        refs: List[Dict[str, Any]] = []
        gender = None
        for emotion, count in emotion_counts.items():
            selected = choose_evenly(candidates.get((speaker, emotion), []), count, rng)
            if len(selected) < count:
                missing.append({"speaker_id": speaker, "emotion": emotion, "needed": count, "found": len(selected)})
            for idx, item in enumerate(selected):
                gender = item.get("gender") or gender
                source_name = Path(str(item["audio"].get("path") or "sample.wav")).name
                wav_name = f"{speaker}_{emotion}_{idx:02d}_{source_name}"
                wav_path = wav_dir / wav_name
                wav_path.write_bytes(item["audio"]["bytes"])
                ref = {
                    "path": str(wav_path),
                    "text": item["text"],
                    "duration": item["duration"],
                    "snr": None,
                    "emotion": emotion,
                    "source_dataset": "esd",
                    "source_path": item["audio"].get("path"),
                    "gender": item.get("gender"),
                }
                refs.append(ref)
                flat_rows.append({"spk": f"esd_{speaker}", "speaker_id": speaker, **ref})
        bank_rows.append({"spk": f"esd_{speaker}", "speaker_id": speaker, "gender": gender, "lang": args.language, "dataset": "esd", "refs": refs})

    (out_dir / "voice_bank.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in bank_rows),
        encoding="utf-8",
    )
    (out_dir / "refs_flat.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in flat_rows),
        encoding="utf-8",
    )
    stats = {
        "out_dir": str(out_dir),
        "voice_bank": str(out_dir / "voice_bank.jsonl"),
        "speakers": len(bank_rows),
        "refs": len(flat_rows),
        "emotion_counts_per_speaker": emotion_counts,
        "missing": missing,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
