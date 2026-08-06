#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import random
import re
import tarfile
import sys
import wave
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List


BACKCHANNEL_TAG = "<BACKCHANNEL>"
ALLOWED_BACKCHANNEL_TEXTS = frozenset({
    "是的",
    "对对对",
    "对啊",
    "对呀",
    "对",
    "嗯对",
    "嗯",
    "啊",
    "哦",
    "是",
    "是吧",
    "对对",
    "哎",
    "呃",
    "对吧",
    "嗯嗯",
    "诶",
    "啊对",
    "是啊",
    "对的",
    "嗯是",
    "好",
    "哎呀",
    "哦对",
    "噢",
    "嗯对对对",
    "嗯嗯嗯",
    "嗯是的",
    "确实",
    "行",
    "好的",
    "知道",
    "明白",
    "就是",
    "没错",
})


@dataclass
class Clip:
    clip_id: str
    archive: str
    wav_member: str
    text: str
    duration_sec: float
    sample_rate: int
    channels: int
    sample_width: int
    speaker: str
    gender: str
    emotion: str
    language: str
    task: str


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_member_text(tf: tarfile.TarFile, members: Dict[str, tarfile.TarInfo], name: str) -> str:
    member = members.get(name)
    if member is None:
        return ""
    fileobj = tf.extractfile(member)
    if fileobj is None:
        return ""
    return fileobj.read().decode("utf-8", errors="ignore").strip()


def clean_text(raw: str) -> str:
    text = raw.replace(BACKCHANNEL_TAG, "")
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[\+\]", "", text)
    return re.sub(r"\s+", "", text).strip()


def text_chars(text: str) -> int:
    return len(re.sub(r"[\s，。！？、,.!?;；:：~～…]+", "", text))


def balance_key(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：~～…]+", "", text).lower()


def wav_info(tf: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[int, int, int, int]:
    fileobj = tf.extractfile(member)
    if fileobj is None:
        raise ValueError("missing wav payload")
    with wave.open(fileobj, "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()


def has_energy_speech(tf: tarfile.TarFile, member: tarfile.TarInfo, sample_rate: int, channels: int, sample_width: int) -> bool:
    if sample_width != 2 or sample_rate <= 0 or channels <= 0:
        return False
    fileobj = tf.extractfile(member)
    if fileobj is None:
        return False
    with wave.open(fileobj, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    frame_n = max(1, int(round(sample_rate * 0.02)) * channels)
    energies = []
    for start in range(0, len(samples), frame_n):
        frame = samples[start:start + frame_n]
        if frame:
            energies.append(math.sqrt(sum(value * value for value in frame) / len(frame)))
    if not energies or max(energies) <= 0:
        return False
    peak = max(energies)
    floor = sorted(energies)[min(len(energies) - 1, int(len(energies) * 0.2))]
    threshold = max(peak * 0.08, floor * 2.5, 80.0)
    return any(value >= threshold for value in energies)


def clip_id(archive: Path, wav_member: str) -> str:
    payload = f"{archive.name}:{wav_member}".encode("utf-8")
    return "ramc_" + hashlib.sha256(payload).hexdigest()[:20]


def scan_archive(path: Path, args: argparse.Namespace) -> tuple[List[Clip], Counter]:
    clips: List[Clip] = []
    rejected: Counter = Counter()
    with tarfile.open(path, "r") as tf:
        members = {member.name: member for member in tf.getmembers() if member.isfile()}
        wav_members = sorted(
            (member for member in members.values() if member.name.lower().endswith(".wav")),
            key=lambda member: member.name,
        )
        for wav_member in wav_members:
            base = wav_member.name[:-4]
            raw_text = read_member_text(tf, members, base + ".txt")
            task = read_member_text(tf, members, base + ".task")
            language = read_member_text(tf, members, base + ".lang")
            text = clean_text(raw_text)
            residual = re.sub(r"[\u3400-\u9fff\s，。！？、,.!?;；:：~～…]+", "", text.lower())
            if residual and not re.fullmatch(r"(?:ok)+", residual):
                rejected["non_chinese_text"] += 1
                continue

            if BACKCHANNEL_TAG not in raw_text and BACKCHANNEL_TAG not in task:
                rejected["not_backchannel"] += 1
                continue
            if not args.allow_non_cn and "<CN>" not in language:
                rejected["not_cn"] += 1
                continue
            if balance_key(text) not in ALLOWED_BACKCHANNEL_TEXTS:
                rejected["text_not_allowed"] += 1
                continue
            chars = text_chars(text)
            if chars < args.min_text_chars:
                rejected["text_too_short"] += 1
                continue
            if args.max_text_chars > 0 and chars > args.max_text_chars:
                rejected["text_too_long"] += 1
                continue
            try:
                sample_rate, channels, sample_width, frames = wav_info(tf, wav_member)
            except Exception:
                rejected["bad_wav"] += 1
                continue
            if sample_rate <= 0 or channels <= 0 or sample_width <= 0:
                rejected["bad_wav"] += 1
                continue
            duration_sec = frames / sample_rate
            if duration_sec < args.min_duration_sec:
                rejected["audio_too_short"] += 1
                continue
            if args.max_duration_sec > 0 and duration_sec > args.max_duration_sec:
                rejected["audio_too_long"] += 1
                continue
            if not has_energy_speech(tf, wav_member, sample_rate, channels, sample_width):
                rejected["audio_no_speech"] += 1
                continue
            clips.append(
                Clip(
                    clip_id=clip_id(path, wav_member.name),
                    archive=str(path.resolve()),
                    wav_member=wav_member.name,
                    text=text,
                    duration_sec=round(duration_sec, 6),
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_width=sample_width,
                    speaker=read_member_text(tf, members, base + ".speaker"),
                    gender=read_member_text(tf, members, base + ".gender"),
                    emotion=read_member_text(tf, members, base + ".emotion"),
                    language=language,
                    task=task,
                )
            )
    return clips, rejected


def balanced_selection(clips: List[Clip], max_per_text: int, max_total: int, seed: int) -> List[Clip]:
    rng = random.Random(seed)
    by_text: Dict[str, List[Clip]] = defaultdict(list)
    for clip in clips:
        by_text[balance_key(clip.text)].append(clip)
    selected: List[Clip] = []
    for key in sorted(by_text):
        group = by_text[key]
        rng.shuffle(group)
        selected.extend(group[:max_per_text] if max_per_text > 0 else group)
    rng.shuffle(selected)
    if max_total > 0:
        selected = selected[:max_total]
    return sorted(selected, key=lambda clip: clip.clip_id)


def extract_audio(clips: List[Clip], wav_dir: Path) -> tuple[int, int]:
    wav_dir.mkdir(parents=True, exist_ok=True)
    by_archive: Dict[str, List[Clip]] = defaultdict(list)
    for clip in clips:
        by_archive[clip.archive].append(clip)

    written = cached = 0
    for archive, archive_clips in sorted(by_archive.items()):
        with tarfile.open(archive, "r") as tf:
            for clip in archive_clips:
                out = wav_dir / f"{clip.clip_id}.wav"
                if out.exists() and out.stat().st_size > 44:
                    cached += 1
                    continue
                fileobj = tf.extractfile(clip.wav_member)
                if fileobj is None:
                    raise FileNotFoundError(f"{archive}:{clip.wav_member}")
                tmp = out.with_name(out.name + ".tmp")
                tmp.write_bytes(fileobj.read())
                tmp.replace(out)
                written += 1
    return written, cached


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract balanced Chinese backchannel clips from MagicData RAMC tar shards.")
    ap.add_argument(
        "--input_dir",
        default="/data/haifengjia/datasets/Easy-Turn-Trainset/trainset/magicdata_ramc",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_duration_sec", type=float, default=0.2)
    ap.add_argument("--max_duration_sec", type=float, default=2.0)
    ap.add_argument("--min_text_chars", type=int, default=1)
    ap.add_argument("--max_text_chars", type=int, default=8)
    ap.add_argument("--max_per_text", type=int, default=200)
    ap.add_argument("--max_total", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--allow_non_cn", action="store_true")
    ap.add_argument("--index_only", action="store_true")
    args = ap.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    input_dir = Path(args.input_dir)
    archives = sorted(input_dir.glob("backchannel_*.tar"))
    if not archives:
        raise FileNotFoundError(f"no backchannel_*.tar under {input_dir}")

    rejected: Counter = Counter()
    clips: List[Clip] = []
    if args.workers == 1:
        scan_results = (scan_archive(archive, args) for archive in archives)
        for archive_clips, archive_rejected in scan_results:
            clips.extend(archive_clips)
            rejected.update(archive_rejected)
    else:
        max_workers = min(args.workers, len(archives))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for archive_clips, archive_rejected in executor.map(scan_archive, archives, repeat(args)):
                clips.extend(archive_clips)
                rejected.update(archive_rejected)

    selected = balanced_selection(clips, args.max_per_text, args.max_total, args.seed)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    written = cached = 0
    if not args.index_only:
        written, cached = extract_audio(selected, wav_dir)

    rows = []
    for clip in selected:
        row = asdict(clip)
        row["audio"] = str((wav_dir / f"{clip.clip_id}.wav").resolve())
        row["source_dataset"] = "magicdata_ramc"
        rows.append(row)
    manifest = out_dir / "backchannels.jsonl"
    write_jsonl(manifest, rows)

    all_text_counts = Counter(clip.text for clip in clips)
    selected_text_counts = Counter(clip.text for clip in selected)
    stats = {
        "input_dir": str(input_dir),
        "archives": len(archives),
        "eligible_before_balance": len(clips),
        "selected": len(selected),
        "unique_texts_before_balance": len(all_text_counts),
        "unique_texts_selected": len(selected_text_counts),
        "allowed_texts": sorted(ALLOWED_BACKCHANNEL_TEXTS),
        "rejected": dict(rejected),
        "top_texts_before_balance": all_text_counts.most_common(30),
        "top_texts_selected": selected_text_counts.most_common(30),
        "audio_written": written,
        "audio_cached": cached,
        "index_only": bool(args.index_only),
        "manifest": str(manifest),
        "wav_dir": str(wav_dir),
        "args": vars(args),
    }
    stats_path = out_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
