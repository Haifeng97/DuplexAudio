#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import threading
import wave
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from duplex_label_protocol import (
    EOR,
    FD_A_ANSWER,
    FD_D_WAIT,
    FD_F_WAIT,
    FD_G_INTERRUPT,
    FD_H_CONTINUE,
    FD_IDLE,
    LEGACY_LABEL_MAP,
    PROTOCOL_NAME,
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_int(text: str, seed: int) -> int:
    value = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16]
    return int(value, 16)


def valid_wav(path: Path, sample_rate: int, frames: int) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wav:
            return (
                wav.getnchannels() == 1
                and wav.getsampwidth() == 2
                and wav.getframerate() == sample_rate
                and wav.getnframes() == frames
            )
    except (EOFError, OSError, wave.Error):
        return False


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def read_pcm16(path: Path, sample_rate: int) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono PCM16")
        if wav.getframerate() != sample_rate:
            raise ValueError(f"{path}: sample_rate={wav.getframerate()}, expected={sample_rate}")
        return wav.readframes(wav.getnframes())


def write_pcm16_atomic(path: Path, raw: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    tmp.replace(path)


def noise_pcm16(samples: int, rms: float, seed: int) -> bytes:
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(samples)]
    current = (sum(value * value for value in values) / max(1, len(values))) ** 0.5 or 1.0
    scale = rms * 32767.0 / current
    integers = [max(-32768, min(32767, int(round(value * scale)))) for value in values]
    return struct.pack(f"<{len(integers)}h", *integers)


def state_entry(label: str, kind: str, source: str, turn_id: int) -> Dict[str, Any]:
    return {
        "kind": kind,
        "label_type": "state",
        "label": label,
        "audio_source": source,
        "turn_id": turn_id,
    }


def reindex_timeline(timeline: List[Dict[str, Any]], chunk_n: int, chunk_ms: int) -> None:
    for idx, item in enumerate(timeline):
        item["idx"] = idx
        item["start_sec"] = round(idx * chunk_ms / 1000.0, 6)
        item["end_sec"] = round((idx + 1) * chunk_ms / 1000.0, 6)
        item["start_sample"] = idx * chunk_n
        item["end_sample"] = (idx + 1) * chunk_n


def clarification_turn_id(row: Dict[str, Any]) -> int:
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    value = source_row.get("inserted_turn_id")
    if value:
        return int(value)
    for turn in source_row.get("turns") or []:
        if isinstance(turn, dict) and turn.get("source") == "inserted_incomplete_query":
            return int(turn.get("turn_id") or 0)
    return 0


def mark_f_wait_before_pause(timeline: List[Dict[str, Any]], pause_idx: int, turn_id: int) -> None:
    for idx in range(pause_idx - 1, -1, -1):
        item = timeline[idx]
        if turn_id and int(item.get("turn_id") or 0) != turn_id:
            continue
        if item.get("label") == FD_D_WAIT:
            item["label"] = FD_F_WAIT
            item["kind"] = "incomplete_query_detected"
            return
    raise ValueError("query chunk before incomplete pause not found")


def upgrade_timeline(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[int], str]:
    timeline = row.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        raise ValueError(f"{row.get('id')}: missing timeline")

    for item in timeline:
        if item.get("kind") == "eor":
            item["label"] = EOR
            item["token_text"] = EOR
        elif item.get("label_type") != "text":
            item["label"] = LEGACY_LABEL_MAP.get(str(item.get("label")), str(item.get("label")))

    scenario = str(row.get("scenario") or "")
    insert_audio_idx: Optional[int] = None
    mode = "labels_only"

    if scenario in {"incomplete_query", "incomplete_query_candidate"}:
        pause_indices = [
            idx for idx, item in enumerate(timeline)
            if item.get("kind") == "incomplete_pause_wait"
        ]
        if not pause_indices:
            raise ValueError(f"{row.get('id')}: incomplete pause groups={len(pause_indices)}")
        first_pause = pause_indices[0]
        turn_id = int(timeline[first_pause].get("turn_id") or 0)
        mark_f_wait_before_pause(timeline, first_pause, turn_id)
        for idx in pause_indices:
            timeline[idx]["label"] = FD_IDLE
        mode = "incomplete_labels"

    elif scenario == "incomplete_query_clarification":
        pause_indices = [
            idx for idx, item in enumerate(timeline)
            if item.get("kind") == "clarification_wait"
        ]
        if not pause_indices:
            raise ValueError(f"{row.get('id')}: clarification pause not found")
        turn_id = clarification_turn_id(row) or int(timeline[pause_indices[0]].get("turn_id") or 0)
        mark_f_wait_before_pause(timeline, pause_indices[0], turn_id)
        for idx in pause_indices:
            timeline[idx]["label"] = FD_IDLE
        mode = "clarification_labels"

    elif scenario == "player_backchannel":
        backchannel_indices = [
            idx for idx, item in enumerate(timeline)
            if item.get("audio_source") == "backchannel_audio"
        ]
        if not backchannel_indices:
            raise ValueError(f"{row.get('id')}: backchannel audio region not found")
        if not any(item.get("label") == FD_H_CONTINUE for item in timeline):
            search_from = backchannel_indices[-1] + 1
            answer_idx = next(
                (
                    idx for idx in range(search_from, len(timeline))
                    if timeline[idx].get("label") == FD_A_ANSWER
                ),
                -1,
            )
            if answer_idx < 0:
                raise ValueError(f"{row.get('id')}: continuation ANSWER not found")
            turn_id = int(timeline[answer_idx].get("turn_id") or 0)
            timeline.insert(
                answer_idx,
                state_entry(FD_H_CONTINUE, "backchannel_continue", "backchannel_continue_gn", turn_id),
            )
            insert_audio_idx = answer_idx
        mode = "backchannel_continue"

    return timeline, insert_audio_idx, mode


def process_row(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Tuple[Dict[str, Any], str]:
    sample_rate = int(row.get("sample_rate") or 0)
    chunk_ms = int(row.get("chunk_ms") or 0)
    if sample_rate <= 0 or chunk_ms <= 0:
        raise ValueError(f"{row.get('id')}: invalid sample_rate/chunk_ms")
    chunk_n = int(round(sample_rate * chunk_ms / 1000.0))
    src = Path(str(row.get("audio") or ""))
    if not src.exists():
        raise FileNotFoundError(src)

    old_chunks = len(row.get("timeline") or [])
    timeline, insert_idx, mode = upgrade_timeline(row)
    expected_frames = len(timeline) * chunk_n

    if not valid_wav(out_wav, sample_rate, expected_frames):
        if insert_idx is None:
            copy_atomic(src, out_wav)
        else:
            raw = read_pcm16(src, sample_rate)
            if len(raw) != old_chunks * chunk_n * 2:
                raise ValueError(f"{row.get('id')}: source audio/timeline length mismatch")
            byte_idx = insert_idx * chunk_n * 2
            inserted = noise_pcm16(
                chunk_n,
                args.noise_rms,
                stable_int(f"continue:{row.get('id')}", args.seed),
            )
            write_pcm16_atomic(out_wav, raw[:byte_idx] + inserted + raw[byte_idx:], sample_rate)

    reindex_timeline(timeline, chunk_n, chunk_ms)
    row["timeline"] = timeline
    row["audio"] = str(out_wav.resolve())
    row["label_protocol"] = PROTOCOL_NAME
    row["label_protocol_upgrade"] = {
        "from_manifest": str(args.input_manifest),
        "protocol": PROTOCOL_NAME,
        "mode": mode,
        "seed": args.seed,
    }
    row["stats"] = {
        "timeline_chunks": len(timeline),
        "audio_samples": expected_frames,
        "duration_sec": round(expected_frames / sample_rate, 6),
    }
    return row, mode


def write_rows(absolute: Any, relative: Any, row: Dict[str, Any]) -> None:
    absolute.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    relative_row = dict(row)
    relative_row["audio"] = str(Path("wav") / Path(str(row["audio"])).name)
    relative.write(json.dumps(relative_row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade a final manifest to fd_control_v1 labels.")
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--noise_rms", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    args.input_manifest = Path(args.input_manifest).resolve()
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    absolute_tmp = out_dir / "manifest.jsonl.tmp"
    relative_tmp = out_dir / "manifest_relative.jsonl.tmp"
    counts: Counter = Counter()
    scenarios: Counter = Counter()
    pending: deque[Future[Tuple[Dict[str, Any], str]]] = deque()
    total = 0

    def consume(future: Future[Tuple[Dict[str, Any], str]], absolute: Any, relative: Any, bar: tqdm) -> None:
        nonlocal total
        row, mode = future.result()
        write_rows(absolute, relative, row)
        counts[mode] += 1
        scenarios[str(row.get("scenario") or "unknown")] += 1
        total += 1
        bar.update(1)

    with absolute_tmp.open("w", encoding="utf-8") as absolute, relative_tmp.open("w", encoding="utf-8") as relative:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(unit="row", dynamic_ncols=True, desc=f"upgrade {out_dir.name}", disable=args.quiet) as bar:
                for idx, row in enumerate(read_jsonl(args.input_manifest)):
                    if args.limit and idx >= args.limit:
                        break
                    out_wav = wav_dir / Path(str(row.get("audio") or f"{row.get('id')}.wav")).name
                    pending.append(executor.submit(process_row, row, out_wav, args))
                    if len(pending) >= args.workers * 4:
                        consume(pending.popleft(), absolute, relative, bar)
                while pending:
                    consume(pending.popleft(), absolute, relative, bar)

    manifest = out_dir / "manifest.jsonl"
    relative_manifest = out_dir / "manifest_relative.jsonl"
    absolute_tmp.replace(manifest)
    relative_tmp.replace(relative_manifest)
    stats = {
        "input_manifest": str(args.input_manifest),
        "out_dir": str(out_dir.resolve()),
        "rows": total,
        "scenarios": dict(scenarios),
        "modes": dict(counts),
        "label_protocol": PROTOCOL_NAME,
        "seed": args.seed,
        "workers": args.workers,
        "manifest": str(manifest.resolve()),
        "relative_manifest": str(relative_manifest.resolve()),
        "wav_dir": str(wav_dir.resolve()),
    }
    (out_dir / "upgrade_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "README.txt").write_text(
        "Final duplex dataset using fd_control_v1 labels.\n"
        "manifest.jsonl uses absolute audio paths; manifest_relative.jsonl uses relative paths.\n"
        "WAV files are direct copies except backchannel rows, which add one 180 ms CONTINUE control chunk.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
