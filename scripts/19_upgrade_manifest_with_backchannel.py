#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import threading
import wave
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def stable_int(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16], 16)


def valid_wav(path: Path, sample_rate: int = 0) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wf:
            return (
                wf.getnchannels() == 1
                and wf.getsampwidth() == 2
                and wf.getnframes() > 0
                and (not sample_rate or wf.getframerate() == sample_rate)
            )
    except (EOFError, OSError, wave.Error):
        return False


def read_pcm16(path: Path, expected_sample_rate: int) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path}: sample_width={width}, expected PCM16")
    if sample_rate != expected_sample_rate:
        raise ValueError(f"{path}: sample_rate={sample_rate}, expected={expected_sample_rate}")
    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1).astype(np.int16)
    return samples.copy()


def write_pcm16_atomic(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with wave.open(str(tmp), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    tmp.replace(path)


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


def noise_samples(count: int, seed: int, rms: float) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int16)
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(count).astype(np.float64)
    current = float(np.sqrt(np.mean(values * values))) or 1.0
    values *= rms * 32767.0 / current
    return np.clip(values, -32768, 32767).astype(np.int16)


def reindex_timeline(timeline: List[Dict[str, Any]], chunk_n: int, chunk_ms: int) -> None:
    for idx, item in enumerate(timeline):
        item["idx"] = idx
        item["start_sec"] = round(idx * chunk_ms / 1000.0, 6)
        item["end_sec"] = round((idx + 1) * chunk_ms / 1000.0, 6)
        item["start_sample"] = idx * chunk_n
        item["end_sample"] = (idx + 1) * chunk_n


def state_entry(label: str, kind: str, source: str, turn_id: int) -> Dict[str, Any]:
    return {
        "kind": kind,
        "label_type": "state",
        "label": label,
        "audio_source": source,
        "turn_id": turn_id,
    }


def inserted_turn_id(row: Dict[str, Any]) -> int:
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    value = source_row.get("inserted_turn_id")
    if value:
        return int(value)
    for turn in source_row.get("turns") or []:
        if isinstance(turn, dict) and turn.get("source") == "inserted_incomplete_query":
            return int(turn.get("turn_id") or 0)
    return 0


def answer_segments(row: Dict[str, Any]) -> List[Tuple[int, int, List[int]]]:
    timeline = row.get("timeline") if isinstance(row.get("timeline"), list) else []
    segments: List[Tuple[int, int, List[int]]] = []
    for answer_idx, item in enumerate(timeline):
        if item.get("label") != "ANSWER":
            continue
        turn_id = int(item.get("turn_id") or 0)
        text_positions: List[int] = []
        eor_idx = -1
        for idx in range(answer_idx + 1, len(timeline)):
            current = timeline[idx]
            if current.get("label") == "ANSWER":
                break
            if current.get("label") == "<EOR>" and int(current.get("turn_id") or 0) == turn_id:
                eor_idx = idx
                break
            if current.get("label_type") == "text" and current.get("kind") == "text_token":
                text_positions.append(idx)
        if eor_idx >= 0 and len(text_positions) >= 4:
            segments.append((turn_id, eor_idx, text_positions))
    return segments


def eligible_backchannel(row: Dict[str, Any]) -> bool:
    return row.get("scenario") == "normal_qa" and bool(answer_segments(row))


class BackchannelAudio:
    def __init__(self, manifest: Path, sample_rate: int, chunk_n: int, noise_rms: float, seed: int):
        self.rows = list(read_jsonl(manifest))
        if not self.rows:
            raise ValueError(f"empty backchannel manifest: {manifest}")
        self.sample_rate = sample_rate
        self.chunk_n = chunk_n
        self.noise_rms = noise_rms
        self.seed = seed
        self.cache: Dict[str, Tuple[np.ndarray, Dict[str, Any]]] = {}
        self.invalid_clip_ids: set[str] = set()
        self.lock = threading.Lock()

    def pick(self, sample_id: str) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
        start = stable_int(sample_id, self.seed) % len(self.rows)
        for offset in range(len(self.rows)):
            clip = self.rows[(start + offset) % len(self.rows)]
            clip_id = str(clip.get("clip_id") or clip.get("audio"))
            with self.lock:
                if clip_id in self.invalid_clip_ids:
                    continue
                cached = self.cache.get(clip_id)
            if cached is None:
                try:
                    samples, meta = self._prepare(clip)
                except ValueError:
                    with self.lock:
                        self.invalid_clip_ids.add(clip_id)
                    continue
                with self.lock:
                    cached = self.cache.setdefault(clip_id, (samples, meta))
            return clip, cached[0], dict(cached[1])
        raise RuntimeError("no backchannel audio passes energy VAD")

    def _prepare(self, clip: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        import soundfile as sf  # type: ignore
        import torch  # type: ignore
        from torchaudio.functional import resample  # type: ignore

        data, source_rate = sf.read(str(clip["audio"]), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if int(source_rate) != self.sample_rate:
            tensor = torch.from_numpy(mono)
            mono = resample(tensor, int(source_rate), self.sample_rate).numpy()
        pcm = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        start, end = self._energy_bounds(pcm)
        if start is None or end is None or start >= end:
            raise ValueError(f"backchannel has no speech: {clip.get('audio')}")
        trimmed = pcm[start:end]
        chunks = int(math.ceil(len(trimmed) / self.chunk_n))
        padding = chunks * self.chunk_n - len(trimmed)
        if padding:
            pad = noise_samples(padding, stable_int(str(clip.get("clip_id")), self.seed), self.noise_rms)
            trimmed = np.concatenate([trimmed, pad])
        meta = {
            "clip_id": clip.get("clip_id"),
            "source_audio": clip.get("audio"),
            "source_sample_rate": int(source_rate),
            "processed_sample_rate": self.sample_rate,
            "trim_start_sample": start,
            "trim_end_sample": end,
            "processed_samples": len(trimmed),
            "chunks": chunks,
            "padding_samples": padding,
        }
        return trimmed, meta

    def _energy_bounds(self, samples: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        frame_n = max(1, int(round(self.sample_rate * 0.02)))
        energies = []
        for start in range(0, len(samples), frame_n):
            frame = samples[start:start + frame_n].astype(np.float64)
            if len(frame):
                energies.append(float(np.sqrt(np.mean(frame * frame))))
        if not energies or max(energies) <= 0:
            return None, None
        peak = max(energies)
        floor = sorted(energies)[min(len(energies) - 1, int(len(energies) * 0.2))]
        threshold = max(peak * 0.08, floor * 2.5, 80.0)
        voiced = [idx for idx, value in enumerate(energies) if value >= threshold]
        if not voiced:
            return None, None
        return voiced[0] * frame_n, min(len(samples), (voiced[-1] + 1) * frame_n)


def transform_clarification(
    row: Dict[str, Any], samples: np.ndarray, chunk_n: int, chunk_ms: int,
    wait_min: float, wait_max: float, noise_rms: float, seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    timeline = row["timeline"]
    if any(item.get("kind") == "clarification_wait" for item in timeline):
        return row, samples
    turn_id = inserted_turn_id(row)
    if not turn_id:
        raise ValueError(f"{row.get('id')}: missing inserted_turn_id")
    answer_idx = next(
        (idx for idx, item in enumerate(timeline) if item.get("label") == "ANSWER" and int(item.get("turn_id") or 0) == turn_id),
        -1,
    )
    if answer_idx < 0:
        raise ValueError(f"{row.get('id')}: clarification ANSWER not found")
    rng = random.Random(stable_int(str(row["id"]), seed))
    duration = rng.uniform(min(wait_min, wait_max), max(wait_min, wait_max))
    chunks = max(1, int(round(duration * 1000.0 / chunk_ms)))
    source = f"gn_before_turn{turn_id}_clarification"
    inserted = [state_entry("WAIT", "clarification_wait", source, turn_id) for _ in range(chunks)]
    timeline[answer_idx:answer_idx] = inserted
    audio_insert = noise_samples(chunks * chunk_n, stable_int(f"clarification:{row['id']}", seed), noise_rms)
    samples = np.concatenate([samples[:answer_idx * chunk_n], audio_insert, samples[answer_idx * chunk_n:]])
    wait_meta = {
        "turn_id": turn_id,
        "chunks": chunks,
        "duration_sec": round(chunks * chunk_ms / 1000.0, 6),
        "range_sec": [wait_min, wait_max],
        "audio_source": source,
    }
    row["clarification_wait"] = [wait_meta]
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    policy = source_row.get("gn_policy") if isinstance(source_row.get("gn_policy"), dict) else {}
    policy["clarification_wait_range_sec"] = [wait_min, wait_max]
    source_row["gn_policy"] = policy
    row["source_row"] = source_row
    return row, samples


def transform_backchannel(
    row: Dict[str, Any], samples: np.ndarray, bank: BackchannelAudio,
    chunk_n: int, chunk_ms: int, noise_rms: float, seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    segments = answer_segments(row)
    if not segments:
        raise ValueError(f"{row.get('id')}: no eligible answer segment")
    turn_id, _, text_positions = segments[-1]
    n_tokens = len(text_positions)
    lo = max(1, int(math.ceil(n_tokens * 0.3)))
    hi = min(n_tokens - 1, int(math.floor(n_tokens * 0.7)))
    if hi < lo:
        lo, hi = 1, n_tokens - 1
    rng = random.Random(stable_int(str(row["id"]), seed))
    split = rng.randint(lo, hi)
    insert_idx = text_positions[split - 1] + 1
    prefix_text = "".join(str(row["timeline"][idx].get("token_text") or "") for idx in text_positions[:split])
    remaining_text = "".join(str(row["timeline"][idx].get("token_text") or "") for idx in text_positions[split:])
    clip, backchannel, vad_meta = bank.pick(str(row["id"]))
    backchannel_chunks = len(backchannel) // chunk_n
    bc_entries = [
        state_entry("INTERRUPT" if idx == 0 else "WAIT", "interrupt" if idx == 0 else "wait", "backchannel_audio", turn_id)
        for idx in range(backchannel_chunks)
    ]
    continuation = state_entry("ANSWER", "answer_trigger", "answer_continuation_gn", turn_id)
    row["timeline"][insert_idx:insert_idx] = bc_entries + [continuation]
    continuation_noise = noise_samples(chunk_n, stable_int(f"continuation:{row['id']}", seed), noise_rms)
    audio_insert = np.concatenate([backchannel, continuation_noise])
    samples = np.concatenate([samples[:insert_idx * chunk_n], audio_insert, samples[insert_idx * chunk_n:]])
    row["scenario"] = "player_backchannel"
    row["backchannel_text"] = str(clip.get("text") or "")
    row["backchannel_vad"] = vad_meta
    row["backchannel"] = {
        "turn_id": turn_id,
        "answer_token_split": split,
        "answer_token_count": n_tokens,
        "answer_prefix_text": prefix_text,
        "answer_remaining_text": remaining_text,
        "text": clip.get("text"),
        "clip_id": clip.get("clip_id"),
        "speaker": clip.get("speaker"),
        "gender": clip.get("gender"),
        "source_dataset": clip.get("source_dataset", "magicdata_ramc"),
        "chunks": backchannel_chunks,
    }
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    source_row.update({
        "scenario": "player_backchannel",
        "backchannel_turn_id": turn_id,
        "answer_prefix_text": prefix_text,
        "answer_remaining_text": remaining_text,
        "backchannel_text": clip.get("text"),
        "backchannel_audio": {
            "path": clip.get("audio"),
            "clip_id": clip.get("clip_id"),
            "speaker": clip.get("speaker"),
            "gender": clip.get("gender"),
            "source_dataset": clip.get("source_dataset", "magicdata_ramc"),
        },
    })
    row["source_row"] = source_row
    return row, samples


def finalize_row(row: Dict[str, Any], samples: np.ndarray, out_wav: Path, chunk_n: int, chunk_ms: int) -> Dict[str, Any]:
    timeline = row.get("timeline") or []
    reindex_timeline(timeline, chunk_n, chunk_ms)
    expected = len(timeline) * chunk_n
    if len(samples) != expected:
        raise ValueError(f"{row.get('id')}: audio samples={len(samples)}, timeline expects={expected}")
    row["audio"] = str(out_wav.resolve())
    row["stats"] = {
        "timeline_chunks": len(timeline),
        "audio_samples": len(samples),
        "duration_sec": round(len(samples) / int(row["sample_rate"]), 6),
    }
    return row


def process_row(
    row: Dict[str, Any], out_wav: Path, selected_backchannels: set[str], bank: BackchannelAudio,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], str]:
    sample_rate = int(row.get("sample_rate") or 0)
    chunk_ms = int(row.get("chunk_ms") or 0)
    if sample_rate <= 0 or chunk_ms <= 0:
        raise ValueError(f"{row.get('id')}: invalid sample_rate/chunk_ms")
    chunk_n = int(round(sample_rate * chunk_ms / 1000.0))
    src = Path(str(row.get("audio") or ""))
    if not src.exists():
        raise FileNotFoundError(src)
    mode = "copy"
    must_transform = str(row.get("id")) in selected_backchannels or row.get("scenario") == "incomplete_query_clarification"
    if must_transform:
        samples = read_pcm16(src, sample_rate)
        if len(samples) != len(row.get("timeline") or []) * chunk_n:
            raise ValueError(f"{row.get('id')}: source audio/timeline length mismatch")
        if str(row.get("id")) in selected_backchannels:
            row, samples = transform_backchannel(row, samples, bank, chunk_n, chunk_ms, args.noise_rms, args.seed)
            mode = "backchannel"
        else:
            row, samples = transform_clarification(
                row, samples, chunk_n, chunk_ms,
                args.clarification_wait_min, args.clarification_wait_max,
                args.noise_rms, args.seed,
            )
            mode = "clarification"
        row["upgrade"] = {
            "from_manifest": str(args.input_manifest),
            "mode": mode,
            "seed": args.seed,
        }
        row = finalize_row(row, samples, out_wav, chunk_n, chunk_ms)
        if args.overwrite_audio or not valid_wav(out_wav, sample_rate):
            write_pcm16_atomic(out_wav, samples, sample_rate)
    else:
        row["audio"] = str(out_wav.resolve())
        if args.overwrite_audio or not valid_wav(out_wav, sample_rate):
            copy_atomic(src, out_wav)
    return row, mode


def write_rows(absolute_f: Any, relative_f: Any, row: Dict[str, Any], out_dir: Path) -> None:
    absolute_f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    relative = dict(row)
    relative["audio"] = str(Path("wav") / Path(str(row["audio"])).name)
    relative_f.write(json.dumps(relative, ensure_ascii=False, separators=(",", ":")) + "\n")


def scan_input(path: Path, ratio: float, seed: int, limit: int) -> Tuple[Counter, set[str], int]:
    counts: Counter = Counter()
    eligible: List[Tuple[int, str]] = []
    total = 0
    for row in read_jsonl(path):
        if limit and total >= limit:
            break
        total += 1
        counts[str(row.get("scenario") or "unknown")] += 1
        if eligible_backchannel(row):
            sample_id = str(row.get("id") or "")
            eligible.append((stable_int(sample_id, seed), sample_id))
    target = min(len(eligible), int(round(total * ratio)))
    eligible.sort()
    return counts, {sample_id for _, sample_id in eligible[:target]}, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Upgrade a final duplex manifest with real backchannel and clarification WAIT.")
    ap.add_argument("--input_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--backchannel_manifest", default="outputs/roleplay_zh_v3/backchannel_corpus/backchannels.jsonl")
    ap.add_argument("--backchannel_ratio", type=float, default=0.10)
    ap.add_argument("--clarification_wait_min", type=float, default=3.0)
    ap.add_argument("--clarification_wait_max", type=float, default=5.0)
    ap.add_argument("--noise_rms", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=20260730)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite_audio", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.backchannel_ratio <= 1:
        raise SystemExit("--backchannel_ratio must be in [0, 1]")
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    input_manifest = Path(args.input_manifest)
    args.input_manifest = input_manifest.resolve()
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    before_counts, selected_backchannels, total = scan_input(
        input_manifest, args.backchannel_ratio, args.seed, args.limit,
    )
    bank = BackchannelAudio(
        Path(args.backchannel_manifest), 24000, int(round(24000 * 0.18)), args.noise_rms, args.seed,
    )
    manifest_tmp = out_dir / "manifest.jsonl.tmp"
    relative_tmp = out_dir / "manifest_relative.jsonl.tmp"
    mode_counts: Counter = Counter()
    output_counts: Counter = Counter()
    pending: deque[Future[Tuple[Dict[str, Any], str]]] = deque()

    def consume(future: Future[Tuple[Dict[str, Any], str]], absolute_f: Any, relative_f: Any, bar: tqdm) -> None:
        row, mode = future.result()
        write_rows(absolute_f, relative_f, row, out_dir)
        mode_counts[mode] += 1
        output_counts[str(row.get("scenario") or "unknown")] += 1
        bar.update(1)
        if sum(mode_counts.values()) % 100 == 0:
            bar.set_postfix(dict(mode_counts), refresh=False)

    with manifest_tmp.open("w", encoding="utf-8") as absolute_f, relative_tmp.open("w", encoding="utf-8") as relative_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            with tqdm(total=total, unit="row", dynamic_ncols=True, desc=f"upgrade {out_dir.name}", disable=args.quiet) as bar:
                for idx, row in enumerate(read_jsonl(input_manifest)):
                    if args.limit and idx >= args.limit:
                        break
                    out_wav = wav_dir / Path(str(row.get("audio") or f"{row.get('id')}.wav")).name
                    pending.append(executor.submit(
                        process_row, row, out_wav, selected_backchannels, bank, args,
                    ))
                    if len(pending) >= args.workers * 4:
                        consume(pending.popleft(), absolute_f, relative_f, bar)
                while pending:
                    consume(pending.popleft(), absolute_f, relative_f, bar)

    manifest = out_dir / "manifest.jsonl"
    relative_manifest = out_dir / "manifest_relative.jsonl"
    manifest_tmp.replace(manifest)
    relative_tmp.replace(relative_manifest)
    stats = {
        "input_manifest": str(input_manifest),
        "out_dir": str(out_dir),
        "rows": total,
        "before_scenarios": dict(before_counts),
        "after_scenarios": dict(output_counts),
        "selected_backchannels": len(selected_backchannels),
        "modes": dict(mode_counts),
        "backchannel_ratio": args.backchannel_ratio,
        "clarification_wait_range_sec": [args.clarification_wait_min, args.clarification_wait_max],
        "seed": args.seed,
        "workers": args.workers,
        "backchannel_invalid_audio": len(bank.invalid_clip_ids),
        "manifest": str(manifest),
        "relative_manifest": str(relative_manifest),
        "wav_dir": str(wav_dir),
    }
    (out_dir / "upgrade_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        "Final duplex dataset upgraded with 10% real RAMC backchannel and 3-5s clarification WAIT.\n"
        "manifest.jsonl uses absolute NFS audio paths; manifest_relative.jsonl uses paths relative to this directory.\n"
        "WAV files are direct copies or rewritten files, not symlinks or hardlinks.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
