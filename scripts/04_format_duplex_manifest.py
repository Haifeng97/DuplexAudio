#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def float_to_pcm16(x: float) -> int:
    return int(max(-32768, min(32767, round(max(-1.0, min(1.0, float(x))) * 32767.0))))


def read_wav_with_soundfile(path: Path, sample_rate: int) -> List[int]:
    import soundfile as sf  # type: ignore

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(f"{path} sample_rate={sr}, expected {sample_rate}")
    if data.size == 0:
        return []
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = data.mean(axis=1)
    return [float_to_pcm16(x) for x in mono]


def read_wav_pcm16_fallback(path: Path, sample_rate: int) -> List[int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        if sr != sample_rate:
            raise ValueError(f"{path} sample_rate={sr}, expected {sample_rate}")
        raw = wf.readframes(wf.getnframes())
    if width != 2:
        raise ValueError(f"{path} sampwidth={width}, expected 2")
    vals = list(struct.unpack("<" + "h" * (len(raw) // 2), raw))
    if channels == 1:
        return vals
    mono = []
    for i in range(0, len(vals), channels):
        mono.append(int(sum(vals[i:i + channels]) / channels))
    return mono


def read_wav_mono_pcm16(path: Path, sample_rate: int) -> List[int]:
    try:
        return read_wav_with_soundfile(path, sample_rate)
    except ImportError:
        return read_wav_pcm16_fallback(path, sample_rate)


def write_wav_pcm16(path: Path, samples: List[int], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(struct.pack("<h", max(-32768, min(32767, int(x)))) for x in samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def pad_to_chunks(samples: List[int], chunk_n: int) -> Tuple[List[int], int]:
    chunks = int(math.ceil(len(samples) / chunk_n)) if samples else 0
    need = chunks * chunk_n
    if len(samples) < need:
        samples = samples + [0] * (need - len(samples))
    return samples, chunks


def gaussian_noise(chunks: int, chunk_n: int, rng: random.Random, rms: float) -> List[int]:
    n = max(0, chunks * chunk_n)
    if n == 0:
        return []
    vals = [rng.gauss(0.0, 1.0) for _ in range(n)]
    cur = math.sqrt(sum(x * x for x in vals) / len(vals)) or 1.0
    scale = rms * 32767.0 / cur
    return [int(max(-32768, min(32767, x * scale))) for x in vals]


def text_units(text: str) -> List[str]:
    return [ch for ch in str(text).replace("\r", "").replace("\n", "") if ch.strip()]


def entry(idx: int, label: str, kind: str, chunk_n: int, chunk_ms: int, source: str, turn_id: int = 0) -> Dict[str, Any]:
    return {
        "idx": idx,
        "kind": kind,
        "label_type": "state",
        "label": label,
        "start_sec": round(idx * chunk_ms / 1000.0, 6),
        "end_sec": round((idx + 1) * chunk_ms / 1000.0, 6),
        "start_sample": idx * chunk_n,
        "end_sample": (idx + 1) * chunk_n,
        "audio_source": source,
        "turn_id": turn_id,
    }


def text_entry(idx: int, token_text: str, text_idx: int, chunk_n: int, chunk_ms: int, source: str, turn_id: int) -> Dict[str, Any]:
    ent = entry(idx, token_text, "text_token", chunk_n, chunk_ms, source, turn_id)
    ent.update({
        "label_type": "text",
        "token_id": None,
        "token_text": token_text,
        "text_token_idx": text_idx,
    })
    return ent


class Builder:
    def __init__(self, sample_rate: int, chunk_ms: int, noise_rms: float, seed: int):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_n = int(round(sample_rate * chunk_ms / 1000.0))
        self.noise_rms = noise_rms
        self.rng = random.Random(seed)
        self.audio: List[int] = []
        self.timeline: List[Dict[str, Any]] = []

    def idx(self) -> int:
        return len(self.timeline)

    def add_noise(self, chunks: int, label: str, kind: str, source: str, turn_id: int = 0) -> None:
        samples = gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms)
        self.audio.extend(samples)
        for _ in range(chunks):
            self.timeline.append(entry(self.idx(), label, kind, self.chunk_n, self.chunk_ms, source, turn_id))

    def add_query_audio(self, path: str, turn_id: int, source: str) -> None:
        samples = read_wav_mono_pcm16(Path(path), self.sample_rate)
        samples, chunks = pad_to_chunks(samples, self.chunk_n)
        self.audio.extend(samples)
        for _ in range(chunks):
            self.timeline.append(entry(self.idx(), "WAIT", "wait", self.chunk_n, self.chunk_ms, source, turn_id))

    def add_answer(self, text: str, turn_id: int, source: str, *, prefix_only: bool = False, min_chunks: int = 0) -> None:
        units = text_units(text)
        need = 1 + len(units) + (0 if prefix_only else 1)
        chunks = max(min_chunks, need)
        self.audio.extend(gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms))
        self.timeline.append(entry(self.idx(), "ANSWER", "answer_trigger", self.chunk_n, self.chunk_ms, source, turn_id))
        for j, unit in enumerate(units):
            self.timeline.append(text_entry(self.idx(), unit, j, self.chunk_n, self.chunk_ms, source, turn_id))
        if not prefix_only:
            self.timeline.append(text_entry(self.idx(), "<EOR>", len(units), self.chunk_n, self.chunk_ms, source, turn_id))
            self.timeline[-1]["kind"] = "eor"
            self.timeline[-1]["label"] = "<EOR>"
            self.timeline[-1]["token_text"] = "<EOR>"
        while len(self.timeline) < len(self.audio) // self.chunk_n:
            self.timeline.append(entry(self.idx(), "IDLE", "answer_tail_idle", self.chunk_n, self.chunk_ms, source, turn_id))


def answer_region_chunks(text: str) -> int:
    return max(1, int(math.ceil(len(str(text)) * 1.1)))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def build_normal(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(args.sample_rate, args.chunk_ms, args.noise_rms, stable_seed(row["id"]))
    b.add_noise(args.initial_idle_chunks, "IDLE", "initial_idle", "gn_before")
    assets = row["tts_assets"]
    turns = row.get("turns") or [{"question_text": row["question_text"], "answer_text": row["answer_text"]}]
    for i, turn in enumerate(turns, start=1):
        key = "query" if len(turns) == 1 else f"turn{i:03d}_query"
        b.add_query_audio(assets[key]["audio"], i, f"turn{i}_query_audio")
        b.add_answer(turn["answer_text"], i, f"turn{i}_answer_gn", min_chunks=answer_region_chunks(turn["answer_text"]))
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "normal_qa")


def build_interrupt(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(args.sample_rate, args.chunk_ms, args.noise_rms, stable_seed(row["id"]))
    b.add_noise(args.initial_idle_chunks, "IDLE", "initial_idle", "gn_before")
    assets = row["tts_assets"]
    b.add_query_audio(assets["base_query"]["audio"], 1, "base_query_audio")
    b.add_answer(row["base"]["answer_prefix_text"], 1, "base_answer_prefix_gn", prefix_only=True)
    b.add_query_audio(assets["donor_query"]["audio"], 2, "donor_query_audio")
    donor_answer = row["donor"]["answer_text"]
    b.add_answer(donor_answer, 2, "donor_answer_gn", min_chunks=answer_region_chunks(donor_answer))
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "player_interrupts_ai")


def build_incomplete(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(args.sample_rate, args.chunk_ms, args.noise_rms, stable_seed(row["id"]))
    b.add_noise(args.initial_idle_chunks, "IDLE", "initial_idle", "gn_before")
    assets = row["tts_assets"]
    b.add_query_audio(assets["query_part1"]["audio"], 1, "query_part1_audio")
    between_chunks = max(1, int(round(float(row["gn_policy"]["between_query_parts_sec"]) * 1000.0 / args.chunk_ms)))
    b.add_noise(between_chunks, "WAIT", "incomplete_pause_wait", "gn_between_query_parts", 1)
    b.add_query_audio(assets["query_part2"]["audio"], 1, "query_part2_audio")
    answer = row["answer_text_if_complete"]
    b.add_answer(answer, 1, "answer_gn", min_chunks=answer_region_chunks(answer))
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "incomplete_query")


def common_manifest(row: Dict[str, Any], out_wav: Path, b: Builder, scenario: str) -> Dict[str, Any]:
    answer = row.get("answer_text") or row.get("answer_text_if_complete") or row.get("donor", {}).get("answer_text", "")
    question = row.get("question_text") or row.get("full_question_text") or row.get("donor", {}).get("question_text", "")
    return {
        "id": row["id"],
        "source": "cgame_duplex",
        "scenario": scenario,
        "task": "duplex_qa",
        "audio": str(out_wav),
        "sample_rate": b.sample_rate,
        "chunk_ms": b.chunk_ms,
        "sysprompt": row.get("sysprompt") or row.get("base", {}).get("sysprompt", ""),
        "question_text": question,
        "answer_text": answer,
        "text": answer,
        "target_text": answer,
        "text_query": question,
        "asr_text": question,
        "timeline": b.timeline,
        "stats": {
            "timeline_chunks": len(b.timeline),
            "audio_samples": len(b.audio),
            "duration_sec": round(len(b.audio) / b.sample_rate, 6),
        },
        "source_row": row,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Format duplex scenario index into wav + timeline manifest.")
    ap.add_argument("--index", required=True, help="scenario_index.jsonl from 02_make_turn_tts_tasks.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--chunk_ms", type=int, default=180)
    ap.add_argument("--noise_rms", type=float, default=0.003)
    ap.add_argument("--initial_idle_chunks", type=int, default=2)
    ap.add_argument("--final_idle_chunks", type=int, default=2)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.index))
    out_path = Path(args.out)
    wav_dir = Path(args.wav_dir) if args.wav_dir else out_path.parent / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "normal_qa": build_normal,
        "player_interrupts_ai": build_interrupt,
        "incomplete_query_candidate": build_incomplete,
    }
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            scenario = row.get("scenario")
            if scenario not in builders:
                continue
            out_wav = wav_dir / f"{row['id']}.wav"
            manifest = builders[scenario](row, out_wav, args)
            f.write(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    print(json.dumps({"index": args.index, "out": str(out_path), "wav_dir": str(wav_dir), "n": n}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
