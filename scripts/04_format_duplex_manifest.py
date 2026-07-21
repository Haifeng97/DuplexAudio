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

from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    return gaussian_noise_samples(n, rng, rms)


def gaussian_noise_samples(n: int, rng: random.Random, rms: float) -> List[int]:
    n = max(0, n)
    if n == 0:
        return []
    vals = [rng.gauss(0.0, 1.0) for _ in range(n)]
    cur = math.sqrt(sum(x * x for x in vals) / len(vals)) or 1.0
    scale = rms * 32767.0 / cur
    return [int(max(-32768, min(32767, x * scale))) for x in vals]


class TextTokenizer:
    def __init__(self, tokenizer_json: str = ""):
        self.path = tokenizer_json
        self.mode = "char"
        self._tokenizer = None
        if tokenizer_json:
            try:
                from tokenizers import Tokenizer  # type: ignore
            except ImportError as exc:
                raise RuntimeError("tokenizers is required when --tokenizer_json is set") from exc
            self._tokenizer = Tokenizer.from_file(tokenizer_json)
            self.mode = "tokenizer_json"

    def encode(self, text: str) -> List[Dict[str, Any]]:
        clean = str(text).replace("\r", "").replace("\n", "")
        if not self._tokenizer:
            return [
                {"token_id": None, "token_text": ch}
                for ch in clean
                if ch.strip()
            ]
        enc = self._tokenizer.encode(clean, add_special_tokens=False)
        out: List[Dict[str, Any]] = []
        for token_id, raw_token in zip(enc.ids, enc.tokens):
            token_text = self._decode_token(token_id, raw_token)
            if token_text:
                out.append({"token_id": token_id, "token_text": token_text, "raw_token": raw_token})
        return out

    def _decode_token(self, token_id: int, raw_token: str) -> str:
        assert self._tokenizer is not None
        try:
            text = self._tokenizer.decode([token_id], skip_special_tokens=False)
        except TypeError:
            text = self._tokenizer.decode([token_id])
        return text or raw_token

    def metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "path": self.path,
        }


class VadSilenceReplacer:
    def __init__(self, mode: str, sample_rate: int, noise_rms: float):
        self.requested_mode = mode
        self.sample_rate = sample_rate
        self.vad_sample_rate = 16000
        self.noise_rms = noise_rms
        self.backend = "off"
        self._silero_model = None
        self._silero_get_speech_timestamps = None
        if mode in {"auto", "silero"}:
            self._load_silero()
        if self.backend == "off" and mode in {"auto", "energy"}:
            self.backend = "energy"
        if self.backend == "off" and mode == "silero":
            raise RuntimeError("silero_vad is not installed; use --vad_mode auto or install silero-vad")

    def _load_silero(self) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad  # type: ignore
        except ImportError:
            return
        self._silero_model = load_silero_vad()
        self._silero_get_speech_timestamps = get_speech_timestamps
        self.backend = "silero"

    def process(self, samples: List[int], rng: random.Random) -> Tuple[List[int], Dict[str, Any]]:
        if self.backend == "off" or not samples:
            return samples, self._meta(len(samples), None, None)
        if self.backend == "silero":
            start, end = self._silero_bounds(samples)
        else:
            start, end = self._energy_bounds(samples)
        if start is None or end is None or start >= end:
            return samples, self._meta(len(samples), None, None)
        replaced = list(samples)
        if start > 0:
            replaced[:start] = gaussian_noise_samples(start, rng, self.noise_rms)
        if end < len(replaced):
            replaced[end:] = gaussian_noise_samples(len(replaced) - end, rng, self.noise_rms)
        return replaced, self._meta(len(samples), start, end)

    def _silero_bounds(self, samples: List[int]) -> Tuple[Optional[int], Optional[int]]:
        assert self._silero_get_speech_timestamps is not None
        assert self._silero_model is not None
        import torch  # type: ignore

        audio = torch.tensor([max(-1.0, min(1.0, x / 32768.0)) for x in samples], dtype=torch.float32)
        vad_audio = self._resample_for_silero(audio)
        timestamps = self._silero_get_speech_timestamps(vad_audio, self._silero_model, sampling_rate=self.vad_sample_rate)
        if not timestamps:
            return None, None
        scale = self.sample_rate / self.vad_sample_rate
        start = int(round(int(timestamps[0]["start"]) * scale))
        end = int(round(int(timestamps[-1]["end"]) * scale))
        return max(0, min(start, len(samples))), max(0, min(end, len(samples)))

    def _resample_for_silero(self, audio: Any) -> Any:
        if self.sample_rate == self.vad_sample_rate:
            return audio
        try:
            import torchaudio.functional as F  # type: ignore

            return F.resample(audio, self.sample_rate, self.vad_sample_rate)
        except Exception:
            import torch.nn.functional as Fnn  # type: ignore

            src = audio.view(1, 1, -1)
            out_len = max(1, int(round(audio.numel() * self.vad_sample_rate / self.sample_rate)))
            return Fnn.interpolate(src, size=out_len, mode="linear", align_corners=False).view(-1)

    def _energy_bounds(self, samples: List[int]) -> Tuple[Optional[int], Optional[int]]:
        frame_n = max(1, int(round(self.sample_rate * 0.02)))
        energies: List[float] = []
        for i in range(0, len(samples), frame_n):
            frame = samples[i:i + frame_n]
            if not frame:
                continue
            energies.append(math.sqrt(sum(x * x for x in frame) / len(frame)))
        if not energies:
            return None, None
        peak = max(energies)
        if peak <= 0:
            return None, None
        floor = sorted(energies)[min(len(energies) - 1, max(0, int(len(energies) * 0.2)))]
        threshold = max(peak * 0.08, floor * 2.5, 80.0)
        voiced = [i for i, e in enumerate(energies) if e >= threshold]
        if not voiced:
            return None, None
        start = max(0, voiced[0] * frame_n)
        end = min(len(samples), (voiced[-1] + 1) * frame_n)
        return start, end

    def _meta(self, sample_count: int, start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
        leading = start if start is not None else 0
        trailing = sample_count - end if end is not None else 0
        return {
            "requested_mode": self.requested_mode,
            "backend": self.backend,
            "vad_sample_rate": self.vad_sample_rate if self.backend == "silero" else self.sample_rate,
            "speech_start_sample": start,
            "speech_end_sample": end,
            "leading_replaced_sec": round(leading / self.sample_rate, 6),
            "trailing_replaced_sec": round(trailing / self.sample_rate, 6),
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "backend": self.backend,
            "vad_sample_rate": self.vad_sample_rate if self.backend == "silero" else self.sample_rate,
            "noise_rms": self.noise_rms,
        }


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


def text_entry(idx: int, token: Dict[str, Any], text_idx: int, chunk_n: int, chunk_ms: int, source: str, turn_id: int) -> Dict[str, Any]:
    token_text = token["token_text"]
    ent = entry(idx, token_text, "text_token", chunk_n, chunk_ms, source, turn_id)
    ent.update({
        "label_type": "text",
        "token_id": token.get("token_id"),
        "token_text": token_text,
        "raw_token": token.get("raw_token"),
        "text_token_idx": text_idx,
    })
    return ent


class Builder:
    def __init__(
        self,
        sample_rate: int,
        chunk_ms: int,
        noise_rms: float,
        seed: int,
        tokenizer: TextTokenizer,
        vad: VadSilenceReplacer,
        min_query_audio_sec: float,
    ):
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.chunk_n = int(round(sample_rate * chunk_ms / 1000.0))
        self.noise_rms = noise_rms
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.vad = vad
        self.min_query_audio_sec = min_query_audio_sec
        self.audio: List[int] = []
        self.timeline: List[Dict[str, Any]] = []
        self.query_vad: List[Dict[str, Any]] = []

    def idx(self) -> int:
        return len(self.timeline)

    def add_noise(self, chunks: int, label: str, kind: str, source: str, turn_id: int = 0) -> None:
        samples = gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms)
        self.audio.extend(samples)
        for _ in range(chunks):
            self.timeline.append(entry(self.idx(), label, kind, self.chunk_n, self.chunk_ms, source, turn_id))

    def add_query_audio(self, path: str, turn_id: int, source: str, *, first_label: str = "WAIT") -> None:
        samples = read_wav_mono_pcm16(Path(path), self.sample_rate)
        duration_sec = len(samples) / self.sample_rate if self.sample_rate else 0.0
        if self.min_query_audio_sec > 0 and duration_sec < self.min_query_audio_sec:
            raise ValueError(
                f"query_audio_too_short path={path} "
                f"duration_sec={duration_sec:.6f} "
                f"min_query_audio_sec={self.min_query_audio_sec:.6f}"
            )
        samples, vad_meta = self.vad.process(samples, self.rng)
        vad_meta.update({"path": path, "source": source, "turn_id": turn_id})
        chunks = int(math.ceil(len(samples) / self.chunk_n)) if samples else 0
        need = chunks * self.chunk_n
        pad_samples = max(0, need - len(samples))
        if pad_samples:
            samples = samples + gaussian_noise_samples(pad_samples, self.rng, self.noise_rms)
        vad_meta["chunk_padding_noise_sec"] = round(pad_samples / self.sample_rate, 6)
        self.query_vad.append(vad_meta)
        self.audio.extend(samples)
        for i in range(chunks):
            label = first_label if i == 0 else "WAIT"
            kind = "wait" if label == "WAIT" else label.lower()
            self.timeline.append(entry(self.idx(), label, kind, self.chunk_n, self.chunk_ms, source, turn_id))

    def add_answer(self, text: str, turn_id: int, source: str, *, prefix_only: bool = False, min_chunks: int = 0) -> None:
        units = self.tokenizer.encode(text)
        need = 1 + len(units) + (0 if prefix_only else 1)
        chunks = max(min_chunks, need)
        self.audio.extend(gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms))
        self.timeline.append(entry(self.idx(), "ANSWER", "answer_trigger", self.chunk_n, self.chunk_ms, source, turn_id))
        for j, unit in enumerate(units):
            self.timeline.append(text_entry(self.idx(), unit, j, self.chunk_n, self.chunk_ms, source, turn_id))
        if not prefix_only:
            self.timeline.append(text_entry(self.idx(), {"token_id": None, "token_text": "<EOR>"}, len(units), self.chunk_n, self.chunk_ms, source, turn_id))
            self.timeline[-1]["kind"] = "eor"
            self.timeline[-1]["label"] = "<EOR>"
            self.timeline[-1]["token_text"] = "<EOR>"
        while len(self.timeline) < len(self.audio) // self.chunk_n:
            self.timeline.append(entry(self.idx(), "IDLE", "answer_tail_idle", self.chunk_n, self.chunk_ms, source, turn_id))

    def add_answer_continuation(self, text: str, turn_id: int, source: str, *, text_idx_offset: int = 0, min_chunks: int = 0) -> None:
        units = self.tokenizer.encode(text)
        need = len(units) + 1
        chunks = max(min_chunks, need)
        self.audio.extend(gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms))
        for j, unit in enumerate(units):
            self.timeline.append(text_entry(self.idx(), unit, text_idx_offset + j, self.chunk_n, self.chunk_ms, source, turn_id))
        self.timeline.append(text_entry(self.idx(), {"token_id": None, "token_text": "<EOR>"}, text_idx_offset + len(units), self.chunk_n, self.chunk_ms, source, turn_id))
        self.timeline[-1]["kind"] = "eor"
        self.timeline[-1]["label"] = "<EOR>"
        self.timeline[-1]["token_text"] = "<EOR>"
        while len(self.timeline) < len(self.audio) // self.chunk_n:
            self.timeline.append(entry(self.idx(), "IDLE", "answer_tail_idle", self.chunk_n, self.chunk_ms, source, turn_id))


    def answer_region_chunks(self, text: str) -> int:
        return max(1, int(math.ceil(len(self.tokenizer.encode(text)) * 1.1)))

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))


def random_duration_chunks(rng: random.Random, min_sec: float, max_sec: float, chunk_ms: int) -> int:
    lo = min(min_sec, max_sec)
    hi = max(min_sec, max_sec)
    sec = rng.uniform(lo, hi)
    return max(1, int(round(sec * 1000.0 / chunk_ms)))


def add_initial_idle(b: Builder, args: argparse.Namespace) -> None:
    if args.initial_idle_chunks > 0:
        chunks = args.initial_idle_chunks
    else:
        chunks = random_duration_chunks(b.rng, args.initial_idle_sec_min, args.initial_idle_sec_max, args.chunk_ms)
    b.add_noise(chunks, "IDLE", "initial_idle", "gn_before")


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def build_normal(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(
        args.sample_rate,
        args.chunk_ms,
        args.noise_rms,
        stable_seed(row["id"]),
        args.text_tokenizer,
        args.vad_processor,
        args.min_query_audio_sec,
    )
    add_initial_idle(b, args)
    assets = row["tts_assets"]
    turns = row.get("turns") or [{"question_text": row["question_text"], "answer_text": row["answer_text"]}]
    for i, turn in enumerate(turns, start=1):
        key = "query" if len(turns) == 1 else f"turn{i:03d}_query"
        b.add_query_audio(assets[key]["audio"], i, f"turn{i}_query_audio")
        b.add_answer(turn["answer_text"], i, f"turn{i}_answer_gn", min_chunks=b.answer_region_chunks(turn["answer_text"]))
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "normal_qa")


def build_interrupt(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(
        args.sample_rate,
        args.chunk_ms,
        args.noise_rms,
        stable_seed(row["id"]),
        args.text_tokenizer,
        args.vad_processor,
        args.min_query_audio_sec,
    )
    add_initial_idle(b, args)
    assets = row["tts_assets"]
    b.add_query_audio(assets["base_query"]["audio"], 1, "base_query_audio")
    b.add_answer(row["base"]["answer_prefix_text"], 1, "base_answer_prefix_gn", prefix_only=True)
    b.add_query_audio(assets["donor_query"]["audio"], 2, "donor_query_audio", first_label="INTERRUPT")
    donor_answer = row["donor"]["answer_text"]
    b.add_answer(donor_answer, 2, "donor_answer_gn", min_chunks=b.answer_region_chunks(donor_answer))
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "player_interrupts_ai")


def build_backchannel(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(
        args.sample_rate,
        args.chunk_ms,
        args.noise_rms,
        stable_seed(row["id"]),
        args.text_tokenizer,
        args.vad_processor,
        args.min_query_audio_sec,
    )
    add_initial_idle(b, args)
    assets = row["tts_assets"]
    b.add_query_audio(assets["query"]["audio"], 1, "query_audio")
    answer_prefix = row["answer_prefix_text"]
    answer_remaining = row["answer_remaining_text"]
    b.add_answer(answer_prefix, 1, "answer_prefix_gn", prefix_only=True)
    b.add_query_audio(assets["backchannel"]["audio"], 1, "backchannel_audio", first_label="BACKCHANNEL")
    b.add_answer_continuation(
        answer_remaining,
        1,
        "answer_remaining_gn",
        text_idx_offset=b.token_count(answer_prefix),
        min_chunks=b.answer_region_chunks(answer_remaining),
    )
    b.add_noise(args.final_idle_chunks, "IDLE", "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "player_backchannel")


def build_incomplete(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
    b = Builder(
        args.sample_rate,
        args.chunk_ms,
        args.noise_rms,
        stable_seed(row["id"]),
        args.text_tokenizer,
        args.vad_processor,
        args.min_query_audio_sec,
    )
    add_initial_idle(b, args)
    assets = row["tts_assets"]
    b.add_query_audio(assets["query_part1"]["audio"], 1, "query_part1_audio")
    between_chunks = max(1, int(round(float(row["gn_policy"]["between_query_parts_sec"]) * 1000.0 / args.chunk_ms)))
    b.add_noise(between_chunks, "WAIT", "incomplete_pause_wait", "gn_between_query_parts", 1)
    b.add_query_audio(assets["query_part2"]["audio"], 1, "query_part2_audio")
    answer = row["answer_text_if_complete"]
    b.add_answer(answer, 1, "answer_gn", min_chunks=b.answer_region_chunks(answer))
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
        "backchannel_text": row.get("backchannel_text", ""),
        "timeline": b.timeline,
        "tokenizer": b.tokenizer.metadata(),
        "vad": b.vad.metadata(),
        "query_vad": b.query_vad,
        "stats": {
            "timeline_chunks": len(b.timeline),
            "audio_samples": len(b.audio),
            "duration_sec": round(len(b.audio) / b.sample_rate, 6),
        },
        "source_row": row,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Format duplex scenario index into wav + timeline manifest.")
    ap.add_argument("--index", required=True, help="scenario_index.jsonl from 02_make_turn_tts_tasks.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--sample_rate", type=int, default=24000)
    ap.add_argument("--chunk_ms", type=int, default=180)
    ap.add_argument("--noise_rms", type=float, default=0.003)
    ap.add_argument("--initial_idle_chunks", type=int, default=0, help=">0 keeps legacy fixed initial GN chunks; 0 uses random initial GN seconds.")
    ap.add_argument("--initial_idle_sec_min", type=float, default=0.5)
    ap.add_argument("--initial_idle_sec_max", type=float, default=1.5)
    ap.add_argument("--final_idle_chunks", type=int, default=2)
    ap.add_argument("--tokenizer_json", default="tokenizers/qwen3_8b/tokenizer.json")
    ap.add_argument("--vad_mode", choices=["silero", "auto", "energy", "off"], default="silero")
    ap.add_argument("--min_query_audio_sec", type=float, default=1.0, help="Skip a sample if any required query wav is shorter than this; 0 disables.")
    args = ap.parse_args()

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else None
    if tokenizer_json and not tokenizer_json.is_absolute():
        tokenizer_json = Path.cwd() / tokenizer_json
    args.text_tokenizer = TextTokenizer(str(tokenizer_json) if tokenizer_json else "")
    args.vad_processor = VadSilenceReplacer(args.vad_mode, args.sample_rate, args.noise_rms)

    rows = read_jsonl(Path(args.index))
    out_path = Path(args.out)
    wav_dir = Path(args.wav_dir) if args.wav_dir else out_path.parent / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "normal_qa": build_normal,
        "player_interrupts_ai": build_interrupt,
        "player_backchannel": build_backchannel,
        "incomplete_query_candidate": build_incomplete,
    }
    n = 0
    skipped: List[Dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        progress = tqdm(rows, total=len(rows), dynamic_ncols=True, unit="row", desc=f"format {out_path.parent.name}")
        for row in progress:
            scenario = row.get("scenario")
            if scenario not in builders:
                skipped.append({
                    "id": row.get("id"),
                    "scenario": scenario,
                    "error": "unsupported_scenario",
                })
                progress.set_postfix(written=n, skipped=len(skipped), refresh=False)
                continue
            out_wav = wav_dir / f"{row['id']}.wav"
            try:
                manifest = builders[scenario](row, out_wav, args)
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "id": row.get("id"),
                    "scenario": scenario,
                    "error": type(exc).__name__,
                    "message": str(exc),
                })
                progress.set_postfix(written=n, skipped=len(skipped), refresh=False)
                continue
            f.write(json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
            if n % 100 == 0:
                progress.set_postfix(written=n, skipped=len(skipped), refresh=False)
    skipped_path = out_path.with_suffix(out_path.suffix + ".skipped.jsonl")
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    if skipped:
        write_jsonl(skipped_path, skipped)
    stats = {
        "index": args.index,
        "out": str(out_path),
        "wav_dir": str(wav_dir),
        "input_rows": len(rows),
        "n": n,
        "skipped": len(skipped),
        "skipped_path": str(skipped_path) if skipped else "",
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
