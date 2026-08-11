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

from duplex_label_protocol import (
    EOR,
    FD_A_ANSWER,
    FD_D_WAIT,
    FD_F_WAIT,
    FD_G_INTERRUPT,
    FD_H_CONTINUE,
    FD_IDLE,
    FD_J_ACTIVE,
    PROTOCOL_NAME,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def float_to_pcm16(x: float) -> int:
    return int(max(-32768, min(32767, round(max(-1.0, min(1.0, float(x))) * 32767.0))))


def read_wav_with_soundfile(path: Path, sample_rate: int, *, allow_resample: bool = False) -> List[int]:
    import soundfile as sf  # type: ignore

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate:
        if not allow_resample:
            raise ValueError(f"{path} sample_rate={sr}, expected {sample_rate}")
        import torch  # type: ignore
        from torchaudio.functional import resample  # type: ignore

        waveform = torch.from_numpy(data.T)
        data = resample(waveform, int(sr), int(sample_rate)).T.numpy()
    if data.size == 0:
        return []
    if data.shape[1] == 1:
        mono = data[:, 0]
    else:
        mono = data.mean(axis=1)
    return [float_to_pcm16(x) for x in mono]


def read_wav_pcm16_fallback(path: Path, sample_rate: int, *, allow_resample: bool = False) -> List[int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        if sr != sample_rate:
            if allow_resample:
                raise RuntimeError("soundfile and torchaudio are required to resample backchannel audio")
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


def read_wav_mono_pcm16(path: Path, sample_rate: int, *, allow_resample: bool = False) -> List[int]:
    try:
        return read_wav_with_soundfile(path, sample_rate, allow_resample=allow_resample)
    except ImportError:
        return read_wav_pcm16_fallback(path, sample_rate, allow_resample=allow_resample)


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

    def process(self, samples: List[int], rng: random.Random, *, trim_silence: bool = False) -> Tuple[List[int], Dict[str, Any]]:
        sample_count = len(samples)
        if self.backend == "off" or not samples:
            if trim_silence:
                raise ValueError(f"query_audio_no_speech sample_count={sample_count} vad_backend={self.backend}")
            return samples, self._meta(sample_count, None, None)
        if self.backend == "silero":
            start, end = self._silero_bounds(samples)
        else:
            start, end = self._energy_bounds(samples)
        if start is None or end is None or start >= end:
            if trim_silence:
                raise ValueError(f"query_audio_no_speech sample_count={sample_count} vad_backend={self.backend}")
            return samples, self._meta(sample_count, None, None)
        if trim_silence:
            trimmed = samples[start:end]
            return trimmed, self._meta(
                sample_count,
                start,
                end,
                processed_count=len(trimmed),
                trim_silence_requested=True,
                trim_applied=True,
            )
        replaced = list(samples)
        if start > 0:
            replaced[:start] = gaussian_noise_samples(start, rng, self.noise_rms)
        if end < len(replaced):
            replaced[end:] = gaussian_noise_samples(len(replaced) - end, rng, self.noise_rms)
        return replaced, self._meta(sample_count, start, end)

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

    def _meta(
        self,
        sample_count: int,
        start: Optional[int],
        end: Optional[int],
        *,
        processed_count: Optional[int] = None,
        trim_silence_requested: bool = False,
        trim_applied: bool = False,
    ) -> Dict[str, Any]:
        processed_count = sample_count if processed_count is None else processed_count
        leading = start if start is not None else 0
        trailing = sample_count - end if end is not None else 0
        return {
            "requested_mode": self.requested_mode,
            "backend": self.backend,
            "vad_sample_rate": self.vad_sample_rate if self.backend == "silero" else self.sample_rate,
            "speech_start_sample": start,
            "speech_end_sample": end,
            "original_sample_count": sample_count,
            "processed_sample_count": processed_count,
            "original_duration_sec": round(sample_count / self.sample_rate, 6),
            "processed_duration_sec": round(processed_count / self.sample_rate, 6),
            "trim_silence_requested": trim_silence_requested,
            "trim_applied": trim_applied,
            "trim_start_sample": start if trim_applied else None,
            "trim_end_sample": end if trim_applied else None,
            "leading_trimmed_sec": round(leading / self.sample_rate, 6) if trim_applied else 0.0,
            "trailing_trimmed_sec": round(trailing / self.sample_rate, 6) if trim_applied else 0.0,
            "leading_replaced_sec": 0.0 if trim_applied else round(leading / self.sample_rate, 6),
            "trailing_replaced_sec": 0.0 if trim_applied else round(trailing / self.sample_rate, 6),
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
        self.inter_turn_idle: List[Dict[str, Any]] = []
        self.clarification_wait: List[Dict[str, Any]] = []

    def idx(self) -> int:
        return len(self.timeline)

    def add_noise(self, chunks: int, label: str, kind: str, source: str, turn_id: int = 0) -> None:
        samples = gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms)
        self.audio.extend(samples)
        for _ in range(chunks):
            self.timeline.append(entry(self.idx(), label, kind, self.chunk_n, self.chunk_ms, source, turn_id))

    def add_query_audio(
        self,
        path: str,
        turn_id: int,
        source: str,
        *,
        first_label: str = FD_D_WAIT,
        last_label: Optional[str] = None,
        trim_silence: bool = False,
        min_audio_sec: Optional[float] = None,
        allow_resample: bool = False,
        vad_processor: Optional[VadSilenceReplacer] = None,
    ) -> None:
        samples = read_wav_mono_pcm16(Path(path), self.sample_rate, allow_resample=allow_resample)
        effective_min_sec = self.min_query_audio_sec if min_audio_sec is None else min_audio_sec
        duration_sec = len(samples) / self.sample_rate if self.sample_rate else 0.0
        if effective_min_sec > 0 and duration_sec < effective_min_sec:
            raise ValueError(
                f"query_audio_too_short path={path} "
                f"duration_sec={duration_sec:.6f} "
                f"min_audio_sec={effective_min_sec:.6f}"
            )
        processor = vad_processor or self.vad
        samples, vad_meta = processor.process(samples, self.rng, trim_silence=trim_silence)
        processed_duration_sec = len(samples) / self.sample_rate if self.sample_rate else 0.0
        if effective_min_sec > 0 and processed_duration_sec < effective_min_sec:
            raise ValueError(
                f"query_audio_too_short_after_vad path={path} "
                f"duration_sec={processed_duration_sec:.6f} "
                f"min_audio_sec={effective_min_sec:.6f}"
            )
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
            label = first_label if i == 0 else FD_D_WAIT
            if last_label is not None and i == chunks - 1:
                label = last_label
            kind_by_label = {
                FD_D_WAIT: "wait",
                FD_F_WAIT: "incomplete_query_detected",
                FD_G_INTERRUPT: "interrupt",
            }
            kind = kind_by_label.get(label, "control")
            self.timeline.append(entry(self.idx(), label, kind, self.chunk_n, self.chunk_ms, source, turn_id))

    def add_answer(self, text: str, turn_id: int, source: str, *, prefix_only: bool = False, min_chunks: int = 0) -> None:
        units = self.tokenizer.encode(text)
        need = 1 + len(units) + (0 if prefix_only else 1)
        chunks = max(min_chunks, need)
        self.audio.extend(gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms))
        self.timeline.append(entry(self.idx(), FD_A_ANSWER, "answer_trigger", self.chunk_n, self.chunk_ms, source, turn_id))
        for j, unit in enumerate(units):
            self.timeline.append(text_entry(self.idx(), unit, j, self.chunk_n, self.chunk_ms, source, turn_id))
        if not prefix_only:
            self.timeline.append(text_entry(self.idx(), {"token_id": None, "token_text": EOR}, len(units), self.chunk_n, self.chunk_ms, source, turn_id))
            self.timeline[-1]["kind"] = "eor"
            self.timeline[-1]["label"] = EOR
            self.timeline[-1]["token_text"] = EOR
        while len(self.timeline) < len(self.audio) // self.chunk_n:
            self.timeline.append(entry(self.idx(), FD_IDLE, "answer_tail_idle", self.chunk_n, self.chunk_ms, source, turn_id))

    def add_answer_continuation(self, text: str, turn_id: int, source: str, *, text_idx_offset: int = 0, min_chunks: int = 0) -> None:
        units = self.tokenizer.encode(text)
        need = 1 + len(units) + 1
        chunks = max(min_chunks, need)
        self.audio.extend(gaussian_noise(chunks, self.chunk_n, self.rng, self.noise_rms))
        self.timeline.append(entry(self.idx(), FD_A_ANSWER, "answer_trigger", self.chunk_n, self.chunk_ms, source, turn_id))
        for j, unit in enumerate(units):
            self.timeline.append(text_entry(self.idx(), unit, text_idx_offset + j, self.chunk_n, self.chunk_ms, source, turn_id))
        self.timeline.append(text_entry(self.idx(), {"token_id": None, "token_text": EOR}, text_idx_offset + len(units), self.chunk_n, self.chunk_ms, source, turn_id))
        self.timeline[-1]["kind"] = "eor"
        self.timeline[-1]["label"] = EOR
        self.timeline[-1]["token_text"] = EOR
        while len(self.timeline) < len(self.audio) // self.chunk_n:
            self.timeline.append(entry(self.idx(), FD_IDLE, "answer_tail_idle", self.chunk_n, self.chunk_ms, source, turn_id))


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
    b.add_noise(chunks, FD_IDLE, "initial_idle", "gn_before")


def row_has_original_history(row: Dict[str, Any]) -> bool:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if int(meta.get("history_turn_count") or 0) > 0:
        return True
    turns = row.get("turns")
    return isinstance(turns, list) and any(isinstance(t, dict) and t.get("source") == "history" for t in turns)


def add_inter_turn_idle(b: Builder, args: argparse.Namespace, row: Dict[str, Any], prev_turn_id: int, next_turn_id: int) -> None:
    if args.disable_inter_turn_idle or not (row_has_original_history(row) or row.get("force_inter_turn_idle")):
        return
    chunks = random_duration_chunks(b.rng, args.inter_turn_idle_sec_min, args.inter_turn_idle_sec_max, args.chunk_ms)
    source = f"gn_between_turn{prev_turn_id}_turn{next_turn_id}"
    b.add_noise(chunks, FD_IDLE, "between_turn_idle", source, prev_turn_id)
    b.inter_turn_idle.append({
        "after_turn_id": prev_turn_id,
        "before_turn_id": next_turn_id,
        "chunks": chunks,
        "duration_sec": round(chunks * args.chunk_ms / 1000.0, 6),
        "range_sec": [args.inter_turn_idle_sec_min, args.inter_turn_idle_sec_max],
        "audio_source": source,
    })


def add_clarification_wait(b: Builder, args: argparse.Namespace, row: Dict[str, Any], turn_id: int) -> None:
    policy = row.get("gn_policy") if isinstance(row.get("gn_policy"), dict) else {}
    wait_range = policy.get("clarification_wait_range_sec", [3.0, 5.0])
    if not isinstance(wait_range, list) or len(wait_range) != 2:
        raise ValueError("clarification_wait_range_sec must contain [min_sec, max_sec]")
    min_sec, max_sec = float(wait_range[0]), float(wait_range[1])
    if min_sec < 0 or max_sec < 0:
        raise ValueError("clarification wait range must be non-negative")
    chunks = max(1, random_duration_chunks(b.rng, min_sec, max_sec, args.chunk_ms))
    source = f"gn_before_turn{turn_id}_clarification"
    idle_chunks = chunks - 1
    if idle_chunks:
        b.add_noise(idle_chunks, FD_IDLE, "clarification_wait", source, turn_id)
    b.add_noise(1, FD_J_ACTIVE, "clarification_active", source, turn_id)
    b.clarification_wait.append({
        "turn_id": turn_id,
        "chunks": chunks,
        "idle_chunks": idle_chunks,
        "active_chunks": 1,
        "duration_sec": round(chunks * args.chunk_ms / 1000.0, 6),
        "range_sec": [min_sec, max_sec],
        "audio_source": source,
    })


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def find_turn_index(turns: List[Dict[str, Any]], turn_id: Any, default: int) -> int:
    for i, turn in enumerate(turns):
        if isinstance(turn, dict) and turn.get("turn_id") == turn_id:
            return i
    return default


def turn_query_asset_key(turns: List[Dict[str, Any]], idx: int) -> str:
    return "query" if len(turns) == 1 else f"turn{idx + 1:03d}_query"


def add_complete_turn(b: Builder, assets: Dict[str, Any], turn: Dict[str, Any], local_idx: int, key: str) -> None:
    turn_id = local_idx + 1
    b.add_query_audio(assets[key]["audio"], turn_id, f"turn{turn_id}_query_audio", trim_silence=True)
    answer_text = str(turn.get("answer_text", ""))
    b.add_answer(answer_text, turn_id, f"turn{turn_id}_answer_gn", min_chunks=b.answer_region_chunks(answer_text))


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
    for idx, turn in enumerate(turns):
        add_complete_turn(b, assets, turn, idx, turn_query_asset_key(turns, idx))
        if idx + 1 < len(turns):
            add_inter_turn_idle(b, args, row, idx + 1, idx + 2)
    b.add_noise(args.final_idle_chunks, FD_IDLE, "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "normal_qa")


def build_incomplete_clarification(row: Dict[str, Any], out_wav: Path, args: argparse.Namespace) -> Dict[str, Any]:
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
    turns = row.get("turns") or []
    if not turns:
        raise ValueError("incomplete_query_clarification requires non-empty turns")
    inserted_turn_id = int(row.get("inserted_turn_id") or 0)
    for idx, turn in enumerate(turns):
        key = f"turn{idx + 1:03d}_query"
        turn_id = idx + 1
        is_inserted = turn_id == inserted_turn_id or turn.get("source") == "inserted_incomplete_query"
        b.add_query_audio(
            assets[key]["audio"],
            turn_id,
            f"turn{turn_id}_query_audio",
            last_label=FD_F_WAIT if is_inserted else None,
            trim_silence=True,
        )
        if is_inserted:
            add_clarification_wait(b, args, row, turn_id)
        answer_text = str(turn.get("answer_text", ""))
        b.add_answer(answer_text, turn_id, f"turn{turn_id}_answer_gn", min_chunks=b.answer_region_chunks(answer_text))
        if idx + 1 < len(turns):
            add_inter_turn_idle(b, args, row, turn_id, turn_id + 1)
    b.add_noise(args.final_idle_chunks, FD_IDLE, "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "incomplete_query_clarification")


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
    turns = [t for t in (row.get("turns") or []) if isinstance(t, dict)]
    if not turns:
        turns = [row.get("base", {}), row.get("donor", {})]
    base_idx = find_turn_index(turns, row.get("base", {}).get("turn_id"), max(0, len(turns) - 2))
    donor_idx = find_turn_index(turns, row.get("donor", {}).get("turn_id"), max(0, len(turns) - 1))
    if donor_idx <= base_idx:
        base_idx = max(0, len(turns) - 2)
        donor_idx = max(0, len(turns) - 1)

    for idx in range(0, base_idx):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    base_turn_id = base_idx + 1
    b.add_query_audio(assets["base_query"]["audio"], base_turn_id, "base_query_audio", trim_silence=True)
    b.add_answer(row["base"]["answer_prefix_text"], base_turn_id, "base_answer_prefix_gn", prefix_only=True)

    donor_turn_id = donor_idx + 1
    b.add_query_audio(assets["donor_query"]["audio"], donor_turn_id, "donor_query_audio", first_label=FD_G_INTERRUPT, trim_silence=True)
    donor_answer = row["donor"]["answer_text"]
    b.add_answer(donor_answer, donor_turn_id, "donor_answer_gn", min_chunks=b.answer_region_chunks(donor_answer))

    if donor_idx + 1 < len(turns):
        add_inter_turn_idle(b, args, row, donor_turn_id, donor_turn_id + 1)
    for idx in range(donor_idx + 1, len(turns)):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        if idx + 1 < len(turns):
            add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    b.add_noise(args.final_idle_chunks, FD_IDLE, "final_idle", "gn_after")
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
    turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    if not turns:
        turns = [{"turn_id": 1, "question_text": row.get("question_text", ""), "answer_text": row.get("answer_text", "")}]
    special_idx = int(row.get("backchannel_turn_index") or 0) - 1
    if not (0 <= special_idx < len(turns)):
        special_idx = find_turn_index(turns, row.get("backchannel_turn_id"), len(turns) - 1)

    for idx in range(special_idx):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    turn_id = special_idx + 1
    b.add_query_audio(assets["query"]["audio"], turn_id, "backchannel_turn_query_audio", trim_silence=True)
    answer_prefix = row["answer_prefix_text"]
    answer_remaining = row["answer_remaining_text"]
    b.add_answer(answer_prefix, turn_id, "answer_prefix_gn", prefix_only=True)
    b.add_query_audio(
        assets["backchannel"]["audio"],
        turn_id,
        "backchannel_audio",
        first_label=FD_G_INTERRUPT,
        trim_silence=True,
        min_audio_sec=args.min_backchannel_audio_sec,
        allow_resample=True,
        vad_processor=args.backchannel_vad_processor,
    )
    b.add_noise(1, FD_H_CONTINUE, "backchannel_continue", "backchannel_continue_gn", turn_id)
    b.add_answer_continuation(
        answer_remaining,
        turn_id,
        "answer_remaining_gn",
        text_idx_offset=b.token_count(answer_prefix),
        min_chunks=b.answer_region_chunks(answer_remaining),
    )
    if special_idx + 1 < len(turns):
        add_inter_turn_idle(b, args, row, turn_id, turn_id + 1)
    for idx in range(special_idx + 1, len(turns)):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        if idx + 1 < len(turns):
            add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    b.add_noise(args.final_idle_chunks, FD_IDLE, "final_idle", "gn_after")
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
    turns = [t for t in (row.get("turns") or []) if isinstance(t, dict)]
    special_idx = int(row.get("incomplete_turn_index") or 0) - 1
    if turns and not (0 <= special_idx < len(turns)):
        special_idx = find_turn_index(turns, row.get("incomplete_turn_id"), len(turns) - 1)
    if not turns:
        turns = [{"turn_id": 1, "question_text": row.get("full_question_text", ""), "answer_text": row.get("answer_text_if_complete", "")}]
        special_idx = 0

    for idx in range(0, special_idx):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    special_turn_id = special_idx + 1
    b.add_query_audio(
        assets["query_part1"]["audio"],
        special_turn_id,
        "query_part1_audio",
        last_label=FD_F_WAIT,
        trim_silence=True,
    )
    between_chunks = max(1, int(round(float(row["gn_policy"]["between_query_parts_sec"]) * 1000.0 / args.chunk_ms)))
    b.add_noise(between_chunks, FD_IDLE, "incomplete_pause_wait", "gn_between_query_parts", special_turn_id)
    b.add_query_audio(assets["query_part2"]["audio"], special_turn_id, "query_part2_audio", trim_silence=True)
    answer = row["answer_text_if_complete"]
    b.add_answer(answer, special_turn_id, "answer_gn", min_chunks=b.answer_region_chunks(answer))

    if special_idx + 1 < len(turns):
        add_inter_turn_idle(b, args, row, special_turn_id, special_turn_id + 1)
    for idx in range(special_idx + 1, len(turns)):
        add_complete_turn(b, assets, turns[idx], idx, f"turn{idx + 1:03d}_query")
        if idx + 1 < len(turns):
            add_inter_turn_idle(b, args, row, idx + 1, idx + 2)

    b.add_noise(args.final_idle_chunks, FD_IDLE, "final_idle", "gn_after")
    write_wav_pcm16(out_wav, b.audio, args.sample_rate)
    return common_manifest(row, out_wav, b, "incomplete_query")


def common_manifest(row: Dict[str, Any], out_wav: Path, b: Builder, scenario: str) -> Dict[str, Any]:
    answer = row.get("answer_text") or row.get("answer_text_if_complete") or row.get("donor", {}).get("answer_text", "")
    question = row.get("question_text") or row.get("full_question_text") or row.get("donor", {}).get("question_text", "")
    return {
        "id": row["id"],
        "source": "cgame_duplex",
        "scenario": scenario,
        "label_protocol": PROTOCOL_NAME,
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
        "inter_turn_idle": b.inter_turn_idle,
        "clarification_wait": b.clarification_wait,
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
    ap.add_argument("--inter_turn_idle_sec_min", type=float, default=1.0, help="Random IDLE floor inserted after EOR between original multi-turn dialogue turns.")
    ap.add_argument("--inter_turn_idle_sec_max", type=float, default=3.0, help="Random IDLE ceiling inserted after EOR between original multi-turn dialogue turns.")
    ap.add_argument("--disable_inter_turn_idle", action="store_true", help="Disable random IDLE between original multi-turn dialogue turns.")
    ap.add_argument("--tokenizer_json", default="tokenizers/qwen3_8b/tokenizer.json")
    ap.add_argument("--vad_mode", choices=["silero", "auto", "energy", "off"], default="silero")
    ap.add_argument("--quiet", action="store_true", help="Disable the per-row tqdm progress bar.")
    ap.add_argument("--backchannel_vad_mode", choices=["energy", "silero"], default="energy", help="VAD used for short recorded backchannel clips.")
    ap.add_argument("--min_query_audio_sec", type=float, default=1.0, help="Skip a sample if any required query wav is shorter than this; 0 disables.")
    ap.add_argument("--min_backchannel_audio_sec", type=float, default=0.08, help="Minimum voiced backchannel duration after VAD; 0 disables.")
    args = ap.parse_args()

    tokenizer_json = Path(args.tokenizer_json) if args.tokenizer_json else None
    if tokenizer_json and not tokenizer_json.is_absolute():
        tokenizer_json = Path.cwd() / tokenizer_json
    args.text_tokenizer = TextTokenizer(str(tokenizer_json) if tokenizer_json else "")
    args.vad_processor = VadSilenceReplacer(args.vad_mode, args.sample_rate, args.noise_rms)
    args.backchannel_vad_processor = VadSilenceReplacer(
        args.backchannel_vad_mode,
        args.sample_rate,
        args.noise_rms,
    )

    rows = read_jsonl(Path(args.index))
    out_path = Path(args.out)
    wav_dir = Path(args.wav_dir) if args.wav_dir else out_path.parent / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    builders = {
        "normal_qa": build_normal,
        "incomplete_query_clarification": build_incomplete_clarification,
        "player_interrupts_ai": build_interrupt,
        "player_backchannel": build_backchannel,
        "incomplete_query_candidate": build_incomplete,
    }
    n = 0
    skipped: List[Dict[str, Any]] = []
    with out_path.open("w", encoding="utf-8") as f:
        progress = tqdm(rows, total=len(rows), dynamic_ncols=True, unit="row", desc=f"format {out_path.parent.name}", disable=args.quiet)
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
