#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import struct
import threading
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from duplex_label_protocol import EOR, FD_G_INTERRUPT


SCHEMA_VERSION = "duplex_multimodal_enrichment_v1"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def stable_int(text: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()[:16], 16)


def clean_description(value: Any, field: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        raise ValueError(f"{field} is empty")
    if len(text) > max_chars:
        raise ValueError(f"{field} has {len(text)} chars, max={max_chars}")
    return text


def clean_action(value: Any) -> str:
    text = clean_description(value, "action_expression", max_chars=160)
    pairs = (("（", "）"), ("(", ")"))
    changed = True
    while changed:
        changed = False
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right):
                text = text[len(left):-len(right)].strip()
                changed = True
    if not text:
        raise ValueError("action_expression is empty after removing parentheses")
    return text


class TextTokenizer:
    def __init__(self, tokenizer_json: str):
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("tokenizers is required") from exc
        self.tokenizer = Tokenizer.from_file(tokenizer_json)

    def encode(self, text: str) -> List[Dict[str, Any]]:
        encoding = self.tokenizer.encode(text.replace("\r", "").replace("\n", ""), add_special_tokens=False)
        output: List[Dict[str, Any]] = []
        for token_id, raw_token in zip(encoding.ids, encoding.tokens):
            try:
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except TypeError:
                token_text = self.tokenizer.decode([token_id])
            if token_text or raw_token:
                output.append({
                    "token_id": token_id,
                    "token_text": token_text or raw_token,
                    "raw_token": raw_token,
                })
        return output


def expected_answer_turn_ids(row: Dict[str, Any]) -> List[int]:
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    turns = source_row.get("turns") if isinstance(source_row.get("turns"), list) else []
    output = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("answer_text") or "").strip() and turn.get("train_answer") is not False:
            output.append(int(turn.get("turn_id") or 0))
    if output:
        return output
    return [1] if str(row.get("answer_text") or "").strip() else []


def load_excluded_ids(paths: List[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        for row in iter_jsonl(Path(value)):
            sample_id = str(row.get("source_id") or row.get("id") or "")
            if sample_id:
                excluded.add(sample_id)
    return excluded


def load_filled(
    path: Path,
    *,
    excluded_ids: set[str],
    normalize_voice_prefix: bool,
) -> Tuple[Dict[str, Dict[str, Any]], Counter]:
    rows: Dict[str, Dict[str, Any]] = {}
    counts: Counter = Counter()
    for row in iter_jsonl(path):
        counts["input_rows"] += 1
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{row.get('id')}: invalid schema_version")
        sample_id = str(row.get("source_id") or row.get("id") or "")
        if not sample_id:
            raise ValueError("empty source_id")
        if sample_id in excluded_ids:
            counts["excluded_rows"] += 1
            continue
        if sample_id in rows:
            raise ValueError(f"duplicate source_id: {sample_id!r}")
        scene = clean_description(row.get("scene_description"), "scene_description", max_chars=500)
        voice = clean_description(row.get("voice_description"), "voice_description", max_chars=240)
        if not voice.startswith("说话时的声音特征：") and not voice.startswith("说话时的声音特征:"):
            if not normalize_voice_prefix:
                raise ValueError(f"{sample_id}: voice_description must start with 说话时的声音特征：")
            voice = f"说话时的声音特征：{voice}"
            counts["normalized_voice_prefix"] += 1
            if len(voice) > 240:
                raise ValueError(f"{sample_id}: normalized voice_description has {len(voice)} chars, max=240")
        descriptions = row.get("turn_descriptions")
        if not isinstance(descriptions, list) or not descriptions:
            raise ValueError(f"{sample_id}: turn_descriptions must be non-empty")
        actions: Dict[int, str] = {}
        for item in descriptions:
            if not isinstance(item, dict):
                raise ValueError(f"{sample_id}: invalid turn_descriptions item")
            turn_id = int(item.get("turn_id") or 0)
            if turn_id <= 0 or turn_id in actions:
                raise ValueError(f"{sample_id}: duplicate or invalid turn_id={turn_id}")
            actions[turn_id] = clean_action(item.get("action_expression"))
        rows[sample_id] = {
            "scene_description": scene,
            "voice_description": voice,
            "actions": actions,
        }
        counts["loaded_rows"] += 1
    return rows, counts

def read_wav_chunks(path: Path, sample_rate: int, chunk_n: int, expected_chunks: int) -> List[bytes]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono PCM16")
        if wav.getframerate() != sample_rate:
            raise ValueError(f"{path}: sample_rate mismatch")
        raw = wav.readframes(wav.getnframes())
    chunk_bytes = chunk_n * 2
    if len(raw) != expected_chunks * chunk_bytes:
        raise ValueError(
            f"{path}: audio chunks={len(raw) / chunk_bytes:.3f}, timeline chunks={expected_chunks}"
        )
    return [raw[start:start + chunk_bytes] for start in range(0, len(raw), chunk_bytes)]


def write_wav_atomic(path: Path, chunks: List[bytes], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(chunks))
    tmp.replace(path)


def noise_chunk(chunk_n: int, rms: float, seed: int) -> bytes:
    rng = random.Random(seed)
    values = [rng.gauss(0.0, 1.0) for _ in range(chunk_n)]
    current = math.sqrt(sum(value * value for value in values) / len(values)) or 1.0
    scale = rms * 32767.0 / current
    pcm = [max(-32768, min(32767, int(value * scale))) for value in values]
    return struct.pack(f"<{len(pcm)}h", *pcm)



class NoiseChunkBank:
    def __init__(self, seed: int, bank_size: int):
        if bank_size <= 0:
            raise ValueError("noise bank size must be > 0")
        self.seed = seed
        self.bank_size = bank_size
        self.cache: Dict[Tuple[int, float, int], bytes] = {}

    def get(self, chunk_n: int, rms: float, key: str) -> bytes:
        slot = stable_int(key, self.seed) % self.bank_size
        cache_key = (chunk_n, rms, slot)
        if cache_key not in self.cache:
            bank_seed = stable_int(f"noise-bank:{chunk_n}:{rms}:{slot}", self.seed)
            self.cache[cache_key] = noise_chunk(chunk_n, rms, bank_seed)
        return self.cache[cache_key]

def insertion_positions(timeline: List[Dict[str, Any]], turn_ids: List[int]) -> Dict[int, Tuple[int, str]]:
    positions: Dict[int, Tuple[int, str]] = {}
    for turn_id in turn_ids:
        eors = [
            idx for idx, item in enumerate(timeline)
            if item.get("label") == EOR and int(item.get("turn_id") or 0) == turn_id
        ]
        if len(eors) == 1:
            positions[turn_id] = (eors[0], "before_eor")
            continue
        answer_indices = [
            idx for idx, item in enumerate(timeline)
            if item.get("kind") == "answer_trigger" and int(item.get("turn_id") or 0) == turn_id
        ]
        if not answer_indices:
            raise ValueError(f"turn_id={turn_id}: answer trigger not found")
        interrupt_idx = next(
            (
                idx for idx in range(answer_indices[-1] + 1, len(timeline))
                if timeline[idx].get("label") == FD_G_INTERRUPT
            ),
            -1,
        )
        if interrupt_idx < 0:
            raise ValueError(f"turn_id={turn_id}: neither EOR nor following interrupt found")
        positions[turn_id] = (interrupt_idx, "before_interrupt")
    return positions


def text_entry(token: Dict[str, Any], text_idx: int, turn_id: int) -> Dict[str, Any]:
    token_text = token["token_text"]
    return {
        "kind": "text_token",
        "label_type": "text",
        "label": token_text,
        "token_id": token.get("token_id"),
        "token_text": token_text,
        "raw_token": token.get("raw_token"),
        "text_token_idx": text_idx,
        "audio_source": "action_expression_gn",
        "turn_id": turn_id,
        "multimodal_field": "action_expression",
    }


def reindex_timeline(timeline: List[Dict[str, Any]], chunk_n: int, chunk_ms: int) -> None:
    for idx, item in enumerate(timeline):
        item["idx"] = idx
        item["start_sec"] = round(idx * chunk_ms / 1000.0, 6)
        item["end_sec"] = round((idx + 1) * chunk_ms / 1000.0, 6)
        item["start_sample"] = idx * chunk_n
        item["end_sample"] = (idx + 1) * chunk_n


def upgrade_row(
    row: Dict[str, Any],
    filled: Dict[str, Any],
    tokenizer: TextTokenizer,
    out_wav: Path,
    noise_rms: float,
    seed: int,
    noise_bank: NoiseChunkBank,
) -> Tuple[Dict[str, Any], Counter]:
    expected_turn_ids = expected_answer_turn_ids(row)
    actions = filled["actions"]
    if set(actions) != set(expected_turn_ids):
        raise ValueError(
            f"turn descriptions mismatch: expected={expected_turn_ids}, filled={sorted(actions)}"
        )
    sample_rate = int(row.get("sample_rate") or 0)
    chunk_ms = int(row.get("chunk_ms") or 0)
    if sample_rate <= 0 or chunk_ms <= 0:
        raise ValueError("invalid sample_rate or chunk_ms")
    chunk_n = int(round(sample_rate * chunk_ms / 1000.0))
    timeline = [dict(item) for item in row.get("timeline") or []]
    chunks = read_wav_chunks(Path(str(row["audio"])), sample_rate, chunk_n, len(timeline))
    positions = insertion_positions(timeline, expected_turn_ids)
    counts: Counter = Counter()

    for turn_id, (position, mode) in sorted(positions.items(), key=lambda item: item[1][0], reverse=True):
        wrapped = f"（{actions[turn_id]}）"
        tokens = tokenizer.encode(wrapped)
        if not tokens:
            raise ValueError(f"turn_id={turn_id}: action tokenization is empty")
        existing_text_indices = [
            int(item.get("text_token_idx"))
            for item in timeline[:position]
            if int(item.get("turn_id") or 0) == turn_id and item.get("text_token_idx") is not None
        ]
        text_idx = max(existing_text_indices, default=-1) + 1
        inserted_timeline = [text_entry(token, text_idx + idx, turn_id) for idx, token in enumerate(tokens)]
        inserted_chunks = [
            noise_bank.get(chunk_n, noise_rms, f"{row['id']}:{turn_id}:{idx}")
            for idx in range(len(tokens))
        ]
        timeline[position:position] = inserted_timeline
        chunks[position:position] = inserted_chunks
        following_idx = position + len(tokens)
        if mode == "before_eor" and timeline[following_idx].get("label") == EOR:
            timeline[following_idx]["text_token_idx"] = text_idx + len(tokens)
        counts[mode] += 1
        counts["inserted_action_tokens"] += len(tokens)

    reindex_timeline(timeline, chunk_n, chunk_ms)
    write_wav_atomic(out_wav, chunks, sample_rate)
    output = copy.deepcopy(row)
    output["audio"] = str(out_wav.resolve())
    output["timeline"] = timeline
    output["scene_description"] = filled["scene_description"]
    output["voice_description"] = filled["voice_description"]
    output["action_expression_descriptions"] = [
        {
            "turn_id": turn_id,
            "action_expression": actions[turn_id],
            "wrapped_text": f"（{actions[turn_id]}）",
            "insertion": positions[turn_id][1],
        }
        for turn_id in expected_turn_ids
    ]

    source_row = output.get("source_row") if isinstance(output.get("source_row"), dict) else {}
    source_turns = source_row.get("turns") if isinstance(source_row.get("turns"), list) else []
    original_answers: Dict[int, str] = {}
    augmented_answers: Dict[int, str] = {}
    for turn in source_turns:
        if not isinstance(turn, dict):
            continue
        turn_id = int(turn.get("turn_id") or 0)
        if turn_id not in actions:
            continue
        original = str(turn.get("answer_text") or "")
        augmented = original + f"（{actions[turn_id]}）"
        turn["original_answer_text"] = original
        turn["answer_text"] = augmented
        turn["action_expression"] = actions[turn_id]
        original_answers[turn_id] = original
        augmented_answers[turn_id] = augmented

    current_turn_id = max(expected_turn_ids, default=0)
    if current_turn_id in augmented_answers:
        original_current = original_answers[current_turn_id]
        augmented_current = augmented_answers[current_turn_id]
        output["original_answer_text"] = str(output.get("answer_text") or original_current)
        output["answer_text"] = augmented_current
        for field in ("text", "target_text"):
            if output.get(field) == original_current:
                output[field] = augmented_current
        if source_row.get("answer_text") == original_current:
            source_row["original_answer_text"] = original_current
            source_row["answer_text"] = augmented_current
        if source_row.get("answer_text_if_complete") == original_current:
            source_row["original_answer_text_if_complete"] = original_current
            source_row["answer_text_if_complete"] = augmented_current

    output["multimodal_enrichment"] = {
        "schema_version": SCHEMA_VERSION,
        "action_token_count": counts["inserted_action_tokens"],
        "audio_padding": "gaussian_noise_per_inserted_action_token",
        "noise_rms": noise_rms,
    }
    stats = dict(output.get("stats") or {})
    stats["chunks"] = len(timeline)
    stats["duration_sec"] = round(len(timeline) * chunk_ms / 1000.0, 6)
    output["stats"] = stats
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply filled scene, AI voice, and per-answer action annotations to sampled duplex rows."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--filled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--tokenizer_json", default="tokenizers/qwen3_8b/tokenizer.json")
    parser.add_argument("--noise_rms", type=float, default=0.003)
    parser.add_argument("--noise_bank_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--exclude_jsonl", action="append", default=[])
    parser.add_argument("--normalize_voice_prefix", action="store_true")
    parser.add_argument("--skip_invalid", action="store_true")
    parser.add_argument("--rejected", default="")
    parser.add_argument("--progress_every", type=int, default=1000)
    args = parser.parse_args()

    excluded_ids = load_excluded_ids(args.exclude_jsonl)
    filled, filled_counts = load_filled(
        Path(args.filled),
        excluded_ids=excluded_ids,
        normalize_voice_prefix=args.normalize_voice_prefix,
    )
    tokenizer = TextTokenizer(args.tokenizer_json)
    pending = set(filled)
    totals: Counter = Counter()
    wav_dir = Path(args.wav_dir)
    noise_bank = NoiseChunkBank(args.seed, args.noise_bank_size)
    used_wav_names: set[str] = set()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected = Path(args.rejected) if args.rejected else out.with_suffix(out.suffix + ".rejected.jsonl")
    rejected.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = out.with_name(f"{out.name}.tmp-{os.getpid()}")
    rejected_tmp = rejected.with_name(f"{rejected.name}.tmp-{os.getpid()}")
    with out_tmp.open("w", encoding="utf-8") as output_handle, rejected_tmp.open("w", encoding="utf-8") as rejected_handle:
        for row in iter_jsonl(Path(args.manifest)):
            sample_id = str(row.get("id") or "")
            if sample_id not in pending:
                continue
            try:
                digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
                wav_name = f"{digest}__{Path(str(row['audio'])).name}"
                if wav_name in used_wav_names:
                    raise ValueError(f"duplicate output WAV name: {wav_name}")
                used_wav_names.add(wav_name)
                upgraded, counts = upgrade_row(
                    row,
                    filled[sample_id],
                    tokenizer,
                    wav_dir / wav_name,
                    args.noise_rms,
                    args.seed,
                    noise_bank,
                )
            except Exception as exc:
                if not args.skip_invalid:
                    raise
                rejected_handle.write(json.dumps({
                    "source_id": sample_id,
                    "scenario": row.get("scenario"),
                    "stage": "upgrade",
                    "error": f"{type(exc).__name__}: {exc}",
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                totals["rejected_upgrade"] += 1
                pending.remove(sample_id)
                continue
            output_handle.write(json.dumps(upgraded, ensure_ascii=False, separators=(",", ":")) + "\n")
            totals.update(counts)
            totals[f"scenario:{row.get('scenario')}"] += 1
            totals["written"] += 1
            pending.remove(sample_id)
            if args.progress_every > 0 and totals["written"] % args.progress_every == 0:
                print(
                    f"progress written={totals['written']} rejected={totals['rejected_upgrade']} pending={len(pending)}",
                    flush=True,
                )
            if not pending:
                break
        if pending:
            if not args.skip_invalid:
                raise ValueError(f"filled IDs missing from manifest: {sorted(pending)[:20]}")
            for sample_id in sorted(pending):
                rejected_handle.write(json.dumps({
                    "source_id": sample_id,
                    "stage": "source_lookup",
                    "error": "filled ID missing from manifest",
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
                totals["rejected_missing_manifest"] += 1
            pending.clear()
    out_tmp.replace(out)
    rejected_tmp.replace(rejected)

    stats = {
        "manifest": args.manifest,
        "filled": args.filled,
        "out": str(out),
        "wav_dir": str(wav_dir),
        "rejected": str(rejected),
        "written": totals["written"],
        "counts": dict(totals),
        "filled_counts": dict(filled_counts),
        "exclude_jsonl": args.exclude_jsonl,
        "excluded_ids": len(excluded_ids),
        "tokenizer_json": args.tokenizer_json,
        "noise_rms": args.noise_rms,
        "noise_bank_size": args.noise_bank_size,
        "seed": args.seed,
    }
    out.with_suffix(out.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
