#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("task_id") or "") for row in iter_jsonl(path) if row.get("task_id")}


def levenshtein(a: list[int], b: list[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def is_attachable_punctuation(char: str) -> bool:
    return bool(char) and (unicodedata.category(char).startswith("P") or char in "~～…")


def transcript_tokens(text: str, vocab: dict[str, int], special_ids: set[int], converter) -> tuple[list[int], list[int]]:
    token_ids: list[int] = []
    char_indices: list[int] = []
    for index, char in enumerate(text):
        normalized = unicodedata.normalize("NFKC", converter.convert(char)).lower()
        token_id = vocab.get(normalized) if len(normalized) == 1 else None
        if token_id is None or token_id in special_ids or char.isspace():
            continue
        token_ids.append(int(token_id))
        char_indices.append(index)
    return token_ids, char_indices


def load_audio(task: dict[str, Any], sample_rate: int):
    import torch
    import torchaudio

    path = str(task["audio_path"])
    info = torchaudio.info(path)
    source_rate = int(info.sample_rate)
    start_sec = float(task.get("audio_start_sec") or 0.0)
    end_value = task.get("audio_end_sec")
    frame_offset = max(0, int(round(start_sec * source_rate)))
    if end_value is None:
        num_frames = -1
    else:
        num_frames = max(1, int(round(float(end_value) * source_rate)) - frame_offset)
    waveform, loaded_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError("empty audio")
    waveform = waveform.mean(dim=0)
    if int(loaded_rate) != sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(loaded_rate), sample_rate)
    waveform = waveform.to(torch.float32)
    return waveform, source_rate


def greedy_ids(frame_ids: list[int], ignored_ids: set[int]) -> list[int]:
    output: list[int] = []
    previous = None
    for token_id in frame_ids:
        if token_id != previous and token_id not in ignored_ids:
            output.append(token_id)
        previous = token_id
    return output


def align_one(
    task: dict[str, Any],
    log_probs,
    audio_duration: float,
    vocab: dict[str, int],
    special_ids: set[int],
    blank_id: int,
    boundary_tolerance_sec: float,
    converter,
) -> dict[str, Any]:
    import torch
    import torchaudio

    full_query = str(task["full_query"])
    target_ids, char_indices = transcript_tokens(full_query, vocab, special_ids, converter)
    if not target_ids:
        raise ValueError("transcript has no alignable characters")
    targets = torch.tensor(target_ids, dtype=torch.int32, device=log_probs.device).unsqueeze(0)
    paths, scores = torchaudio.functional.forced_align(
        log_probs.unsqueeze(0), targets, blank=blank_id,
    )
    spans = torchaudio.functional.merge_tokens(paths[0], scores[0], blank=blank_id)
    if len(spans) != len(target_ids):
        raise ValueError(f"aligned span count {len(spans)} != target count {len(target_ids)}")

    frame_count = int(log_probs.shape[0])
    seconds_per_frame = audio_duration / frame_count
    aligned_chars: list[dict[str, Any]] = []
    for source_index, token_id, span in zip(char_indices, target_ids, spans):
        raw_score = float(span.score)
        score = math.exp(raw_score) if raw_score <= 0 else raw_score
        aligned_chars.append({
            "char": full_query[source_index],
            "source_index": source_index,
            "token_id": token_id,
            "start_sec": float(span.start) * seconds_per_frame,
            "end_sec": float(span.end) * seconds_per_frame,
            "score": score,
        })

    cut_sec = min(max(float(task["cut_sec"]), 0.0), audio_duration)
    completed = [item for item in aligned_chars if item["end_sec"] <= cut_sec + boundary_tolerance_sec]
    if not completed:
        raise ValueError("cut occurs before the first aligned character ends")
    last = completed[-1]
    prefix_end = int(last["source_index"]) + 1
    while prefix_end < len(full_query) and is_attachable_punctuation(full_query[prefix_end]):
        prefix_end += 1
    partial_query = full_query[:prefix_end].strip()
    next_char = next((item for item in aligned_chars if item["source_index"] > last["source_index"]), None)

    ignored_ids = set(special_ids)
    delimiter_id = vocab.get("|")
    if delimiter_id is not None:
        ignored_ids.add(delimiter_id)
    predicted = greedy_ids(log_probs.argmax(dim=-1).tolist(), ignored_ids)
    cer = levenshtein(predicted, target_ids) / max(1, len(target_ids))
    supported_non_punct = sum(
        1 for char in full_query if not char.isspace() and not is_attachable_punctuation(char)
    )
    support_ratio = len(target_ids) / max(1, supported_non_punct)
    boundary_chars = completed[-3:] + ([next_char] if next_char else [])
    boundary_scores = [float(item["score"]) for item in boundary_chars if item]
    all_scores = [float(item["score"]) for item in aligned_chars]
    quality_reasons: list[str] = []
    if support_ratio < 0.70:
        quality_reasons.append("low_character_coverage")
    if cer > 0.65:
        quality_reasons.append("high_greedy_cer")
    if sum(all_scores) / len(all_scores) < 0.08:
        quality_reasons.append("low_alignment_score")
    if sum(boundary_scores) / max(1, len(boundary_scores)) < 0.01:
        quality_reasons.append("low_boundary_score")
    if float(task["cut_sec"]) > audio_duration + 0.15:
        quality_reasons.append("cut_after_audio_end")

    return {
        **task,
        "schema_version": "external_fwait_alignment_result_v1",
        "status": "uncertain" if quality_reasons else "ok",
        "partial_query": partial_query,
        "partial_query_char_count": sum(
            1 for char in partial_query if not char.isspace() and not is_attachable_punctuation(char)
        ),
        "last_completed_char": last,
        "next_char": next_char,
        "audio_duration_sec": audio_duration,
        "effective_cut_sec": cut_sec,
        "alignment": {
            "alignable_chars": len(target_ids),
            "supported_character_ratio": support_ratio,
            "mean_score": sum(all_scores) / len(all_scores),
            "min_score": min(all_scores),
            "boundary_mean_score": sum(boundary_scores) / max(1, len(boundary_scores)),
            "greedy_cer": cer,
            "quality_reasons": quality_reasons,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover F_WAIT partial text with batched Chinese CTC forced alignment.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bucket_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary_tolerance_sec", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bucket_size <= 0:
        raise SystemExit("--batch_size and --bucket_size must be > 0")

    import torch
    from opencc import OpenCC
    from transformers import AutoModelForCTC, AutoProcessor

    tasks_path = Path(args.tasks)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else completed_ids(results_path)
    tasks = [row for row in iter_jsonl(tasks_path) if str(row.get("task_id") or "") not in done]
    rng = random.Random(args.seed)
    windows: list[list[dict[str, Any]]] = []
    for offset in range(0, len(tasks), args.bucket_size):
        window = tasks[offset:offset + args.bucket_size]
        window.sort(key=lambda row: float(row.get("audio_duration_hint_sec") or 9999))
        windows.append(window)
    rng.shuffle(windows)
    tasks = [task for window in windows for task in window]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(args.device).eval()
    sample_rate = int(processor.feature_extractor.sampling_rate)
    vocab = {str(key).lower(): int(value) for key, value in processor.tokenizer.get_vocab().items()}
    blank_id = int(model.config.pad_token_id if model.config.pad_token_id is not None else 0)
    special_ids = set(int(value) for value in processor.tokenizer.all_special_ids)
    converter = OpenCC("t2s")
    mode = "w" if args.overwrite else "a"
    written = errors = 0
    with results_path.open(mode, encoding="utf-8") as output:
        for batch_start in range(0, len(tasks), args.batch_size):
            batch_tasks = tasks[batch_start:batch_start + args.batch_size]
            waveforms = []
            valid_tasks = []
            for task in batch_tasks:
                try:
                    waveform, _ = load_audio(task, sample_rate)
                    waveforms.append(waveform.numpy())
                    valid_tasks.append(task)
                except Exception as exc:
                    result = {**task, "schema_version": "external_fwait_alignment_result_v1", "status": "error", "error": str(exc)}
                    output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output.flush()
                    written += 1
                    errors += 1
            if not valid_tasks:
                continue
            inputs = processor(waveforms, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device) if "attention_mask" in inputs else None
            with torch.inference_mode():
                logits = model(input_values, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits, dim=-1)
            if attention_mask is None:
                input_lengths = torch.tensor([len(waveform) for waveform in waveforms], device=logits.device)
            else:
                input_lengths = attention_mask.sum(dim=-1)
            output_lengths = model._get_feat_extract_output_lengths(input_lengths).to(torch.int64)
            for task, waveform, emission, output_length in zip(valid_tasks, waveforms, log_probs, output_lengths):
                try:
                    result = align_one(
                        task,
                        emission[: int(output_length)].contiguous(),
                        len(waveform) / sample_rate,
                        vocab,
                        special_ids,
                        blank_id,
                        args.boundary_tolerance_sec,
                        converter,
                    )
                except Exception as exc:
                    result = {**task, "schema_version": "external_fwait_alignment_result_v1", "status": "error", "error": str(exc)}
                    errors += 1
                output.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
            output.flush()
            if args.progress_every and written % args.progress_every < len(valid_tasks):
                print(json.dumps({"written_this_run": written, "remaining": len(tasks) - written, "errors": errors}), flush=True)

    print(json.dumps({"tasks": len(tasks), "resumed": len(done), "written": written, "errors": errors, "results": str(results_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
