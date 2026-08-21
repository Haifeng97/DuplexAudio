#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
    if not path.is_file():
        return set()
    return {str(row.get("task_id") or "") for row in iter_jsonl(path) if row.get("task_id")}


def normalize_text(text: str, converter: Any) -> str:
    simplified = converter.convert(unicodedata.normalize("NFKC", text)).lower()
    return "".join(
        char for char in simplified
        if not char.isspace() and not unicodedata.category(char).startswith(("P", "S"))
    )


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def load_audio(path: Path, sample_rate: int):
    from math import gcd

    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise ValueError("empty_audio")
    waveform = audio.mean(axis=1)
    if int(source_rate) != sample_rate:
        factor = gcd(int(source_rate), sample_rate)
        waveform = resample_poly(waveform, sample_rate / factor, int(source_rate) / factor).astype("float32", copy=False)
    return torch.from_numpy(waveform.copy()).to(torch.float32)


def write_result(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable batched CTC verification for generated query WAV files."
    )
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--bucket_size", type=int, default=1024)
    parser.add_argument("--max_cer", type=float, default=0.65)
    parser.add_argument("--min_coverage", type=float, default=0.25)
    parser.add_argument("--min_target_chars", type=int, default=4)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress_every", type=int, default=1000)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bucket_size <= 0:
        parser.error("--batch_size and --bucket_size must be > 0")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("invalid shard selection")
    if not 0 <= args.max_cer <= 1 or not 0 <= args.min_coverage <= 1:
        parser.error("CER and coverage thresholds must be within [0, 1]")

    import torch
    from opencc import OpenCC
    from transformers import AutoModelForCTC, AutoProcessor

    result_path = Path(args.results)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else completed_ids(result_path)
    tasks = [
        row for line_index, row in enumerate(iter_jsonl(Path(args.tasks)))
        if line_index % args.num_shards == args.shard_index
        and str(row.get("id") or "") not in done
    ]
    windows = []
    for offset in range(0, len(tasks), args.bucket_size):
        window = tasks[offset:offset + args.bucket_size]
        window.sort(key=lambda row: (len(str(row.get("text") or "")), str(row.get("id") or "")))
        windows.append(window)
    random.Random(args.seed + args.shard_index).shuffle(windows)
    tasks = [task for window in windows for task in window]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).to(args.device).eval()
    sample_rate = int(processor.feature_extractor.sampling_rate)
    converter = OpenCC("t2s")
    mode = "w" if args.overwrite else "a"
    written = accepted = rejected = errors = 0

    with result_path.open(mode, encoding="utf-8") as output:
        for batch_start in range(0, len(tasks), args.batch_size):
            batch_tasks = tasks[batch_start:batch_start + args.batch_size]
            waveforms = []
            valid_tasks = []
            for task in batch_tasks:
                task_id = str(task.get("id") or "")
                try:
                    audio_path = Path(str(task.get("out") or ""))
                    if not audio_path.is_file():
                        raise FileNotFoundError(audio_path)
                    waveform = load_audio(audio_path, sample_rate)
                except Exception as exc:
                    write_result(output, {
                        "schema_version": "qwen3_tts_asr_validation_v1",
                        "task_id": task_id,
                        "status": "error",
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    written += 1
                    errors += 1
                    continue
                valid_tasks.append(task)
                waveforms.append(waveform.numpy())

            if valid_tasks:
                inputs = processor(
                    waveforms,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                input_values = inputs.input_values.to(args.device)
                attention_mask = (
                    inputs.attention_mask.to(args.device)
                    if "attention_mask" in inputs else None
                )
                with torch.inference_mode():
                    logits = model(input_values, attention_mask=attention_mask).logits
                transcripts = processor.batch_decode(logits.argmax(dim=-1).cpu())
                for task, transcript in zip(valid_tasks, transcripts):
                    target = normalize_text(str(task.get("text") or ""), converter)
                    hypothesis = normalize_text(str(transcript or ""), converter)
                    if not target:
                        status, reason, cer, coverage = "error", "empty_normalized_target", 1.0, 0.0
                    elif len(target) < args.min_target_chars:
                        status, reason = "ok_short", "short_target_duration_checks_only"
                        cer = levenshtein(target, hypothesis) / len(target)
                        coverage = min(1.0, len(hypothesis) / len(target))
                    else:
                        cer = levenshtein(target, hypothesis) / len(target)
                        coverage = min(1.0, len(hypothesis) / len(target))
                        reasons = []
                        if cer > args.max_cer:
                            reasons.append("high_cer")
                        if coverage < args.min_coverage:
                            reasons.append("low_transcript_coverage")
                        status = "rejected" if reasons else "ok"
                        reason = ",".join(reasons)
                    write_result(output, {
                        "schema_version": "qwen3_tts_asr_validation_v1",
                        "task_id": str(task["id"]),
                        "status": status,
                        "reason": reason,
                        "target": target,
                        "transcript": hypothesis,
                        "cer": round(cer, 6),
                        "coverage": round(coverage, 6),
                        "audio": str(task.get("out") or ""),
                    })
                    written += 1
                    if status in {"ok", "ok_short"}:
                        accepted += 1
                    elif status == "rejected":
                        rejected += 1
                    else:
                        errors += 1
            output.flush()
            if args.progress_every and written % args.progress_every < len(batch_tasks):
                print(json.dumps({
                    "shard": args.shard_index,
                    "written": written,
                    "remaining": len(tasks) - written,
                    "accepted": accepted,
                    "rejected": rejected,
                    "errors": errors,
                }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "tasks_this_run": len(tasks),
        "resumed": len(done),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "written": written,
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors,
        "results": str(result_path.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
