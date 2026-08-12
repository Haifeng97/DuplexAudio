#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
import random
import wave
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_MAX_NEW_TOKENS_CAP = 2048


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def valid_wav(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 44:
        return False
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnchannels() > 0 and wf.getframerate() > 0 and wf.getnframes() > 0
    except (EOFError, OSError, wave.Error):
        return False


def validate_tasks(tasks: List[Dict[str, Any]]) -> None:
    errors: List[str] = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        for key in ("id", "text", "out", "ref_wav", "ref_text"):
            if not str(task.get(key) or "").strip():
                errors.append(f"{task_id or '<missing-id>'}: missing {key}")
                break
        ref_wav = Path(str(task.get("ref_wav") or ""))
        if str(task.get("ref_wav") or "") and not ref_wav.exists():
            errors.append(f"{task_id}: missing ref_wav {ref_wav}")
        if len(errors) >= 20:
            break
    if errors:
        raise ValueError("invalid Qwen3-TTS tasks:\n" + "\n".join(errors))


def progress_line(done: int, total: int, ok: int, errors: int, cached: int) -> str:
    pct = (100.0 * done / total) if total else 100.0
    return f"[overall] {done}/{total} ({pct:.2f}%) ok={ok} cached={cached} errors={errors}"


def torch_dtype(torch_module: Any, name: str) -> Any:
    values = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    return values[name]


def estimated_task_length(task: Dict[str, Any]) -> float:
    text_units = sum(1 for char in str(task.get("text") or "") if not char.isspace())
    try:
        ref_duration = max(0.0, float(task.get("ref_duration") or 0.0))
    except (TypeError, ValueError):
        ref_duration = 0.0
    return text_units + ref_duration * 6.0


def text_units(task: Dict[str, Any]) -> int:
    return sum(1 for char in str(task.get("text") or "") if not char.isspace())


def batch_max_new_tokens(
    tasks: List[Dict[str, Any]],
    *,
    fixed_max_new_tokens: int,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
    generation_guard_sec: float,
    codec_frame_rate: float,
    max_new_tokens_cap: int,
) -> int:
    if fixed_max_new_tokens > 0:
        return fixed_max_new_tokens
    longest_text = max((text_units(task) for task in tasks), default=1)
    max_audio_sec = max(max_audio_floor_sec, longest_text * max_sec_per_char)
    derived = max(1, int(math.ceil((max_audio_sec + generation_guard_sec) * codec_frame_rate)))
    return min(derived, max_new_tokens_cap)


def max_audio_sec_for_task(
    task: Dict[str, Any],
    *,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
) -> float:
    return max(max_audio_floor_sec, text_units(task) * max_sec_per_char)


def audio_quality_error(
    task: Dict[str, Any],
    duration_sec: float,
    *,
    min_audio_sec: float,
    max_audio_floor_sec: float,
    max_sec_per_char: float,
    token_limit: int = 0,
    codec_frame_rate: float = 12.0,
) -> str:
    if duration_sec < min_audio_sec:
        return "audio_too_short"
    if token_limit > 0 and duration_sec >= token_limit / codec_frame_rate - 0.25:
        return "generation_reached_token_limit"
    if duration_sec > max_audio_sec_for_task(
        task,
        max_audio_floor_sec=max_audio_floor_sec,
        max_sec_per_char=max_sec_per_char,
    ):
        return "audio_too_long_for_text"
    return ""


def length_sorted(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        tasks,
        key=lambda task: (
            estimated_task_length(task),
            len(str(task.get("text") or "")),
            str(task.get("id") or ""),
        ),
    )


def batches(tasks: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [tasks[start:start + batch_size] for start in range(0, len(tasks), batch_size)]


def make_task_batches(
    tasks: List[Dict[str, Any]],
    batch_size: int,
    *,
    shuffle_batches: bool,
    shuffle_seed: int,
) -> List[List[Dict[str, Any]]]:
    grouped = batches(length_sorted(tasks), batch_size)
    if shuffle_batches:
        random.Random(shuffle_seed).shuffle(grouped)
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description="Run resumable batched Qwen3-TTS Base voice-clone tasks.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL, help="Hugging Face model id or local model directory")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--expected_sample_rate", type=int, default=24000)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--batch_size", type=int, default=128, help="Native Qwen3-TTS inference batch size.")
    ap.add_argument("--max_new_tokens", type=int, default=0, help="Fixed codec-token limit; 0 derives a limit per batch.")
    ap.add_argument("--min_audio_sec", type=float, default=1.0)
    ap.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    ap.add_argument("--max_sec_per_char", type=float, default=1.2)
    ap.add_argument("--generation_guard_sec", type=float, default=5.0)
    ap.add_argument("--codec_frame_rate", type=float, default=12.0)
    ap.add_argument("--max_new_tokens_cap", type=int, default=DEFAULT_MAX_NEW_TOKENS_CAP)
    ap.add_argument("--shuffle_batches", action="store_true", help="Shuffle batch order after length bucketing.")
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args()
    if args.batch_size <= 0:
        ap.error("--batch_size must be > 0")
    if args.max_new_tokens < 0:
        ap.error("--max_new_tokens must be >= 0")
    if min(
        args.min_audio_sec,
        args.max_audio_floor_sec,
        args.max_sec_per_char,
        args.codec_frame_rate,
    ) <= 0 or args.generation_guard_sec < 0:
        ap.error("audio quality and dynamic max_new_tokens parameters are invalid")
    if args.max_new_tokens_cap <= 0:
        ap.error("--max_new_tokens_cap must be > 0")

    tasks_path = Path(args.tasks)
    tasks = read_jsonl(tasks_path)
    validate_tasks(tasks)
    result_path = Path(args.results) if args.results else tasks_path.with_name("tts_results.jsonl")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    pending: List[Dict[str, Any]] = []
    ok = errors = rejected = cached = 0
    for task in tasks:
        out = Path(str(task["out"]))
        duration_sec = None
        if valid_wav(out):
            with wave.open(str(out), "rb") as wav:
                duration_sec = wav.getnframes() / wav.getframerate()
        cache_error = audio_quality_error(
            task,
            duration_sec,
            min_audio_sec=args.min_audio_sec,
            max_audio_floor_sec=args.max_audio_floor_sec,
            max_sec_per_char=args.max_sec_per_char,
        ) if duration_sec is not None else "invalid_wav"
        if duration_sec is not None and not cache_error and not args.overwrite:
            append_jsonl(
                result_path,
                {
                    "id": task["id"],
                    "status": "cached",
                    "out": str(out),
                    "duration_sec": duration_sec,
                },
            )
            cached += 1
        else:
            if duration_sec is not None and cache_error:
                out.unlink()
            pending.append(task)

    if not pending:
        if args.progress_every and tasks:
            print(progress_line(len(tasks), len(tasks), 0, 0, cached), flush=True)
        print(json.dumps({
            "tasks": len(tasks),
            "pending": 0,
            "batch_size": args.batch_size,
            "batches": 0,
            "ok_or_cached": cached,
            "rejected": 0,
            "errors": 0,
            "results": str(result_path),
            "model": args.model_dir,
            "voice_prompts_cached": 0,
        }, ensure_ascii=False, indent=2))
        return

    import soundfile as sf  # type: ignore
    import torch  # type: ignore
    from qwen_tts import Qwen3TTSModel  # type: ignore

    model = Qwen3TTSModel.from_pretrained(
        args.model_dir,
        device_map="cuda:0",
        dtype=torch_dtype(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    prompt_cache: Dict[tuple[str, str], Any] = {}
    task_batches = make_task_batches(
        pending,
        args.batch_size,
        shuffle_batches=args.shuffle_batches,
        shuffle_seed=args.shuffle_seed,
    )
    processed = 0

    def record(row: Dict[str, Any]) -> None:
        append_jsonl(result_path, row)

    for batch_index, task_batch in enumerate(task_batches, start=1):
        ready_tasks: List[Dict[str, Any]] = []
        prompts: List[Any] = []
        for task in task_batch:
            try:
                ref_wav = str(task["ref_wav"])
                ref_text = str(task["ref_text"]).strip()
                prompt_key = (ref_wav, ref_text)
                prompt = prompt_cache.get(prompt_key)
                if prompt is None:
                    created = model.create_voice_clone_prompt(
                        ref_audio=ref_wav,
                        ref_text=ref_text,
                        x_vector_only_mode=False,
                    )
                    if not isinstance(created, list) or len(created) != 1:
                        raise RuntimeError(
                            f"create_voice_clone_prompt returned {type(created).__name__} "
                            f"with length={len(created) if isinstance(created, list) else 'unknown'}"
                        )
                    prompt = created[0]
                    prompt_cache[prompt_key] = prompt
                ready_tasks.append(task)
                prompts.append(prompt)
            except Exception as exc:
                record({
                    "id": task["id"],
                    "status": "error",
                    "out": str(task["out"]),
                    "error": f"voice_prompt_error: {exc!r}",
                    "batch_index": batch_index,
                })
                errors += 1
                processed += 1

        if ready_tasks:
            token_limit = batch_max_new_tokens(
                ready_tasks,
                fixed_max_new_tokens=args.max_new_tokens,
                max_audio_floor_sec=args.max_audio_floor_sec,
                max_sec_per_char=args.max_sec_per_char,
                generation_guard_sec=args.generation_guard_sec,
                codec_frame_rate=args.codec_frame_rate,
                max_new_tokens_cap=args.max_new_tokens_cap,
            )
            print(
                f"[batch] {batch_index}/{len(task_batches)} size={len(ready_tasks)} "
                f"max_text_units={max(text_units(task) for task in ready_tasks)} "
                f"max_new_tokens={token_limit}",
                flush=True,
            )
            try:
                wavs, sample_rate = model.generate_voice_clone(
                    text=[str(task["text"]) for task in ready_tasks],
                    language=[args.language] * len(ready_tasks),
                    voice_clone_prompt=prompts,
                    max_new_tokens=token_limit,
                )
                if int(sample_rate) != args.expected_sample_rate:
                    raise RuntimeError(
                        f"Qwen3-TTS sample_rate={sample_rate}, expected {args.expected_sample_rate}"
                    )
                if not isinstance(wavs, (list, tuple)) or len(wavs) != len(ready_tasks):
                    raise RuntimeError(
                        f"Qwen3-TTS returned {len(wavs) if isinstance(wavs, (list, tuple)) else 'non-list'} "
                        f"audios for {len(ready_tasks)} tasks"
                    )
            except Exception as exc:
                for task in ready_tasks:
                    record({
                        "id": task["id"],
                        "status": "error",
                        "out": str(task["out"]),
                        "error": f"batch_inference_error: {exc!r}",
                        "batch_index": batch_index,
                        "batch_size": len(ready_tasks),
                        "max_new_tokens": token_limit,
                    })
                    errors += 1
                    processed += 1
            else:
                for task, wav_data in zip(ready_tasks, wavs):
                    out = Path(str(task["out"]))
                    tmp = out.with_name(f"{out.stem}.tmp-{os.getpid()}.wav")
                    duration_sec = len(wav_data) / sample_rate
                    quality_error = audio_quality_error(
                        task,
                        duration_sec,
                        min_audio_sec=args.min_audio_sec,
                        max_audio_floor_sec=args.max_audio_floor_sec,
                        max_sec_per_char=args.max_sec_per_char,
                        token_limit=token_limit,
                        codec_frame_rate=args.codec_frame_rate,
                    )
                    if quality_error:
                        record({
                            "id": task["id"],
                            "status": "rejected",
                            "out": str(out),
                            "error": quality_error,
                            "duration_sec": duration_sec,
                            "max_audio_sec": max_audio_sec_for_task(
                                task,
                                max_audio_floor_sec=args.max_audio_floor_sec,
                                max_sec_per_char=args.max_sec_per_char,
                            ),
                            "batch_index": batch_index,
                            "batch_size": len(ready_tasks),
                            "max_new_tokens": token_limit,
                        })
                        rejected += 1
                        processed += 1
                        continue
                    try:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        sf.write(str(tmp), wav_data, sample_rate, subtype="PCM_16")
                        tmp.replace(out)
                        record({
                            "id": task["id"],
                            "status": "ok",
                            "out": str(out),
                            "sample_rate": int(sample_rate),
                            "engine": "qwen3_tts",
                            "model": args.model_dir,
                            "batch_index": batch_index,
                            "batch_size": len(ready_tasks),
                            "max_new_tokens": token_limit,
                            "duration_sec": duration_sec,
                        })
                        ok += 1
                    except Exception as exc:
                        if tmp.exists():
                            tmp.unlink()
                        record({
                            "id": task["id"],
                            "status": "error",
                            "out": str(out),
                            "error": f"wav_write_error: {exc!r}",
                            "batch_index": batch_index,
                        })
                        errors += 1
                    processed += 1

        done = cached + processed
        if args.progress_every and (
            done == len(tasks)
            or done // args.progress_every != (done - len(task_batch)) // args.progress_every
        ):
            print(progress_line(done, len(tasks), ok, errors, cached), flush=True)

    print(json.dumps({
        "tasks": len(tasks),
        "pending": len(pending),
        "batch_size": args.batch_size,
        "batches": len(task_batches),
        "length_sort": "target_chars_plus_ref_duration",
        "shuffle_batches": args.shuffle_batches,
        "shuffle_seed": args.shuffle_seed,
        "max_new_tokens": args.max_new_tokens or "dynamic",
        "min_audio_sec": args.min_audio_sec,
        "max_audio_floor_sec": args.max_audio_floor_sec,
        "max_sec_per_char": args.max_sec_per_char,
        "generation_guard_sec": args.generation_guard_sec,
        "codec_frame_rate": args.codec_frame_rate,
        "max_new_tokens_cap": args.max_new_tokens_cap,
        "ok_or_cached": ok + cached,
        "rejected": rejected,
        "errors": errors,
        "results": str(result_path),
        "model": args.model_dir,
        "voice_prompts_cached": len(prompt_cache),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
