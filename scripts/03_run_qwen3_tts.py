#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run resumable Qwen3-TTS Base voice-clone tasks.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--model_dir", default=DEFAULT_MODEL, help="Hugging Face model id or local model directory")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--expected_sample_rate", type=int, default=24000)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    tasks = read_jsonl(tasks_path)
    validate_tasks(tasks)
    result_path = Path(args.results) if args.results else tasks_path.with_name("tts_results.jsonl")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    if not args.overwrite and all(valid_wav(Path(task["out"])) for task in tasks):
        for task in tasks:
            append_jsonl(
                result_path,
                {"id": task["id"], "status": "cached", "out": str(task["out"])},
            )
        if args.progress_every and tasks:
            print(progress_line(len(tasks), len(tasks), 0, 0, len(tasks)), flush=True)
        print(json.dumps({
            "tasks": len(tasks),
            "ok_or_cached": len(tasks),
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
    results: List[Dict[str, Any]] = []
    ok = errors = cached = 0

    for idx, task in enumerate(tasks, start=1):
        out = Path(task["out"])
        if valid_wav(out) and not args.overwrite:
            row = {"id": task["id"], "status": "cached", "out": str(out)}
            cached += 1
        else:
            tmp = out.with_name(f"{out.stem}.tmp-{os.getpid()}.wav")
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                ref_wav = str(task["ref_wav"])
                ref_text = str(task["ref_text"]).strip()
                prompt_key = (ref_wav, ref_text)
                prompt = prompt_cache.get(prompt_key)
                if prompt is None:
                    prompt = model.create_voice_clone_prompt(
                        ref_audio=ref_wav,
                        ref_text=ref_text,
                        x_vector_only_mode=False,
                    )
                    prompt_cache[prompt_key] = prompt
                wavs, sample_rate = model.generate_voice_clone(
                    text=str(task["text"]),
                    language=args.language,
                    voice_clone_prompt=prompt,
                )
                if not wavs:
                    raise RuntimeError("Qwen3-TTS returned no audio")
                if int(sample_rate) != args.expected_sample_rate:
                    raise RuntimeError(
                        f"Qwen3-TTS sample_rate={sample_rate}, expected {args.expected_sample_rate}"
                    )
                sf.write(str(tmp), wavs[0], sample_rate, subtype="PCM_16")
                tmp.replace(out)
                row = {
                    "id": task["id"],
                    "status": "ok",
                    "out": str(out),
                    "sample_rate": int(sample_rate),
                    "engine": "qwen3_tts",
                    "model": args.model_dir,
                }
                ok += 1
            except Exception as exc:
                if tmp.exists():
                    tmp.unlink()
                row = {"id": task["id"], "status": "error", "out": str(out), "error": repr(exc)}
                errors += 1
        results.append(row)
        append_jsonl(result_path, row)
        if args.progress_every and (idx % args.progress_every == 0 or idx == len(tasks)):
            print(progress_line(idx, len(tasks), ok, errors, cached), flush=True)

    print(json.dumps({
        "tasks": len(tasks),
        "ok_or_cached": ok + cached,
        "errors": errors,
        "results": str(result_path),
        "model": args.model_dir,
        "voice_prompts_cached": len(prompt_cache),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
