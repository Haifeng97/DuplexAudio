#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List


V3_EOP = "<|endofprompt|>"
V3_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_wav_pcm16(path: Path, samples: Iterable[int], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(struct.pack("<h", max(-32768, min(32767, int(x)))) for x in samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data)


def looks_like_cosyvoice3(model_dir: str) -> bool:
    name = Path(model_dir).name.lower()
    return "cosyvoice3" in name or "cv3" in name or "fun-cosyvoice3" in name


def prep_ref_text(ref_text: str, model_dir: str) -> str:
    if not looks_like_cosyvoice3(model_dir):
        return ref_text
    if V3_EOP in (ref_text or ""):
        return ref_text
    return f"{V3_DEFAULT_SYSTEM_PROMPT}{V3_EOP}{ref_text or ''}"


def mock_tts(text: str, out: Path, sample_rate: int) -> None:
    duration = max(0.45, min(8.0, 0.16 * len(text) + 0.25))
    n = int(round(duration * sample_rate))
    freq = 180 + (abs(hash(text)) % 120)
    amp = 2600
    samples = []
    for i in range(n):
        env = min(1.0, i / max(1, int(0.05 * sample_rate)), (n - i) / max(1, int(0.05 * sample_rate)))
        val = amp * env * math.sin(2.0 * math.pi * freq * i / sample_rate)
        samples.append(int(val))
    write_wav_pcm16(out, samples, sample_rate)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def progress_line(done: int, total: int, ok: int, errors: int, cached: int) -> str:
    pct = (100.0 * done / total) if total else 100.0
    return f"[overall] {done}/{total} ({pct:.2f}%) ok={ok} cached={cached} errors={errors}"


def run_cosyvoice(tasks: List[Dict[str, Any]], args: argparse.Namespace, result_path: Path) -> List[Dict[str, Any]]:
    sys.path.insert(0, args.cosyvoice_repo)
    sys.path.insert(0, str(Path(args.cosyvoice_repo) / "third_party" / "Matcha-TTS"))
    from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore
    import torchaudio  # type: ignore

    model = AutoModel(model_dir=args.model_dir)
    results = []
    ok = errors = cached = 0
    for i, task in enumerate(tasks, start=1):
        out = Path(task["out"])
        if out.exists() and out.stat().st_size > 1000 and not args.overwrite:
            row = {"id": task["id"], "status": "cached", "out": str(out)}
            cached += 1
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                last = None
                ref_text = prep_ref_text(str(task.get("ref_text", "")), args.model_dir)
                for last in model.inference_zero_shot(
                    task["text"],
                    ref_text,
                    task["ref_wav"],
                    stream=False,
                ):
                    pass
                if last is None:
                    raise RuntimeError("CosyVoice returned no audio")
                torchaudio.save(str(out), last["tts_speech"].cpu(), model.sample_rate)
                row = {"id": task["id"], "status": "ok", "out": str(out), "sample_rate": model.sample_rate}
                ok += 1
            except Exception as exc:
                row = {"id": task["id"], "status": "error", "out": str(out), "error": repr(exc)}
                errors += 1
        results.append(row)
        append_jsonl(result_path, row)
        if args.progress_every and (i % args.progress_every == 0 or i == len(tasks)):
            print(progress_line(i, len(tasks), ok, errors, cached), flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Run TTS tasks for duplex query audio.")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--results", default="")
    ap.add_argument("--mock_tts", action="store_true", help="Generate synthetic speech-like wavs for pipeline testing")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cosyvoice_repo", default="/home/haifeng/Projects/CosyVoice")
    ap.add_argument("--model_dir", default="/home/haifeng/Projects/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
    ap.add_argument("--progress_every", type=int, default=50)
    args = ap.parse_args()

    tasks_path = Path(args.tasks)
    tasks = read_jsonl(tasks_path)
    result_path = Path(args.results) if args.results else tasks_path.with_name("tts_results.jsonl")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        result_path.unlink()

    if args.mock_tts:
        results = []
        ok = errors = cached = 0
        for i, task in enumerate(tasks, start=1):
            out = Path(task["out"])
            if out.exists() and out.stat().st_size > 1000 and not args.overwrite:
                row = {"id": task["id"], "status": "cached", "out": str(out)}
                cached += 1
            else:
                try:
                    mock_tts(str(task["text"]), out, args.sample_rate)
                    row = {"id": task["id"], "status": "mock_ok", "out": str(out), "sample_rate": args.sample_rate}
                    ok += 1
                except Exception as exc:
                    row = {"id": task["id"], "status": "error", "out": str(out), "error": repr(exc)}
                    errors += 1
            results.append(row)
            append_jsonl(result_path, row)
            if args.progress_every and (i % args.progress_every == 0 or i == len(tasks)):
                print(progress_line(i, len(tasks), ok, errors, cached), flush=True)
    else:
        results = run_cosyvoice(tasks, args, result_path)

    ok = sum(1 for row in results if row.get("status") in {"ok", "mock_ok", "cached"})
    print(json.dumps({
        "tasks": len(tasks),
        "ok_or_cached": ok,
        "errors": len(results) - ok,
        "results": str(result_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
