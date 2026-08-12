#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter

from tqdm import tqdm
from pathlib import Path
from typing import Dict, List


def read_nonempty_lines(path: Path) -> List[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def status_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                counts["bad_json"] += 1
                continue
            counts[str(row.get("status") or "missing_status")] += 1
    return counts


def fmt_seconds(sec: float | None) -> str:
    if sec is None or sec < 0:
        return "unknown"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def progress_snapshot(shard_paths: List[Path], result_dir: Path) -> Dict[str, object]:
    total = done = 0
    counts: Counter = Counter()
    for shard_path in shard_paths:
        suffix = shard_path.stem.removeprefix("tts_tasks_")
        result_path = result_dir / f"tts_results_{suffix}.jsonl"
        total += count_lines(shard_path)
        done += count_lines(result_path)
        counts.update(status_counts(result_path))
    return {
        "done": done,
        "total": total,
        "pct": (100.0 * done / total) if total else 100.0,
        "status": dict(counts),
    }


def read_new_statuses(path: Path, state: Dict[str, object]) -> Counter:
    counts: Counter = Counter()
    offset = int(state.get("offset", 0))
    partial = str(state.get("partial", ""))
    if not path.exists():
        state["offset"] = 0
        state["partial"] = ""
        return counts
    size = path.stat().st_size
    if size < offset:
        offset = 0
        partial = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        chunk = f.read()
        state["offset"] = f.tell()
    if not chunk:
        state["partial"] = partial
        return counts
    text = partial + chunk
    if text.endswith("\n"):
        lines = text.splitlines()
        state["partial"] = ""
    else:
        lines = text.splitlines()
        state["partial"] = lines.pop() if lines else text
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            counts["bad_json"] += 1
            continue
        counts[str(row.get("status") or "missing_status")] += 1
    return counts


def monitor_progress(
    shard_paths: List[Path],
    result_dir: Path,
    interval: float,
    stop_event: threading.Event,
) -> None:
    total = sum(count_lines(path) for path in shard_paths)
    result_paths = []
    for shard_path in shard_paths:
        suffix = shard_path.stem.removeprefix("tts_tasks_")
        result_paths.append(result_dir / f"tts_results_{suffix}.jsonl")

    states: Dict[Path, Dict[str, object]] = {path: {"offset": 0, "partial": ""} for path in result_paths}
    done = 0
    counts: Counter = Counter()
    bar = tqdm(total=total, dynamic_ncols=True, unit="task", desc="TOTAL", mininterval=0.2)

    try:
        while not stop_event.wait(interval):
            delta = 0
            for path in result_paths:
                new_counts = read_new_statuses(path, states[path])
                if new_counts:
                    new_done = sum(new_counts.values())
                    delta += new_done
                    counts.update(new_counts)
            if delta:
                done += delta
                bar.update(delta)
                bar.set_postfix(dict(counts), refresh=True)
    finally:
        bar.close()


def split_tasks(lines: List[str], out_dir: Path, prefix: str, shards: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx in range(shards):
        part = lines[idx::shards]
        path = out_dir / f"{prefix}_{idx:02d}.jsonl"
        path.write_text("\n".join(part) + ("\n" if part else ""), encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Split TTS tasks and run one or more TTS workers per GPU.")
    ap.add_argument("--tasks", required=True, help="Combined tts_tasks.jsonl")
    ap.add_argument("--work_dir", default="", help="Where to write auto shards/results/logs. Default: tasks file directory.")
    ap.add_argument("--gpus", default="4,5,6,7", help="Comma-separated physical GPU ids.")
    ap.add_argument("--procs_per_gpu", type=int, default=1)
    ap.add_argument("--engine", choices=["cosyvoice", "qwen3_tts"], default="cosyvoice")
    ap.add_argument("--cosyvoice_repo", default="/data/haifengjia/models/CosyVoice")
    ap.add_argument("--model_dir", default="", help="Local model directory or Hugging Face model id")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--attn_implementation", choices=["flash_attention_2", "sdpa", "eager"], default="flash_attention_2")
    ap.add_argument("--batch_size", type=int, default=128, help="Native Qwen3-TTS batch size per worker.")
    ap.add_argument("--max_new_tokens", type=int, default=0, help="Fixed Qwen codec-token limit; 0 derives per batch.")
    ap.add_argument("--min_audio_sec", type=float, default=1.0)
    ap.add_argument("--max_audio_floor_sec", type=float, default=10.0)
    ap.add_argument("--max_sec_per_char", type=float, default=1.2)
    ap.add_argument("--generation_guard_sec", type=float, default=5.0)
    ap.add_argument("--codec_frame_rate", type=float, default=12.0)
    ap.add_argument("--max_new_tokens_cap", type=int, default=2048)
    ap.add_argument("--shuffle_batches", action="store_true", help="Shuffle length-bucketed batch order.")
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--project", default=os.environ.get("PROJECT", ""))
    ap.add_argument("--python", default=sys.executable, help="Python executable to run workers; default is current Python.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--monitor_every", type=float, default=0.5, help="Refresh aggregate progress every N seconds; 0 disables.")
    args = ap.parse_args()

    if not args.model_dir:
        if args.engine == "qwen3_tts":
            args.model_dir = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        else:
            args.model_dir = "/data/haifengjia/models/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
    tasks_path = Path(args.tasks)
    work_dir = Path(args.work_dir) if args.work_dir else tasks_path.parent
    shard_dir = work_dir / "auto_shards"
    log_dir = work_dir / "logs"
    result_dir = work_dir / "auto_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    if args.procs_per_gpu <= 0:
        raise SystemExit("--procs_per_gpu must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be > 0")
    if args.max_new_tokens < 0:
        raise SystemExit("--max_new_tokens must be >= 0")
    if min(
        args.min_audio_sec,
        args.max_audio_floor_sec,
        args.max_sec_per_char,
        args.codec_frame_rate,
    ) <= 0 or args.generation_guard_sec < 0:
        raise SystemExit("audio quality and dynamic max_new_tokens parameters are invalid")
    if args.max_new_tokens_cap <= 0:
        raise SystemExit("--max_new_tokens_cap must be > 0")
    worker_count = len(gpus) * args.procs_per_gpu

    lines = read_nonempty_lines(tasks_path)
    shard_paths = split_tasks(lines, shard_dir, "tts_tasks", worker_count)

    commands = []
    for worker_idx, shard_path in enumerate(shard_paths):
        gpu = gpus[worker_idx % len(gpus)]
        result_path = result_dir / f"tts_results_{worker_idx:02d}.jsonl"
        log_path = log_dir / f"tts_worker_{worker_idx:02d}_gpu{gpu}.log"
        worker_script = "scripts/03_run_qwen3_tts.py" if args.engine == "qwen3_tts" else "scripts/03_run_tts.py"
        cmd = [
            args.python,
            worker_script,
            "--tasks", str(shard_path),
            "--results", str(result_path),
            "--model_dir", args.model_dir,
            "--progress_every", str(args.progress_every),
        ]
        if args.engine == "cosyvoice":
            cmd.extend(["--cosyvoice_repo", args.cosyvoice_repo])
        else:
            cmd.extend([
                "--language", args.language,
                "--dtype", args.dtype,
                "--attn_implementation", args.attn_implementation,
                "--batch_size", str(args.batch_size),
                "--max_new_tokens", str(args.max_new_tokens),
                "--min_audio_sec", str(args.min_audio_sec),
                "--max_audio_floor_sec", str(args.max_audio_floor_sec),
                "--max_sec_per_char", str(args.max_sec_per_char),
                "--generation_guard_sec", str(args.generation_guard_sec),
                "--codec_frame_rate", str(args.codec_frame_rate),
                "--max_new_tokens_cap", str(args.max_new_tokens_cap),
                "--shuffle_seed", str(args.shuffle_seed + worker_idx),
            ])
            if args.shuffle_batches:
                cmd.append("--shuffle_batches")
        if args.overwrite:
            cmd.append("--overwrite")
        commands.append((worker_idx, gpu, cmd, log_path, shard_path, result_path))

    summary = {
        "tasks": str(tasks_path),
        "engine": args.engine,
        "model_dir": args.model_dir,
        "total_tasks": len(lines),
        "work_dir": str(work_dir),
        "gpus": gpus,
        "procs_per_gpu": args.procs_per_gpu,
        "batch_size": args.batch_size if args.engine == "qwen3_tts" else None,
        "max_new_tokens": args.max_new_tokens if args.engine == "qwen3_tts" else None,
        "min_audio_sec": args.min_audio_sec if args.engine == "qwen3_tts" else None,
        "max_audio_floor_sec": args.max_audio_floor_sec if args.engine == "qwen3_tts" else None,
        "max_sec_per_char": args.max_sec_per_char if args.engine == "qwen3_tts" else None,
        "generation_guard_sec": args.generation_guard_sec if args.engine == "qwen3_tts" else None,
        "codec_frame_rate": args.codec_frame_rate if args.engine == "qwen3_tts" else None,
        "max_new_tokens_cap": args.max_new_tokens_cap if args.engine == "qwen3_tts" else None,
        "shuffle_batches": args.shuffle_batches if args.engine == "qwen3_tts" else None,
        "shuffle_seed": args.shuffle_seed if args.engine == "qwen3_tts" else None,
        "worker_shuffle_seeds": (
            [args.shuffle_seed + idx for idx in range(worker_count)]
            if args.engine == "qwen3_tts" and args.shuffle_batches
            else None
        ),
        "workers": worker_count,
        "project": args.project,
        "shards": [str(p) for p in shard_paths],
    }
    (work_dir / "multi_gpu_launch.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        for worker_idx, gpu, cmd, log_path, shard_path, result_path in commands:
            print(f"worker={worker_idx} gpu={gpu} tasks={shard_path} results={result_path} log={log_path}")
            print(" ".join(cmd))
        return

    for _, _, _, _, _, result_path in commands:
        if result_path.exists():
            result_path.unlink()

    procs = []
    for worker_idx, gpu, cmd, log_path, shard_path, result_path in commands:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        if args.project:
            env["PROJECT"] = args.project
        log_file = log_path.open("w", encoding="utf-8")
        print(f"start worker={worker_idx} gpu={gpu} tasks={shard_path} results={result_path} log={log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=Path.cwd(), env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        procs.append((worker_idx, proc, log_file))

    stop_event = threading.Event()
    monitor_thread = None
    if args.monitor_every > 0:
        monitor_thread = threading.Thread(
            target=monitor_progress,
            args=(shard_paths, result_dir, args.monitor_every, stop_event),
            daemon=True,
        )
        monitor_thread.start()

    failed = 0
    try:
        for worker_idx, proc, log_file in procs:
            code = proc.wait()
            log_file.close()
            print(f"done worker={worker_idx} exit={code}", flush=True)
            if code != 0:
                failed += 1
    except KeyboardInterrupt:
        print("received KeyboardInterrupt; terminating workers", flush=True)
        for _, proc, log_file in procs:
            proc.terminate()
            log_file.close()
        raise
    finally:
        stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)
        snap = progress_snapshot(shard_paths, result_dir)
        print(f"TOTAL {snap['done']}/{snap['total']} ({float(snap['pct']):.2f}%) status={snap['status']}", flush=True)
    if failed:
        raise SystemExit(failed)


if __name__ == "__main__":
    main()
