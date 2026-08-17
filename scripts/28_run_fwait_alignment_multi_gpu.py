#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def nonempty_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [line for line in handle if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for line in handle if line.strip())


def update_line_count(path: Path, state: dict[str, int]) -> int:
    if not path.exists():
        state["offset"] = 0
        state["count"] = 0
        return 0
    size = path.stat().st_size
    if size < state["offset"]:
        state["offset"] = 0
        state["count"] = 0
    if size == state["offset"]:
        return state["count"]
    with path.open("rb") as handle:
        handle.seek(state["offset"])
        chunk = handle.read()
        state["offset"] = handle.tell()
    state["count"] += chunk.count(b"\n")
    return state["count"]


def split_tasks(tasks: Path, shard_dir: Path, count: int) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths = [shard_dir / f"tasks_{index:02d}.jsonl" for index in range(count)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    try:
        for index, line in enumerate(nonempty_lines(tasks)):
            handles[index % count].write(line)
    finally:
        for handle in handles:
            handle.close()
    return paths


def merge_results(paths: list[Path], output: Path) -> int:
    tmp = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8") as target:
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="ignore") as source:
                for line in source:
                    if line.strip():
                        target.write(line)
                        count += 1
    tmp.replace(output)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run F_WAIT forced alignment on one worker per GPU.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bucket_size", type=int, default=512)
    parser.add_argument("--monitor_every", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tasks = Path(args.tasks)
    work_dir = Path(args.work_dir)
    shard_dir = work_dir / "shards"
    result_dir = work_dir / "results"
    log_dir = work_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus is empty")
    shards = split_tasks(tasks, shard_dir, len(gpus))
    processes = []
    result_paths = []
    for index, (gpu, shard) in enumerate(zip(gpus, shards)):
        result = result_dir / f"results_{index:02d}.jsonl"
        log = log_dir / f"worker_{index:02d}_gpu{gpu}.log"
        result_paths.append(result)
        command = [
            sys.executable,
            str(Path(__file__).with_name("27_run_fwait_forced_alignment.py")),
            "--tasks", str(shard),
            "--results", str(result),
            "--model", args.model,
            "--device", "cuda:0",
            "--batch_size", str(args.batch_size),
            "--bucket_size", str(args.bucket_size),
            "--progress_every", "0",
            "--seed", str(42 + index),
        ]
        if args.overwrite:
            command.append("--overwrite")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        stream = log.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        processes.append((process, stream, gpu, log))
        print(f"start worker={index} gpu={gpu} log={log}", flush=True)

    total = count_lines(tasks)
    last_done = -1
    progress_states = {path: {"offset": 0, "count": 0} for path in result_paths}
    try:
        while any(process.poll() is None for process, _, _, _ in processes):
            done = sum(update_line_count(path, progress_states[path]) for path in result_paths)
            if done != last_done:
                pct = 100.0 * done / total if total else 100.0
                print(f"TOTAL {done}/{total} ({pct:.2f}%)", flush=True)
                last_done = done
            time.sleep(max(0.2, args.monitor_every))
    finally:
        for process, stream, _, _ in processes:
            process.wait()
            stream.close()
    failed = [{"gpu": gpu, "returncode": process.returncode, "log": str(log)} for process, _, gpu, log in processes if process.returncode]
    if failed:
        raise SystemExit(json.dumps({"worker_failures": failed}, ensure_ascii=False, indent=2))
    merged_path = work_dir / "alignment_results.jsonl"
    merged = merge_results(result_paths, merged_path)
    print(json.dumps({"total": total, "merged": merged, "results": str(merged_path.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
