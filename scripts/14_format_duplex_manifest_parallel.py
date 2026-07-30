#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from tqdm import tqdm


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line for line in f if line.strip()]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_lines(path: Path, lines: Iterable[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")
            n += 1
    return n


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_jsonl(out: Path, parts: List[Path]) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as dst:
        for part in parts:
            if not part.exists():
                continue
            with part.open("r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    if line.strip():
                        dst.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def merge_skipped(out: Path, parts: List[Path]) -> int:
    skipped_parts = [p.with_suffix(p.suffix + ".skipped.jsonl") for p in parts]
    skipped_parts = [p for p in skipped_parts if p.exists()]
    if not skipped_parts:
        return 0
    return merge_jsonl(out.with_suffix(out.suffix + ".skipped.jsonl"), skipped_parts)


def load_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel wrapper for 04_format_duplex_manifest.py.")
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample_rate", type=int, default=24000)
    ap.add_argument("--chunk_ms", type=int, default=180)
    ap.add_argument("--noise_rms", type=float, default=0.003)
    ap.add_argument("--initial_idle_chunks", type=int, default=0)
    ap.add_argument("--initial_idle_sec_min", type=float, default=0.5)
    ap.add_argument("--initial_idle_sec_max", type=float, default=1.5)
    ap.add_argument("--final_idle_chunks", type=int, default=2)
    ap.add_argument("--inter_turn_idle_sec_min", type=float, default=1.0)
    ap.add_argument("--inter_turn_idle_sec_max", type=float, default=3.0)
    ap.add_argument("--disable_inter_turn_idle", action="store_true")
    ap.add_argument("--tokenizer_json", default="tokenizers/qwen3_8b/tokenizer.json")
    ap.add_argument("--vad_mode", choices=["silero", "auto", "energy", "off"], default="silero")
    ap.add_argument("--min_query_audio_sec", type=float, default=1.0)
    ap.add_argument("--monitor_every", type=float, default=2.0)
    ap.add_argument("--script", default="scripts/04_format_duplex_manifest.py")
    ap.add_argument("--quiet", action="store_true", help="Do not print live tqdm progress.")
    args = ap.parse_args()

    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    index = Path(args.index)
    out = Path(args.out)
    wav_dir = Path(args.wav_dir)
    work_dir = out.parent / "_format_shards"
    index_dir = work_dir / "index"
    manifest_dir = work_dir / "manifest"
    log_dir = work_dir / "logs"
    for d in (index_dir, manifest_dir, log_dir, wav_dir):
        d.mkdir(parents=True, exist_ok=True)

    lines = read_lines(index)
    shard_paths: List[Path] = []
    for worker in range(args.workers):
        shard = index_dir / f"index_{worker:02d}.jsonl"
        write_lines(shard, lines[worker::args.workers])
        shard_paths.append(shard)

    manifest_parts: List[Path] = []
    procs = []
    for worker, shard in enumerate(shard_paths):
        part = manifest_dir / f"manifest_{worker:02d}.jsonl"
        manifest_parts.append(part)
        log_path = log_dir / f"format_{worker:02d}.log"
        cmd = [
            sys.executable,
            args.script,
            "--index", str(shard),
            "--out", str(part),
            "--wav_dir", str(wav_dir / f"shard_{worker:02d}"),
            "--sample_rate", str(args.sample_rate),
            "--chunk_ms", str(args.chunk_ms),
            "--noise_rms", str(args.noise_rms),
            "--initial_idle_chunks", str(args.initial_idle_chunks),
            "--initial_idle_sec_min", str(args.initial_idle_sec_min),
            "--initial_idle_sec_max", str(args.initial_idle_sec_max),
            "--final_idle_chunks", str(args.final_idle_chunks),
            "--inter_turn_idle_sec_min", str(args.inter_turn_idle_sec_min),
            "--inter_turn_idle_sec_max", str(args.inter_turn_idle_sec_max),
            "--tokenizer_json", args.tokenizer_json,
            "--vad_mode", args.vad_mode,
            "--min_query_audio_sec", str(args.min_query_audio_sec),
        ]
        if args.disable_inter_turn_idle:
            cmd.append("--disable_inter_turn_idle")
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        procs.append((worker, proc, log_file, log_path))

    summary = {
        "index": str(index),
        "out": str(out),
        "wav_dir": str(wav_dir),
        "workers": args.workers,
        "input_rows": len(lines),
        "work_dir": str(work_dir),
    }
    write_json(out.parent / "parallel_format_launch.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    bar = None if args.quiet else tqdm(total=len(lines), dynamic_ncols=True, unit="row", desc=f"format {out.parent.name}")
    last_done = 0
    failed = 0
    try:
        while True:
            done = sum(count_lines(p) + count_lines(p.with_suffix(p.suffix + ".skipped.jsonl")) for p in manifest_parts)
            if done > last_done:
                if bar is not None:
                    bar.update(done - last_done)
                last_done = done
            alive = [proc for _, proc, _, _ in procs if proc.poll() is None]
            if not alive:
                break
            time.sleep(args.monitor_every)
    finally:
        if bar is not None:
            bar.close()

    for worker, proc, log_file, log_path in procs:
        code = proc.wait()
        log_file.close()
        if code != 0:
            failed += 1
            print(f"worker {worker} failed exit={code} log={log_path}", flush=True)
    if failed:
        raise SystemExit(failed)

    written = merge_jsonl(out, manifest_parts)
    skipped = merge_skipped(out, manifest_parts)
    scenario_counts: Counter[str] = Counter()
    for part in manifest_parts:
        stats = load_stats(part.with_suffix(part.suffix + ".stats.json"))
        for key, value in (stats.get("scenario_counts") or {}).items() if isinstance(stats.get("scenario_counts"), dict) else []:
            scenario_counts[str(key)] += int(value)
    stats = {
        **summary,
        "n": written,
        "skipped": skipped,
        "parts": [str(p) for p in manifest_parts],
        "scenario_counts": dict(scenario_counts),
    }
    write_json(out.with_suffix(out.suffix + ".stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
