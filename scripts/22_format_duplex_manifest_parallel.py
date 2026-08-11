#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterator, List


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield line


def count_lines(path: Path) -> int:
    return sum(1 for _ in iter_lines(path))


def split_index(index: Path, shard_dir: Path, workers: int, total: int) -> List[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_count = min(workers, total) if total else 0
    if not shard_count:
        return []
    per_shard = int(math.ceil(total / shard_count))
    paths = [shard_dir / f"scenario_index_{idx:03d}.jsonl" for idx in range(shard_count)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    try:
        for line_no, line in enumerate(iter_lines(index)):
            shard_idx = min(shard_count - 1, line_no // per_shard)
            handles[shard_idx].write(line)
    finally:
        for handle in handles:
            handle.close()
    return [path for path in paths if path.stat().st_size]


def read_stats(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return row if isinstance(row, dict) else None


def part_complete(index: Path, manifest: Path) -> bool:
    stats = read_stats(manifest.with_suffix(manifest.suffix + ".stats.json"))
    if not stats:
        return False
    return int(stats.get("input_rows") or -1) == count_lines(index)


def formatter_command(index: Path, manifest: Path, wav_dir: Path, args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("04_format_duplex_manifest.py")),
        "--index", str(index),
        "--out", str(manifest),
        "--wav_dir", str(wav_dir),
        "--sample_rate", str(args.sample_rate),
        "--chunk_ms", str(args.chunk_ms),
        "--tokenizer_json", str(args.tokenizer_json),
        "--vad_mode", args.vad_mode,
        "--backchannel_vad_mode", args.backchannel_vad_mode,
        "--min_query_audio_sec", str(args.min_query_audio_sec),
        "--min_backchannel_audio_sec", str(args.min_backchannel_audio_sec),
        "--quiet",
    ]


def run_part(index: Path, part_dir: Path, wav_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    part_dir.mkdir(parents=True, exist_ok=True)
    manifest = part_dir / "manifest.jsonl"
    if not args.overwrite_parts and part_complete(index, manifest):
        return {"index": str(index), "manifest": str(manifest), "cached": True}
    log = part_dir / "format.log"
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            formatter_command(index, manifest, wav_dir, args),
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"formatter failed for {index}; see {log}")
    return {"index": str(index), "manifest": str(manifest), "cached": False}


def append_file(source: Path, output: Any) -> int:
    count = 0
    if not source.exists():
        return count
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                output.write(line)
                count += 1
    return count


def merge_parts(part_dirs: List[Path], out_dir: Path) -> Dict[str, Any]:
    manifest_tmp = out_dir / "manifest.jsonl.tmp"
    skipped_tmp = out_dir / "manifest.jsonl.skipped.jsonl.tmp"
    written = skipped = 0
    with manifest_tmp.open("w", encoding="utf-8") as manifest_out, skipped_tmp.open("w", encoding="utf-8") as skipped_out:
        for part_dir in part_dirs:
            written += append_file(part_dir / "manifest.jsonl", manifest_out)
            skipped += append_file(part_dir / "manifest.jsonl.skipped.jsonl", skipped_out)
    manifest = out_dir / "manifest.jsonl"
    skipped_path = out_dir / "manifest.jsonl.skipped.jsonl"
    manifest_tmp.replace(manifest)
    skipped_tmp.replace(skipped_path)
    return {
        "manifest": str(manifest.resolve()),
        "written": written,
        "skipped": skipped,
        "skipped_path": str(skipped_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Format a scenario index with parallel quiet 04 workers.")
    parser.add_argument("--index", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--chunk_ms", type=int, default=180)
    parser.add_argument("--tokenizer_json", default="tokenizers/qwen3_8b/tokenizer.json")
    parser.add_argument("--vad_mode", choices=["silero", "auto", "energy", "off"], default="silero")
    parser.add_argument("--backchannel_vad_mode", choices=["energy", "silero"], default="energy")
    parser.add_argument("--min_query_audio_sec", type=float, default=1.0)
    parser.add_argument("--min_backchannel_audio_sec", type=float, default=0.08)
    parser.add_argument("--overwrite_parts", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0")

    index = Path(args.index).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    total = count_lines(index)
    shard_paths = split_index(index, out_dir / "_work" / "shards", args.workers, total)
    part_dirs = [out_dir / "_work" / "parts" / f"part_{idx:03d}" for idx in range(len(shard_paths))]

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_part, shard, part_dir, wav_dir, args): idx
            for idx, (shard, part_dir) in enumerate(zip(shard_paths, part_dirs))
        }
        for future in as_completed(futures):
            results.append(future.result())

    merged = merge_parts(part_dirs, out_dir)
    stats = {
        "index": str(index),
        "out_dir": str(out_dir.resolve()),
        "input_rows": total,
        "workers": args.workers,
        "parts": len(part_dirs),
        "cached_parts": sum(bool(row["cached"]) for row in results),
        **merged,
    }
    (out_dir / "parallel_format_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
