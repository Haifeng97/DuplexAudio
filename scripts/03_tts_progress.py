#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List


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


def fmt_seconds(sec: float) -> str:
    if sec < 0 or sec == float("inf"):
        return "unknown"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def snapshot(work_dir: Path) -> Dict[str, object]:
    shard_dir = work_dir / "auto_shards"
    result_dir = work_dir / "auto_results"
    shard_paths = sorted(shard_dir.glob("tts_tasks_*.jsonl"))
    rows: List[Dict[str, object]] = []
    total = done = 0
    all_status: Counter = Counter()
    for shard_path in shard_paths:
        suffix = shard_path.stem.removeprefix("tts_tasks_")
        result_path = result_dir / f"tts_results_{suffix}.jsonl"
        shard_total = count_lines(shard_path)
        shard_done = count_lines(result_path)
        counts = status_counts(result_path)
        total += shard_total
        done += shard_done
        all_status.update(counts)
        rows.append({
            "worker": suffix,
            "done": shard_done,
            "total": shard_total,
            "pct": round(100.0 * shard_done / shard_total, 2) if shard_total else 100.0,
            "status": dict(counts),
        })
    return {
        "work_dir": str(work_dir),
        "workers": len(shard_paths),
        "done": done,
        "total": total,
        "pct": round(100.0 * done / total, 2) if total else 100.0,
        "status": dict(all_status),
        "workers_detail": rows,
    }


def print_snapshot(data: Dict[str, object], *, previous_done: int | None, previous_time: float | None) -> None:
    now = time.time()
    done = int(data["done"])
    total = int(data["total"])
    pct = float(data["pct"])
    rate = None
    eta = None
    if previous_done is not None and previous_time is not None and now > previous_time:
        delta_done = done - previous_done
        delta_time = now - previous_time
        if delta_done > 0:
            rate = delta_done / delta_time
            eta = (total - done) / rate if rate > 0 else None
    rate_text = f" rate={rate:.3f}/s" if rate is not None else ""
    eta_text = f" eta={fmt_seconds(eta)}" if eta is not None else ""
    print(f"TOTAL {done}/{total} ({pct:.2f}%) status={data['status']}{rate_text}{eta_text}")
    for row in data["workers_detail"]:
        print(f"  worker {row['worker']}: {row['done']}/{row['total']} ({row['pct']}%) {row['status']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Show progress for scripts/03_run_tts_multi_gpu.py outputs.")
    ap.add_argument("--work_dir", default="outputs/final_all")
    ap.add_argument("--watch", type=float, default=0.0, help="Refresh interval seconds; 0 prints once.")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    previous_done = None
    previous_time = None
    while True:
        data = snapshot(work_dir)
        print_snapshot(data, previous_done=previous_done, previous_time=previous_time)
        previous_done = int(data["done"])
        previous_time = time.time()
        if args.watch <= 0:
            break
        print("-" * 80, flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
