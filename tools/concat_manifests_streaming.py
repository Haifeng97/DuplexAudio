#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def iter_jsonl_lines(path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically concatenate large manifests with streaming ID checks.")
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stats", default="")
    parser.add_argument("--progress_every", type=int, default=100000)
    parser.add_argument("--require_absolute_audio", action="store_true")
    args = parser.parse_args()

    inputs = [Path(value).resolve() for value in args.manifest]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output = Path(args.out).resolve()
    stats_path = Path(args.stats).resolve() if args.stats else output.with_name("concat_stats.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    db_path = output.with_name(f".{output.name}.ids-{os.getpid()}.sqlite3")

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY, source TEXT NOT NULL)")
    input_counts: Dict[str, int] = {}
    scenarios = Counter()
    total = 0
    id_digest = hashlib.sha256()
    try:
        with output_tmp.open("w", encoding="utf-8") as output_handle:
            for source in inputs:
                current = 0
                for original_line, row in iter_jsonl_lines(source):
                    sample_id = str(row.get("id") or "")
                    if not sample_id:
                        raise ValueError(f"missing id in {source}")
                    try:
                        connection.execute("INSERT INTO ids VALUES (?,?)", (sample_id, str(source)))
                    except sqlite3.IntegrityError as exc:
                        previous = connection.execute(
                            "SELECT source FROM ids WHERE id=?", (sample_id,)
                        ).fetchone()[0]
                        raise ValueError(f"duplicate id {sample_id}: {previous} and {source}") from exc
                    audio = Path(str(row.get("audio") or ""))
                    if args.require_absolute_audio and not audio.is_absolute():
                        raise ValueError(f"non-absolute audio path for {sample_id}: {audio}")
                    output_handle.write(original_line if original_line.endswith("\n") else original_line + "\n")
                    id_digest.update(sample_id.encode("utf-8"))
                    id_digest.update(b"\n")
                    scenarios[str(row.get("scenario") or "")] += 1
                    current += 1
                    total += 1
                    if total % 10000 == 0:
                        connection.commit()
                    if args.progress_every > 0 and total % args.progress_every == 0:
                        print(f"CONCAT {total} source={source}", flush=True)
                input_counts[str(source)] = current
        connection.commit()
        output_tmp.replace(output)
    except BaseException:
        output_tmp.unlink(missing_ok=True)
        raise
    finally:
        connection.close()
        db_path.unlink(missing_ok=True)

    stats = {
        "manifest": str(output),
        "rows": total,
        "inputs": input_counts,
        "scenario_counts": dict(scenarios),
        "duplicate_ids": 0,
        "id_order_sha256": id_digest.hexdigest(),
        "audio_paths": "absolute" if args.require_absolute_audio else "unchecked",
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
