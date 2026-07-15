#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", text)
    return text.strip("._-")[:180] or "sample"


def add_task(
    tasks: List[Dict[str, Any]],
    asset: Dict[str, Any],
    *,
    sample_id: str,
    key: str,
    text: str,
    wav_dir: Path,
    ref_wav: str,
    ref_text: str,
) -> None:
    task_id = f"{safe_name(sample_id)}__{key}"
    out = wav_dir / f"{task_id}.wav"
    task = {
        "id": task_id,
        "sample_id": sample_id,
        "key": key,
        "text": text,
        "out": str(out),
        "ref_wav": ref_wav,
        "ref_text": ref_text,
    }
    tasks.append(task)
    asset[key] = {
        "task_id": task_id,
        "text": text,
        "audio": str(out),
    }


def attach_assets(row: Dict[str, Any], wav_dir: Path, ref_wav: str, ref_text: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = dict(row)
    sample_id = str(out.get("id"))
    scenario = out.get("scenario")
    assets: Dict[str, Any] = {}

    if scenario == "normal_qa":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query",
            text=str(out.get("question_text", "")),
            wav_dir=wav_dir,
            ref_wav=ref_wav,
            ref_text=ref_text,
        )
    elif scenario == "player_interrupts_ai":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="base_query",
            text=str(out.get("base", {}).get("question_text", "")),
            wav_dir=wav_dir,
            ref_wav=ref_wav,
            ref_text=ref_text,
        )
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="donor_query",
            text=str(out.get("donor", {}).get("question_text", "")),
            wav_dir=wav_dir,
            ref_wav=ref_wav,
            ref_text=ref_text,
        )
    elif scenario == "incomplete_query_candidate":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query_part1",
            text=str(out.get("query_part1_text", "")),
            wav_dir=wav_dir,
            ref_wav=ref_wav,
            ref_text=ref_text,
        )
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query_part2",
            text=str(out.get("query_part2_text", "")),
            wav_dir=wav_dir,
            ref_wav=ref_wav,
            ref_text=ref_text,
        )
    else:
        turns = out.get("turns")
        if isinstance(turns, list):
            for idx, turn in enumerate(turns, start=1):
                if isinstance(turn, dict) and turn.get("needs_tts", True):
                    add_task(
                        tasks,
                        assets,
                        sample_id=sample_id,
                        key=f"turn{idx:03d}_query",
                        text=str(turn.get("question_text", "")),
                        wav_dir=wav_dir,
                        ref_wav=ref_wav,
                        ref_text=ref_text,
                    )
    out["tts_assets"] = assets
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Create turn-level TTS tasks for duplex scenario candidates.")
    ap.add_argument("--input", required=True, help="Scenario candidate JSONL or selected turns JSONL")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_wav", default="/home/haifeng/Projects/CosyVoice/asset/zero_shot_prompt.wav")
    ap.add_argument("--ref_text", default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "query_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]

    tasks: List[Dict[str, Any]] = []
    index_rows = [
        attach_assets(row, wav_dir, args.ref_wav, args.ref_text, tasks)
        for row in rows
    ]

    tasks_path = out_dir / "tts_tasks.jsonl"
    index_path = out_dir / "scenario_index.jsonl"
    n_tasks = write_jsonl(tasks_path, tasks)
    n_index = write_jsonl(index_path, index_rows)
    print(json.dumps({
        "input": str(in_path),
        "out_dir": str(out_dir),
        "tasks": n_tasks,
        "index_rows": n_index,
        "tasks_path": str(tasks_path),
        "index_path": str(index_path),
        "wav_dir": str(wav_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
