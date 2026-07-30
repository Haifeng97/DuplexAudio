#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def safe_name(text: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text))
    return text.strip("._-")[:180] or "sample"


def terminal_punct_text(text: str, *, question_heuristic: bool = True) -> str:
    text = str(text or "").strip()
    if not text or re.search(r"[。！？!?；;：:，,、…]$", text):
        return text
    if question_heuristic and re.search(r"(吗|么|呢|嘛|吧)$", text):
        return text + "？"
    return text + "。"


def history_turn_count(row: Dict[str, Any]) -> int:
    sr = row.get("source_row") if isinstance(row.get("source_row"), dict) else row
    meta = sr.get("meta") if isinstance(sr.get("meta"), dict) else {}
    try:
        return int(meta.get("history_turn_count") or 0)
    except Exception:
        return 0


def add_delta_task(tasks: List[Dict[str, Any]], assets: Dict[str, Any], *, sample_id: str, key: str, text: str, wav_dir: Path, voice: Dict[str, Any]) -> None:
    task_id = f"{safe_name(sample_id)}__{key}"
    out = wav_dir / f"{task_id}.wav"
    tts_text = terminal_punct_text(text, question_heuristic=True)
    ref_wav = voice.get("ref_wav")
    ref_text = voice.get("ref_text")
    if not ref_wav or ref_text is None:
        raise ValueError(f"missing ref_wav/ref_text for {sample_id} {key}")
    task = {
        "id": task_id,
        "sample_id": sample_id,
        "key": key,
        "text": tts_text,
        "source_text": str(text),
        "out": str(out),
        "ref_wav": ref_wav,
        "ref_text": ref_text,
        **{k: voice.get(k) for k in ["voice_id", "voice_spk", "voice_lang", "voice_dataset", "voice_ref_index", "ref_duration", "ref_snr", "ref_emotion"]},
    }
    tasks.append(task)
    assets[key] = {
        "task_id": task_id,
        "text": str(text),
        "tts_text": tts_text,
        "audio": str(out),
        "voice": {k: voice.get(k) for k in ["voice_id", "voice_spk", "voice_lang", "voice_dataset", "voice_ref_index", "ref_wav", "ref_text", "ref_duration", "ref_snr", "ref_emotion"]},
    }


def source_row(row: Dict[str, Any]) -> Dict[str, Any]:
    sr = row.get("source_row")
    return dict(sr) if isinstance(sr, dict) else dict(row)


def load_selected_turns(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        out[str(row.get("id"))] = row
    return out


def find_turn_idx(turns: List[Dict[str, Any]], question: str) -> int:
    for i, turn in enumerate(turns):
        if str(turn.get("question_text") or "") == str(question or ""):
            return i
    return max(0, len(turns) - 1)


def find_turn_id_idx(turns: List[Dict[str, Any]], turn_id: Any, default: int) -> int:
    for i, turn in enumerate(turns):
        if turn.get("turn_id") == turn_id:
            return i
    return default


def ensure_voice(assets: Dict[str, Any], preferred: str) -> Dict[str, Any]:
    if preferred in assets and isinstance(assets[preferred].get("voice"), dict):
        return dict(assets[preferred]["voice"])
    for asset in assets.values():
        if isinstance(asset, dict) and isinstance(asset.get("voice"), dict):
            return dict(asset["voice"])
    raise ValueError("no reusable voice metadata found")


def prepare_normal(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    single_manifest: List[Dict[str, Any]] = []
    multi_index: List[Dict[str, Any]] = []
    hist = Counter()
    for row in iter_jsonl(Path(args.normal_manifest)):
        h = history_turn_count(row)
        hist[h] += 1
        if h > 0:
            multi_index.append(source_row(row))
        else:
            single_manifest.append(row)
    return {
        "normal_single_manifest": write_jsonl(out_dir / "reuse" / "normal_single_manifest.jsonl", single_manifest),
        "normal_multi_index": write_jsonl(out_dir / "normal_multi" / "scenario_index.jsonl", multi_index),
        "normal_history_dist": dict(sorted(hist.items())),
    }


def prepare_incomplete(args: argparse.Namespace, out_dir: Path, selected: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    single_manifest: List[Dict[str, Any]] = []
    multi_index: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    wav_dir = out_dir / "incomplete_multi" / "query_wav"
    hist = Counter()
    for manifest_row in iter_jsonl(Path(args.incomplete_manifest)):
        h = history_turn_count(manifest_row)
        hist[h] += 1
        if h <= 0:
            single_manifest.append(manifest_row)
            continue
        row = source_row(manifest_row)
        source_id = str(row.get("source_id") or "")
        selected_row = selected.get(source_id)
        if not selected_row:
            raise ValueError(f"selected row not found for incomplete source_id={source_id}")
        turns = [dict(t) for t in (selected_row.get("turns") or []) if isinstance(t, dict)]
        if not turns:
            raise ValueError(f"selected row has no turns for incomplete source_id={source_id}")
        special_idx = find_turn_idx(turns, row.get("full_question_text", ""))
        row["turns"] = turns
        row["incomplete_turn_index"] = special_idx + 1
        row["incomplete_turn_id"] = turns[special_idx].get("turn_id")
        row.setdefault("tts_assets", {})
        assets = row["tts_assets"]
        voice = ensure_voice(assets, "query_part1")
        for idx, turn in enumerate(turns):
            if idx == special_idx:
                continue
            key = f"turn{idx + 1:03d}_query"
            if key not in assets:
                add_delta_task(tasks, assets, sample_id=str(row["id"]), key=key, text=str(turn.get("question_text", "")), wav_dir=wav_dir, voice=voice)
        multi_index.append(row)
    return {
        "incomplete_single_manifest": write_jsonl(out_dir / "reuse" / "incomplete_single_manifest.jsonl", single_manifest),
        "incomplete_multi_index": write_jsonl(out_dir / "incomplete_multi" / "scenario_index.jsonl", multi_index),
        "incomplete_delta_tts_tasks": write_jsonl(out_dir / "incomplete_multi" / "tts_tasks.jsonl", tasks),
        "incomplete_history_dist": dict(sorted(hist.items())),
    }


def prepare_interrupt(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    index_rows: List[Dict[str, Any]] = []
    tasks: List[Dict[str, Any]] = []
    wav_dir = out_dir / "interrupt" / "query_wav"
    hist = Counter()
    for manifest_row in iter_jsonl(Path(args.interrupt_manifest)):
        h = history_turn_count(manifest_row)
        hist[h] += 1
        row = source_row(manifest_row)
        turns = [dict(t) for t in (row.get("turns") or []) if isinstance(t, dict)]
        if not turns:
            index_rows.append(row)
            continue
        base_idx = find_turn_id_idx(turns, row.get("base", {}).get("turn_id"), max(0, len(turns) - 2))
        donor_idx = find_turn_id_idx(turns, row.get("donor", {}).get("turn_id"), max(0, len(turns) - 1))
        if donor_idx <= base_idx:
            base_idx = max(0, len(turns) - 2)
            donor_idx = max(0, len(turns) - 1)
        row.setdefault("tts_assets", {})
        assets = row["tts_assets"]
        voice = ensure_voice(assets, "base_query")
        for idx, turn in enumerate(turns):
            if idx in {base_idx, donor_idx}:
                continue
            key = f"turn{idx + 1:03d}_query"
            if key not in assets:
                add_delta_task(tasks, assets, sample_id=str(row["id"]), key=key, text=str(turn.get("question_text", "")), wav_dir=wav_dir, voice=voice)
        index_rows.append(row)
    return {
        "interrupt_index": write_jsonl(out_dir / "interrupt" / "scenario_index.jsonl", index_rows),
        "interrupt_delta_tts_tasks": write_jsonl(out_dir / "interrupt" / "tts_tasks.jsonl", tasks),
        "interrupt_history_dist": dict(sorted(hist.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare v2 delta inputs by reusing existing single-turn manifests and special TTS audio.")
    ap.add_argument("--out_dir", default="outputs/final_v2_delta")
    ap.add_argument("--selected", default="outputs/selected_turns/selected.jsonl")
    ap.add_argument("--normal_manifest", default="outputs/final_trimmed/final_normal/manifest.jsonl")
    ap.add_argument("--incomplete_manifest", default="outputs/final_trimmed/final_incomplete/manifest.jsonl")
    ap.add_argument("--interrupt_manifest", default="outputs/final_trimmed/final_interrupt/manifest.jsonl")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    selected = load_selected_turns(Path(args.selected))
    stats: Dict[str, Any] = {"out_dir": str(out_dir), "selected_rows": len(selected)}
    stats.update(prepare_normal(args, out_dir))
    stats.update(prepare_incomplete(args, out_dir, selected))
    stats.update(prepare_interrupt(args, out_dir))
    stats_path = out_dir / "prepare_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
