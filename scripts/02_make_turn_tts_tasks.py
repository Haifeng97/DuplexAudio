#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
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


def stable_int(text: str, seed: int = 0) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def read_yaml_section(path: str, section: str) -> Dict[str, Any]:
    if not path:
        return {}
    import yaml  # type: ignore

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        return {}
    selected = data.get(section, data)
    return selected if isinstance(selected, dict) else {}


def terminal_punct_text(text: str, *, question_heuristic: bool = True) -> str:
    text = str(text or "").strip()
    if not text or re.search(r"[。！？!?；;：:，,、…]$", text):
        return text
    if question_heuristic and re.search(r"(吗|么|呢|嘛|吧)$", text):
        return text + "？"
    return text + "。"


def load_voice_refs(
    path: str,
    max_voice_refs: int = 0,
    max_ref_text_chars: int = 0,
    min_ref_snr: float = 0.0,
) -> List[Dict[str, Any]]:
    if not path:
        return []
    refs: List[Dict[str, Any]] = []
    for row in read_jsonl(Path(path)):
        if isinstance(row.get("refs"), list):
            spk = str(row.get("spk") or row.get("voice_id") or "voice")
            for idx, ref in enumerate(row.get("refs") or []):
                if not isinstance(ref, dict) or not ref.get("path"):
                    continue
                refs.append({
                    "voice_id": f"{spk}:ref{idx}",
                    "voice_spk": spk,
                    "voice_lang": row.get("lang"),
                    "voice_dataset": row.get("dataset") or ref.get("source_dataset"),
                    "voice_ref_index": idx,
                    "ref_wav": str(ref.get("path")),
                    "ref_text": str(ref.get("text") or ""),
                    "ref_duration": ref.get("duration"),
                    "ref_snr": ref.get("snr"),
                    "ref_emotion": ref.get("emotion"),
                })
        elif row.get("ref_wav") or row.get("path"):
            voice_id = str(row.get("voice_id") or row.get("spk") or f"voice{len(refs):04d}")
            refs.append({
                "voice_id": voice_id,
                "voice_spk": row.get("spk") or voice_id,
                "voice_lang": row.get("lang"),
                "voice_dataset": row.get("dataset") or row.get("source_dataset"),
                "voice_ref_index": row.get("voice_ref_index", 0),
                "ref_wav": str(row.get("ref_wav") or row.get("path")),
                "ref_text": str(row.get("ref_text") or row.get("text") or ""),
                "ref_duration": row.get("duration"),
                "ref_snr": row.get("snr"),
                "ref_emotion": row.get("emotion"),
            })
    if max_ref_text_chars and max_ref_text_chars > 0:
        refs = [r for r in refs if len(str(r.get("ref_text") or "")) <= max_ref_text_chars]
    if min_ref_snr and min_ref_snr > 0:
        refs = [r for r in refs if float(r.get("ref_snr") or -999.0) >= min_ref_snr]
    if max_voice_refs and max_voice_refs > 0:
        refs = refs[:max_voice_refs]
    return refs


class VoicePicker:
    def __init__(
        self,
        refs: List[Dict[str, Any]],
        *,
        fallback_ref_wav: str,
        fallback_ref_text: str,
        strategy: str,
        seed: int,
    ):
        self.refs = refs
        self.strategy = strategy
        self.seed = seed
        self.next_idx = 0
        self.sample_cache: Dict[str, Dict[str, Any]] = {}
        self.fallback = {
            "voice_id": "default",
            "voice_spk": "default",
            "voice_lang": None,
            "voice_dataset": None,
            "voice_ref_index": 0,
            "ref_wav": fallback_ref_wav,
            "ref_text": fallback_ref_text,
            "ref_duration": None,
            "ref_snr": None,
            "ref_emotion": None,
        }

    def pick(self, sample_id: str, key: str) -> Dict[str, Any]:
        if not self.refs:
            return dict(self.fallback)
        if self.strategy == "sample_hash":
            cached = self.sample_cache.get(sample_id)
            if cached is None:
                cached = self.refs[stable_int(sample_id, self.seed) % len(self.refs)]
                self.sample_cache[sample_id] = cached
            return dict(cached)
        if self.strategy == "task_hash":
            return dict(self.refs[stable_int(f"{sample_id}:{key}", self.seed) % len(self.refs)])
        if self.strategy == "round_robin":
            ref = self.refs[self.next_idx % len(self.refs)]
            self.next_idx += 1
            return dict(ref)
        if self.strategy == "random":
            ref = self.refs[stable_int(f"random:{sample_id}:{key}:{self.next_idx}", self.seed) % len(self.refs)]
            self.next_idx += 1
            return dict(ref)
        raise ValueError(f"unknown voice strategy: {self.strategy}")


def voice_metadata(ref: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "voice_id": ref.get("voice_id"),
        "voice_spk": ref.get("voice_spk"),
        "voice_lang": ref.get("voice_lang"),
        "voice_dataset": ref.get("voice_dataset"),
        "voice_ref_index": ref.get("voice_ref_index"),
        "ref_wav": ref.get("ref_wav"),
        "ref_text": ref.get("ref_text"),
        "ref_duration": ref.get("ref_duration"),
        "ref_snr": ref.get("ref_snr"),
        "ref_emotion": ref.get("ref_emotion"),
    }


def add_task(
    tasks: List[Dict[str, Any]],
    asset: Dict[str, Any],
    *,
    sample_id: str,
    key: str,
    text: str,
    wav_dir: Path,
    voice: Dict[str, Any],
    tts_text_punct: bool,
) -> None:
    task_id = f"{safe_name(sample_id)}__{key}"
    out = wav_dir / f"{task_id}.wav"
    tts_text = terminal_punct_text(text, question_heuristic=(key != "backchannel")) if tts_text_punct else str(text)
    task = {
        "id": task_id,
        "sample_id": sample_id,
        "key": key,
        "text": tts_text,
        "source_text": str(text),
        "out": str(out),
        "ref_wav": voice["ref_wav"],
        "ref_text": voice["ref_text"],
        **voice_metadata(voice),
    }
    tasks.append(task)
    asset[key] = {
        "task_id": task_id,
        "text": text,
        "tts_text": tts_text,
        "audio": str(out),
        "voice": voice_metadata(voice),
    }


def attach_assets(
    row: Dict[str, Any],
    wav_dir: Path,
    picker: VoicePicker,
    tasks: List[Dict[str, Any]],
    *,
    tts_text_punct: bool,
) -> Dict[str, Any]:
    out = dict(row)
    sample_id = str(out.get("id"))
    scenario = out.get("scenario")
    assets: Dict[str, Any] = {}

    if scenario == "normal_qa":
        turns = out.get("turns")
        if isinstance(turns, list) and len(turns) > 1:
            for idx, turn in enumerate(turns, start=1):
                if isinstance(turn, dict) and turn.get("needs_tts", True):
                    key = f"turn{idx:03d}_query"
                    add_task(
                        tasks,
                        assets,
                        sample_id=sample_id,
                        key=key,
                        text=str(turn.get("question_text", "")),
                        wav_dir=wav_dir,
                        voice=picker.pick(sample_id, key),
                        tts_text_punct=tts_text_punct,
                    )
        else:
            add_task(
                tasks,
                assets,
                sample_id=sample_id,
                key="query",
                text=str(out.get("question_text", "")),
                wav_dir=wav_dir,
                voice=picker.pick(sample_id, "query"),
                tts_text_punct=tts_text_punct,
            )
    elif scenario == "player_interrupts_ai":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="base_query",
            text=str(out.get("base", {}).get("question_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "base_query"),
            tts_text_punct=tts_text_punct,
        )
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="donor_query",
            text=str(out.get("donor", {}).get("question_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "donor_query"),
            tts_text_punct=tts_text_punct,
        )
    elif scenario == "player_backchannel":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query",
            text=str(out.get("question_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "query"),
            tts_text_punct=tts_text_punct,
        )
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="backchannel",
            text=str(out.get("backchannel_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "backchannel"),
            tts_text_punct=tts_text_punct,
        )
    elif scenario == "incomplete_query_candidate":
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query_part1",
            text=str(out.get("query_part1_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "query_part1"),
            tts_text_punct=tts_text_punct,
        )
        add_task(
            tasks,
            assets,
            sample_id=sample_id,
            key="query_part2",
            text=str(out.get("query_part2_text", "")),
            wav_dir=wav_dir,
            voice=picker.pick(sample_id, "query_part2"),
            tts_text_punct=tts_text_punct,
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
                        voice=picker.pick(sample_id, f"turn{idx:03d}_query"),
                        tts_text_punct=tts_text_punct,
                    )
    out["tts_assets"] = assets
    return out


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/esd_voice_tts.yaml", help="YAML config section: tts_tasks")
    known, _ = pre.parse_known_args()
    defaults = read_yaml_section(known.config, "tts_tasks")

    ap = argparse.ArgumentParser(
        description="Create turn-level TTS tasks for duplex scenario candidates.",
        parents=[pre],
    )
    ap.add_argument("--input", required=True, help="Scenario candidate JSONL or selected turns JSONL")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_wav", default="/home/haifeng/Projects/CosyVoice/asset/zero_shot_prompt.wav")
    ap.add_argument("--ref_text", default="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。")
    ap.add_argument("--voice_bank", default="", help="voice_gen voice_bank.jsonl or flat JSONL with ref_wav/ref_text")
    ap.add_argument("--voice_strategy", choices=["sample_hash", "task_hash", "round_robin", "random"], default="sample_hash")
    ap.add_argument("--voice_seed", type=int, default=42)
    ap.add_argument("--max_voice_refs", type=int, default=0, help="Use first N refs from voice bank; 0 means all")
    ap.add_argument("--max_ref_text_chars", type=int, default=0, help="Drop refs with longer transcripts; 0 disables")
    ap.add_argument("--min_ref_snr", type=float, default=0.0, help="Drop refs below this SNR; 0 disables")
    ap.add_argument("--tts_text_punct", action="store_true", help="Add terminal punctuation to TTS text only")
    ap.add_argument("--limit", type=int, default=0)
    ap.set_defaults(**defaults)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "query_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(in_path)
    if args.limit:
        rows = rows[: args.limit]

    tasks: List[Dict[str, Any]] = []
    voice_refs = load_voice_refs(
        args.voice_bank,
        max_voice_refs=args.max_voice_refs,
        max_ref_text_chars=args.max_ref_text_chars,
        min_ref_snr=args.min_ref_snr,
    )
    picker = VoicePicker(
        voice_refs,
        fallback_ref_wav=args.ref_wav,
        fallback_ref_text=args.ref_text,
        strategy=args.voice_strategy,
        seed=args.voice_seed,
    )
    index_rows = [
        attach_assets(row, wav_dir, picker, tasks, tts_text_punct=bool(args.tts_text_punct))
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
        "voice_bank": str(args.voice_bank or ""),
        "voice_refs": len(voice_refs) if voice_refs else 1,
        "voice_strategy": args.voice_strategy if voice_refs else "single_ref",
        "max_ref_text_chars": int(args.max_ref_text_chars),
        "min_ref_snr": float(args.min_ref_snr),
        "tts_text_punct": bool(args.tts_text_punct),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
