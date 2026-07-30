#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


DEFAULT_ROLES = ("吉莉", "伞兵", "阿梅", "花傲天")
V3_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
V3_EOP = "<|endofprompt|>"


def safe_name(text: Any, fallback: str = "sample") -> str:
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text or "")).strip("._-")
    return s[:120] or fallback


def terminal_punct_text(text: str) -> str:
    text = str(text or "").strip()
    if not text or re.search(r"[。！？!?；;：:，,、…]$", text):
        return text
    if re.search(r"(吗|么|呢|嘛|吧)$", text):
        return text + "？"
    return text + "。"


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def role_pattern(roles: List[str]) -> re.Pattern[str]:
    return re.compile(r"^\s*请你扮演\s*(" + "|".join(map(re.escape, roles)) + r")\s*与我对话")


def detect_role(row: Dict[str, Any], pattern: re.Pattern[str]) -> str:
    sysprompt = str(row.get("sysprompt") or row.get("system") or "")
    m = pattern.search(sysprompt)
    return m.group(1) if m else ""


def load_ref_texts(path: str) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--ref_text_json must be a JSON object mapping role to text")
    return {str(k): str(v) for k, v in data.items()}


def extract_refs(ref_zip: Path, refs_dir: Path, roles: List[str], ref_texts: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    refs_dir.mkdir(parents=True, exist_ok=True)
    role_refs: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(ref_zip) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            role = next((r for r in roles if r in name), "")
            if not role:
                continue
            out = refs_dir / name
            with zf.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            role_refs[role] = {
                "voice_id": role,
                "voice_spk": role,
                "voice_lang": "zh",
                "voice_dataset": "ai_partner_0726_ref",
                "voice_ref_index": 0,
                "ref_wav": f"refs/{name}",
                "ref_text": ref_texts.get(role, ""),
                "ref_duration": None,
                "ref_snr": None,
                "ref_emotion": None,
            }
    missing = [role for role in roles if role not in role_refs]
    if missing:
        raise RuntimeError(f"missing reference wavs for roles: {missing}")
    return role_refs


def make_task(
    *,
    out_dir: Path,
    row_idx: int,
    sample_id: str,
    role: str,
    turn: Dict[str, Any],
    voice: Dict[str, Any],
    tts_text_punct: bool,
) -> Dict[str, Any]:
    turn_id = turn.get("turn_id")
    source = str(turn.get("source") or "")
    key = f"answer_t{turn_id}"
    task_id = f"reply_{row_idx:06d}_{safe_name(sample_id, f'sample_{row_idx:06d}')[:48]}__{key}"
    text = str(turn.get("answer_text") or "").strip()
    tts_text = terminal_punct_text(text) if tts_text_punct else text
    wav_rel = f"wav/{role}/{task_id}.wav"
    return {
        "id": task_id,
        "sample_id": sample_id,
        "row_idx": row_idx,
        "key": key,
        "turn_id": turn_id,
        "turn_source": source,
        "role": role,
        "text": tts_text,
        "source_text": text,
        "out": wav_rel,
        "ref_wav": voice["ref_wav"],
        "ref_text": voice["ref_text"],
        "voice_id": voice["voice_id"],
        "voice_spk": voice["voice_spk"],
        "voice_lang": voice["voice_lang"],
        "voice_dataset": voice["voice_dataset"],
        "voice_ref_index": voice["voice_ref_index"],
        "ref_duration": voice["ref_duration"],
        "ref_snr": voice["ref_snr"],
        "ref_emotion": voice["ref_emotion"],
    }


def build_tasks(args: argparse.Namespace) -> Dict[str, Any]:
    selected = Path(args.selected)
    out_dir = Path(args.out_dir)
    refs_dir = out_dir / "refs"
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    role_re = role_pattern(roles)
    ref_texts = load_ref_texts(args.ref_text_json)
    role_refs = extract_refs(Path(args.ref_zip), refs_dir, roles, ref_texts)

    tasks: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    role_task_counts: Counter[str] = Counter()
    role_sample_counts: Counter[str] = Counter()
    turn_source_counts: Counter[str] = Counter()

    for row_idx, row in enumerate(read_jsonl(selected), start=1):
        counters["rows_read"] += 1
        role = detect_role(row, role_re)
        if not role:
            counters["skip_role_unmatched"] += 1
            continue
        role_sample_counts[role] += 1
        sample_id = str(row.get("id") or f"row_{row_idx:06d}")
        turns = row.get("turns")
        if not isinstance(turns, list):
            counters["skip_missing_turns"] += 1
            continue
        selected_turns = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            if not args.include_history and turn.get("source") != "current":
                continue
            text = str(turn.get("answer_text") or "").strip()
            if not text:
                counters["skip_empty_answer"] += 1
                continue
            selected_turns.append(turn)
        if not selected_turns:
            counters["skip_no_selected_turns"] += 1
            continue
        for turn in selected_turns:
            task = make_task(
                out_dir=out_dir,
                row_idx=row_idx,
                sample_id=sample_id,
                role=role,
                turn=turn,
                voice=role_refs[role],
                tts_text_punct=not args.no_tts_text_punct,
            )
            tasks.append(task)
            role_task_counts[role] += 1
            turn_source_counts[str(turn.get("source") or "")] += 1
            index_rows.append({
                "id": task["id"],
                "sample_id": sample_id,
                "row_idx": row_idx,
                "role": role,
                "turn_id": task["turn_id"],
                "turn_source": task["turn_source"],
                "question_text": str(turn.get("question_text") or ""),
                "answer_text": task["source_text"],
                "tts_text": task["text"],
                "audio": task["out"],
                "voice": {
                    "voice_id": task["voice_id"],
                    "voice_spk": task["voice_spk"],
                    "ref_wav": task["ref_wav"],
                    "ref_text": task["ref_text"],
                },
                "source_meta": row.get("meta", {}),
            })

    tasks_path = out_dir / "tts_tasks.jsonl"
    index_path = out_dir / "reply_index.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(index_path, index_rows)

    scripts_dir = out_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for script in ("03_run_tts.py", "03_run_tts_multi_gpu.py"):
        shutil.copy2(Path("scripts") / script, scripts_dir / script)

    (out_dir / "run_reply_tts.sh").write_text("""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${PROJECT:=Cgame_aimate_haifengjia}"
: "${GPUS:=0,1,2,3}"
: "${PROCS_PER_GPU:=2}"
: "${TTS_PYTHON:=/data/haifengjia/rp_v2_normal_tts_20260724/runtime/cosyvoice_env/bin/python}"
: "${COSYVOICE_REPO:=/data/haifengjia/rp_v2_normal_tts_20260724/runtime/CosyVoice}"
: "${MODEL_DIR:=$COSYVOICE_REPO/pretrained_models/Fun-CosyVoice3-0.5B}"

PROJECT="$PROJECT" "$TTS_PYTHON" scripts/03_run_tts_multi_gpu.py \
  --tasks tts_tasks.jsonl \
  --work_dir . \
  --gpus "$GPUS" \
  --procs_per_gpu "$PROCS_PER_GPU" \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$MODEL_DIR" \
  --monitor_every 0.5 \
  --progress_every 50
""", encoding="utf-8")
    (out_dir / "run_reply_tts.sh").chmod(0o755)

    readme = f"""AI partner reply TTS package

Input selected data:
  {selected}

Scope:
  Only rows already kept by the previous AI-partner filtering step.
  Roles are detected from sysprompt prefix: 请你扮演<role>与我对话。
  Each selected turn's answer_text is synthesized. History answers are included.

Files:
  tts_tasks.jsonl: CosyVoice tasks for AI replies.
  reply_index.jsonl: task-to-source mapping.
  refs/: four role reference wavs.
  wav/: generated output wavs.
  scripts/: copied TTS runner scripts.

Run on GPU machine:
  cd {out_dir}
  GPUS=0,1,2,3 PROCS_PER_GPU=2 ./run_reply_tts.sh

If the portable CosyVoice runtime is somewhere else, override:
  TTS_PYTHON=/path/to/runtime/cosyvoice_env/bin/python \\
  COSYVOICE_REPO=/path/to/runtime/CosyVoice \\
  GPUS=0,1,2,3 PROCS_PER_GPU=2 ./run_reply_tts.sh

Existing wav files are reused unless --overwrite is passed through the runner.
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

    stats = {
        "selected": str(selected),
        "out_dir": str(out_dir),
        "ref_zip": str(args.ref_zip),
        "include_history": bool(args.include_history),
        "roles": roles,
        "tasks": len(tasks),
        "index_rows": len(index_rows),
        "counters": dict(counters),
        "role_sample_counts": dict(role_sample_counts),
        "role_task_counts": dict(role_task_counts),
        "turn_source_counts": dict(turn_source_counts),
        "ref_wavs": role_refs,
        "tasks_path": str(tasks_path),
        "index_path": str(index_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build reply-only TTS tasks for filtered AI-partner rows.")
    ap.add_argument("--selected", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_zip", required=True)
    ap.add_argument("--roles", default=",".join(DEFAULT_ROLES))
    ap.add_argument("--ref_text_json", default="", help="Optional JSON mapping role name to reference transcript.")
    ap.add_argument("--include_history", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--no_tts_text_punct", action="store_true")
    args = ap.parse_args()
    stats = build_tasks(args)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
