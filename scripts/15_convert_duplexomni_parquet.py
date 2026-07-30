#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import struct
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

import pyarrow.parquet as pq  # type: ignore
from tqdm import tqdm


SAMPLE_RATE = 24000
CHUNK_BYTES = 23040
CHUNK_SAMPLES = CHUNK_BYTES // 2
CHUNK_MS = 480


def safe_name(text: str, fallback: str) -> str:
    import re

    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", text).strip("._-")
    return name[:120] or fallback


def stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def write_wav_from_pcm_chunks(path: Path, chunks: List[bytes], sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for chunk in chunks:
            wf.writeframes(chunk)


def pcm_rms(chunk: bytes) -> float:
    if not chunk:
        return 0.0
    n = len(chunk) // 2
    if n <= 0:
        return 0.0
    vals = struct.unpack("<" + "h" * n, chunk)
    return (sum(float(v) * float(v) for v in vals) / n) ** 0.5


def parse_payload(content: Any) -> Dict[str, Any]:
    if not isinstance(content, str) or not content.startswith("{"):
        return {}
    try:
        obj = ast.literal_eval(content)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def decode_codec_summary(blob: bytes) -> Dict[str, Any]:
    if not blob:
        return {}
    try:
        import torch  # type: ignore

        codec = torch.load(io.BytesIO(blob), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        return {"decode_error": repr(exc), "bytes": len(blob)}
    out: Dict[str, Any] = {"bytes": len(blob), "keys": list(codec.keys()) if isinstance(codec, dict) else []}
    if isinstance(codec, dict):
        codes = codec.get("codes")
        if hasattr(codes, "shape"):
            out["codes_shape"] = list(codes.shape)
        for key in ("turn_boundaries", "assistant_turn_indices", "turn_audio_indices"):
            value = codec.get(key)
            if hasattr(value, "shape"):
                out[key + "_shape"] = list(value.shape)
                out[key + "_head"] = value[:10].tolist()
    return out


def iter_rows(path: Path, *, columns: List[str], limit: int, row_group_start: int, row_group_end: int) -> Iterator[Dict[str, Any]]:
    pf = pq.ParquetFile(path)
    end = pf.num_row_groups if row_group_end < 0 else min(row_group_end, pf.num_row_groups)
    yielded = 0
    for rg in range(row_group_start, end):
        table = pf.read_row_group(rg, columns=columns)
        for row in table.to_pylist():
            yield row
            yielded += 1
            if limit and yielded >= limit:
                return


def build_timeline(messages: List[Dict[str, Any]], audios: List[bytes]) -> Dict[str, Any]:
    timeline: List[Dict[str, Any]] = []
    full_asr: List[str] = []
    full_tts: List[str] = []
    counts: Counter[str] = Counter()
    previous_tts_delta = ""
    previous_tts_control = ""
    previous_system2_control = ""
    for idx, chunk in enumerate(audios):
        assistant_msg_idx = 2 + idx * 2
        payload: Dict[str, Any] = {}
        if assistant_msg_idx < len(messages):
            payload = parse_payload(messages[assistant_msg_idx].get("content"))
        asr_delta = str(payload.get("asr") or "")
        tts_delta = str(payload.get("tts") or "")
        tts_control = str(payload.get("tts_control") or "")
        system2_control = str(payload.get("system2_control") or "")
        display_tts_delta = previous_tts_delta
        display_tts_control = previous_tts_control
        display_system2_control = previous_system2_control
        if asr_delta:
            full_asr.append(asr_delta)
            counts["asr_nonempty"] += 1
        if tts_delta:
            full_tts.append(tts_delta)
            counts["tts_nonempty"] += 1
        if tts_control:
            counts["tts_control_nonempty"] += 1
        if system2_control:
            counts["system2_control_nonempty"] += 1
        active: List[str] = []
        if asr_delta:
            active.append(f"USER:{asr_delta}")
        if display_tts_delta:
            active.append(f"ASSISTANT:{display_tts_delta}")
        if display_tts_control:
            active.append(f"TTS_CTRL:{display_tts_control}")
        if display_system2_control:
            active.append(f"S2_CTRL:{display_system2_control}")
        label = " | ".join(active) if active else "NO_OUTPUT"
        if asr_delta and display_tts_delta:
            kind = "asr_tts_overlap"
        elif asr_delta:
            kind = "asr_delta"
        elif display_tts_delta:
            kind = "tts_delta"
        elif display_tts_control:
            kind = "tts_control"
        elif display_system2_control:
            kind = "system2_control"
        else:
            kind = "no_output"
        label_type = "text" if asr_delta or display_tts_delta else "state"
        start_sample = idx * CHUNK_SAMPLES
        end_sample = start_sample + CHUNK_SAMPLES
        timeline.append({
            "idx": idx,
            "kind": kind,
            "label_type": label_type,
            "label": label,
            "start_sec": round(start_sample / SAMPLE_RATE, 6),
            "end_sec": round(end_sample / SAMPLE_RATE, 6),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "causal_output_start_sec": round(end_sample / SAMPLE_RATE, 6),
            "causal_output_end_sec": round((end_sample + CHUNK_SAMPLES) / SAMPLE_RATE, 6),
            "audio_source": "duplexomni_pcm_chunk",
            "chunk_idx": idx,
            "asr_delta": asr_delta,
            "tts_delta": tts_delta,
            "tts_control": tts_control,
            "system2_control": system2_control,
            "display_tts_delta": display_tts_delta,
            "display_tts_control": display_tts_control,
            "display_system2_control": display_system2_control,
            "rms": round(pcm_rms(chunk), 4),
        })
        previous_tts_delta = tts_delta
        previous_tts_control = tts_control
        previous_system2_control = system2_control
    return {
        "timeline": timeline,
        "asr_text": "".join(full_asr),
        "tts_text": "".join(full_tts),
        "counts": dict(counts),
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def load_selection(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    selected: Dict[str, Dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id") or row.get("id") or "")
            if not sample_id:
                raise ValueError(f"missing sample_id in {path}:{line_no}")
            selected[sample_id] = row
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert DuplexOmni parquet rows into wav + timeline manifest for inspection.")
    ap.add_argument("--parquet", default="dataset/duplexomni/train_e2e_codec_000000.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav_dir", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--row_group_start", type=int, default=0)
    ap.add_argument("--row_group_end", type=int, default=-1)
    ap.add_argument("--selection_jsonl", default="", help="Only convert sample IDs listed in this JSONL.")
    ap.add_argument("--decode_codec", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    wav_dir = Path(args.wav_dir) if args.wav_dir else out_path.parent / "wav"
    columns = [
        "session_id",
        "sample_id",
        "split",
        "system_prompt",
        "voice_plan",
        "agent_speaker",
        "user_speaker",
        "messages",
        "audios",
    ]
    if args.decode_codec:
        columns.append("codec")

    rows: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "parquet": args.parquet,
        "out": str(out_path),
        "wav_dir": str(wav_dir),
        "limit": args.limit,
        "row_group_start": args.row_group_start,
        "row_group_end": args.row_group_end,
    }
    scenario_counts: Counter[str] = Counter()
    chunk_counts: Counter[int] = Counter()
    bad: List[Dict[str, Any]] = []

    selection = load_selection(args.selection_jsonl)
    remaining_ids: Set[str] = set(selection)
    source_iter = iter_rows(
        Path(args.parquet),
        columns=columns,
        limit=0 if selection else args.limit,
        row_group_start=args.row_group_start,
        row_group_end=args.row_group_end,
    )
    total = len(selection) if selection else (args.limit if args.limit else None)
    for row_idx, row in enumerate(tqdm(source_iter, total=total, dynamic_ncols=True, unit="row", desc="duplexomni"), start=1):
        sample_id = str(row.get("sample_id") or row.get("session_id") or f"duplexomni_{row_idx:06d}")
        if selection and sample_id not in remaining_ids:
            continue
        audios = row.get("audios") or []
        if not isinstance(audios, list) or not audios:
            bad.append({"sample_id": sample_id, "error": "missing_audios"})
            continue
        if any(not isinstance(chunk, (bytes, bytearray)) for chunk in audios):
            bad.append({"sample_id": sample_id, "error": "non_bytes_audio_chunk"})
            continue
        if any(len(chunk) != CHUNK_BYTES for chunk in audios):
            bad.append({
                "sample_id": sample_id,
                "error": "unexpected_chunk_bytes",
                "lengths_head": [len(chunk) for chunk in audios[:10]],
            })
            continue
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        timeline_info = build_timeline(messages, audios)
        wav_name = safe_name(sample_id, f"duplexomni_{row_idx:06d}") + ".wav"
        wav_path = wav_dir / wav_name
        write_wav_from_pcm_chunks(wav_path, audios)
        selection_meta = selection.get(sample_id, {})
        preview_categories = selection_meta.get("preview_categories") or []
        primary_category = str(selection_meta.get("primary_category") or "")
        split = str(row.get("split") or "duplexomni")
        scenario = f"duplexomni_{primary_category}" if primary_category else split
        manifest = {
            "id": sample_id,
            "source": "duplexomni",
            "scenario": scenario,
            "task": "duplexomni_e2e_inbound",
            "split": split,
            "audio": str(wav_path),
            "sample_rate": SAMPLE_RATE,
            "chunk_ms": CHUNK_MS,
            "sysprompt": row.get("system_prompt") or "",
            "system_prompt": row.get("system_prompt") or "",
            "question_text": timeline_info["asr_text"],
            "answer_text": timeline_info["tts_text"],
            "text_query": timeline_info["asr_text"],
            "asr_text": timeline_info["asr_text"],
            "text": timeline_info["tts_text"],
            "target_text": timeline_info["tts_text"],
            "timeline": timeline_info["timeline"],
            "messages": messages,
            "voice_plan": row.get("voice_plan") or "",
            "agent_speaker": row.get("agent_speaker") or "",
            "user_speaker": row.get("user_speaker") or "",
            "preview_categories": preview_categories,
            "stats": {
                "audio_chunks": len(audios),
                "duration_sec": round(len(audios) * CHUNK_MS / 1000.0, 6),
                "message_count": len(messages),
                **timeline_info["counts"],
            },
        }
        if args.decode_codec:
            manifest["codec"] = decode_codec_summary(row.get("codec") or b"")
        rows.append(manifest)
        scenario_counts[scenario] += 1
        chunk_counts[len(audios)] += 1
        remaining_ids.discard(sample_id)
        if selection and not remaining_ids:
            break

    n = write_jsonl(out_path, rows)
    if bad:
        write_jsonl(out_path.with_suffix(out_path.suffix + ".bad.jsonl"), bad)
    stats.update({
        "n": n,
        "bad": len(bad),
        "scenario_counts": dict(scenario_counts),
        "audio_chunk_count_min": min(chunk_counts) if chunk_counts else 0,
        "audio_chunk_count_max": max(chunk_counts) if chunk_counts else 0,
        "audio_chunk_count_top": dict(chunk_counts.most_common(20)),
        "selection_jsonl": args.selection_jsonl,
        "selection_missing": sorted(remaining_ids),
    })
    out_path.with_suffix(out_path.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
