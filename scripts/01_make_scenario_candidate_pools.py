#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def first_turn(row: Dict[str, Any]) -> Dict[str, Any] | None:
    turns = row.get("turns")
    if isinstance(turns, list) and turns:
        turn = turns[-1]
        if isinstance(turn, dict):
            return turn
    return None


def has_train_history(row: Dict[str, Any]) -> bool:
    turns = row.get("turns")
    return isinstance(turns, list) and len(turns) >= 2


def has_sysprompt_history(row: Dict[str, Any]) -> bool:
    meta = row.get("meta")
    if isinstance(meta, dict) and "sysprompt_ds_history_present" in meta:
        return bool(meta.get("sysprompt_ds_history_present"))
    sysprompt = str(row.get("sysprompt") or "")
    return "<｜User｜>" in sysprompt and "<｜Assistant｜>" in sysprompt


def selection_flag(row: Dict[str, Any], name: str) -> bool:
    selection = row.get("selection")
    if isinstance(selection, dict) and name in selection:
        return bool(selection.get(name))
    turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    if not turns:
        return False
    if name in {"can_normal", "can_interrupt_donor"}:
        return any(str(turn.get("question_text") or "") and str(turn.get("answer_text") or "") for turn in turns)
    if name == "can_interrupt_base":
        return any(len(str(turn.get("answer_text") or "")) >= 8 for turn in turns)
    if name == "can_incomplete_query":
        return any(len(str(turn.get("question_text") or "")) >= 8 for turn in turns)
    return False


def source_group_id(row: Dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    return str(meta.get("source_group_id") or row.get("source_group_id") or row.get("source_id") or row.get("id"))


def normalize_row(
    row: Dict[str, Any],
    *,
    min_question_chars: int,
    max_question_chars: int,
    max_answer_chars: int,
    max_turns: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    turns = [dict(turn) for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    stats = {"history_turns_dropped": 0, "leading_turns_truncated": 0}
    if not turns:
        return None, {**stats, "missing_turns": 1}

    current = turns[-1]
    current_q = str(current.get("question_text") or "")
    current_a = str(current.get("answer_text") or "")
    if not current_q or not current_a:
        return None, {**stats, "missing_current_text": 1}
    if len(current_q) < min_question_chars:
        return None, {**stats, "current_question_too_short": 1}
    if max_question_chars > 0 and len(current_q) > max_question_chars:
        return None, {**stats, "current_question_too_long": 1}
    if max_answer_chars > 0 and len(current_a) > max_answer_chars:
        return None, {**stats, "current_answer_too_long": 1}

    kept_history: List[Dict[str, Any]] = []
    for turn in turns[:-1]:
        question = str(turn.get("question_text") or "")
        answer = str(turn.get("answer_text") or "")
        valid = bool(question and answer)
        valid = valid and len(question) >= min_question_chars
        valid = valid and (max_question_chars <= 0 or len(question) <= max_question_chars)
        valid = valid and (max_answer_chars <= 0 or len(answer) <= max_answer_chars)
        if valid:
            kept_history.append(turn)
        else:
            stats["history_turns_dropped"] += 1

    normalized_turns = kept_history + [current]
    if max_turns > 0 and len(normalized_turns) > max_turns:
        stats["leading_turns_truncated"] = len(normalized_turns) - max_turns
        normalized_turns = normalized_turns[-max_turns:]
    for turn_id, turn in enumerate(normalized_turns, start=1):
        turn["turn_id"] = turn_id

    out = dict(row)
    out["turns"] = normalized_turns
    out["question_text"] = current_q
    out["answer_text"] = current_a
    meta = dict(row.get("meta") or {})
    meta["turn_count"] = len(normalized_turns)
    meta["history_turn_count"] = max(0, len(normalized_turns) - 1)
    out["meta"] = meta
    return out, stats


def answer_gn_chunks(answer_text: str) -> int:
    return max(1, int(len(answer_text) * 1.1 + 0.999999))


def normal_candidate(row: Dict[str, Any], chunk_ms: int) -> Dict[str, Any]:
    turn = first_turn(row) or {}
    answer_text = str(turn.get("answer_text", row.get("answer_text", "")))
    return {
        "id": row["id"],
        "scenario": "normal_qa",
        "source_id": row["id"],
        "source_group_id": source_group_id(row),
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "question_text": turn.get("question_text", row.get("question_text", "")),
        "answer_text": answer_text,
        "audio_plan": [
            "gn_before",
            "query_audio",
            "gn_answer_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_audio -> WAIT",
            "gn_answer_region -> ANSWER + answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "chunk_ms": chunk_ms,
            "answer_gn_chunks": answer_gn_chunks(answer_text),
            "answer_gn_duration_sec": round(answer_gn_chunks(answer_text) * chunk_ms / 1000.0, 6),
            "answer_gn_formula": "ceil(len(answer_text) * 1.1) chunks",
        },
        "meta": row.get("meta", {}),
    }


def interrupt_candidate(base: Dict[str, Any], donor: Dict[str, Any], prefix_chars: int) -> Dict[str, Any]:
    base_turn = first_turn(base) or {}
    donor_turn = first_turn(donor) or {}
    base_answer = str(base_turn.get("answer_text", ""))
    prefix_chars = max(1, min(prefix_chars, max(1, len(base_answer) - 1)))
    return {
        "id": f"interrupt__{base['id']}__{donor['id']}",
        "scenario": "player_interrupts_ai",
        "source_group_ids": [source_group_id(base), source_group_id(donor)],
        "source": "candidate_pair_only_no_timeline_yet",
        "base": {
            "id": base["id"],
            "sysprompt": base.get("sysprompt", ""),
            "question_text": base_turn.get("question_text", ""),
            "answer_text": base_answer,
            "answer_prefix_text": base_answer[:prefix_chars],
            "prefix_chars": prefix_chars,
            "meta": base.get("meta", {}),
        },
        "donor": {
            "id": donor["id"],
            "sysprompt": donor.get("sysprompt", ""),
            "question_text": donor_turn.get("question_text", ""),
            "answer_text": donor_turn.get("answer_text", ""),
            "meta": donor.get("meta", {}),
        },
        "audio_plan": [
            "gn_before",
            "base_query_audio",
            "gn_base_answer_prefix_region",
            "donor_query_audio",
            "gn_donor_answer_region",
            "gn_after",
        ],
        "intended_timeline": [
            "gn_before -> IDLE",
            "base question TTS -> D_WAIT",
            "short gaussian region -> A_ANSWER + base answer prefix tokens, no EOR",
            "donor question starts while base answer is unfinished -> D_WAIT",
            "donor remaining question TTS -> D_WAIT",
            "donor answer over gaussian -> A_ANSWER + donor answer tokens + EOR",
            "gn_after -> IDLE",
        ],
    }


def same_row_interrupt_candidate(row: Dict[str, Any], prefix_chars: int) -> Dict[str, Any] | None:
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        return None
    base_turn = turns[-2]
    donor_turn = turns[-1]
    if not isinstance(base_turn, dict) or not isinstance(donor_turn, dict):
        return None
    base_answer = str(base_turn.get("answer_text", ""))
    donor_question = str(donor_turn.get("question_text", ""))
    donor_answer = str(donor_turn.get("answer_text", ""))
    if not base_answer or not donor_question or not donor_answer:
        return None
    prefix_chars = max(1, min(prefix_chars, max(1, len(base_answer) - 1)))
    return {
        "id": f"interrupt_same_row__{row['id']}__t{base_turn.get('turn_id', 'prev')}_t{donor_turn.get('turn_id', 'next')}",
        "scenario": "player_interrupts_ai",
        "source": "same_row_previous_turn_interrupted_by_next_turn",
        "source_id": row["id"],
        "source_group_id": source_group_id(row),
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "base": {
            "id": row["id"],
            "turn_id": base_turn.get("turn_id"),
            "sysprompt": row.get("sysprompt", ""),
            "question_text": base_turn.get("question_text", ""),
            "answer_text": base_answer,
            "answer_prefix_text": base_answer[:prefix_chars],
            "prefix_chars": prefix_chars,
            "meta": row.get("meta", {}),
        },
        "donor": {
            "id": row["id"],
            "turn_id": donor_turn.get("turn_id"),
            "sysprompt": row.get("sysprompt", ""),
            "question_text": donor_question,
            "answer_text": donor_answer,
            "meta": row.get("meta", {}),
        },
        "audio_plan": [
            "gn_before",
            "base_query_audio",
            "gn_base_answer_prefix_region",
            "donor_query_audio",
            "gn_donor_answer_region",
            "gn_after",
        ],
        "intended_timeline": [
            "gn_before -> IDLE",
            "previous turn question TTS -> WAIT",
            "short gaussian region -> ANSWER + previous answer prefix tokens, no EOR",
            "next turn question starts while previous answer is unfinished -> INTERRUPT then WAIT",
            "next turn answer over gaussian -> ANSWER + next answer tokens + EOR",
            "gn_after -> IDLE",
        ],
        "meta": row.get("meta", {}),
    }


def backchannel_candidate(
    row: Dict[str, Any],
    backchannel: Dict[str, Any],
    prefix_chars: int,
    chunk_ms: int,
) -> Dict[str, Any]:
    turn = first_turn(row) or {}
    answer_text = str(turn.get("answer_text", ""))
    prefix_chars = max(1, min(prefix_chars, max(1, len(answer_text) - 1)))
    answer_prefix = answer_text[:prefix_chars]
    answer_remaining = answer_text[prefix_chars:]
    return {
        "id": f"backchannel__{row['id']}__p{prefix_chars}",
        "scenario": "player_backchannel",
        "source_id": row["id"],
        "source_group_id": source_group_id(row),
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "backchannel_turn_id": turn.get("turn_id"),
        "backchannel_turn_index": len(row.get("turns", []) or []),
        "question_text": turn.get("question_text", ""),
        "answer_text": answer_text,
        "answer_prefix_text": answer_prefix,
        "answer_remaining_text": answer_remaining,
        "prefix_chars": prefix_chars,
        "backchannel_text": str(backchannel.get("text") or ""),
        "backchannel_audio": {
            "path": str(backchannel.get("audio") or ""),
            "clip_id": backchannel.get("clip_id"),
            "duration_sec": backchannel.get("duration_sec"),
            "sample_rate": backchannel.get("sample_rate"),
            "speaker": backchannel.get("speaker"),
            "gender": backchannel.get("gender"),
            "source_dataset": backchannel.get("source_dataset", "magicdata_ramc"),
        },
        "audio_plan": [
            "gn_before",
            "query_audio",
            "gn_answer_prefix_region",
            "backchannel_audio",
            "gn_answer_remaining_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_audio -> WAIT",
            "gn_answer_prefix_region -> ANSWER + answer prefix tokens, no EOR",
            "player backchannel audio -> INTERRUPT then WAIT",
            "gn_answer_remaining_region -> ANSWER + remaining answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "chunk_ms": chunk_ms,
            "answer_prefix_gn_chunks": answer_gn_chunks(answer_prefix),
            "answer_remaining_gn_chunks": answer_gn_chunks(answer_remaining),
            "answer_split_policy": "random prefix, remaining answer continues after player backchannel",
        },
        "meta": row.get("meta", {}),
    }


def choose_query_split(q: str, rng: random.Random, min_prefix_chars: int) -> int | None:
    if len(q) < max(min_prefix_chars + 2, 8):
        return None
    lo = min_prefix_chars
    hi = max(min_prefix_chars + 1, min(len(q) - 1, int(len(q) * 0.7)))
    if hi <= lo:
        return None
    return rng.randint(lo, hi)


def incomplete_candidate(row: Dict[str, Any], rng: random.Random, min_prefix_chars: int) -> Dict[str, Any] | None:
    turn = first_turn(row)
    if not turn:
        return None
    q = str(turn.get("question_text", ""))
    cut = choose_query_split(q, rng, min_prefix_chars)
    if cut is None:
        return None
    gn_between_sec = round(rng.uniform(0.5, 2.0), 3)
    part1 = q[:cut]
    part2 = q[cut:]
    return {
        "id": f"incomplete__{row['id']}__cut{cut}",
        "scenario": "incomplete_query_candidate",
        "source_id": row["id"],
        "source_group_id": source_group_id(row),
        "sysprompt": row.get("sysprompt", ""),
        "turns": row.get("turns", []),
        "incomplete_turn_id": turn.get("turn_id"),
        "incomplete_turn_index": len(row.get("turns", []) or []),
        "query_part1_text": part1,
        "query_part2_text": part2,
        "partial_question_text": part1,
        "full_question_text": q,
        "answer_text_if_complete": turn.get("answer_text", ""),
        "split": {
            "cut_char_index": cut,
            "unicode_codepoint_boundary": True,
        },
        "audio_plan": [
            "gn_before",
            "query_part1_audio",
            "gn_between_query_parts",
            "query_part2_audio",
            "gn_answer_region",
            "gn_after",
        ],
        "timeline_plan": [
            "gn_before -> IDLE",
            "query_part1_audio -> WAIT",
            "gn_between_query_parts -> WAIT",
            "query_part2_audio -> WAIT",
            "gn_answer_region -> ANSWER + answer text tokens + EOR",
            "gn_after -> IDLE",
        ],
        "gn_policy": {
            "between_query_parts_sec": gn_between_sec,
            "between_query_parts_range_sec": [0.5, 2.0],
        },
        "meta": row.get("meta", {}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build small candidate pools for normal QA, player interrupt, and incomplete-query scenarios.")
    ap.add_argument("--input", required=True, help="Selected turns JSONL from 00_select_duplex_turns.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit_each", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260715)
    ap.add_argument("--min_interrupt_answer_chars", type=int, default=8)
    ap.add_argument("--min_backchannel_answer_chars", type=int, default=8)
    ap.add_argument("--min_incomplete_prefix_chars", type=int, default=3)
    ap.add_argument("--min_question_chars", type=int, default=1)
    ap.add_argument("--max_question_chars", type=int, default=240)
    ap.add_argument("--max_answer_chars", type=int, default=360)
    ap.add_argument("--max_turns", type=int, default=8)
    ap.add_argument("--backchannel_manifest", default="", help="JSONL from 16_extract_magicdata_backchannels.py")
    ap.add_argument("--chunk_ms", type=int, default=180)
    ap.add_argument("--require_history", action="store_true")
    ap.add_argument("--require_sysprompt_history", action="store_true")
    ap.add_argument("--interrupt_pair_mode", choices=["cross_row", "same_row_previous"], default="cross_row")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    input_rows = read_jsonl(Path(args.input))
    rows = []
    normalization_stats: Dict[str, int] = {}
    for row in input_rows:
        normalized, row_stats = normalize_row(
            row,
            min_question_chars=args.min_question_chars,
            max_question_chars=args.max_question_chars,
            max_answer_chars=args.max_answer_chars,
            max_turns=args.max_turns,
        )
        for key, value in row_stats.items():
            normalization_stats[key] = normalization_stats.get(key, 0) + value
        if normalized is not None:
            rows.append(normalized)
    if args.require_history:
        rows = [r for r in rows if has_train_history(r)]
    if args.require_sysprompt_history:
        rows = [r for r in rows if has_sysprompt_history(r)]
    out_dir = Path(args.out_dir)

    normal_pool = [r for r in rows if selection_flag(r, "can_normal")]
    interrupt_source_rows = [r for r in rows if has_train_history(r)]
    interrupt_base_pool = [
        r for r in interrupt_source_rows
        if selection_flag(r, "can_interrupt_base")
        and len(str((first_turn(r) or {}).get("answer_text", ""))) >= args.min_interrupt_answer_chars
    ]
    same_row_interrupt_pool = [
        r for r in interrupt_source_rows
        if selection_flag(r, "can_interrupt_base")
        and len(str((r.get("turns") or [])[-2].get("answer_text", ""))) >= args.min_interrupt_answer_chars
    ]
    interrupt_donor_pool = [r for r in interrupt_source_rows if selection_flag(r, "can_interrupt_donor")]
    backchannel_pool = [
        r for r in rows
        if selection_flag(r, "can_normal")
        and len(str((first_turn(r) or {}).get("answer_text", ""))) >= args.min_backchannel_answer_chars
    ]
    incomplete_pool = [r for r in rows if selection_flag(r, "can_incomplete_query")]

    rng.shuffle(normal_pool)
    rng.shuffle(interrupt_base_pool)
    rng.shuffle(same_row_interrupt_pool)
    rng.shuffle(interrupt_donor_pool)
    rng.shuffle(backchannel_pool)
    rng.shuffle(incomplete_pool)

    limit_each = args.limit_each if args.limit_each > 0 else None
    normal_rows = [normal_candidate(r, args.chunk_ms) for r in normal_pool[:limit_each]]

    interrupt_rows = []
    if args.interrupt_pair_mode == "same_row_previous":
        for row in same_row_interrupt_pool:
            turns = row.get("turns") or []
            base_answer = str(turns[-2].get("answer_text", ""))
            max_prefix = min(len(base_answer) - 1, 12)
            prefix_chars = rng.randint(1, max(1, max_prefix))
            cand = same_row_interrupt_candidate(row, prefix_chars)
            if cand is not None:
                interrupt_rows.append(cand)
            if limit_each is not None and len(interrupt_rows) >= limit_each:
                break
    else:
        pair_count = min(limit_each or min(len(interrupt_base_pool), len(interrupt_donor_pool)), len(interrupt_base_pool), len(interrupt_donor_pool))
        for i in range(pair_count):
            base = interrupt_base_pool[i]
            donor = interrupt_donor_pool[-(i + 1)]
            if base["id"] == donor["id"] and len(interrupt_donor_pool) > 1:
                donor = interrupt_donor_pool[-(i + 2)]
            base_answer = str((first_turn(base) or {}).get("answer_text", ""))
            max_prefix = min(len(base_answer) - 1, 12)
            prefix_chars = rng.randint(1, max(1, max_prefix))
            interrupt_rows.append(interrupt_candidate(base, donor, prefix_chars))

    backchannel_clips = read_jsonl(Path(args.backchannel_manifest)) if args.backchannel_manifest else []
    rng.shuffle(backchannel_clips)
    backchannel_rows = []
    for idx, row in enumerate(backchannel_pool):
        if not backchannel_clips:
            break
        answer_text = str((first_turn(row) or {}).get("answer_text", ""))
        max_prefix = min(len(answer_text) - 1, 12)
        prefix_chars = rng.randint(1, max(1, max_prefix))
        backchannel = backchannel_clips[idx % len(backchannel_clips)]
        backchannel_rows.append(backchannel_candidate(row, backchannel, prefix_chars, args.chunk_ms))
        if limit_each is not None and len(backchannel_rows) >= limit_each:
            break

    incomplete_rows = []
    for row in incomplete_pool:
        cand = incomplete_candidate(row, rng, args.min_incomplete_prefix_chars)
        if cand is not None:
            incomplete_rows.append(cand)
        if limit_each is not None and len(incomplete_rows) >= limit_each:
            break

    counts = {
        "input": str(Path(args.input)),
        "out_dir": str(out_dir),
        "input_rows": len(input_rows),
        "rows": len(rows),
        "normalization": {
            "min_question_chars": args.min_question_chars,
            "max_question_chars": args.max_question_chars,
            "max_answer_chars": args.max_answer_chars,
            "max_turns": args.max_turns,
            "stats": normalization_stats,
        },
        "require_history": bool(args.require_history),
        "require_sysprompt_history": bool(args.require_sysprompt_history),
        "interrupt_pair_mode": args.interrupt_pair_mode,
        "interrupt_requires_train_history": True,
        "interrupt_source_rows": len(interrupt_source_rows),
        "normal_pool": len(normal_pool),
        "interrupt_base_pool": len(interrupt_base_pool),
        "same_row_interrupt_pool": len(same_row_interrupt_pool),
        "interrupt_donor_pool": len(interrupt_donor_pool),
        "backchannel_pool": len(backchannel_pool),
        "backchannel_manifest": args.backchannel_manifest,
        "backchannel_clips": len(backchannel_clips),
        "incomplete_pool": len(incomplete_pool),
        "normal_written": write_jsonl(out_dir / "normal_qa_candidates.jsonl", normal_rows),
        "interrupt_written": write_jsonl(out_dir / "player_interrupt_candidates.jsonl", interrupt_rows),
        "backchannel_written": write_jsonl(out_dir / "player_backchannel_candidates.jsonl", backchannel_rows),
        "incomplete_written": write_jsonl(out_dir / "incomplete_query_candidates.jsonl", incomplete_rows),
    }
    (out_dir / "candidate_pool_stats.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
