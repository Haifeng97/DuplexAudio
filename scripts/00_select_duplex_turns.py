#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


FULLWIDTH_USER = "<｜User｜>"
FULLWIDTH_ASSISTANT = "<｜Assistant｜>"
FULLWIDTH_EOS = "<｜end▁of▁sentence｜>"


def norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_dialog_text(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)
    s = s.replace(FULLWIDTH_EOS, " ")
    s = re.sub(r"<\|im_start\|>\s*(?:system|user|assistant)?", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|im_end\|>|<\|begin_of_text\|>|<\|end_of_text\|>|<\|eot_id\|>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|start_header_id\|>\s*(?:system|user|assistant)\s*<\|end_header_id\|>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"</?\s*(?:assistant|user|system|human|bot)\b[^>]*>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\[/?INST\]|<<SYS>>|<</SYS>>", " ", s, flags=re.IGNORECASE)
    s = s.replace(FULLWIDTH_USER, " ").replace(FULLWIDTH_ASSISTANT, " ")
    return norm_spaces(s)


def clean_sysprompt(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)

    replacements = [
        (FULLWIDTH_USER, "\n玩家："),
        (FULLWIDTH_ASSISTANT, "\n吉莉："),
        (FULLWIDTH_EOS, "\n"),
        ("<|im_start|>user", "\n玩家："),
        ("<|im_start|>assistant", "\n吉莉："),
        ("<|im_start|>system", "\n"),
        ("<|im_end|>", "\n"),
    ]
    for old, new in replacements:
        s = s.replace(old, new)

    s = re.sub(r"</?\s*user\b[^>]*>", "\n玩家：", s, flags=re.IGNORECASE)
    s = re.sub(r"</?\s*assistant\b[^>]*>", "\n吉莉：", s, flags=re.IGNORECASE)
    s = re.sub(r"</?\s*system\b[^>]*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|[^>]+?\|>", " ", s)
    s = re.sub(r"\[/?INST\]|<<SYS>>|<</SYS>>", " ", s, flags=re.IGNORECASE)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def safe_id(raw: Any, idx: int) -> str:
    base = str(raw or f"duplex_{idx:08d}")
    base = re.sub(r"[^0-9A-Za-z_.-]+", "_", base).strip("._-")
    return base[:180] or f"duplex_{idx:08d}"


def detect_input_kind(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            ch = f.read(1)
            if not ch:
                raise ValueError(f"empty input: {path}")
            if ch.isspace():
                continue
            if ch == "[":
                return "json_array"
            if ch == "{":
                return "jsonl"
            raise ValueError(f"unsupported input, first non-space char={ch!r}")


def iter_json_array_objects(path: Path) -> Iterator[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch.isspace():
                continue
            if ch != "[":
                raise ValueError(f"expected top-level JSON array, got {ch!r}")
            break

        buf = ""
        pos = 0
        while True:
            while True:
                if pos >= len(buf):
                    more = f.read(chunk_size)
                    if not more:
                        return
                    buf = more
                    pos = 0
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf) and buf[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buf) and buf[pos] == "]":
                    return
                if pos < len(buf):
                    break

            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    if isinstance(obj, dict):
                        yield obj
                    pos = end
                    if pos > chunk_size:
                        buf = buf[pos:]
                        pos = 0
                    break
                except json.JSONDecodeError:
                    more = f.read(chunk_size)
                    if not more:
                        raise
                    if pos:
                        buf = buf[pos:]
                        pos = 0
                    buf += more


def iter_jsonl_objects(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj


def iter_input_objects(path: Path) -> Iterator[Dict[str, Any]]:
    if detect_input_kind(path) == "json_array":
        yield from iter_json_array_objects(path)
    else:
        yield from iter_jsonl_objects(path)


def first_nonempty(*values: Any) -> str:
    for value in values:
        cleaned = clean_dialog_text(value)
        if cleaned:
            return cleaned
    return ""


def query_from_obj(obj: Dict[str, Any]) -> Tuple[str, str]:
    q = first_nonempty(obj.get("input"))
    if q:
        return q, "input"
    q = first_nonempty(obj.get("instruction"))
    if q:
        return q, "instruction"
    queries = obj.get("queries_with_punc")
    if isinstance(queries, list):
        for item in queries:
            q = clean_dialog_text(item)
            if q:
                return q, "queries_with_punc"
    q = first_nonempty(obj.get("question"), obj.get("query"), obj.get("text_query"))
    if q:
        return q, "fallback"
    return "", "missing"


def answer_from_obj(obj: Dict[str, Any]) -> str:
    return first_nonempty(obj.get("output"), obj.get("answer"), obj.get("response"), obj.get("target"), obj.get("target_text"))


def history_pairs(history: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not isinstance(history, list):
        return pairs
    for item in history:
        user = ""
        assistant = ""
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user = clean_dialog_text(item[0])
            assistant = clean_dialog_text(item[1])
        elif isinstance(item, dict):
            user = first_nonempty(item.get("user"), item.get("input"), item.get("question"))
            assistant = first_nonempty(item.get("assistant"), item.get("output"), item.get("answer"), item.get("response"))
        if user and assistant:
            pairs.append((user, assistant))
    return pairs


def category_values(obj: Dict[str, Any]) -> Dict[str, str]:
    meta_info = obj.get("meta_info") if isinstance(obj.get("meta_info"), dict) else {}
    return {
        "meta_category": str(meta_info.get("category") or ""),
        "meta_data_type": str(meta_info.get("data_type") or ""),
        "top_category": str(obj.get("category") or ""),
        "raw_category": str(obj.get("raw_category") or ""),
        "chat_type": str(obj.get("chat_type") or ""),
        "role_name": str(meta_info.get("role_name") or ""),
        "usage_scene": str(meta_info.get("usage_scene") or ""),
    }


def matches_category(obj: Dict[str, Any], mode: str, contains: str) -> bool:
    vals = category_values(obj)
    meta_category = vals["meta_category"]
    meta_data_type = vals["meta_data_type"]
    combined = "\n".join(vals.values())
    if mode == "all":
        return True
    if mode == "game_state":
        return meta_category.startswith("以游戏为中心-基于状态的闲聊")
    if mode == "state":
        return "状态" in meta_category or "状态" in meta_data_type or "state" in combined.lower()
    if mode == "contains":
        return bool(contains and contains in combined)
    raise ValueError(f"unknown category mode: {mode}")


def build_sysprompt(obj: Dict[str, Any], include_reference: bool) -> Tuple[str, bool]:
    # The source system prompt is expected to be complete. meta_info.reference_text
    # is kept in meta only and must not be appended here, otherwise state rows
    # duplicate 【相关知识】/【游戏对局状态】 blocks.
    system = clean_sysprompt(obj.get("system", ""))
    return system, False


def build_example(
    obj: Dict[str, Any],
    idx: int,
    *,
    include_reference: bool,
    category_mode: str,
    min_question_chars: int,
    max_question_chars: int,
    max_answer_chars: int,
    max_turns: int,
) -> Tuple[Dict[str, Any] | None, str]:
    current_q, q_source = query_from_obj(obj)
    current_a = answer_from_obj(obj)
    if not current_q:
        return None, "missing_query"
    if not current_a:
        return None, "missing_answer"
    if len(current_q) < min_question_chars:
        return None, "question_too_short"
    if max_question_chars > 0 and len(current_q) > max_question_chars:
        return None, "question_too_long"
    if max_answer_chars > 0 and len(current_a) > max_answer_chars:
        return None, "answer_too_long"

    pairs = history_pairs(obj.get("history"))
    turns: List[Dict[str, Any]] = []
    for user, assistant in pairs:
        if max_question_chars > 0 and len(user) > max_question_chars:
            continue
        if max_answer_chars > 0 and len(assistant) > max_answer_chars:
            continue
        turns.append(
            {
                "turn_id": len(turns) + 1,
                "source": "history",
                "question_text": user,
                "answer_text": assistant,
                "needs_tts": True,
                "train_answer": True,
            }
        )
    turns.append(
        {
            "turn_id": len(turns) + 1,
            "source": "current",
            "question_text": current_q,
            "answer_text": current_a,
            "needs_tts": True,
            "train_answer": True,
        }
    )
    if max_turns > 0 and len(turns) > max_turns:
        turns = turns[-max_turns:]
        for i, turn in enumerate(turns, start=1):
            turn["turn_id"] = i

    vals = category_values(obj)
    meta_info = obj.get("meta_info") if isinstance(obj.get("meta_info"), dict) else {}
    sysprompt, reference_text_appended = build_sysprompt(obj, include_reference)
    sid = safe_id(obj.get("hash_str") or obj.get("hash_key") or obj.get("id"), idx)
    answer_char_max = max(len(t["answer_text"]) for t in turns)
    question_char_max = max(len(t["question_text"]) for t in turns)
    train_turns = len(turns)
    example = {
        "id": sid,
        "source": "cgame_duplex_turns",
        "selection": {
            "category_mode": category_mode,
            "can_normal": train_turns >= 1,
            "can_interrupt_base": answer_char_max >= 8,
            "can_interrupt_donor": train_turns >= 1,
            "can_incomplete_query": question_char_max >= 8,
        },
        "sysprompt": sysprompt,
        "turns": turns,
        "question_text": current_q,
        "answer_text": current_a,
        "text_query": current_q,
        "asr_text": current_q,
        "text": current_a,
        "target_text": current_a,
        "meta": {
            "source_index": idx,
            "hash_str": obj.get("hash_str"),
            "hash_key": obj.get("hash_key"),
            "query_source": q_source,
            "turn_count": train_turns,
            "history_turn_count": max(0, train_turns - 1),
            "meta_category": vals["meta_category"],
            "meta_data_type": vals["meta_data_type"],
            "top_category": vals["top_category"],
            "raw_category": vals["raw_category"],
            "chat_type": vals["chat_type"],
            "role_name": vals["role_name"],
            "usage_scene": vals["usage_scene"],
            "reference_text_present": bool(meta_info.get("reference_text")),
            "reference_text_appended": bool(reference_text_appended),
            "config": obj.get("config"),
            "user_profile": obj.get("user_profile"),
        },
    }
    return example, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Select usable chat/state rows and normalize them into sysprompt + trainable turns.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats_out", default="")
    ap.add_argument("--category_mode", choices=["game_state", "state", "all", "contains"], default="game_state")
    ap.add_argument("--category_contains", default="")
    ap.add_argument("--include_reference", action="store_true", help="Deprecated no-op; sysprompt uses system only")
    ap.add_argument("--min_question_chars", type=int, default=1)
    ap.add_argument("--max_question_chars", type=int, default=240)
    ap.add_argument("--max_answer_chars", type=int, default=360)
    ap.add_argument("--max_turns", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=20000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.stats_out) if args.stats_out else out_path.with_suffix(out_path.suffix + ".stats.json")

    counters = Counter()
    category_counts = Counter()
    data_type_counts = Counter()
    query_source_counts = Counter()
    turn_count_hist = Counter()
    seen_ids = set()
    duplicate_ids = 0
    read_n = wrote_n = 0

    with out_path.open("w", encoding="utf-8") as fw:
        for read_n, obj in enumerate(iter_input_objects(in_path), start=1):
            counters["read"] += 1
            if not matches_category(obj, args.category_mode, args.category_contains):
                counters["skip_category"] += 1
                continue
            ex, reason = build_example(
                obj,
                read_n,
                include_reference=args.include_reference,
                category_mode=args.category_mode,
                min_question_chars=args.min_question_chars,
                max_question_chars=args.max_question_chars,
                max_answer_chars=args.max_answer_chars,
                max_turns=args.max_turns,
            )
            if ex is None:
                counters[f"skip_{reason}"] += 1
                continue

            base_id = ex["id"]
            sid = base_id
            suffix = 1
            while sid in seen_ids:
                duplicate_ids += 1
                suffix += 1
                sid = f"{base_id}_{suffix}"
            seen_ids.add(sid)
            ex["id"] = sid

            meta = ex["meta"]
            category_counts[meta["meta_category"]] += 1
            data_type_counts[meta["meta_data_type"]] += 1
            query_source_counts[meta["query_source"]] += 1
            turn_count_hist[str(meta["turn_count"])] += 1
            counters["can_normal"] += int(bool(ex["selection"]["can_normal"]))
            counters["can_interrupt_base"] += int(bool(ex["selection"]["can_interrupt_base"]))
            counters["can_interrupt_donor"] += int(bool(ex["selection"]["can_interrupt_donor"]))
            counters["can_incomplete_query"] += int(bool(ex["selection"]["can_incomplete_query"]))
            counters["with_sysprompt"] += int(bool(ex["sysprompt"]))
            counters["with_reference_text"] += int(bool(meta["reference_text_present"]))
            counters["with_reference_text_appended"] += int(bool(meta["reference_text_appended"]))

            fw.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
            wrote_n += 1
            counters["wrote"] = wrote_n
            if args.limit and wrote_n >= args.limit:
                break
            if args.progress_every > 0 and read_n % args.progress_every == 0:
                print(json.dumps({"read": read_n, "wrote": wrote_n, "counters": dict(counters)}, ensure_ascii=False), flush=True)

    stats = {
        "input": str(in_path),
        "out": str(out_path),
        "read": read_n,
        "wrote": wrote_n,
        "duplicate_ids": duplicate_ids,
        "args": vars(args),
        "counters": dict(counters),
        "query_source_counts": dict(query_source_counts.most_common()),
        "turn_count_hist": dict(turn_count_hist.most_common()),
        "top_meta_categories": dict(category_counts.most_common(80)),
        "top_meta_data_types": dict(data_type_counts.most_common(80)),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
