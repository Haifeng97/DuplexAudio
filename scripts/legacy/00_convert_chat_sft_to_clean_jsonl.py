#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


ROLE_NAMES = {
    "user": "玩家",
    "human": "玩家",
    "player": "玩家",
    "assistant": "吉莉",
    "gilly": "吉莉",
    "bot": "吉莉",
}


SPECIAL_TOKEN_PATTERNS = [
    r"<\|im_start\|>\s*(?:system|user|assistant)?",
    r"<\|im_end\|>",
    r"<\|begin_of_text\|>",
    r"<\|end_of_text\|>",
    r"<\|start_header_id\|>\s*(?:system|user|assistant)\s*<\|end_header_id\|>",
    r"<\|eot_id\|>",
    r"</?\s*(?:assistant|user|system|human|bot)\b[^>]*>",
    r"\[/?INST\]",
    r"<<SYS>>",
    r"<</SYS>>",
]


def norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_special_tokens(text: Any, *, keep_newlines: bool = False, strip_think: bool = True) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    if strip_think:
        s = re.sub(r"<think\b[^>]*>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL)
    for pat in SPECIAL_TOKEN_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    if keep_newlines:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
        return "\n".join(line for line in lines if line).strip()
    return norm_spaces(s)


def safe_id(raw: Any, idx: int) -> str:
    base = str(raw or f"chat_{idx:08d}")
    base = re.sub(r"[^0-9A-Za-z_.-]+", "_", base).strip("._-")
    return base[:160] or f"chat_{idx:08d}"


def iter_json_array_objects(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield objects from a pretty-printed top-level JSON array without loading it."""
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
                    buf = ""
                    pos = 0
                    buf += more
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


def iter_input_objects(path: Path) -> Iterator[Dict[str, Any]]:
    kind = detect_input_kind(path)
    if kind == "json_array":
        yield from iter_json_array_objects(path)
    else:
        yield from iter_jsonl_objects(path)


def history_pairs(history: Any, *, strip_think: bool) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not isinstance(history, list):
        return pairs

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user = strip_special_tokens(item[0], strip_think=strip_think)
            assistant = strip_special_tokens(item[1], strip_think=strip_think)
            if user or assistant:
                pairs.append((user, assistant))
        elif isinstance(item, dict):
            user = strip_special_tokens(
                item.get("user") or item.get("input") or item.get("question"),
                strip_think=strip_think,
            )
            assistant = strip_special_tokens(
                item.get("assistant") or item.get("output") or item.get("answer"),
                strip_think=strip_think,
            )
            if user or assistant:
                pairs.append((user, assistant))
    return pairs


def build_history_text(pairs: Iterable[Tuple[str, str]]) -> str:
    lines: List[str] = ["【对话历史】（不参与训练的部分）"]
    n = 0
    for user, assistant in pairs:
        if user:
            lines.append(f"玩家：{user}")
        if assistant:
            lines.append(f"吉莉：{assistant}")
        n += 1
    return "\n".join(lines) if n else ""


def build_messages(system: str, pairs: List[Tuple[str, str]], current_input: str, current_output: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system, "train": False})
    for user, assistant in pairs:
        if user:
            messages.append({"role": "user", "name": ROLE_NAMES["user"], "content": user, "train": False})
        if assistant:
            messages.append({"role": "assistant", "name": ROLE_NAMES["assistant"], "content": assistant, "train": True})
    if current_input:
        messages.append({"role": "user", "name": ROLE_NAMES["user"], "content": current_input, "train": False})
    if current_output:
        messages.append({"role": "assistant", "name": ROLE_NAMES["assistant"], "content": current_output, "train": True})
    return messages


def build_prompt_text(system: str, history_text: str, current_input: str) -> str:
    parts: List[str] = []
    if system:
        parts.append(system)
    if history_text:
        parts.append(history_text)
    if current_input:
        parts.append("【当前输入】\n玩家：" + current_input)
    return "\n\n".join(parts).strip()


def convert_obj(obj: Dict[str, Any], idx: int, *, strip_think: bool, keep_judge: bool) -> Dict[str, Any] | None:
    system = strip_special_tokens(obj.get("system", ""), keep_newlines=True, strip_think=strip_think)
    current_input = strip_special_tokens(
        obj.get("input") or obj.get("instruction") or obj.get("question") or obj.get("query"),
        strip_think=strip_think,
    )
    current_output = strip_special_tokens(
        obj.get("output") or obj.get("answer") or obj.get("response") or obj.get("target"),
        strip_think=strip_think,
    )
    if not current_input or not current_output:
        return None

    pairs = history_pairs(obj.get("history"), strip_think=strip_think)
    history_text = build_history_text(pairs)
    prompt_text = build_prompt_text(system, history_text, current_input)
    sid = safe_id(obj.get("hash_str") or obj.get("id"), idx)

    out: Dict[str, Any] = {
        "id": sid,
        "source": "cgame_aipartner_sft_clean",
        "system": system,
        "history": [[u, a] for u, a in pairs],
        "history_text": history_text,
        "input": current_input,
        "output": current_output,
        "question": current_input,
        "answer": current_output,
        "question_text": current_input,
        "answer_text": current_output,
        "text_query": current_input,
        "target_text": current_output,
        "text": current_output,
        "asr_text": current_input,
        "prompt_text": prompt_text,
        "messages": build_messages(system, pairs, current_input, current_output),
        "meta": {
            "source_index": idx,
            "hash_str": obj.get("hash_str"),
            "is_clean": obj.get("is_clean"),
            "user_profile": obj.get("user_profile"),
            "config": obj.get("config"),
        },
    }
    if keep_judge:
        out["meta"]["judge_result"] = obj.get("judge_result")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stream-convert chat SFT JSON/JSONL into cleaned generic JSONL for duplex data generation."
    )
    ap.add_argument("--input", required=True, help="Input JSON array or JSONL file")
    ap.add_argument("--out", required=True, help="Output cleaned JSONL")
    ap.add_argument("--limit", type=int, default=0, help="Convert at most N valid rows")
    ap.add_argument("--keep_think", action="store_true", help="Keep <think>...</think> blocks instead of stripping them")
    ap.add_argument("--keep_judge", action="store_true", help="Keep judge_result in meta")
    ap.add_argument("--progress_every", type=int, default=10000)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    read_n = wrote_n = skipped_n = duplicate_n = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for read_n, obj in enumerate(iter_input_objects(in_path), start=1):
            ex = convert_obj(obj, read_n, strip_think=not args.keep_think, keep_judge=args.keep_judge)
            if ex is None:
                skipped_n += 1
                continue
            base_id = ex["id"]
            sid = base_id
            suffix = 1
            while sid in seen:
                duplicate_n += 1
                suffix += 1
                sid = f"{base_id}_{suffix}"
            seen.add(sid)
            ex["id"] = sid
            fw.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")
            wrote_n += 1
            if args.limit and wrote_n >= args.limit:
                break
            if args.progress_every > 0 and read_n % args.progress_every == 0:
                print(json.dumps({"read": read_n, "wrote": wrote_n, "skipped": skipped_n}, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "input": str(in_path),
                "out": str(out_path),
                "read": read_n,
                "wrote": wrote_n,
                "skipped": skipped_n,
                "duplicate_ids": duplicate_n,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
