from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash
from .llm import parse_json_response, response_text
from .text import effective_char_count


INCOMPLETE_SCENARIOS = {"incomplete_query", "incomplete_query_clarification"}


def _current_turn(row: Dict[str, Any]) -> Dict[str, Any]:
    turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    if not turns:
        raise ValueError(f"row {row.get('id')} has no turns")
    return turns[-1]


def _request_id(job: str, sample_id: str) -> str:
    return f"{job}__{stable_hash({'job': job, 'sample_id': sample_id}, length=24)}"


def export_split_requests(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    source = run_dir / "02_plan" / "customized_selected.jsonl"
    out_dir = run_dir / "03_llm" / "incomplete_split"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "requests.jsonl"
    llm = dict(config.get("llm") or {})
    candidate_count = int(llm.get("candidate_count", 5))
    min_chars = int(llm.get("min_incomplete_prefix_effective_chars", 3))
    max_chars = int(llm.get("max_incomplete_prefix_effective_chars", 14))
    written = skipped = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            if row.get("primary_scenario") not in INCOMPLETE_SCENARIOS:
                continue
            turn = _current_turn(row)
            question = str(turn.get("question_text") or "")
            if effective_char_count(question) <= min_chars:
                skipped += 1
                continue
            prompt = (
                "为一条中文语音 query 找出最自然的未说完位置。输出严格 JSON，不要 markdown。\n"
                f"请给出 {candidate_count} 个不同候选，每个候选包含 cut_char_index 和 reason。\n"
                "cut_char_index 是 Python 字符串切片索引；原文必须满足 text[:cut]+text[cut:]==text，禁止改字。\n"
                f"前半句去掉标点和空白后必须为 {min_chars}-{max_chars} 个字符。\n"
                "候选应在语法或语义上明显悬而未决，让听者自然期待后半句；不要选已经可以独立回答的位置。\n"
                f"query：{question}\n"
                f"输出格式：{{\"candidates\":[{{\"cut_char_index\":整数,\"reason\":\"简短理由\"}}]}}"
            )
            request = {
                "request_id": _request_id("incomplete_candidates", str(row["id"])),
                "job_type": "incomplete_candidates",
                "sample_id": row["id"],
                "source_version": row.get("source_version"),
                "primary_scenario": row.get("primary_scenario"),
                "input": {"question_text": question, "candidate_count": candidate_count, "min_effective_chars": min_chars, "max_effective_chars": max_chars},
                "messages": [{"role": "user", "content": prompt}],
                "response_text": "",
            }
            handle.write(canonical_json(request) + "\n")
            written += 1
    result = {"source": str(source), "output": str(output), "written": written, "skipped": skipped}
    atomic_write_json(out_dir / "stats.json", result)
    return result


def _load_request_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["request_id"]): row for row in iter_jsonl(path)}


def _response_key(row: Dict[str, Any]) -> str:
    return str(row.get("request_id") or row.get("id") or "")


def valid_cut(question: str, cut: Any, min_chars: int, max_chars: int) -> Optional[int]:
    if isinstance(cut, bool):
        return None
    try:
        value = int(cut)
    except (TypeError, ValueError):
        return None
    if value <= 0 or value >= len(question):
        return None
    prefix_count = effective_char_count(question[:value])
    if not min_chars <= prefix_count <= max_chars:
        return None
    if effective_char_count(question[value:]) == 0:
        return None
    return value


def export_rank_requests(config: Dict[str, Any], run_dir: Path, filled_path: Path) -> Dict[str, Any]:
    split_dir = run_dir / "03_llm" / "incomplete_split"
    request_index = _load_request_index(split_dir / "requests.jsonl")
    out_dir = run_dir / "03_llm" / "incomplete_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "requests.jsonl"
    rejected = out_dir / "rejected.jsonl"
    if rejected.exists():
        rejected.unlink()
    llm = dict(config.get("llm") or {})
    min_chars = int(llm.get("min_incomplete_prefix_effective_chars", 3))
    max_chars = int(llm.get("max_incomplete_prefix_effective_chars", 14))
    counts: Counter = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for response in iter_jsonl(filled_path):
            original = request_index.get(_response_key(response))
            if original is None:
                counts["unknown_request"] += 1
                continue
            try:
                parsed = parse_json_response(response_text(response))
            except Exception as exc:
                with rejected.open("a", encoding="utf-8") as reject_handle:
                    reject_handle.write(canonical_json({"request_id": original["request_id"], "reason": "invalid_json", "error": repr(exc)}) + "\n")
                counts["invalid_json"] += 1
                continue
            raw_candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []
            question = str(original["input"]["question_text"])
            candidates: List[Dict[str, Any]] = []
            seen = set()
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                cut = valid_cut(question, item.get("cut_char_index"), min_chars, max_chars)
                if cut is None or cut in seen:
                    continue
                seen.add(cut)
                candidates.append({
                    "cut_char_index": cut,
                    "part1": question[:cut],
                    "part2": question[cut:],
                    "reason": str(item.get("reason") or ""),
                })
            if not candidates:
                with rejected.open("a", encoding="utf-8") as reject_handle:
                    reject_handle.write(canonical_json({"request_id": original["request_id"], "sample_id": original["sample_id"], "reason": "no_valid_candidates"}) + "\n")
                counts["no_valid_candidates"] += 1
                continue
            prompt = (
                "从候选截断中选择最像真实玩家说到一半停下的位置。输出严格 JSON，不要 markdown。\n"
                "优先标准：前半句语法/语义明显未完成；后半句能自然补全；前半句不能已经是可独立回答的问题。\n"
                "对每个候选给 0-100 分，并给出 selected_cut_char_index。\n"
                f"完整 query：{question}\n候选：{json.dumps(candidates, ensure_ascii=False)}\n"
                "格式：{\"selected_cut_char_index\":整数,\"scores\":[{\"cut_char_index\":整数,\"score\":整数,\"reason\":\"理由\"}]}"
            )
            rank_request = {
                "request_id": _request_id("incomplete_rank", str(original["sample_id"])),
                "parent_request_id": original["request_id"],
                "job_type": "incomplete_rank",
                "sample_id": original["sample_id"],
                "primary_scenario": original["primary_scenario"],
                "input": {"question_text": question, "candidates": candidates},
                "messages": [{"role": "user", "content": prompt}],
                "response_text": "",
            }
            handle.write(canonical_json(rank_request) + "\n")
            counts["written"] += 1
    result = {"input": str(filled_path), "output": str(output), "rejected": str(rejected), "counts": dict(counts)}
    atomic_write_json(out_dir / "stats.json", result)
    return result


def apply_rank_results(config: Dict[str, Any], run_dir: Path, filled_path: Path) -> Dict[str, Any]:
    rank_dir = run_dir / "03_llm" / "incomplete_rank"
    requests = _load_request_index(rank_dir / "requests.jsonl")
    output = rank_dir / "selected_splits.jsonl"
    rejected = rank_dir / "apply_rejected.jsonl"
    if rejected.exists():
        rejected.unlink()
    counts: Counter = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for response in iter_jsonl(filled_path):
            request = requests.get(_response_key(response))
            if request is None:
                counts["unknown_request"] += 1
                continue
            try:
                parsed = parse_json_response(response_text(response))
                selected = int(parsed["selected_cut_char_index"])
            except Exception as exc:
                with rejected.open("a", encoding="utf-8") as reject_handle:
                    reject_handle.write(canonical_json({"sample_id": request["sample_id"], "reason": "invalid_rank_response", "error": repr(exc)}) + "\n")
                counts["invalid_rank_response"] += 1
                continue
            candidate = next((item for item in request["input"]["candidates"] if int(item["cut_char_index"]) == selected), None)
            if candidate is None:
                with rejected.open("a", encoding="utf-8") as reject_handle:
                    reject_handle.write(canonical_json({"sample_id": request["sample_id"], "reason": "selected_cut_not_candidate", "selected": selected}) + "\n")
                counts["selected_cut_not_candidate"] += 1
                continue
            handle.write(canonical_json({
                "sample_id": request["sample_id"],
                "primary_scenario": request["primary_scenario"],
                "question_text": request["input"]["question_text"],
                "cut_char_index": selected,
                "query_part1_text": candidate["part1"],
                "query_part2_text": candidate["part2"],
                "rank_response": parsed,
            }) + "\n")
            counts["selected"] += 1
    result = {"input": str(filled_path), "output": str(output), "rejected": str(rejected), "counts": dict(counts)}
    atomic_write_json(rank_dir / "apply_stats.json", result)
    return result


def export_clarification_requests(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    selected = {
        str(row["sample_id"]): row
        for row in iter_jsonl(run_dir / "03_llm" / "incomplete_rank" / "selected_splits.jsonl")
        if row.get("primary_scenario") == "incomplete_query_clarification"
    }
    source = run_dir / "02_plan" / "customized_selected.jsonl"
    out_dir = run_dir / "03_llm" / "clarification"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "requests.jsonl"
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            split = selected.get(str(row["id"]))
            if split is None:
                continue
            turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
            context = turns[:-1]
            history = "\n".join(f"玩家：{turn.get('question_text', '')}\nAI：{turn.get('answer_text', '')}" for turn in context) or "无"
            partial = split["query_part1_text"]
            prompt = (
                "玩家说了半句话后停住超过3秒。请生成角色自然、简短的澄清追问。\n"
                "只能追问/确认/轻微催促，不得猜测并直接回答完整意图；answer_text不超过20个汉字。\n"
                "action_expression只描述角色当下可见的表情、眼神、头部或上肢动作，不超过80字。\n"
                f"角色设定：{row.get('sysprompt', '')}\n对话历史：{history}\n玩家半句：{partial}\n"
                "只输出JSON：{\"answer_text\":\"澄清回复\",\"action_expression\":\"动作\"}"
            )
            request = {
                "request_id": _request_id("clarification", str(row["id"])),
                "job_type": "clarification",
                "sample_id": row["id"],
                "input": {"partial_question_text": partial, "full_question_text": split["question_text"], "cut_char_index": split["cut_char_index"]},
                "messages": [{"role": "user", "content": prompt}],
                "response_text": "",
            }
            handle.write(canonical_json(request) + "\n")
            written += 1
    result = {"source": str(source), "output": str(output), "written": written}
    atomic_write_json(out_dir / "stats.json", result)
    return result
