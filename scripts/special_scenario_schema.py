#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


SCHEMA_VERSION = "duplex_special_v1"
AI_INTERVENES_USER = "ai_intervenes_user"
PLAYER_COMPLETE = "player_complete"
SPECIAL_SCENARIOS = {AI_INTERVENES_USER, PLAYER_COMPLETE}
VIOLATION_CATEGORIES = {
    "abuse",
    "harassment",
    "hate",
    "sexual",
    "violence",
    "illegal",
    "self_harm",
    "other",
}
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class SpecialScenarioError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def fail(code: str, message: str) -> None:
    raise SpecialScenarioError(code, message)


def is_special_row(row: Dict[str, Any]) -> bool:
    return str(row.get("scenario") or "") in SPECIAL_SCENARIOS


def cjk_count(text: Any) -> int:
    return len(CJK_RE.findall(str(text or "")))


def special_turn(row: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    turns = row.get("turns")
    if not isinstance(turns, list) or not turns:
        fail("missing_turns", "special scenario requires non-empty turns")
    normalized_turns = [turn for turn in turns if isinstance(turn, dict)]
    if len(normalized_turns) != len(turns):
        fail("invalid_turn", "every turns item must be an object")
    event_indices = [
        idx for idx, turn in enumerate(normalized_turns)
        if isinstance(turn.get("event"), dict)
    ]
    if event_indices != [len(normalized_turns) - 1]:
        fail("invalid_event_position", "exactly the final current turn must contain event")
    current = normalized_turns[-1]
    if current.get("source") != "current":
        fail("invalid_current_source", "special turn must use source=current")
    expected_ids = list(range(1, len(normalized_turns) + 1))
    actual_ids = [turn.get("turn_id") for turn in normalized_turns]
    if actual_ids != expected_ids:
        fail("invalid_turn_ids", f"turn_id must be contiguous: {actual_ids!r}")
    if any(turn.get("source") != "history" for turn in normalized_turns[:-1]):
        fail("invalid_history_source", "all preceding turns must use source=history")
    return normalized_turns, current, dict(current["event"])


def validate_special_row(row: Dict[str, Any], *, min_intervene_suffix_cjk: int = 6) -> Dict[str, Any]:
    if row.get("schema_version") != SCHEMA_VERSION:
        fail("invalid_schema_version", f"schema_version must be {SCHEMA_VERSION}")
    scenario = str(row.get("scenario") or "")
    if scenario not in SPECIAL_SCENARIOS:
        fail("unsupported_special_scenario", f"unsupported scenario: {scenario!r}")
    if not str(row.get("id") or "").strip():
        fail("missing_id", "id is required")
    if not str(row.get("sysprompt") or "").strip():
        fail("missing_sysprompt", "sysprompt with the complete persona is required")

    turns, current, event = special_turn(row)
    question = str(current.get("question_text") or "")
    if not question:
        fail("missing_current_question", "current question_text is required")
    if current.get("needs_tts") is not True:
        fail("current_needs_tts", "special current turn must set needs_tts=true")

    if scenario == AI_INTERVENES_USER:
        if event.get("type") != "intervene":
            fail("invalid_intervene_event", "Intervene event.type must be intervene")
        before = str(event.get("user_text_until_trigger") or "")
        after = str(event.get("user_text_after_trigger") or "")
        if question != before + after:
            fail("intervene_text_mismatch", "question_text must equal trigger prefix plus suffix")
        if cjk_count(after) < min_intervene_suffix_cjk:
            fail(
                "intervene_suffix_too_short",
                f"user_text_after_trigger needs at least {min_intervene_suffix_cjk} CJK characters",
            )
        category = str(event.get("violation_category") or "")
        if category not in VIOLATION_CATEGORIES:
            fail("invalid_violation_category", f"invalid violation_category: {category!r}")
        if not str(current.get("answer_text") or "").strip():
            fail("missing_intervene_answer", "Intervene answer_text is required")
        if current.get("train_answer") is not True:
            fail("intervene_train_answer", "Intervene must set train_answer=true")
    else:
        if event.get("type") != "complete":
            fail("invalid_complete_event", "Complete event.type must be complete")
        completion_type = str(event.get("completion_type") or "")
        response_mode = str(event.get("response_mode") or "")
        answer = current.get("answer_text")
        if (completion_type, response_mode) == ("normal_closing", "acknowledge"):
            if not str(answer or "").strip():
                fail("missing_complete_answer", "acknowledge Complete requires answer_text")
            if current.get("train_answer") is not True:
                fail("complete_train_answer", "acknowledge Complete must set train_answer=true")
            if not str(current.get("answer_speaker") or "").strip():
                fail("missing_answer_speaker", "acknowledge Complete requires answer_speaker")
        elif (completion_type, response_mode) == ("force_stop", "silent"):
            if answer not in (None, ""):
                fail("silent_complete_answer", "silent Complete requires answer_text=null")
            if current.get("train_answer") is not False:
                fail("silent_complete_train_answer", "silent Complete must set train_answer=false")
            if current.get("answer_speaker") not in (None, ""):
                fail("silent_complete_speaker", "silent Complete requires answer_speaker=null")
        else:
            fail(
                "invalid_complete_mode",
                f"invalid Complete mode: completion_type={completion_type!r}, response_mode={response_mode!r}",
            )

    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    if "turn_count" in meta and int(meta["turn_count"]) != len(turns):
        fail("meta_turn_count", "meta.turn_count does not match turns")
    if "history_turn_count" in meta and int(meta["history_turn_count"]) != len(turns) - 1:
        fail("meta_history_turn_count", "meta.history_turn_count does not match turns")
    return {
        "scenario": scenario,
        "turns": turns,
        "current": current,
        "event": event,
        "current_index": len(turns) - 1,
    }


def normalize_special_row(
    row: Dict[str, Any],
    *,
    min_question_chars: int,
    max_question_chars: int,
    max_answer_chars: int,
    max_turns: int,
    min_intervene_suffix_cjk: int = 6,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    turns = [dict(turn) for turn in (row.get("turns") or []) if isinstance(turn, dict)]
    stats = {"history_turns_dropped": 0, "leading_turns_truncated": 0}
    if not turns:
        fail("missing_turns", "special scenario requires non-empty turns")
    current = turns[-1]
    question = str(current.get("question_text") or "")
    if len(question) < min_question_chars:
        fail("current_question_too_short", "current question is too short")
    if max_question_chars > 0 and len(question) > max_question_chars:
        fail("current_question_too_long", "current question is too long")
    answer = current.get("answer_text")
    if answer is not None and max_answer_chars > 0 and len(str(answer)) > max_answer_chars:
        fail("current_answer_too_long", "current answer is too long")

    kept_history: List[Dict[str, Any]] = []
    for turn in turns[:-1]:
        history_question = str(turn.get("question_text") or "")
        history_answer = str(turn.get("answer_text") or "")
        valid = bool(history_question and history_answer)
        valid = valid and len(history_question) >= min_question_chars
        valid = valid and (max_question_chars <= 0 or len(history_question) <= max_question_chars)
        valid = valid and (max_answer_chars <= 0 or len(history_answer) <= max_answer_chars)
        if valid:
            kept_history.append(turn)
        else:
            stats["history_turns_dropped"] += 1

    normalized_turns = kept_history + [current]
    if max_turns > 0 and len(normalized_turns) > max_turns:
        stats["leading_turns_truncated"] = len(normalized_turns) - max_turns
        normalized_turns = normalized_turns[-max_turns:]
    for idx, turn in enumerate(normalized_turns, start=1):
        turn["turn_id"] = idx
        turn["source"] = "current" if idx == len(normalized_turns) else "history"

    out = dict(row)
    out["turns"] = normalized_turns
    out["question_text"] = str(current.get("question_text") or "")
    out["answer_text"] = current.get("answer_text")
    out["special_turn_id"] = len(normalized_turns)
    out["special_turn_index"] = len(normalized_turns)
    meta = dict(row.get("meta") or {})
    meta["turn_count"] = len(normalized_turns)
    meta["history_turn_count"] = len(normalized_turns) - 1
    out["meta"] = meta
    validate_special_row(out, min_intervene_suffix_cjk=min_intervene_suffix_cjk)
    return out, stats
