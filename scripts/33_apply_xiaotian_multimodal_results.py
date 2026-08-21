#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_pipeline.llm import parse_json_response, response_text


SCHEMA_VERSION = "duplex_xiaotian_forward_v1"
ENRICHMENT_SCHEMA = "duplex_multimodal_enrichment_v1"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_action(value: Any) -> str:
    text = clean_text(value)
    while len(text) >= 2 and ((text[0], text[-1]) in {("（", "）"), ("(", ")")}):
        text = text[1:-1].strip()
    return text


def validate_response(request: Dict[str, Any], result: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    try:
        parsed = parse_json_response(response_text(result))
    except Exception as exc:
        return [], [f"invalid_json:{exc!r}"]
    turns = parsed.get("turns") if isinstance(parsed, dict) else None
    expected = int(request["turn_count"])
    if not isinstance(turns, list) or len(turns) != expected:
        return [], [f"turn_count_mismatch:expected={expected},actual={len(turns) if isinstance(turns, list) else 'invalid'}"]
    cleaned: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            errors.append(f"turn{index}:not_object")
            continue
        try:
            turn_id = int(turn.get("turn_id"))
        except (TypeError, ValueError):
            turn_id = 0
        question = clean_text(turn.get("question_text"))
        answer = clean_text(turn.get("answer_text"))
        action = clean_action(turn.get("action_expression"))
        if turn_id != index:
            errors.append(f"turn{index}:invalid_turn_id={turn_id}")
        if not 2 <= len(question) <= 100:
            errors.append(f"turn{index}:question_chars={len(question)}")
        if not 1 <= len(answer) <= 40:
            errors.append(f"turn{index}:answer_chars={len(answer)}")
        if "（" in answer or "）" in answer or "(" in answer or ")" in answer:
            errors.append(f"turn{index}:answer_contains_action_parentheses")
        if not 4 <= len(action) <= 160:
            errors.append(f"turn{index}:action_chars={len(action)}")
        cleaned.append({
            "turn_id": index,
            "question_text": question,
            "answer_text": answer,
            "action_expression": action,
        })
    return cleaned, errors


def stable_order(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and apply Xiaotian forward multimodal LLM results.")
    parser.add_argument("--config", default="configs/xiaotian_multimodal_1200.json")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--target", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    target = int(config["target_count"]) if args.target == 0 else args.target
    requests = {str(row["request_id"]): row for row in iter_jsonl(Path(args.requests))}
    out_dir = Path(args.out_dir) if args.out_dir else Path(config["run_dir"]) / "04_applied"
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_path = out_dir / "rejected.jsonl"
    valid: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    counts: Counter = Counter()
    seen_dialogues = set()
    with rejected_path.open("w", encoding="utf-8") as rejected:
        for result in iter_jsonl(Path(args.results)):
            request_id = str(result.get("request_id") or result.get("id") or "")
            request = requests.get(request_id)
            if request is None:
                counts["unknown_request"] += 1
                continue
            turns, errors = validate_response(request, result)
            dialogue_hash = hashlib.sha256(json.dumps(turns, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() if turns else ""
            if dialogue_hash and dialogue_hash in seen_dialogues:
                errors.append("duplicate_dialogue")
            if errors:
                rejected.write(json.dumps({"request_id": request_id, "sample_id": request.get("sample_id"), "errors": errors}, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["rejected"] += 1
                for error in errors:
                    counts[error.split(":", 1)[0]] += 1
                continue
            seen_dialogues.add(dialogue_hash)
            valid.append((request, turns))
            counts["valid"] += 1
    valid.sort(key=lambda item: (stable_order(str(item[0]["sample_id"]), int(config["seed"])), str(item[0]["sample_id"])))
    selected = valid if target <= 0 else valid[:target]
    intermediate_path = out_dir / "duplex_intermediate.jsonl"
    enrichment_path = out_dir / "multimodal_enrichment.jsonl"
    with intermediate_path.open("w", encoding="utf-8") as intermediate, enrichment_path.open("w", encoding="utf-8") as enrichment:
        for request, generated_turns in selected:
            turns = []
            descriptions = []
            for index, generated in enumerate(generated_turns, start=1):
                turns.append({
                    "turn_id": index,
                    "source": "current" if index == len(generated_turns) else "history",
                    "question_text": generated["question_text"],
                    "answer_text": generated["answer_text"],
                    "needs_tts": True,
                    "train_answer": True,
                    "question_speaker": "玩家",
                    "answer_speaker": config["ai_role_name"],
                })
                descriptions.append({"turn_id": index, "action_expression": generated["action_expression"]})
            sample_id = str(request["sample_id"])
            row = {
                "id": sample_id,
                "sysprompt": config["sysprompt"],
                "turns": turns,
                "meta": {
                    "dataset": "xiaotian_multimodal_forward",
                    "split": "train",
                    "language": "zh",
                    "role_name": config["ai_role_name"],
                    "player_name": "玩家",
                    "turn_count": len(turns),
                    "history_turn_count": max(0, len(turns) - 1),
                    "text_provenance": "dsv4_forward_generated",
                    "source_group_id": f"xiaotian_multimodal_forward::{sample_id}",
                    "player_role_card_id": request["player_role_card_id"],
                    "query_agent": {
                        "role_card": request["player_role_card"],
                        "agent_type": request.get("player_agent_type", "general"),
                        "emotion": request.get("player_emotion_hint", ""),
                        "world_view_category": request.get("player_world_view_category", ""),
                        "source_file": request.get("player_source_file", ""),
                    },
                },
            }
            intermediate.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            enrichment.write(json.dumps({
                "schema_version": ENRICHMENT_SCHEMA,
                "id": sample_id,
                "source_id": sample_id,
                "scenario": "normal_qa",
                "sysprompt": config["sysprompt"],
                "dialogue_turns": turns,
                "scene_description": config["scene_description"],
                "voice_description": config["voice_description"],
                "turn_descriptions": descriptions,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = {
        "requests": len(requests),
        "result_rows": sum(counts[key] for key in ("valid", "rejected")),
        "valid": len(valid),
        "selected": len(selected),
        "target": target,
        "target_met": target <= 0 or len(selected) >= target,
        "counts": dict(counts),
        "intermediate": str(intermediate_path),
        "enrichment": str(enrichment_path),
        "rejected": str(rejected_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if target > 0 and len(selected) < target:
        raise SystemExit(f"valid rows {len(selected)} are below target {target}; rerun failed requests before TTS")


if __name__ == "__main__":
    main()
