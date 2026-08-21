from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash
from .llm import parse_json_response, response_text


def _turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(turn) for turn in (row.get("turns") or []) if isinstance(turn, dict)]


def _source_group(row: Dict[str, Any]) -> str:
    return str(row.get("root_source_group_id") or row.get("source_group_id") or row.get("id"))


def _answer_chunks(text: str) -> int:
    return max(1, int(len(text) * 1.1 + 0.999999))


def normal_candidate(row: Dict[str, Any], chunk_ms: int) -> Dict[str, Any]:
    turns = _turns(row)
    current = turns[-1]
    answer = str(current.get("answer_text") or "")
    out = dict(row)
    out.update({
        "scenario": "normal_qa",
        "source_id": row["id"],
        "source_group_id": _source_group(row),
        "turns": turns,
        "question_text": current.get("question_text", ""),
        "answer_text": answer,
        "gn_policy": {"chunk_ms": chunk_ms, "answer_gn_chunks": _answer_chunks(answer)},
    })
    return out


def incomplete_candidate(row: Dict[str, Any], split: Dict[str, Any], chunk_ms: int, seed: int) -> Dict[str, Any]:
    turns = _turns(row)
    current = turns[-1]
    cut = int(split["cut_char_index"])
    rng = random.Random(int(stable_hash({"id": row["id"], "seed": seed}, length=15), 16))
    return {
        **row,
        "id": f"incomplete__{row['id']}__cut{cut}",
        "scenario": "incomplete_query_candidate",
        "source_id": row["id"],
        "source_group_id": _source_group(row),
        "turns": turns,
        "incomplete_turn_id": current.get("turn_id"),
        "incomplete_turn_index": len(turns),
        "query_part1_text": split["query_part1_text"],
        "query_part2_text": split["query_part2_text"],
        "partial_question_text": split["query_part1_text"],
        "full_question_text": split["question_text"],
        "answer_text_if_complete": current.get("answer_text", ""),
        "split": {"cut_char_index": cut, "unicode_codepoint_boundary": True, "method": "llm_5_candidates_ranked"},
        "gn_policy": {"between_query_parts_sec": round(rng.uniform(0.5, 2.0), 3), "between_query_parts_range_sec": [0.5, 2.0]},
    }


def clarification_candidate(
    row: Dict[str, Any],
    split: Dict[str, Any],
    clarification: str,
    action_expression: str,
) -> Dict[str, Any]:
    original_turns = _turns(row)
    context = original_turns[:-1]
    current = original_turns[-1]
    turns: List[Dict[str, Any]] = []
    for turn in context:
        clean = dict(turn)
        clean["turn_id"] = len(turns) + 1
        turns.append(clean)
    inserted_turn_id = len(turns) + 1
    turns.append({
        "turn_id": inserted_turn_id,
        "source": "inserted_incomplete_query",
        "question_text": split["query_part1_text"],
        "answer_text": clarification,
        "action_expression": action_expression,
        "needs_tts": True,
        "train_answer": True,
        "full_question_text": split["question_text"],
        "original_answer_text": current.get("answer_text", ""),
        "split": {"cut_char_index": split["cut_char_index"], "method": "llm_5_candidates_ranked"},
    })
    following = dict(current)
    following["turn_id"] = len(turns) + 1
    turns.append(following)
    return {
        **row,
        "id": f"incomplete_clarify__{row['id']}",
        "scenario": "incomplete_query_clarification",
        "source_id": row["id"],
        "source_group_id": _source_group(row),
        "turns": turns,
        "question_text": split["query_part1_text"],
        "answer_text": clarification,
        "partial_question_text": split["query_part1_text"],
        "full_question_text": split["question_text"],
        "clarification_answer_text": clarification,
        "inserted_turn_id": inserted_turn_id,
        "force_inter_turn_idle": True,
        "gn_policy": {
            "clarification_wait_range_sec": [3.0, 5.0],
            "between_turn_idle_range_sec": [1.0, 3.0],
            "force_inter_turn_idle": True,
        },
    }


def interrupt_candidate(row: Dict[str, Any], seed: int) -> Dict[str, Any]:
    turns = _turns(row)
    base = turns[-2]
    donor = turns[-1]
    answer = str(base.get("answer_text") or "")
    max_prefix = max(1, min(len(answer) - 1, 12))
    prefix = 1 + int(stable_hash({"id": row["id"], "seed": seed, "kind": "interrupt"}, length=12), 16) % max_prefix
    return {
        **row,
        "id": f"interrupt_same_row__{row['id']}__t{base.get('turn_id', 'prev')}_t{donor.get('turn_id', 'next')}",
        "scenario": "player_interrupts_ai",
        "source": "same_row_previous_turn_interrupted_by_next_turn",
        "source_id": row["id"],
        "source_group_id": _source_group(row),
        "turns": turns,
        "base": {
            "id": row["id"], "turn_id": base.get("turn_id"), "sysprompt": row.get("sysprompt", ""),
            "question_text": base.get("question_text", ""), "answer_text": answer,
            "answer_prefix_text": answer[:prefix], "prefix_chars": prefix, "meta": row.get("meta", {}),
        },
        "donor": {
            "id": row["id"], "turn_id": donor.get("turn_id"), "sysprompt": row.get("sysprompt", ""),
            "question_text": donor.get("question_text", ""), "answer_text": donor.get("answer_text", ""),
            "meta": row.get("meta", {}),
        },
    }


def backchannel_candidate(row: Dict[str, Any], clip: Dict[str, Any], chunk_ms: int, seed: int) -> Dict[str, Any]:
    turns = _turns(row)
    current = turns[-1]
    answer = str(current.get("answer_text") or "")
    max_prefix = max(1, min(len(answer) - 1, 12))
    prefix = 1 + int(stable_hash({"id": row["id"], "seed": seed, "kind": "backchannel"}, length=12), 16) % max_prefix
    return {
        **row,
        "id": f"backchannel__{row['id']}__p{prefix}",
        "scenario": "player_backchannel",
        "source_id": row["id"],
        "source_group_id": _source_group(row),
        "turns": turns,
        "backchannel_turn_id": current.get("turn_id"),
        "backchannel_turn_index": len(turns),
        "question_text": current.get("question_text", ""),
        "answer_text": answer,
        "answer_prefix_text": answer[:prefix],
        "answer_remaining_text": answer[prefix:],
        "prefix_chars": prefix,
        "backchannel_text": str(clip.get("text") or ""),
        "backchannel_audio": {
            "path": str(clip.get("audio") or ""), "clip_id": clip.get("clip_id"),
            "duration_sec": clip.get("duration_sec"), "sample_rate": clip.get("sample_rate"),
            "speaker": clip.get("speaker"), "gender": clip.get("gender"),
            "source_dataset": clip.get("source_dataset", "magicdata_ramc"),
        },
        "gn_policy": {
            "chunk_ms": chunk_ms,
            "answer_prefix_gn_chunks": _answer_chunks(answer[:prefix]),
            "answer_remaining_gn_chunks": _answer_chunks(answer[prefix:]),
        },
    }


def apply_clarification_results(run_dir: Path, filled_path: Path) -> Dict[str, Any]:
    requests = {
        str(row["request_id"]): row
        for row in iter_jsonl(run_dir / "03_llm" / "clarification" / "requests.jsonl")
    }
    output = run_dir / "03_llm" / "clarification" / "answers.jsonl"
    written = rejected = 0
    with output.open("w", encoding="utf-8") as handle:
        for result in iter_jsonl(filled_path):
            request_id = str(result.get("request_id") or result.get("id") or "")
            request = requests.get(request_id)
            raw = response_text(result).strip()
            try:
                parsed = parse_json_response(raw)
            except Exception:
                parsed = {}
            answer = str(parsed.get("answer_text") or raw).strip().strip('"“”') if isinstance(parsed, dict) else raw.strip().strip('"“”')
            action = str(parsed.get("action_expression") or "").strip() if isinstance(parsed, dict) else ""
            if request is None or not answer or len(answer) > 40 or len(action) > 160:
                rejected += 1
                continue
            handle.write(canonical_json({"sample_id": request["sample_id"], "clarification_answer_text": answer, "action_expression": action}) + "\n")
            written += 1
    stats = {"input": str(filled_path), "output": str(output), "written": written, "rejected": rejected}
    atomic_write_json(output.with_suffix(output.suffix + ".stats.json"), stats)
    return stats


def materialize_scenarios(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    selected_splits_path = run_dir / "03_llm" / "incomplete_rank" / "selected_splits.jsonl"
    split_map = {str(row["sample_id"]): row for row in iter_jsonl(selected_splits_path)} if selected_splits_path.exists() else {}
    clarification_path = run_dir / "03_llm" / "clarification" / "answers.jsonl"
    clarification_map = {
        str(row["sample_id"]): {"answer_text": str(row["clarification_answer_text"]), "action_expression": str(row.get("action_expression") or "")}
        for row in iter_jsonl(clarification_path)
    } if clarification_path.exists() else {}
    backchannel_path = Path(str(config.get("assets", {}).get("backchannel_manifest") or ""))
    clips = list(iter_jsonl(backchannel_path)) if backchannel_path.is_file() else []
    if not clips:
        raise FileNotFoundError("assets.backchannel_manifest must contain recorded backchannel clips")
    seed = int(config.get("planning", {}).get("seed", 20260818))
    chunk_ms = int(config.get("format", {}).get("chunk_ms", 180))
    out_dir = run_dir / "04_scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "normal_qa": "normal_qa.jsonl",
        "player_interrupts_ai": "player_interrupts_ai.jsonl",
        "incomplete_query": "incomplete_query.jsonl",
        "incomplete_query_clarification": "incomplete_query_clarification.jsonl",
        "player_backchannel": "player_backchannel.jsonl",
    }
    handles = {name: (out_dir / filename).open("w", encoding="utf-8") for name, filename in names.items()}
    counts: Counter = Counter()
    try:
        for row in iter_jsonl(run_dir / "02_plan" / "customized_selected.jsonl"):
            assigned = str(row["primary_scenario"])
            actual = assigned
            if assigned == "normal_qa":
                built = normal_candidate(row, chunk_ms)
            elif assigned == "player_interrupts_ai":
                built = interrupt_candidate(row, seed)
            elif assigned == "player_backchannel":
                clip_index = int(stable_hash({"id": row["id"], "seed": seed, "kind": "clip"}, length=12), 16) % len(clips)
                built = backchannel_candidate(row, clips[clip_index], chunk_ms, seed)
            elif assigned == "incomplete_query":
                split = split_map.get(str(row["id"]))
                if split is None:
                    actual = "normal_qa"
                    built = normal_candidate(row, chunk_ms)
                    counts["downgraded_missing_split"] += 1
                else:
                    built = incomplete_candidate(row, split, chunk_ms, seed)
            elif assigned == "incomplete_query_clarification":
                split = split_map.get(str(row["id"]))
                clarification = clarification_map.get(str(row["id"]), {})
                answer = str(clarification.get("answer_text") or "")
                action = str(clarification.get("action_expression") or "")
                if split is None or not answer:
                    actual = "normal_qa"
                    built = normal_candidate(row, chunk_ms)
                    counts["downgraded_missing_clarification"] += 1
                else:
                    built = clarification_candidate(row, split, answer, action)
            else:
                raise ValueError(f"unsupported customized scenario: {assigned}")
            built["primary_scenario"] = actual
            handles[actual].write(canonical_json(built) + "\n")
            counts[actual] += 1
    finally:
        for handle in handles.values():
            handle.close()
    special_output = out_dir / "special.jsonl"
    special_count = 0
    with special_output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(run_dir / "02_plan" / "special_selected.jsonl"):
            handle.write(canonical_json(row) + "\n")
            special_count += 1
    enrichment_path = out_dir / "multimodal_enrichment.jsonl"
    custom_enrichment_path = out_dir / "multimodal_customized.jsonl"
    special_enrichment_path = out_dir / "multimodal_special.jsonl"
    enrichment_count = 0
    custom_enrichment_count = 0
    special_enrichment_count = 0
    with (
        enrichment_path.open("w", encoding="utf-8") as enrichment,
        custom_enrichment_path.open("w", encoding="utf-8") as custom_enrichment,
        special_enrichment_path.open("w", encoding="utf-8") as special_enrichment,
    ):
        scenario_paths = [out_dir / filename for filename in names.values()] + [special_output]
        for scenario_path in scenario_paths:
            is_special = scenario_path == special_output
            group_handle = special_enrichment if is_special else custom_enrichment
            for scenario_row in iter_jsonl(scenario_path):
                scene = str(scenario_row.get("scene_description") or "").strip()
                voice = str(scenario_row.get("voice_description") or "").strip()
                if not scene and not voice:
                    continue
                descriptions = []
                for turn in _turns(scenario_row):
                    if not str(turn.get("answer_text") or "").strip() or turn.get("train_answer") is False:
                        continue
                    action = str(turn.get("action_expression") or "").strip()
                    if not action:
                        raise ValueError(f"{scenario_row.get('id')}: missing action_expression for turn {turn.get('turn_id')}")
                    descriptions.append({"turn_id": int(turn.get("turn_id") or 0), "action_expression": action})
                enrichment_row = {
                    "schema_version": "duplex_multimodal_enrichment_v1",
                    "source_id": scenario_row["id"],
                    "scene_description": scene,
                    "voice_description": voice,
                    "turn_descriptions": descriptions,
                }
                encoded = canonical_json(enrichment_row) + "\n"
                enrichment.write(encoded)
                group_handle.write(encoded)
                enrichment_count += 1
                if is_special:
                    special_enrichment_count += 1
                else:
                    custom_enrichment_count += 1
    result = {
        "counts": dict(counts), "special": special_count,
        "outputs": {name: str(out_dir / filename) for name, filename in names.items()},
        "special_output": str(special_output),
        "multimodal_enrichment": str(enrichment_path),
        "multimodal_customized": str(custom_enrichment_path),
        "multimodal_special": str(special_enrichment_path),
        "multimodal_rows": enrichment_count,
        "multimodal_customized_rows": custom_enrichment_count,
        "multimodal_special_rows": special_enrichment_count,
    }
    atomic_write_json(out_dir / "stats.json", result)
    return result
