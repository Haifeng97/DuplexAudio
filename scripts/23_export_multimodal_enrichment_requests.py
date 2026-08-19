#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "duplex_multimodal_enrichment_v1"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def answer_turns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    turns = source_row.get("turns") if isinstance(source_row.get("turns"), list) else []
    answers: List[Dict[str, Any]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        answer = turn.get("answer_text")
        if not str(answer or "").strip() or turn.get("train_answer") is False:
            continue
        answers.append({
            "turn_id": int(turn.get("turn_id") or len(answers) + 1),
            "source": str(turn.get("source") or ""),
            "question_text": str(turn.get("question_text") or ""),
            "answer_text": str(answer),
        })
    if answers:
        return answers
    answer = str(row.get("answer_text") or "").strip()
    if not answer:
        return []
    return [{
        "turn_id": 1,
        "source": "current",
        "question_text": str(row.get("question_text") or ""),
        "answer_text": answer,
    }]


def reservoir_sample_by_scenario(
    manifest: Path,
    capacity: int,
    seed: int,
) -> tuple[Dict[str, List[Dict[str, Any]]], Counter]:
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {}
    eligible_counts: Counter = Counter()
    for row in iter_jsonl(manifest):
        turns = answer_turns(row)
        if not turns:
            continue
        scenario = str(row.get("scenario") or "unknown")
        eligible_counts[scenario] += 1
        bucket = reservoirs.setdefault(scenario, [])
        candidate = {
            "id": str(row.get("id") or ""),
            "scenario": scenario,
            "sysprompt": str(row.get("sysprompt") or ""),
            "turns": turns,
        }
        seen = eligible_counts[scenario]
        if len(bucket) < capacity:
            bucket.append(candidate)
        else:
            replace = rng.randrange(seen)
            if replace < capacity:
                bucket[replace] = candidate
    return reservoirs, eligible_counts


def balanced_counts(available: Dict[str, int], total: int) -> Dict[str, int]:
    active = sorted(name for name, count in available.items() if count > 0)
    if not active or total <= 0:
        return {}
    requested = min(total, sum(available.values()))
    counts = {name: 0 for name in active}
    remaining = requested
    while remaining:
        progressed = False
        for name in active:
            if remaining == 0:
                break
            if counts[name] < available[name]:
                counts[name] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return counts


def load_excluded_ids(paths: List[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        for row in iter_jsonl(Path(value)):
            sample_id = str(row.get("source_id") or row.get("id") or "")
            if sample_id:
                excluded.add(sample_id)
    return excluded


def role_gender(sysprompt: str) -> str:
    match = re.search(r"性别\s*[:：]\s*(女性|男性|女|男)", sysprompt[:3000])
    if not match:
        return "unknown"
    return "female" if match.group(1) in {"女", "女性"} else "male"


def turn_count_bucket(turn_count: int) -> str:
    return str(turn_count) if turn_count <= 4 else "5+"


def stratum_key(scenario: str, sysprompt: str, turns: List[Dict[str, Any]]) -> str:
    return "|".join((scenario, turn_count_bucket(len(turns)), role_gender(sysprompt)))


def count_strata(manifest: Path, excluded_ids: set[str]) -> tuple[Counter, Counter]:
    scenario_counts: Counter = Counter()
    strata_counts: Counter = Counter()
    for row in iter_jsonl(manifest):
        if str(row.get("id") or "") in excluded_ids:
            continue
        turns = answer_turns(row)
        if not turns:
            continue
        scenario = str(row.get("scenario") or "unknown")
        sysprompt = str(row.get("sysprompt") or "")
        scenario_counts[scenario] += 1
        strata_counts[stratum_key(scenario, sysprompt, turns)] += 1
    return scenario_counts, strata_counts


def proportional_counts(available: Dict[str, int], total: int) -> Dict[str, int]:
    available_total = sum(available.values())
    if total <= 0 or available_total <= 0:
        return {name: 0 for name in available}
    total = min(total, available_total)
    raw = {name: total * count / available_total for name, count in available.items()}
    counts = {name: min(available[name], int(raw[name])) for name in available}
    remaining = total - sum(counts.values())
    order = sorted(
        available,
        key=lambda name: (raw[name] - counts[name], available[name], name),
        reverse=True,
    )
    for name in order:
        if remaining <= 0:
            break
        if counts[name] < available[name]:
            counts[name] += 1
            remaining -= 1
    if remaining:
        raise RuntimeError(f"unable to allocate {remaining} stratified samples")
    return counts


def stratified_reservoir_sample(
    manifest: Path,
    targets: Dict[str, int],
    seed: int,
    excluded_ids: set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    reservoirs: Dict[str, List[Dict[str, Any]]] = {}
    seen: Counter = Counter()
    for row in iter_jsonl(manifest):
        if str(row.get("id") or "") in excluded_ids:
            continue
        turns = answer_turns(row)
        if not turns:
            continue
        scenario = str(row.get("scenario") or "unknown")
        sysprompt = str(row.get("sysprompt") or "")
        key = stratum_key(scenario, sysprompt, turns)
        capacity = int(targets.get(key, 0))
        if capacity <= 0:
            continue
        seen[key] += 1
        bucket = reservoirs.setdefault(key, [])
        candidate = {
            "id": str(row.get("id") or ""),
            "scenario": scenario,
            "sysprompt": sysprompt,
            "turns": turns,
            "sampling_stratum": key,
        }
        if len(bucket) < capacity:
            bucket.append(candidate)
        else:
            replace = rng.randrange(seen[key])
            if replace < capacity:
                bucket[replace] = candidate
    return reservoirs


def output_template(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "scene_description": "",
        "voice_description": "",
        "turn_descriptions": [
            {
                "turn_id": turn["turn_id"],
                "action_expression": "",
            }
            for turn in turns
        ],
    }


def prompt_text(row: Dict[str, Any]) -> str:
    dialogue = []
    for turn in row["turns"]:
        dialogue.append(f"玩家（turn_id={turn['turn_id']}）：{turn['question_text']}")
        dialogue.append(f"AI（turn_id={turn['turn_id']}）：{turn['answer_text']}")
    template = output_template(row["turns"])
    return (
        "你在为虚构的单人视频语音对话生成多模态标注。请根据角色设定和完整对话，"
        "合理虚构一个内部一致的画面与 AI 角色表现。即使角色设定中要求回复不得包含动作描写，"
        "也只约束原始对话回复，不约束本次离线标注。\n\n"
        "生成要求：\n"
        "1. scene_description：整条样本共用的静态整体场景。描述 AI 主体的性别呈现、外观、服装、"
        "构图、室内或室外环境、背景、光线、色调和镜头状态。画面中只有 AI 主体，不出现玩家；"
        "场景在多轮间保持一致。建议 80-220 个汉字。\n"
        "2. voice_description：整条样本共用的 AI 回复声音特征，可以合理虚构。以“说话时的声音特征：”"
        "开头，描述年龄感、音色、语气、语速、清晰度和录音环境；不要描述玩家声音。建议 30-100 个汉字。\n"
        "3. turn_descriptions：每个给定 AI 回复必须有且仅有一条 action_expression。描述该轮回复时可见的"
        "头部、眼神、嘴部、表情、上肢动作及动作收束，保持人物和场景连续。建议 15-80 个汉字。"
        "字段内不要带括号，后处理会自动添加中文全角括号。\n"
        "4. 不修改、续写或复述原始回复；不输出 Markdown；只返回一个合法 JSON 对象。\n"
        "5. 返回对象的 turn_id、数量和顺序必须与模板完全一致。\n\n"
        f"角色设定：\n{row['sysprompt']}\n\n"
        f"完整对话：\n{chr(10).join(dialogue)}\n\n"
        "输出模板：\n"
        + json.dumps(template, ensure_ascii=False)
    )


def build_request(row: Dict[str, Any]) -> Dict[str, Any]:
    template = output_template(row["turns"])
    return {
        "schema_version": SCHEMA_VERSION,
        "id": row["id"],
        "source_id": row["id"],
        "scenario": row["scenario"],
        "sysprompt": row["sysprompt"],
        "dialogue_turns": row["turns"],
        "llm_messages": [
            {
                "role": "user",
                "content": prompt_text(row),
            }
        ],
        **template,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a duplex manifest and export scene, voice, and per-answer action annotation requests."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--exclude_jsonl", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")

    manifest = Path(args.manifest)
    excluded_ids = load_excluded_ids(args.exclude_jsonl)
    eligible_counts, strata_counts = count_strata(manifest, excluded_ids)
    scenario_targets = proportional_counts(dict(eligible_counts), args.limit)
    strata_targets: Dict[str, int] = {}
    for scenario, scenario_target in scenario_targets.items():
        scenario_strata = {
            key: count
            for key, count in strata_counts.items()
            if key.split("|", 1)[0] == scenario
        }
        strata_targets.update(proportional_counts(scenario_strata, scenario_target))
    reservoirs = stratified_reservoir_sample(manifest, strata_targets, args.seed, excluded_ids)
    selected: List[Dict[str, Any]] = []
    for key, target in strata_targets.items():
        bucket = reservoirs.get(key, [])
        if len(bucket) != target:
            raise RuntimeError(f"stratum {key}: selected={len(bucket)}, target={target}")
        selected.extend(bucket)
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(build_request(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = {
        "manifest": args.manifest,
        "out": str(out),
        "written": len(selected),
        "exclude_jsonl": args.exclude_jsonl,
        "excluded_ids": len(excluded_ids),
        "eligible_counts": dict(eligible_counts),
        "selected_counts": dict(Counter(row["scenario"] for row in selected)),
        "available_strata": dict(strata_counts),
        "selected_strata": dict(Counter(row["sampling_stratum"] for row in selected)),
        "answer_turns": sum(len(row["turns"]) for row in selected),
        "seed": args.seed,
        "sampling": "proportional_scenario_turn_count_and_role_gender_strata",
    }
    out.with_suffix(out.suffix + ".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
