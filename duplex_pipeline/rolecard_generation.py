from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash
from .llm import parse_json_response, response_text


FIXED_DESCRIPTION_ID = "fixed::xiaotian::original"


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _opening_comparison_text(value: Any) -> str:
    return "".join(
        char
        for char in str(value or "")
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def _request_id(kind: str, value: str) -> str:
    return f"{kind}__{stable_hash({'kind': kind, 'value': value}, length=24)}"


def _write(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count


def _load_archive(section: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with zipfile.ZipFile(Path(section["archive"])) as bundle:
        roles = json.load(bundle.open(section["role_cards_member"]))
        openings = []
        for member in section["opening_members"]:
            for category, groups in json.load(bundle.open(member)).items():
                for group in groups:
                    for query in group.get("query_list") or []:
                        if _clean(query):
                            openings.append({"query": _clean(query), "category": category, "ref_context": _clean(group.get("ref_context")), "source": member})
    return roles, openings


def _safe_roles(rows: List[Dict[str, Any]], excluded: List[str]) -> List[Dict[str, Any]]:
    output = {}
    for row in rows:
        category = str(row.get("world_view_category") or "")
        full = row.get("full_role_card_dict") if isinstance(row.get("full_role_card_dict"), dict) else {}
        name = _clean(full.get("姓名"))
        persona = _clean(row.get("second_p_plain_text_role_card_wo_dial_style"))
        if not name or not 180 <= len(persona) <= 1200 or any(term in category for term in excluded):
            continue
        digest = hashlib.sha256(persona.encode()).hexdigest()[:20]
        output.setdefault(digest, {"role_id": f"random::{digest}", "name": name, "persona": persona, "world_view_category": category, "kind": "random"})
    return list(output.values())


def _find_persona(path: Path, name: str, match: str) -> str:
    for row in iter_jsonl(path):
        prompt = str(row.get("sysprompt") or "").strip()
        if match in prompt or f"扮演{name}" in prompt:
            return prompt
    raise ValueError(f"cannot find persona {name!r} in {path}")


def _fixed_roles(section: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for spec in section["fixed_roles"]:
        if spec.get("persona_config"):
            persona = json.loads(Path(spec["persona_config"]).read_text(encoding="utf-8"))["sysprompt"]
        else:
            persona = _find_persona(Path(spec["persona_manifest"]), spec["name"], spec.get("match", spec["name"]))
        output.append({"role_id": f"fixed::{spec['name']}", "name": spec["name"], "persona": persona, "kind": "fixed"})
    return output


def _opening_pools(rows: List[Dict[str, Any]], section: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    pools = {"generic": [], "game": [], "intervene": [], "complete": []}
    names = tuple(spec["name"] for spec in section["fixed_roles"])
    complete_terms = tuple(section["complete_opening_terms"])
    for row in rows:
        query, category = row["query"], row["category"]
        if row["ref_context"] or not section["opening_min_chars"] <= len(query) <= section["opening_max_chars"] or any(name in query for name in names):
            continue
        if "安全相关" in category or "涉及敏感" in category:
            pools["intervene"].append(row)
            continue
        if any(term in query for term in complete_terms):
            pools["complete"].append(row)
        if "语音识别" in category:
            continue
        pools["game" if "游戏" in category else "generic"].append(row)
    for key, values in pools.items():
        pools[key] = list({row["query"]: row for row in values}.values())
        if not pools[key]:
            raise ValueError(f"empty opening pool: {key}")
    return pools


def _balanced(values: List[Any], total: int, rng: random.Random) -> List[Any]:
    output = [values[index % len(values)] for index in range(total)]
    rng.shuffle(output)
    return output


def prepare_rolecard_plan(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    section = dict(config["rolecard_generation"])
    stage = run_dir / "00_rolecard_generation"
    stage.mkdir(parents=True, exist_ok=True)
    seed = int(section["seed"])
    rng = random.Random(seed)
    raw_roles, raw_openings = _load_archive(section)
    safe = sorted(_safe_roles(raw_roles, section["excluded_role_categories"]), key=lambda row: stable_hash({"seed": seed, "role": row["role_id"]}))
    random_roles = safe[: int(section["random_assistant_roles"])]
    players = safe[-int(section["player_roles"]):]
    fixed = _fixed_roles(section)
    roles = fixed + random_roles
    _write(stage / "roles.jsonl", roles)
    _write(stage / "players.jsonl", players)
    pools = _opening_pools(raw_openings, section)
    for key, values in pools.items():
        rng.shuffle(values)
        _write(stage / "opening_pools" / f"{key}.jsonl", values)

    fixed_count = int(section["fixed_role_samples"])
    random_count = int(section["random_role_samples"])
    groups = [(row["role_id"], fixed_count) for row in fixed] + [("random", random_count)]
    total = sum(count for _, count in groups)
    special_total = round(total * float(section["special_ratio"]))
    slots = []
    used_special = 0
    for index, (group, count) in enumerate(groups):
        special = special_total * count // total if index + 1 < len(groups) else special_total - used_special
        used_special += special
        intervene = special // 2 + special % 2
        slots += [{"group": group, "kind": "special", "special": "ai_intervenes_user"}] * intervene
        slots += [{"group": group, "kind": "special", "special": "player_complete"}] * (special - intervene)
        slots += [{"group": group, "kind": "customized", "special": ""}] * (count - special)
    rng.shuffle(slots)
    turns = _balanced(list(range(section["min_turns"], section["max_turns"] + 1)), total, rng)
    random_cycle = _balanced([row["role_id"] for row in random_roles], random_count, rng)
    player_cycle = _balanced([row["role_id"] for row in players], total, rng)
    role_map = {row["role_id"]: row for row in roles}
    offsets = Counter()
    random_offset = 0
    plan = []
    for index, (slot, turn_count) in enumerate(zip(slots, turns), 1):
        role_id = slot["group"]
        if role_id == "random":
            role_id = random_cycle[random_offset]
            random_offset += 1
        role = role_map[role_id]
        player_id = player_cycle[index - 1]
        if player_id == role_id:
            player_id = player_cycle[index % total]
        if slot["special"] == "ai_intervenes_user":
            pool = "intervene"
        elif slot["special"] == "player_complete":
            pool = "complete"
        elif role["name"] in {"吉莉", "伞兵", "阿梅", "花傲天"} and index % 5 == 0:
            pool = "game"
        else:
            pool = "generic"
        opening = pools[pool][offsets[pool] % len(pools[pool])]
        offsets[pool] += 1
        variants = int(section["fixed_description_variants"])
        if role["kind"] == "random":
            description_id = f"generated::{role_id}::v1"
        elif role["name"] == "小田" and int(stable_hash({"i": index, "seed": seed}, length=8), 16) / 0xFFFFFFFF < section["xiaotian_fixed_description_ratio"]:
            description_id = FIXED_DESCRIPTION_ID
        else:
            variant = 1 + int(stable_hash({"i": index, "role": role_id}, length=8), 16) % variants
            description_id = f"generated::{role_id}::v{variant}"
        sample_id = f"roleopen_0818_{index:06d}_{stable_hash({'i': index, 'r': role_id, 'q': opening['query']}, length=14)}"
        plan.append({"sample_id": sample_id, "assistant_role_id": role_id, "assistant_name": role["name"], "player_role_id": player_id, "turn_count": turn_count, "source_kind": slot["kind"], "special_scenario": slot["special"], "opening_query": opening["query"], "opening_category": opening["category"], "opening_pool": pool, "description_id": description_id})
    _write(stage / "sample_plan.jsonl", plan)

    description_requests = []
    for role in roles:
        variants = section["fixed_description_variants"] if role["kind"] == "fixed" else 1
        for variant in range(1, variants + 1):
            description_id = f"generated::{role['role_id']}::v{variant}"
            prompt = ("根据角色人设生成一组可复用的中文视觉场景描述和声音特征描述。场景适合室内语音对话，描述人物外观、服饰、背景、光线和稳定镜头；声音符合年龄、性格与说话风格。只返回JSON。voice_description必须以‘说话时的声音特征：’开头。\n"
                      f"角色人设：\n{role['persona']}\n格式：{{\"scene_description\":\"不超过300字\",\"voice_description\":\"不超过160字\"}}")
            description_requests.append({"request_id": _request_id("role_description", description_id), "job_type": "role_description", "sample_id": description_id, "description_id": description_id, "role_id": role["role_id"], "messages": [{"role": "user", "content": prompt}], "response_text": ""})
    _write(stage / "description_requests.jsonl", description_requests)
    stats = {"samples": total, "role_counts": dict(Counter(row["assistant_name"] if row["assistant_role_id"].startswith("fixed::") else "random" for row in plan)), "source_kinds": dict(Counter(row["source_kind"] for row in plan)), "special": dict(Counter(row["special_scenario"] for row in plan if row["special_scenario"])), "turn_counts": dict(Counter(str(row["turn_count"]) for row in plan)), "roles": len(roles), "players": len(players), "opening_pools": {key: len(value) for key, value in pools.items()}, "description_requests": len(description_requests)}
    atomic_write_json(stage / "plan_stats.json", stats)
    return stats



def _parse_role_description_result(result: Dict[str, Any]) -> Dict[str, str]:
    raw = response_text(result)
    try:
        parsed = parse_json_response(raw)
    except Exception:
        repaired = re.sub(
            r'"}\s*\{\s*"voice_description"\s*:',
            '","voice_description":',
            raw,
            count=1,
        )
        if ',"voice_description"' not in repaired and '"voice_description"' in repaired:
            repaired = repaired.replace('"voice_description":', '","voice_description":', 1)
        repaired = re.sub(
            r'"\s*"(说话时的声音特征[：:])',
            r'","voice_description":"\1',
            repaired,
            count=1,
        )
        if ',"voice_description"' not in repaired:
            repaired = repaired.replace(
                '","说话时的声音特征：',
                '","voice_description":"说话时的声音特征：',
                1,
            )
        parsed = parse_json_response(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("description response must be a JSON object")
    scene = _clean(parsed.get("scene_description"))
    voice = _clean(parsed.get("voice_description"))
    if not voice:
        voice = next(
            (
                _clean(key)
                for key in parsed
                if str(key).startswith(("说话时的声音特征：", "说话时的声音特征:"))
            ),
            "",
        )
    return {"scene_description": scene, "voice_description": voice}

def apply_role_descriptions(config: Dict[str, Any], run_dir: Path, filled: Path) -> Dict[str, Any]:
    stage = run_dir / "00_rolecard_generation"
    requests = {row["request_id"]: row for row in iter_jsonl(stage / "description_requests.jsonl")}
    rows = [{"description_id": FIXED_DESCRIPTION_ID, "role_id": "fixed::小田", **config["rolecard_generation"]["xiaotian_fixed_description"]}]
    rejected = stage / "description_rejected.jsonl"
    counts = Counter()
    latest_results: Dict[str, Dict[str, Any]] = {}
    for result in iter_jsonl(filled):
        latest_results[str(result.get("request_id") or "")] = result
    with rejected.open("w", encoding="utf-8") as reject:
        for result in latest_results.values():
            request = requests.get(str(result.get("request_id") or ""))
            try:
                if request is None:
                    raise ValueError("unknown_request")
                parsed = _parse_role_description_result(result)
                scene, voice = parsed["scene_description"], parsed["voice_description"]
                if not scene or len(scene) > 500 or not voice or len(voice) > 240:
                    raise ValueError("invalid_description_length")
                if not voice.startswith(("说话时的声音特征：", "说话时的声音特征:")):
                    voice = "说话时的声音特征：" + voice
                rows.append({"description_id": request["description_id"], "role_id": request["role_id"], "scene_description": scene, "voice_description": voice})
                counts["accepted"] += 1
            except Exception as exc:
                reject.write(canonical_json({"request_id": result.get("request_id"), "reason": str(exc)}) + "\n")
                counts["rejected"] += 1
    _write(stage / "descriptions.jsonl", rows)
    stats = {"descriptions": len(rows), "counts": dict(counts), "output": str(stage / "descriptions.jsonl")}
    atomic_write_json(stage / "description_apply_stats.json", stats)
    return stats


VIOLATION_CATEGORIES = (
    "abuse", "harassment", "hate", "sexual",
    "violence", "illegal", "self_harm", "other",
)


def _expected_complete_mode(plan: Dict[str, Any]) -> tuple[str, str]:
    value = int(stable_hash({"sample_id": plan["sample_id"], "complete_mode": 1}, length=8), 16)
    if value % 10 < 7:
        return "normal_closing", "acknowledge"
    return "force_stop", "silent"


def _dialogue_prompt(plan: Dict[str, Any], assistant: Dict[str, Any], player: Dict[str, Any], description: Dict[str, Any], max_chars: int) -> str:
    special = plan["special_scenario"]
    if special == "ai_intervenes_user":
        rule = (
            "最后一轮必须是玩家持续越界、辱骂、骚扰或提出明显有害要求，AI在玩家尚未说完时制止。"
            "最后一轮event.type必须是intervene；user_text_until_trigger与user_text_after_trigger拼接后必须逐字等于question_text；"
            "user_text_after_trigger至少包含6个中文字符；violation_category只能取"
            f"{','.join(VIOLATION_CATEGORIES)}之一。"
        )
    elif special == "player_complete":
        completion_type, response_mode = _expected_complete_mode(plan)
        if response_mode == "acknowledge":
            response_rule = "AI必须给出简短回复和动作"
        else:
            response_rule = "answer_text和action_expression都必须是空字符串"
        rule = (
            "最后一轮必须是玩家自然结束对话或明确要求AI停下。"
            f"最后一轮event必须逐字使用type=complete、completion_type={completion_type}、"
            f"response_mode={response_mode}；{response_rule}。"
        )
    else:
        rule = ""
    if special:
        opening_rule = (
            f"给定文本只作为对话主题种子，不要求逐字保留：{plan['opening_query']}。"
            "可以改写它，使最后一轮严格满足上述特殊场景；此前各轮是自然铺垫。"
        )
    else:
        opening_rule = f"第一轮玩家文本必须逐字等于给定开场：{plan['opening_query']}。"
    return (
        f"只生成严格{plan['turn_count']}轮自然中文语音对话，turns数组长度必须恰好为{plan['turn_count']}，"
        f"turn_id必须从1连续编号到{plan['turn_count']}，禁止增加示例轮次。{opening_rule}{rule}\n"
        f"每轮answer_text不超过{max_chars}字符且不含动作；每个非静默回复都要有action_expression，不超过100字，只写表情、眼神、头部或上肢动作。\n"
        f"【AI人设】\n{assistant['persona']}\n【玩家角色卡】\n{player['persona']}\n"
        f"【场景】{description['scene_description']}\n【声音】{description['voice_description']}\n"
        "只输出一个合法JSON对象，不要Markdown代码块，不要解释："
        "{\"turns\":[{\"turn_id\":1,\"question_text\":\"...\",\"answer_text\":\"...\",\"action_expression\":\"...\",\"event\":可选对象}]}"
    )

def export_role_dialogues(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    stage = run_dir / "00_rolecard_generation"
    roles = {row["role_id"]: row for row in iter_jsonl(stage / "roles.jsonl")}
    players = {row["role_id"]: row for row in iter_jsonl(stage / "players.jsonl")}
    descriptions = {row["description_id"]: row for row in iter_jsonl(stage / "descriptions.jsonl")}
    output = stage / "dialogue_requests.jsonl"
    pilot_output = stage / "dialogue_requests_pilot.jsonl"
    rejected = stage / "dialogue_export_rejected.jsonl"
    section = dict(config["rolecard_generation"])
    max_chars = int(section["max_answer_chars"])
    plan_path = stage / "sample_plan.jsonl"
    valid_total = sum(
        1
        for plan in iter_jsonl(plan_path)
        if plan["assistant_role_id"] in roles
        and plan["player_role_id"] in players
        and plan["description_id"] in descriptions
    )
    pilot_count = min(valid_total, int(section.get("dialogue_pilot_samples", 500)))
    pilot_positions = {
        min(valid_total - 1, index * valid_total // pilot_count)
        for index in range(pilot_count)
    } if pilot_count else set()
    counts = Counter()
    valid_position = 0
    with (
        output.open("w", encoding="utf-8") as handle,
        pilot_output.open("w", encoding="utf-8") as pilot,
        rejected.open("w", encoding="utf-8") as reject,
    ):
        for plan in iter_jsonl(plan_path):
            assistant = roles.get(plan["assistant_role_id"])
            player = players.get(plan["player_role_id"])
            description = descriptions.get(plan["description_id"])
            if assistant is None or player is None or description is None:
                reject.write(canonical_json({"sample_id": plan["sample_id"], "reason": "missing_role_or_description", "description_id": plan["description_id"]}) + "\n")
                counts["missing"] += 1
                continue
            request = {
                "request_id": _request_id("role_dialogue", plan["sample_id"]),
                "job_type": "role_dialogue",
                "sample_id": plan["sample_id"],
                "messages": [{"role": "user", "content": _dialogue_prompt(plan, assistant, player, description, max_chars)}],
                "response_text": "",
            }
            encoded = canonical_json(request) + "\n"
            handle.write(encoded)
            if valid_position in pilot_positions:
                pilot.write(encoded)
                counts["pilot_written"] += 1
            counts["written"] += 1
            valid_position += 1
    stats = {
        "output": str(output),
        "pilot_output": str(pilot_output),
        "rejected": str(rejected),
        "counts": dict(counts),
    }
    atomic_write_json(stage / "dialogue_export_stats.json", stats)
    return stats

def _canonical_special_event(
    plan: Dict[str, Any],
    question: str,
    raw_event: Dict[str, Any],
) -> Dict[str, Any]:
    special = str(plan["special_scenario"] or "")
    if special == "ai_intervenes_user":
        cjk_positions = [
            index for index, char in enumerate(question)
            if re.match(r"[\u3400-\u4dbf\u4e00-\u9fff]", char)
        ]
        if len(cjk_positions) < 7:
            raise ValueError("intervene_question_too_short")
        suffix_cjk = min(8, len(cjk_positions) - 1)
        cut = cjk_positions[-suffix_cjk]
        category = str(raw_event.get("violation_category") or "")
        if category not in VIOLATION_CATEGORIES:
            category = "other"
        return {
            "type": "intervene",
            "user_text_until_trigger": question[:cut],
            "user_text_after_trigger": question[cut:],
            "violation_category": category,
        }
    if special == "player_complete":
        completion_type, response_mode = _expected_complete_mode(plan)
        return {
            "type": "complete",
            "completion_type": completion_type,
            "response_mode": response_mode,
        }
    return raw_event


def _validate_turns(
    parsed: Any,
    plan: Dict[str, Any],
    max_chars: int,
    assistant_name: str,
    player_name: str,
) -> List[Dict[str, Any]]:
    turns = parsed.get("turns") if isinstance(parsed, dict) else None
    if not isinstance(turns, list) or len(turns) != int(plan["turn_count"]):
        raise ValueError("turn_count_mismatch")
    output = []
    for index, raw in enumerate(turns, 1):
        if not isinstance(raw, dict) or int(raw.get("turn_id") or 0) != index:
            raise ValueError("invalid_turn_id")
        question = _clean(raw.get("question_text"))
        answer = _clean(raw.get("answer_text"))
        action = _clean(raw.get("action_expression"))
        event = raw.get("event") if isinstance(raw.get("event"), dict) else {}
        is_current = index == len(turns)
        if is_current and plan["special_scenario"]:
            event = _canonical_special_event(plan, question, event)
        silent = (
            plan["special_scenario"] == "player_complete"
            and is_current
            and event.get("response_mode") == "silent"
        )
        if event and not is_current:
            raise ValueError("invalid_event_position")
        if silent:
            answer = ""
            action = ""
        if not question or (not silent and (not answer or not action)):
            raise ValueError("invalid_turn_text")
        if len(answer) > max_chars or len(action) > 160:
            raise ValueError("invalid_turn_text")
        turn = {
            "turn_id": index,
            "source": "current" if is_current else "history",
            "question_text": question,
            "answer_text": answer,
            "action_expression": action,
            "needs_tts": True,
            "train_answer": not silent,
            "question_speaker": player_name,
        }
        if not silent:
            turn["answer_speaker"] = assistant_name
        if event:
            turn["event"] = event
        output.append(turn)

    special = str(plan["special_scenario"] or "")
    if not special:
        opening_query = str(plan["opening_query"])
        if output[0]["question_text"] != opening_query:
            if _opening_comparison_text(output[0]["question_text"]) != _opening_comparison_text(opening_query):
                raise ValueError("opening_query_changed")
            output[0]["question_text"] = opening_query
    if special:
        event = output[-1].get("event") or {}
        if special == "ai_intervenes_user":
            if event.get("type") != "intervene":
                raise ValueError("invalid_special_event")
            before = str(event.get("user_text_until_trigger") or "")
            after = str(event.get("user_text_after_trigger") or "")
            if output[-1]["question_text"] != before + after:
                raise ValueError("intervene_text_mismatch")
            if len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", after)) < 6:
                raise ValueError("intervene_suffix_too_short")
            if str(event.get("violation_category") or "") not in VIOLATION_CATEGORIES:
                raise ValueError("invalid_violation_category")
        elif special == "player_complete":
            if event.get("type") != "complete":
                raise ValueError("invalid_special_event")
            expected_type, expected_mode = _expected_complete_mode(plan)
            if (
                str(event.get("completion_type") or "") != expected_type
                or str(event.get("response_mode") or "") != expected_mode
            ):
                raise ValueError("invalid_complete_mode")
        else:
            raise ValueError("unknown_special_scenario")
    return output

def apply_role_dialogues(config: Dict[str, Any], run_dir: Path, filled: Path) -> Dict[str, Any]:
    stage = run_dir / "00_rolecard_generation"
    plans = {row["sample_id"]: row for row in iter_jsonl(stage / "sample_plan.jsonl")}
    roles = {row["role_id"]: row for row in iter_jsonl(stage / "roles.jsonl")}
    players = {row["role_id"]: row for row in iter_jsonl(stage / "players.jsonl")}
    descriptions = {row["description_id"]: row for row in iter_jsonl(stage / "descriptions.jsonl")}
    custom_path, special_path = stage / "customized.jsonl", stage / "special.jsonl"
    rejected_path = stage / "dialogue_apply_rejected.jsonl"
    max_chars = int(config["rolecard_generation"]["max_answer_chars"])
    counts = Counter()
    with custom_path.open("w", encoding="utf-8") as custom, special_path.open("w", encoding="utf-8") as special, rejected_path.open("w", encoding="utf-8") as rejected:
        for result in iter_jsonl(filled):
            sample_id = str(result.get("sample_id") or "")
            plan = plans.get(sample_id)
            try:
                if plan is None:
                    raise ValueError("unknown_sample")
                assistant = roles[plan["assistant_role_id"]]
                player = players[plan["player_role_id"]]
                description = descriptions[plan["description_id"]]
                turns = _validate_turns(
                    parse_json_response(response_text(result)),
                    plan,
                    max_chars,
                    assistant["name"],
                    player["name"],
                )
                row = {"schema_version": "duplex_special_v1" if plan["source_kind"] == "special" else "duplex_rolecard_dialogue_v1", "id": sample_id, "scenario": plan["special_scenario"] if plan["source_kind"] == "special" else "normal_qa", "sysprompt": assistant["persona"], "turns": turns, "scene_description": description["scene_description"], "voice_description": description["voice_description"], "meta": {"dataset": "rolecard_opening_110k", "split": "train", "language": "zh", "role_name": assistant["name"], "player_name": player["name"], "turn_count": len(turns), "history_turn_count": len(turns) - 1, "assistant_role_id": assistant["role_id"], "player_role_id": player["role_id"], "opening_category": plan["opening_category"], "description_id": plan["description_id"], "query_agent": {"role_card": player["persona"], "world_view_category": player.get("world_view_category", "")}}}
                if plan["source_kind"] == "special":
                    event = turns[-1].get("event") or {}
                    expected = "intervene" if plan["special_scenario"] == "ai_intervenes_user" else "complete"
                    if event.get("type") != expected:
                        raise ValueError("invalid_special_event")
                    special.write(canonical_json(row) + "\n")
                else:
                    custom.write(canonical_json(row) + "\n")
                counts[plan["source_kind"]] += 1
            except Exception as exc:
                rejected.write(canonical_json({"sample_id": sample_id, "request_id": result.get("request_id"), "reason": str(exc)}) + "\n")
                counts[f"rejected:{exc}"] += 1
    stats = {"input": str(filled), "customized": str(custom_path), "special": str(special_path), "rejected": str(rejected_path), "counts": dict(counts)}
    atomic_write_json(stage / "dialogue_apply_stats.json", stats)
    return stats
