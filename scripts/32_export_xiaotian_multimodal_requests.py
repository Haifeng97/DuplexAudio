#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCHEMA_VERSION = "duplex_xiaotian_forward_v1"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def card_hash(card: str) -> str:
    return hashlib.sha256(card.encode("utf-8")).hexdigest()


def load_unique_player_cards(path: Path) -> List[Dict[str, Any]]:
    cards: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        query_agent = meta.get("query_agent") if isinstance(meta.get("query_agent"), dict) else {}
        card = str(query_agent.get("role_card") or "").strip()
        if not card:
            continue
        digest = card_hash(card)
        cards.setdefault(digest, {
            "player_role_card_id": digest[:16],
            "player_role_card": card,
            "player_agent_type": str(query_agent.get("agent_type") or "general"),
            "player_emotion_hint": str(query_agent.get("emotion") or ""),
            "player_world_view_category": str(query_agent.get("world_view_category") or ""),
            "player_source_file": str(query_agent.get("source_file") or ""),
        })
    return [cards[key] for key in sorted(cards)]


def output_schema(turn_count: int) -> Dict[str, Any]:
    return {
        "turns": [
            {
                "turn_id": turn_id,
                "question_text": "玩家本轮说的话",
                "answer_text": "小田不超过40字的口语回复，不含动作括号",
                "action_expression": "小田本轮回复时的神态和身体动作，不含括号",
            }
            for turn_id in range(1, turn_count + 1)
        ]
    }


def prompt_text(config: Dict[str, Any], card: Dict[str, Any], turn_count: int) -> str:
    return (
        "请生成一段玩家与 AI 角色“小田”的自然多轮中文语音对话，并为小田每轮回复生成可见动作标注。\n\n"
        "这是正向生成任务：玩家角色卡决定玩家的知识、性格、兴趣和表达方式；小田人设、固定场景和声音特征"
        "共同决定小田的回复内容、语气与动作。不要反向修改场景和声音描述。\n\n"
        f"【玩家角色卡】\n{card['player_role_card']}\n\n"
        f"【小田人设】\n{config['sysprompt']}\n\n"
        f"【固定场景描述】\n{config['scene_description']}\n\n"
        f"【固定声音特征】\n{config['voice_description']}\n\n"
        f"严格生成 {turn_count} 轮，每轮包含玩家一句话和小田一句回复。\n"
        "要求：\n"
        "1. 多轮内容前后连贯，玩家的话符合玩家角色卡，但不要生硬复述角色卡。\n"
        "2. 玩家可以聊职业、生活、兴趣、情绪、美食、宠物、审美或轻松日常；避免敏感和违规话题。\n"
        "3. 小田始终符合人设，称呼玩家为“宝子”要自然，不要求每句都出现。\n"
        "4. 每条 answer_text 为1到2句、最多40个字符，不含括号、动作、旁白或场景描述。\n"
        "5. 每条 action_expression 描述小田当下可见的表情、眼神、头部或上肢动作及收束；"
        "不描述场景、镜头、声音或玩家，不带任何括号。动作在多轮间保持连续且避免每轮重复。\n"
        "6. 不生成具体游戏操作、账号、装备打法或位置报点。\n"
        "7. 只返回合法 JSON；turn_id、数量和顺序必须与模板完全一致，不输出 Markdown。\n\n"
        f"【输出模板】\n{json.dumps(output_schema(turn_count), ensure_ascii=False)}"
    )


def build_jobs(cards: List[Dict[str, Any]], count: int, seed: int, min_turns: int, max_turns: int) -> List[Dict[str, Any]]:
    if count < len(cards):
        raise ValueError(f"request count {count} cannot cover all {len(cards)} cards")
    rng = random.Random(seed)
    jobs: List[Dict[str, Any]] = []
    base_repeats, extra = divmod(count, len(cards))
    extra_cards = set(rng.sample(range(len(cards)), extra)) if extra else set()
    for card_index, card in enumerate(cards):
        repeats = base_repeats + int(card_index in extra_cards)
        for variant in range(1, repeats + 1):
            jobs.append({**card, "variant": variant})
    turn_values = list(range(min_turns, max_turns + 1))
    turn_counts = [turn_values[index % len(turn_values)] for index in range(count)]
    rng.shuffle(jobs)
    rng.shuffle(turn_counts)
    for job, turn_count in zip(jobs, turn_counts):
        job["turn_count"] = turn_count
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export forward Xiaotian dialogue/action generation requests.")
    parser.add_argument("--config", default="configs/xiaotian_multimodal_1200.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = Path(config["source"])
    cards = load_unique_player_cards(source)
    count = int(config["request_count"])
    jobs = build_jobs(
        cards,
        count,
        int(config["seed"]),
        int(config["turn_count_min"]),
        int(config["turn_count_max"]),
    )
    out = Path(args.out) if args.out else Path(config["run_dir"]) / "03_llm" / "requests.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, job in enumerate(jobs, start=1):
            sample_id = f"xiaotian_mm_{index:06d}_{job['player_role_card_id']}_v{job['variant']}"
            request_id = f"xiaotian_forward__{sample_id}"
            row = {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "job_type": "xiaotian_forward_multimodal",
                "sample_id": sample_id,
                "player_role_card_id": job["player_role_card_id"],
                "player_role_card": job["player_role_card"],
                "player_agent_type": job["player_agent_type"],
                "player_emotion_hint": job["player_emotion_hint"],
                "player_world_view_category": job["player_world_view_category"],
                "player_source_file": job["player_source_file"],
                "turn_count": job["turn_count"],
                "sysprompt": config["sysprompt"],
                "scene_description": config["scene_description"],
                "voice_description": config["voice_description"],
                "messages": [{"role": "user", "content": prompt_text(config, job, int(job["turn_count"]))}],
                "response_text": "",
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    stats = {
        "config": str(config_path),
        "source": str(source),
        "unique_player_cards": len(cards),
        "requests": len(jobs),
        "turn_counts": dict(Counter(str(job["turn_count"]) for job in jobs)),
        "role_card_use_counts": dict(Counter(str(job["variant"]) for job in jobs)),
        "out": str(out),
        "seed": int(config["seed"]),
    }
    out.with_suffix(out.suffix + ".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
