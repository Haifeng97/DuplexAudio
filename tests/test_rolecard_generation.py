from __future__ import annotations

import importlib.util
import random
import unittest
from pathlib import Path

from duplex_pipeline.rolecard_generation import (
    _balanced,
    _canonical_special_event,
    _expected_complete_mode,
    _opening_pools,
    _validate_turns,
)


def load_qwen_tts_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "03_run_qwen3_tts.py"
    spec = importlib.util.spec_from_file_location("run_qwen3_tts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RolecardGenerationTests(unittest.TestCase):
    def test_balanced_values_have_exact_counts(self) -> None:
        values = _balanced([1, 2, 3, 4, 5], 20, random.Random(7))
        self.assertEqual({value: values.count(value) for value in set(values)}, {1: 4, 2: 4, 3: 4, 4: 4, 5: 4})

    def test_opening_pools_route_and_filter_queries(self) -> None:
        rows = [
            {"query": "今天过得怎么样", "category": "闲聊主题", "ref_context": ""},
            {"query": "这局游戏怎么玩", "category": "以游戏为中心", "ref_context": ""},
            {"query": "我要骂你", "category": "安全相关-辱骂", "ref_context": ""},
            {"query": "先这样吧回头聊", "category": "闲聊主题", "ref_context": ""},
            {"query": "吉莉你好吗", "category": "闲聊主题", "ref_context": ""},
            {"query": "依赖上下文的问题", "category": "闲聊主题", "ref_context": "上下文"},
        ]
        section = {
            "fixed_roles": [{"name": name} for name in ("小田", "吉莉", "伞兵", "阿梅", "花傲天")],
            "complete_opening_terms": ["回头聊"], "opening_min_chars": 4, "opening_max_chars": 60,
        }
        pools = _opening_pools(rows, section)
        self.assertEqual([row["query"] for row in pools["game"]], ["这局游戏怎么玩"])
        self.assertEqual([row["query"] for row in pools["intervene"]], ["我要骂你"])
        self.assertEqual([row["query"] for row in pools["complete"]], ["先这样吧回头聊"])
        self.assertNotIn("吉莉你好吗", [row["query"] for row in pools["generic"]])

    def test_intervene_metadata_preserves_question_and_suffix(self) -> None:
        plan = {"special_scenario": "ai_intervenes_user"}
        question = "你不要再说这种不尊重人的话了"
        event = _canonical_special_event(plan, question, {"violation_category": "invalid"})
        self.assertEqual(event["user_text_until_trigger"] + event["user_text_after_trigger"], question)
        self.assertGreaterEqual(len(event["user_text_after_trigger"]), 6)
        self.assertEqual(event["violation_category"], "other")

    def test_complete_mode_is_deterministic_and_canonical(self) -> None:
        plan = {"sample_id": "sample-1", "special_scenario": "player_complete"}
        first = _expected_complete_mode(plan)
        self.assertEqual(first, _expected_complete_mode(plan))
        self.assertIn(first, {("normal_closing", "acknowledge"), ("force_stop", "silent")})


    def test_opening_punctuation_change_is_restored(self) -> None:
        plan = {"turn_count": 1, "special_scenario": "", "opening_query": "今天过得怎么样？"}
        parsed = {"turns": [{
            "turn_id": 1,
            "question_text": "今天过得怎么样？？",
            "answer_text": "挺好的。",
            "action_expression": "微笑。",
        }]}
        turns = _validate_turns(parsed, plan, 80, "助手", "玩家")
        self.assertEqual(turns[0]["question_text"], plan["opening_query"])

    def test_opening_semantic_change_is_rejected(self) -> None:
        plan = {"turn_count": 1, "special_scenario": "", "opening_query": "二乘以四的结果是八。"}
        parsed = {"turns": [{
            "turn_id": 1,
            "question_text": "二乘以二的结果是四。",
            "answer_text": "是的。",
            "action_expression": "点头。",
        }]}
        with self.assertRaisesRegex(ValueError, "opening_query_changed"):
            _validate_turns(parsed, plan, 80, "助手", "玩家")


class TtsQualityTests(unittest.TestCase):
    def test_duration_guard_rejects_non_capped_degenerate_audio(self) -> None:
        module = load_qwen_tts_module()
        task = {"text": "一" * 28}
        self.assertAlmostEqual(module.max_audio_sec_for_task(task, max_audio_floor_sec=10, max_sec_per_char=0.4, duration_guard_sec=5), 16.2)
        self.assertEqual(module.audio_quality_error(task, 25.12, min_audio_sec=1, max_audio_floor_sec=10, max_sec_per_char=0.4, duration_guard_sec=5), "audio_too_long_for_text")
        self.assertEqual(module.audio_quality_error(task, 8.0, min_audio_sec=1, max_audio_floor_sec=10, max_sec_per_char=0.4, duration_guard_sec=5), "")


if __name__ == "__main__":
    unittest.main()
