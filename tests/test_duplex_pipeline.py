from __future__ import annotations
from duplex_pipeline.planner import resolve_special_targets

import json
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from duplex_pipeline.incomplete import valid_cut
from duplex_pipeline.llm import run_requests
from duplex_pipeline.normalize import normalize_sources
from duplex_pipeline.offline_incomplete import conservative_split
from duplex_pipeline.orchestrate import _max_rebalanced_targets, publish_base_manifest, release_balanced
from duplex_pipeline.planner import SCENARIOS, largest_remainder, resolve_additions
from duplex_pipeline.snapshot import snapshot_ready_indexes
from duplex_pipeline.text import effective_char_count
from duplex_pipeline.tts import tts_fingerprint


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def custom_row(identifier: str, answer: str = "知道了，我们走后门。") -> dict:
    return {
        "id": identifier,
        "sysprompt": "请你扮演测试角色。",
        "turns": [{"turn_id": 1, "source": "current", "question_text": "我想去仓库那边看看", "answer_text": answer}],
        "meta": {"source_group_id": f"group::{identifier}"},
    }


class TextTests(unittest.TestCase):
    def test_effective_chars_exclude_punctuation_and_spaces(self) -> None:
        self.assertEqual(effective_char_count("我 想去，仓库！"), 5)

    def test_incomplete_cut_uses_effective_prefix_length(self) -> None:
        text = "我想去，仓库那边看看"
        self.assertEqual(valid_cut(text, 4, 3, 14), 4)
        self.assertIsNone(valid_cut(text, len(text), 3, 14))
        self.assertIsNone(valid_cut("一二三四五六七八九十十一十二十三十四十五后半", 15, 3, 14))

    def test_conservative_offline_split_requires_hanging_prefix(self) -> None:
        split = conservative_split("我想去仓库那边看看", 3, 14)
        self.assertIsNotNone(split)
        self.assertEqual(split["query_part1_text"], "我想去")
        self.assertEqual(split["query_part2_text"], "仓库那边看看")
        self.assertIsNone(conservative_split("好的我了解了，那就这样吧", 3, 14))
        self.assertIsNone(conservative_split("这句话已经说完了。但是还有后半句", 3, 14))


class PlannerTests(unittest.TestCase):
    def test_known_0818_allocation(self) -> None:
        base = {
            "normal_qa": 178438,
            "player_interrupts_ai": 40685,
            "incomplete_query": 22288,
            "incomplete_query_clarification": 3434,
            "player_backchannel": 28540,
            "other": 0,
        }
        ratios = {
            "normal_qa": 0.70,
            "player_interrupts_ai": 0.15,
            "incomplete_query": 0.04,
            "incomplete_query_clarification": 0.01,
            "player_backchannel": 0.05,
            "other": 0.05,
        }
        special, target, additions = resolve_additions(base, 1_151_896, 230_342, ratios)
        self.assertEqual(special, 75_015)
        self.assertEqual(sum(target.values()), 1_500_296)
        self.assertEqual(additions["normal_qa"], 871_769)
        self.assertEqual(additions["other"], 75_015)

    def test_largest_remainder_sums_exactly(self) -> None:
        ratios = {name: 1 / len(SCENARIOS) for name in SCENARIOS}
        self.assertEqual(sum(largest_remainder(101, ratios).values()), 101)



class ReleaseBalanceTests(unittest.TestCase):
    def test_post_filter_targets_keep_exact_ratio(self) -> None:
        ratios = {
            "normal_qa": 0.70,
            "player_interrupts_ai": 0.15,
            "incomplete_query": 0.04,
            "incomplete_query_clarification": 0.01,
            "player_backchannel": 0.05,
            "other": 0.05,
        }
        available = {
            "normal_qa": 760,
            "player_interrupts_ai": 150,
            "incomplete_query": 40,
            "incomplete_query_clarification": 10,
            "player_backchannel": 50,
            "other": 50,
        }
        targets = _max_rebalanced_targets(available, ratios)
        self.assertEqual(targets, {
            "normal_qa": 700,
            "player_interrupts_ai": 150,
            "incomplete_query": 40,
            "incomplete_query_clarification": 10,
            "player_backchannel": 50,
            "other": 50,
        })

    def test_release_can_keep_first_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_first = root / "base_first.jsonl"
            base_second = root / "base_second.jsonl"
            customized_root = root / "customized"
            special_root = root / "special"
            balanced_root = root / "balanced"
            write_jsonl(base_first, [{"id": "duplicate", "audio": "/tmp/first.wav", "scenario": "normal_qa"}])
            write_jsonl(base_second, [{"id": "duplicate", "audio": "/tmp/second.wav", "scenario": "normal_qa"}])
            write_jsonl(customized_root / "manifest.jsonl", [{"id": "custom", "audio": "/tmp/custom.wav", "scenario": "normal_qa"}])
            write_jsonl(special_root / "manifest.jsonl", [{"id": "special", "audio": "/tmp/special.wav", "scenario": "player_complete"}])
            config = {
                "base_manifests": [str(base_first), str(base_second)],
                "release": {
                    "customized_root": str(customized_root),
                    "special_root": str(special_root),
                    "balanced_root": str(balanced_root),
                    "absolute_wav_paths": True,
                    "duplicate_id_policy": "keep_first",
                },
            }

            result = release_balanced(config, root / "run")

            rows = [json.loads(line) for line in (balanced_root / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["duplicate", "custom", "special"])
            self.assertEqual(rows[0]["audio"], "/tmp/first.wav")
            self.assertEqual(result["rows"], 3)
            self.assertEqual(result["skipped_duplicate_ids"], {str(base_second): 1})


class NormalizeTests(unittest.TestCase):
    def test_namespaces_versions_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "input.jsonl"
            write_jsonl(source, [custom_row("same", "短回复"), custom_row("same", "这是信息更多而且仍然有效的回复")])
            config = {
                "inputs": {
                    "customized": [{
                        "path": str(source), "namespace": "customized_test",
                        "source_version": "test", "kind": "customized",
                    }],
                    "special": [],
                },
                "normalization": {"max_question_chars": 240, "max_answer_chars": 240},
            }
            stats = normalize_sources(config, root / "run")
            output = Path(stats["outputs"]["customized"]["path"])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["id"], "customized_test::same")
            self.assertEqual(rows[0]["original_id"], "same")
            self.assertEqual(rows[0]["source_version"], "test")
            self.assertEqual(stats["duplicate_counts"], {"customized_test": 1})
            self.assertTrue((root / "run" / "01_normalized" / "special.jsonl").is_file())


class FingerprintTests(unittest.TestCase):
    def test_fingerprint_covers_text_reference_and_generation(self) -> None:
        task = {"text": "你好", "ref_wav": "/tmp/ref.wav", "ref_text": "参考文本"}
        config = {
            "engine": "qwen3_tts", "model": "model", "model_version": "v1",
            "language": "Chinese", "sample_rate": 24000, "generation": {"temperature": 0},
        }
        original = tts_fingerprint(task, config)
        self.assertNotEqual(original, tts_fingerprint({**task, "text": "你好啊"}, config))
        self.assertNotEqual(original, tts_fingerprint({**task, "ref_text": "另一条参考"}, config))
        self.assertNotEqual(original, tts_fingerprint(task, {**config, "model_version": "v2"}))



class BasePublicationTests(unittest.TestCase):
    def test_copies_audio_and_rewrites_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_audio = root / "source.wav"
            source_audio.write_bytes(b"pcm-data")
            source_manifest = root / "source.jsonl"
            write_jsonl(source_manifest, [{"id": "sample-1", "audio": str(source_audio)}])

            result = publish_base_manifest(
                source_manifest,
                root / "published",
                copy_wav=True,
                absolute_wav_paths=True,
                workers=2,
            )

            row = json.loads((root / "published" / "manifest.jsonl").read_text())
            published_audio = Path(row["audio"])
            self.assertTrue(published_audio.is_absolute())
            self.assertEqual(published_audio.read_bytes(), b"pcm-data")
            self.assertEqual(result["counts"]["copied_wav"], 1)


class SpecialPlanningTests(unittest.TestCase):
    def test_special_shortfall_moves_to_available_scenario(self) -> None:
        self.assertEqual(
            resolve_special_targets(5474, {"ai_intervenes_user": 2734, "player_complete": 2749}),
            {"ai_intervenes_user": 2734, "player_complete": 2740},
        )


class TimedLlmTests(unittest.TestCase):
    def test_runtime_limit_stops_scheduling_and_keeps_completed_rows(self) -> None:
        class FakeClient:
            def __init__(self, config: dict) -> None:
                pass

            def complete(self, request: dict) -> str:
                time.sleep(0.03)
                return "{\"ok\":true}"

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            requests = root / "requests.jsonl"
            results = root / "results.jsonl"
            write_jsonl(requests, [
                {
                    "request_id": f"request-{index}",
                    "job_type": "test",
                    "sample_id": f"sample-{index}",
                    "messages": [{"role": "user", "content": "test"}],
                }
                for index in range(5)
            ])
            with patch("duplex_pipeline.llm.Dsv4Client", FakeClient):
                stats = run_requests(
                    {"dsv4": {}},
                    requests,
                    results,
                    resume=False,
                    retries=0,
                    progress_every=0,
                    concurrency=1,
                    quiet=True,
                    max_runtime_sec=0.01,
                )

            self.assertTrue(stats["timed_out"])
            self.assertEqual(stats["processed_this_run"], 1)
            self.assertEqual(stats["remaining"], 4)
            self.assertEqual(len(results.read_text(encoding="utf-8").splitlines()), 1)


class TimedSnapshotTests(unittest.TestCase):
    def test_snapshot_keeps_only_rows_with_complete_valid_tts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            fingerprinted = run_dir / "05_tts" / "fingerprinted"
            results_dir = run_dir / "05_tts" / "run" / "auto_results"
            audio = run_dir / "audio.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(audio), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"\\x00\\x00" * 28800)

            ready_asset = {"task_id": "ready", "audio": str(audio), "text": "测试文本"}
            missing_asset = {"task_id": "missing", "audio": str(audio), "text": "测试文本"}
            write_jsonl(fingerprinted / "customized_index.jsonl", [
                {"id": "keep", "tts_assets": {"query": ready_asset}},
                {"id": "drop", "tts_assets": {"query": missing_asset}},
            ])
            write_jsonl(fingerprinted / "special_index.jsonl", [])
            write_jsonl(results_dir / "tts_results_00.jsonl", [
                {"id": "ready", "status": "ok", "out": str(audio)},
            ])
            stats = snapshot_ready_indexes(
                {
                    "tts": {
                        "min_audio_sec": 1.0,
                        "generation": {
                            "max_audio_floor_sec": 10.0,
                            "max_sec_per_char": 1.2,
                        },
                    },
                },
                run_dir,
            )

            self.assertEqual(stats["ready_task_ids"], 1)
            self.assertEqual(stats["indexes"]["customized"]["counts"]["ready"], 1)
            rows = list(
                json.loads(line)
                for line in (
                    run_dir / "05_tts" / "timed_snapshot" / "customized_index.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual([row["id"] for row in rows], ["keep"])


if __name__ == "__main__":
    unittest.main()
