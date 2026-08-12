#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from duplex_label_protocol import (  # noqa: E402
    EOR,
    FD_A_ANSWER,
    FD_C_INTERVENE,
    FD_D_WAIT,
    FD_IDLE,
    FD_I_COMPLETE,
)
from special_scenario_schema import (  # noqa: E402
    AI_INTERVENES_USER,
    PLAYER_COMPLETE,
    SCHEMA_VERSION,
    SpecialScenarioError,
    validate_special_row,
)


def load_script(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FORMATTER = load_script("format_duplex_manifest", "04_format_duplex_manifest.py")
CANDIDATE_MAKER = load_script("make_scenario_candidate_pools", "01_make_scenario_candidate_pools.py")
VALIDATOR = load_script("validate_duplex_manifest", "05_validate_duplex_manifest.py")
QWEN_RUNNER = load_script("run_qwen3_tts", "03_run_qwen3_tts.py")
TASK_MAKER = load_script("make_turn_tts_tasks", "02_make_turn_tts_tasks.py")


def write_tone(path: Path, voiced_sec: float, sample_rate: int = 24000) -> None:
    silence_n = int(0.25 * sample_rate)
    voiced_n = int(voiced_sec * sample_rate)
    samples = [0] * silence_n
    samples.extend(
        int(5000 * math.sin(2 * math.pi * 220 * idx / sample_rate))
        for idx in range(voiced_n)
    )
    samples.extend([0] * silence_n)
    raw = struct.pack("<" + "h" * len(samples), *samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw)


def base_meta(scenario: str, history_count: int) -> dict[str, Any]:
    return {
        "dataset": "unit_test",
        "split": "train",
        "language": "zh",
        "role_name": "吉莉",
        "player_name": "玩家",
        "turn_count": history_count + 1,
        "history_turn_count": history_count,
        "source_group_id": f"unit::{scenario}",
        "original_dialogue_id": f"unit_{scenario}",
    }


def intervene_row() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": "unit_intervene",
        "scenario": AI_INTERVENES_USER,
        "sysprompt": "请你扮演吉莉与我对话。",
        "turns": [
            {
                "turn_id": 1,
                "source": "history",
                "question_text": "上一局怎么打？",
                "answer_text": "先稳住。",
                "needs_tts": True,
                "train_answer": True,
                "question_speaker": "玩家",
                "answer_speaker": "吉莉",
            },
            {
                "turn_id": 2,
                "source": "current",
                "question_text": "你这个人就是废物你别再继续说话了",
                "answer_text": "打住，这话不尊重人。",
                "needs_tts": True,
                "train_answer": True,
                "question_speaker": "玩家",
                "answer_speaker": "吉莉",
                "event": {
                    "type": "intervene",
                    "user_text_until_trigger": "你这个人就是废物",
                    "user_text_after_trigger": "你别再继续说话了",
                    "violation_category": "abuse",
                },
            },
        ],
        "meta": base_meta(AI_INTERVENES_USER, 1),
    }


def complete_row(response_mode: str) -> dict[str, Any]:
    acknowledge = response_mode == "acknowledge"
    event = {
        "type": "complete",
        "completion_type": "normal_closing" if acknowledge else "force_stop",
        "response_mode": response_mode,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "id": f"unit_complete_{response_mode}",
        "scenario": PLAYER_COMPLETE,
        "sysprompt": "请你扮演吉莉与我对话。",
        "turns": [
            {
                "turn_id": 1,
                "source": "current",
                "question_text": "那先这样，回头聊。",
                "answer_text": "好的。" if acknowledge else None,
                "needs_tts": True,
                "train_answer": acknowledge,
                "question_speaker": "玩家",
                "answer_speaker": "吉莉" if acknowledge else None,
                "event": event,
            }
        ],
        "meta": base_meta(PLAYER_COMPLETE, 0),
    }


class SpecialScenarioPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.history_wav = self.root / "history.wav"
        self.prefix_wav = self.root / "prefix.wav"
        self.suffix_wav = self.root / "suffix.wav"
        self.complete_wav = self.root / "complete.wav"
        write_tone(self.history_wav, 1.1)
        write_tone(self.prefix_wav, 1.2)
        write_tone(self.suffix_wav, 1.4)
        write_tone(self.complete_wav, 1.2)
        self.args = SimpleNamespace(
            sample_rate=24000,
            chunk_ms=180,
            noise_rms=0.003,
            text_tokenizer=FORMATTER.TextTokenizer(""),
            vad_processor=FORMATTER.VadSilenceReplacer("energy", 24000, 0.003),
            min_query_audio_sec=1.0,
            initial_idle_chunks=2,
            initial_idle_sec_min=0.5,
            initial_idle_sec_max=1.5,
            final_idle_chunks=2,
            inter_turn_idle_sec_min=1.0,
            inter_turn_idle_sec_max=3.0,
            disable_inter_turn_idle=False,
            min_intervene_suffix_audio_sec=1.0,
            min_intervene_overlap_sec=0.3,
            intervene_reaction_chunks_min=0,
            intervene_reaction_chunks_max=2,
            intervene_answer_delay_sec_min=0.18,
            intervene_answer_delay_sec_max=0.54,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_schema_rejects_punctuation_only_intervene_suffix(self) -> None:
        row = intervene_row()
        row["turns"][-1]["question_text"] = "你这个人就是废物？！"
        row["turns"][-1]["event"]["user_text_after_trigger"] = "？！"
        with self.assertRaises(SpecialScenarioError) as ctx:
            validate_special_row(row)
        self.assertEqual(ctx.exception.code, "intervene_suffix_too_short")

    def test_mixed_candidate_pool_routes_both_special_scenarios(self) -> None:
        input_path = self.root / "mixed.jsonl"
        out_dir = self.root / "candidates"
        rows = [intervene_row(), complete_row("acknowledge")]
        input_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        argv = [
            "01_make_scenario_candidate_pools.py",
            "--input",
            str(input_path),
            "--out_dir",
            str(out_dir),
            "--limit_each",
            "0",
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            CANDIDATE_MAKER.main()
        combined = [
            json.loads(line)
            for line in (out_dir / "special_scenarios_candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.assertEqual(
            {row["scenario"] for row in combined},
            {AI_INTERVENES_USER, PLAYER_COMPLETE},
        )
        stats = json.loads((out_dir / "candidate_pool_stats.json").read_text(encoding="utf-8"))
        self.assertEqual(stats["special_input_rows"], 2)
        self.assertEqual(stats["special_combined_written"], 2)

    def test_tts_tasks_split_intervene_and_keep_one_voice(self) -> None:
        row = intervene_row()
        tasks: list[dict[str, Any]] = []
        picker = TASK_MAKER.VoicePicker(
            [],
            fallback_ref_wav=str(self.history_wav),
            fallback_ref_text="参考文本。",
            strategy="sample_hash",
            seed=42,
        )
        attached = TASK_MAKER.attach_assets(
            row,
            self.root / "query_wav",
            picker,
            tasks,
            tts_text_punct=True,
        )
        self.assertEqual(
            [task["key"] for task in tasks],
            ["turn001_query", "intervene_query_prefix", "intervene_query_suffix"],
        )
        self.assertEqual(len({task["ref_wav"] for task in tasks}), 1)
        self.assertEqual(
            tasks[-2]["text"],
            row["turns"][-1]["event"]["user_text_until_trigger"],
        )
        self.assertEqual(
            tasks[-1]["text"],
            row["turns"][-1]["event"]["user_text_after_trigger"],
        )
        self.assertEqual(len(attached["tts_assets"]), 3)

    def test_intervene_formatter_overlays_answer_while_player_speaks(self) -> None:
        row = intervene_row()
        row["question_text"] = row["turns"][-1]["question_text"]
        row["answer_text"] = row["turns"][-1]["answer_text"]
        row["tts_assets"] = {
            "turn001_query": {"audio": str(self.history_wav)},
            "intervene_query_prefix": {"audio": str(self.prefix_wav)},
            "intervene_query_suffix": {"audio": str(self.suffix_wav)},
        }
        out = self.root / "intervene_out.wav"
        manifest = FORMATTER.build_intervene(row, out, self.args)
        labels = [item["label"] for item in manifest["timeline"]]
        c_idx = labels.index(FD_C_INTERVENE)
        self.assertEqual(labels[c_idx + 1], FD_A_ANSWER)
        self.assertGreaterEqual(manifest["intervention"]["overlap_sec"], 0.3)
        self.assertGreaterEqual(manifest["intervention"]["reaction_delay_sec"], 0.18)
        self.assertLessEqual(manifest["intervention"]["reaction_delay_sec"], 0.54)
        self.assertLess(
            manifest["intervention"]["answer_start_sec"],
            manifest["intervention"]["player_end_sec"],
        )
        self.assertEqual(VALIDATOR.protocol_errors(manifest), [])
        with wave.open(str(out), "rb") as wf:
            self.assertEqual(
                wf.getnframes(),
                len(manifest["timeline"]) * int(24000 * 0.18),
            )

    def test_complete_acknowledge_and_silent_timelines(self) -> None:
        for response_mode in ("acknowledge", "silent"):
            with self.subTest(response_mode=response_mode):
                row = complete_row(response_mode)
                row["question_text"] = row["turns"][-1]["question_text"]
                row["answer_text"] = row["turns"][-1]["answer_text"]
                row["tts_assets"] = {
                    "complete_query": {"audio": str(self.complete_wav)},
                }
                out = self.root / f"complete_{response_mode}.wav"
                manifest = FORMATTER.build_complete(row, out, self.args)
                labels = [item["label"] for item in manifest["timeline"]]
                i_idx = labels.index(FD_I_COMPLETE)
                expected = FD_A_ANSWER if response_mode == "acknowledge" else FD_IDLE
                self.assertEqual(labels[i_idx + 1], expected)
                if response_mode == "silent":
                    self.assertNotIn(FD_A_ANSWER, labels[i_idx + 1:])
                    self.assertNotIn(EOR, labels[i_idx + 1:])
                self.assertEqual(VALIDATOR.protocol_errors(manifest), [])

    def test_qwen_runner_resumes_entirely_cached_shard_without_model_import(self) -> None:
        task_path = self.root / "tts_tasks.jsonl"
        result_path = self.root / "tts_results.jsonl"
        task = {
            "id": "cached_task",
            "text": "测试。",
            "out": str(self.complete_wav),
            "ref_wav": str(self.history_wav),
            "ref_text": "参考文本。",
        }
        task_path.write_text(json.dumps(task, ensure_ascii=False) + "\n", encoding="utf-8")
        argv = [
            "03_run_qwen3_tts.py",
            "--tasks",
            str(task_path),
            "--results",
            str(result_path),
            "--batch_size",
            "128",
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            QWEN_RUNNER.main()
        rows = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(
            rows,
            [{
                "id": "cached_task",
                "status": "cached",
                "out": str(self.complete_wav),
                "duration_sec": 1.7,
            }],
        )

    def test_length_sort_and_batch_partition(self) -> None:
        tasks = [
            {"id": "long", "text": "一" * 20, "ref_duration": 2.0},
            {"id": "short", "text": "一" * 2, "ref_duration": 1.0},
            {"id": "middle", "text": "一" * 8, "ref_duration": 1.0},
        ]
        ordered = QWEN_RUNNER.length_sorted(tasks)
        self.assertEqual([task["id"] for task in ordered], ["short", "middle", "long"])
        grouped = QWEN_RUNNER.batches(ordered, 2)
        self.assertEqual([len(group) for group in grouped], [2, 1])

    def test_qwen_batch_shuffle_preserves_buckets_and_is_deterministic(self) -> None:
        tasks = [{"id": str(length), "text": "一" * length} for length in range(1, 13)]
        ordered = QWEN_RUNNER.make_task_batches(
            tasks,
            3,
            shuffle_batches=False,
            shuffle_seed=7,
        )
        shuffled = QWEN_RUNNER.make_task_batches(
            tasks,
            3,
            shuffle_batches=True,
            shuffle_seed=7,
        )
        repeated = QWEN_RUNNER.make_task_batches(
            tasks,
            3,
            shuffle_batches=True,
            shuffle_seed=7,
        )
        self.assertNotEqual(
            [[task["id"] for task in batch] for batch in ordered],
            [[task["id"] for task in batch] for batch in shuffled],
        )
        self.assertEqual(
            [[task["id"] for task in batch] for batch in shuffled],
            [[task["id"] for task in batch] for batch in repeated],
        )
        self.assertEqual(
            sorted(task["id"] for batch in shuffled for task in batch),
            sorted(task["id"] for task in tasks),
        )
        for batch in shuffled:
            lengths = [len(task["text"]) for task in batch]
            self.assertLessEqual(max(lengths) - min(lengths), 2)

    def test_qwen_batch_token_limit_tracks_longest_text_and_has_cap(self) -> None:
        kwargs = {
            "fixed_max_new_tokens": 0,
            "max_audio_floor_sec": 10.0,
            "max_sec_per_char": 1.2,
            "generation_guard_sec": 5.0,
            "codec_frame_rate": 12.0,
            "max_new_tokens_cap": 2048,
        }
        self.assertEqual(QWEN_RUNNER.batch_max_new_tokens([{"text": "测试"}], **kwargs), 180)
        self.assertEqual(QWEN_RUNNER.batch_max_new_tokens([{"text": "一" * 20}], **kwargs), 348)
        self.assertEqual(QWEN_RUNNER.batch_max_new_tokens([{"text": "一" * 500}], **kwargs), 2048)
        self.assertEqual(
            QWEN_RUNNER.batch_max_new_tokens(
                [{"text": "测试"}],
                **{**kwargs, "fixed_max_new_tokens": 512},
            ),
            512,
        )

    def test_qwen_audio_quality_rejects_short_long_and_token_limit_outputs(self) -> None:
        task = {"text": "测试"}
        kwargs = {
            "min_audio_sec": 1.0,
            "max_audio_floor_sec": 10.0,
            "max_sec_per_char": 1.2,
        }
        self.assertEqual(
            QWEN_RUNNER.audio_quality_error(task, 0.9, **kwargs),
            "audio_too_short",
        )
        self.assertEqual(
            QWEN_RUNNER.audio_quality_error(task, 10.5, **kwargs),
            "audio_too_long_for_text",
        )
        self.assertEqual(
            QWEN_RUNNER.audio_quality_error(
                task,
                11.9,
                token_limit=144,
                codec_frame_rate=12.0,
                **kwargs,
            ),
            "generation_reached_token_limit",
        )
        self.assertEqual(QWEN_RUNNER.audio_quality_error(task, 2.0, **kwargs), "")


if __name__ == "__main__":
    unittest.main()

