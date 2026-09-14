#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_pipeline.config import load_config
from duplex_pipeline.io import atomic_write_json


def run_logged(
    command: List[str],
    log: Path,
    *,
    env: Dict[str, str],
    allow_failure: bool = False,
) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.returncode


def llm_command(
    config_path: Path,
    input_path: Path,
    output_path: Path,
    concurrency: int,
    max_runtime_sec: float,
    quiet: bool,
) -> List[str]:
    command = [
        sys.executable,
        "scripts/duplex_pipeline.py",
        "llm-run",
        "--config",
        str(config_path),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--concurrency",
        str(concurrency),
        "--max-runtime-sec",
        str(max(0.0, max_runtime_sec)),
        "--resume",
    ]
    if quiet:
        command.append("--quiet")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standard duplex pipeline for a wall-clock budget and publish complete samples."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-runtime-hours", type=float, required=True)
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tts-python", default="/data/haifengjia/miniforge3/envs/qwen3-tts/bin/python")
    parser.add_argument("--format-workers", type=int, default=100)
    parser.add_argument("--finalize-reserve-hours", type=float, default=0.75)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.max_runtime_hours <= 0:
        parser.error("--max-runtime-hours must be > 0")
    if args.llm_concurrency <= 0:
        parser.error("--llm-concurrency must be > 0")
    if args.finalize_reserve_hours < 0 or args.finalize_reserve_hours >= args.max_runtime_hours:
        parser.error("--finalize-reserve-hours must be >= 0 and smaller than the total runtime")

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    run_dir = Path(str(config["run_dir"]))
    timed_dir = run_dir / "09_timed"
    env = dict(os.environ)
    started = time.monotonic()
    deadline = started + args.max_runtime_hours * 3600.0
    finalize_deadline = deadline - args.finalize_reserve_hours * 3600.0
    total_sec = args.max_runtime_hours * 3600.0
    state: Dict[str, Any] = {
        "config": str(config_path),
        "started_epoch": time.time(),
        "max_runtime_hours": args.max_runtime_hours,
        "llm_concurrency": args.llm_concurrency,
        "gpus": args.gpus,
        "procs_per_gpu": args.procs_per_gpu,
        "batch_size": args.batch_size,
        "stages": [],
    }

    def record(name: str, command: List[str], *, allow_failure: bool = False) -> int:
        stage_started = time.monotonic()
        code = run_logged(
            command,
            timed_dir / f"{len(state['stages']):02d}_{name}.log",
            env=env,
            allow_failure=allow_failure,
        )
        state["stages"].append({
            "name": name,
            "returncode": code,
            "elapsed_sec": time.monotonic() - stage_started,
        })
        atomic_write_json(timed_dir / "timed_state.json", state)
        return code

    def phase_budget(end_fraction: float) -> float:
        phase_deadline = started + total_sec * end_fraction
        return max(0.0, phase_deadline - time.monotonic())

    base = [sys.executable, "scripts/duplex_pipeline.py"]
    rolecard = dict(config.get("rolecard_generation") or {})
    if rolecard:
        if bool(rolecard.get("include_multimodal_descriptions", True)):
            raise ValueError(
                "timed standard generation requires "
                "rolecard_generation.include_multimodal_descriptions=false"
            )
        record("rolecard_plan", base + ["rolecard-plan", "--config", str(config_path), "--resume"])
        record(
            "rolecard_export_dialogues",
            base + ["rolecard-export-dialogues", "--config", str(config_path)],
        )
        dialogue_requests = run_dir / "00_rolecard_generation" / "dialogue_requests.jsonl"
        dialogue_results = run_dir / "00_rolecard_generation" / "dialogue_results.jsonl"
        record(
            "llm_dialogues",
            llm_command(
                config_path,
                dialogue_requests,
                dialogue_results,
                args.llm_concurrency,
                phase_budget(0.20),
                args.quiet,
            ),
        )
        record(
            "rolecard_apply_dialogues",
            base + [
                "rolecard-apply-dialogues",
                "--config",
                str(config_path),
                "--input",
                str(dialogue_results),
            ],
        )
        split_fraction, rank_fraction, clarification_fraction = 0.25, 0.30, 0.33
    else:
        split_fraction, rank_fraction, clarification_fraction = 0.14, 0.28, 0.34

    record("prepare", base + ["prepare", "--config", str(config_path), "--resume"])
    record("export_splits", base + ["llm-export-splits", "--config", str(config_path), "--resume"])

    split_requests = run_dir / "03_llm" / "incomplete_split" / "requests.jsonl"
    split_results = run_dir / "03_llm" / "incomplete_split" / "results.jsonl"
    record(
        "llm_splits",
        llm_command(
            config_path,
            split_requests,
            split_results,
            args.llm_concurrency,
            phase_budget(split_fraction),
            args.quiet,
        ),
    )

    rank_requests = run_dir / "03_llm" / "incomplete_rank" / "requests.jsonl"
    rank_results = run_dir / "03_llm" / "incomplete_rank" / "results.jsonl"
    record(
        "export_rank",
        base + [
            "llm-export-rank",
            "--config",
            str(config_path),
            "--input",
            str(split_results),
        ],
    )
    record(
        "llm_rank",
        llm_command(
            config_path,
            rank_requests,
            rank_results,
            args.llm_concurrency,
            phase_budget(rank_fraction),
            args.quiet,
        ),
    )
    record(
        "apply_rank",
        base + [
            "llm-apply-rank",
            "--config",
            str(config_path),
            "--input",
            str(rank_results),
        ],
    )

    record("export_clarification", base + ["llm-export-clarification", "--config", str(config_path)])
    clarification_requests = run_dir / "03_llm" / "clarification" / "requests.jsonl"
    clarification_results = run_dir / "03_llm" / "clarification" / "results.jsonl"
    record(
        "llm_clarification",
        llm_command(
            config_path,
            clarification_requests,
            clarification_results,
            args.llm_concurrency,
            phase_budget(clarification_fraction),
            args.quiet,
        ),
    )
    record(
        "apply_clarification",
        base + [
            "llm-apply-clarification",
            "--config",
            str(config_path),
            "--input",
            str(clarification_results),
        ],
    )

    record("materialize", base + ["materialize", "--config", str(config_path)])
    record("tts_prepare", base + ["tts-prepare", "--config", str(config_path)])

    tts = dict(config["tts"])
    generation = dict(tts.get("generation") or {})
    tts_runtime = max(1.0, finalize_deadline - time.monotonic())
    tts_command = [
        args.tts_python,
        "scripts/03_run_tts_multi_gpu.py",
        "--tasks",
        str((run_dir / "05_tts" / "fingerprinted" / "tts_tasks.jsonl").resolve()),
        "--work_dir",
        str((run_dir / "05_tts" / "run").resolve()),
        "--gpus",
        args.gpus,
        "--procs_per_gpu",
        str(args.procs_per_gpu),
        "--engine",
        "qwen3_tts",
        "--model_dir",
        str(tts["model"]),
        "--batch_size",
        str(args.batch_size),
        "--dtype",
        str(generation.get("dtype", "bfloat16")),
        "--attn_implementation",
        str(generation.get("attn_implementation", "flash_attention_2")),
        "--max_audio_floor_sec",
        str(float(generation.get("max_audio_floor_sec", 10.0))),
        "--max_sec_per_char",
        str(float(generation.get("max_sec_per_char", 1.2))),
        "--generation_guard_sec",
        str(float(generation.get("generation_guard_sec", 5.0))),
        "--max_audio_cap_ratio",
        str(float(generation.get("max_audio_cap_ratio", 0.9))),
        "--codec_frame_rate",
        str(float(generation.get("codec_frame_rate", 12.0))),
        "--max_new_tokens_cap",
        str(int(generation.get("max_new_tokens_cap", 2048))),
        "--max_runtime_sec",
        str(tts_runtime),
        "--stop_grace_sec",
        "30",
        "--monitor_every",
        "5",
        "--progress_every",
        "0",
        "--project",
        env.get("PROJECT", "Cgame_aimate_haifengjia"),
        "--python",
        args.tts_python,
    ]
    if bool(tts.get("shuffle_batches", False)):
        tts_command.extend([
            "--shuffle_batches",
            "--shuffle_seed",
            str(int(tts.get("shuffle_seed", 42))),
        ])
    record("tts", tts_command, allow_failure=True)

    record(
        "finalize_partial",
        base + [
            "finalize-partial",
            "--config",
            str(config_path),
            "--workers",
            str(args.format_workers),
        ],
    )
    final_manifest = Path(str(config["release"]["balanced_root"])) / "manifest.jsonl"
    record(
        "validate",
        [
            sys.executable,
            "scripts/05_validate_duplex_manifest.py",
            "--manifest",
            str(final_manifest),
            "--show",
            "0",
        ],
    )
    state["finished_epoch"] = time.time()
    state["elapsed_sec"] = time.monotonic() - started
    state["manifest"] = str(final_manifest.resolve())
    atomic_write_json(timed_dir / "timed_state.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
