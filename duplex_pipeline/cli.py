from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict

from .config import load_config
from .incomplete import (
    apply_rank_results,
    export_clarification_requests,
    export_rank_requests,
    export_split_requests,
)
from .llm import run_requests
from .orchestrate import format_outputs, prepare_tts, release_balanced
from .scenarios import apply_clarification_results, materialize_scenarios
from .tts import fingerprint_tasks
from .normalize import normalize_sources
from .planner import build_plan
from .rolecard_generation import (
    apply_role_descriptions,
    apply_role_dialogues,
    export_role_dialogues,
    prepare_rolecard_plan,
)
from .state import RunState


StageFunction = Callable[[Dict[str, Any], Path], Dict[str, Any]]


def run_stage(config: Dict[str, Any], name: str, function: StageFunction, *, resume: bool) -> Dict[str, Any]:
    run_dir = Path(str(config["run_dir"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    state = RunState(run_dir)
    if resume and state.is_complete(name):
        return {"stage": name, "status": "skipped_complete"}
    state.update(name, "running", config=config["config_path"])
    try:
        result = function(config, run_dir)
    except Exception as exc:
        state.update(name, "failed", error=repr(exc))
        raise
    state.update(name, "complete", result=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven full-duplex data pipeline.")
    parser.add_argument("command", choices=[
        "normalize", "plan", "prepare", "llm-export-splits", "llm-export-rank",
        "llm-apply-rank", "llm-export-clarification", "llm-apply-clarification",
        "llm-run", "materialize", "tts-fingerprint",
        "tts-prepare", "format", "release",
        "rolecard-plan", "rolecard-apply-descriptions",
        "rolecard-export-dialogues", "rolecard-apply-dialogues",
    ])
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input", default="", help="Filled request JSONL or API request JSONL, depending on command.")
    parser.add_argument("--output", default="", help="API result JSONL for llm-run.")
    parser.add_argument("--index", default="", help="Raw scenario_index.jsonl for tts-fingerprint.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent DSV4 requests for llm-run.")
    parser.add_argument("--quiet", action="store_true", help="Disable live llm-run progress output.")
    parser.add_argument("--workers", type=int, default=100)
    args = parser.parse_args()
    config = load_config(Path(args.config))

    results = []
    run_dir = Path(str(config["run_dir"]))
    if args.command == "rolecard-plan":
        results.append(run_stage(config, "rolecard.plan", prepare_rolecard_plan, resume=args.resume))
    if args.command == "rolecard-apply-descriptions":
        if not args.input:
            parser.error("rolecard-apply-descriptions requires --input")
        results.append(apply_role_descriptions(config, run_dir, Path(args.input)))
    if args.command == "rolecard-export-dialogues":
        results.append(export_role_dialogues(config, run_dir))
    if args.command == "rolecard-apply-dialogues":
        if not args.input:
            parser.error("rolecard-apply-dialogues requires --input")
        results.append(apply_role_dialogues(config, run_dir, Path(args.input)))
    if args.command in {"normalize", "prepare"}:
        results.append(run_stage(config, "normalize", normalize_sources, resume=args.resume))
    if args.command in {"plan", "prepare"}:
        results.append(run_stage(config, "plan", build_plan, resume=args.resume))
    if args.command == "llm-export-splits":
        results.append(run_stage(config, "llm.incomplete_split.export", export_split_requests, resume=args.resume))
    if args.command == "llm-export-rank":
        if not args.input:
            parser.error("llm-export-rank requires --input filled split-candidate JSONL")
        results.append(export_rank_requests(config, run_dir, Path(args.input)))
    if args.command == "llm-apply-rank":
        if not args.input:
            parser.error("llm-apply-rank requires --input filled rank JSONL")
        results.append(apply_rank_results(config, run_dir, Path(args.input)))
    if args.command == "llm-export-clarification":
        results.append(export_clarification_requests(config, run_dir))
    if args.command == "llm-apply-clarification":
        if not args.input:
            parser.error("llm-apply-clarification requires --input filled clarification JSONL")
        results.append(apply_clarification_results(run_dir, Path(args.input)))
    if args.command == "llm-run":
        if not args.input or not args.output:
            parser.error("llm-run requires --input and --output")
        results.append(run_requests(
            dict(config["llm"]), Path(args.input), Path(args.output),
            resume=args.resume, retries=args.retries, progress_every=args.progress_every,
            concurrency=args.concurrency, quiet=args.quiet,
        ))
    if args.command == "materialize":
        results.append(run_stage(config, "materialize", materialize_scenarios, resume=args.resume))
    if args.command == "tts-fingerprint":
        if not args.input or not args.output:
            parser.error("tts-fingerprint requires --input raw tts_tasks.jsonl and --output directory; pass raw scenario_index with --index")
        if not getattr(args, "index", ""):
            parser.error("tts-fingerprint requires --index raw scenario_index.jsonl")
        results.append(fingerprint_tasks(config, Path(args.input), Path(args.index), Path(args.output)))
    if args.command == "tts-prepare":
        results.append(run_stage(config, "tts.prepare", prepare_tts, resume=args.resume))
    if args.command == "format":
        results.append(format_outputs(config, run_dir, args.workers))
    if args.command == "release":
        results.append(release_balanced(config, run_dir))
    print(json.dumps({"command": args.command, "results": results}, ensure_ascii=False, indent=2))
