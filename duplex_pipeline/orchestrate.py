from __future__ import annotations

import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import atomic_write_json, canonical_json, iter_jsonl, stable_hash
from .planner import SCENARIOS, canonical_scenario, largest_remainder
from .tts import fingerprint_tasks


CUSTOM_SCENARIO_FILES = (
    "normal_qa.jsonl",
    "player_interrupts_ai.jsonl",
    "incomplete_query.jsonl",
    "incomplete_query_clarification.jsonl",
    "player_backchannel.jsonl",
)
SPECIAL_SCENARIOS = {"ai_intervenes_user", "player_complete"}


def _run(command: List[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}: {shlex.join(command)}")


def _append_files(paths: Iterable[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as target:
        for path in paths:
            with path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        target.write(line)
                        count += 1
    return count


def _copy_atomic(source: Path, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return False
    tmp = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def publish_base_manifest(
    source_manifest: Path,
    out_root: Path,
    *,
    copy_wav: bool,
    absolute_wav_paths: bool,
    workers: int,
) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    wav_dir = out_root / "wav"
    output = out_root / "manifest.jsonl"
    tmp = output.with_name(f"{output.name}.tmp-{os.getpid()}")
    counts: Counter = Counter()
    worker_count = max(1, workers)
    pending = deque()

    def write_completed(handle: Any) -> None:
        future, row, destination = pending.popleft()
        if future.result():
            counts["copied_wav"] += 1
        else:
            counts["reused_wav"] += 1
        row["audio"] = (
            str(destination.resolve())
            if absolute_wav_paths
            else str(destination.relative_to(out_root))
        )
        handle.write(canonical_json(row) + "\n")
        counts["written"] += 1

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor, tmp.open(
            "w", encoding="utf-8"
        ) as handle:
            for source_row in iter_jsonl(source_manifest):
                row = dict(source_row)
                sample_id = str(row.get("id") or "")
                source_audio = Path(str(row.get("audio") or ""))
                if not sample_id or not source_audio.is_file():
                    raise FileNotFoundError(
                        f"invalid source row id={sample_id!r} audio={source_audio}"
                    )
                if copy_wav:
                    wav_name = f"{stable_hash({'id': sample_id}, length=16)}__{source_audio.name}"
                    destination = wav_dir / wav_name
                    future = executor.submit(_copy_atomic, source_audio, destination)
                    pending.append((future, row, destination))
                    if len(pending) >= worker_count * 4:
                        write_completed(handle)
                else:
                    row["audio"] = str(source_audio.resolve())
                    handle.write(canonical_json(row) + "\n")
                    counts["written"] += 1
                    counts["source_wav_reused"] += 1
            while pending:
                write_completed(handle)
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)
    return {
        "mode": "base_manifest",
        "source_manifest": str(source_manifest),
        "manifest": str(output),
        "wav_dir": str(wav_dir) if copy_wav else "",
        "copy_wav": copy_wav,
        "absolute_wav_paths": absolute_wav_paths,
        "workers": worker_count,
        "counts": dict(counts),
    }


def prepare_tts(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    scenario_dir = run_dir / "04_scenarios"
    raw_root = run_dir / "05_tts" / "raw_parts"
    raw_root.mkdir(parents=True, exist_ok=True)
    tts_tasks = dict(config.get("tts_tasks") or {})
    voice_bank = str(tts_tasks.get("voice_bank") or "")
    if not voice_bank or not Path(voice_bank).is_file():
        raise FileNotFoundError("tts_tasks.voice_bank must point to an existing voice_bank.jsonl")
    inputs = [scenario_dir / name for name in CUSTOM_SCENARIO_FILES] + [scenario_dir / "special.jsonl"]
    part_dirs: List[Path] = []
    for source in inputs:
        if not source.is_file():
            raise FileNotFoundError(source)
        part_dir = raw_root / source.stem
        command = [
            sys.executable,
            "scripts/02_make_turn_tts_tasks.py",
            "--input", str(source),
            "--out_dir", str(part_dir),
            "--voice_bank", voice_bank,
            "--voice_strategy", str(tts_tasks.get("voice_strategy", "sample_hash")),
            "--voice_seed", str(int(tts_tasks.get("voice_seed", 42))),
            "--max_voice_refs", str(int(tts_tasks.get("max_voice_refs", 0))),
            "--max_ref_text_chars", str(int(tts_tasks.get("max_ref_text_chars", 12))),
            "--min_ref_snr", str(float(tts_tasks.get("min_ref_snr", 0.0))),
            "--streaming",
        ]
        if bool(tts_tasks.get("tts_text_punct", True)):
            command.append("--tts_text_punct")
        _run(command, part_dir / "prepare.log")
        part_dirs.append(part_dir)

    raw_combined = run_dir / "05_tts" / "raw_combined"
    raw_tasks = raw_combined / "tts_tasks.jsonl"
    raw_index = raw_combined / "scenario_index.jsonl"
    task_rows = _append_files((part / "tts_tasks.jsonl" for part in part_dirs), raw_tasks)
    index_rows = _append_files((part / "scenario_index.jsonl" for part in part_dirs), raw_index)
    fingerprinted = run_dir / "05_tts" / "fingerprinted"
    fingerprint_stats = fingerprint_tasks(config, raw_tasks, raw_index, fingerprinted)

    custom_index = fingerprinted / "customized_index.jsonl"
    special_index = fingerprinted / "special_index.jsonl"
    custom_count = special_count = 0
    with custom_index.open("w", encoding="utf-8") as custom_handle, special_index.open("w", encoding="utf-8") as special_handle:
        for row in iter_jsonl(fingerprinted / "scenario_index.jsonl"):
            if str(row.get("scenario") or "") in SPECIAL_SCENARIOS:
                special_handle.write(canonical_json(row) + "\n")
                special_count += 1
            else:
                custom_handle.write(canonical_json(row) + "\n")
                custom_count += 1

    commands_path = run_dir / "05_tts" / "NEXT_COMMANDS.sh"
    tts = dict(config["tts"])
    generation = dict(tts.get("generation") or {})
    release = dict(config["release"])
    fmt = dict(config.get("format") or {})
    task_file = (fingerprinted / "tts_tasks.jsonl").resolve()
    work_dir = (run_dir / "05_tts" / "run").resolve()
    validation = dict(config.get("tts_validation") or {})
    validation_enabled = bool(validation.get("enabled", False))
    validation_dir = (run_dir / "05_tts" / "validation").resolve()
    validated_custom_index = validation_dir / "customized_index.jsonl"
    validated_special_index = validation_dir / "special_index.jsonl"
    rejected_index = validation_dir / "rejected_scenarios.jsonl"
    asr_gpus = [str(gpu) for gpu in validation.get("gpus", [4, 5, 6, 7])]
    asr_result_paths = [
        validation_dir / f"asr_results_{index:02d}.jsonl"
        for index in range(len(asr_gpus))
    ]
    base_customized = (run_dir / "06_manifest" / "base" / "customized").resolve()
    base_special = (run_dir / "06_manifest" / "base" / "special").resolve()
    custom_enrichment = (scenario_dir / "multimodal_customized.jsonl").resolve()
    special_enrichment = (scenario_dir / "multimodal_special.jsonl").resolve()
    final_customized = Path(str(release["customized_root"])).resolve()
    final_special = Path(str(release["special_root"])).resolve()
    tts_command = (
        f"python scripts/03_run_tts_multi_gpu.py --tasks {shlex.quote(str(task_file))} "
        f"--work_dir {shlex.quote(str(work_dir))} --gpus 4,5,6,7 --procs_per_gpu 2 "
        f"--engine qwen3_tts --model_dir {shlex.quote(str(tts['model']))} "
        f"--batch_size {int(tts.get('batch_size', 128))} "
        f"--dtype {shlex.quote(str(generation.get('dtype', 'bfloat16')))} "
        f"--attn_implementation {shlex.quote(str(generation.get('attn_implementation', 'flash_attention_2')))} "
        f"--max_audio_floor_sec {float(generation.get('max_audio_floor_sec', 10.0))} "
        f"--max_sec_per_char {float(generation.get('max_sec_per_char', 0.4))} "
        f"--generation_guard_sec {float(generation.get('generation_guard_sec', 5.0))} "
        f"--codec_frame_rate {float(generation.get('codec_frame_rate', 12.0))} "
        f"--max_new_tokens_cap {int(generation.get('max_new_tokens_cap', 2048))} "
        f'--monitor_every 0.5 --progress_every 50 --project "$PROJECT"'
    )
    if bool(tts.get("shuffle_batches", False)):
        tts_command += f" --shuffle_batches --shuffle_seed {int(tts.get('shuffle_seed', 42))}"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'export PROJECT="${PROJECT:-Cgame_aimate_haifengjia}"',
        "",
        "# TTS: resume is content-addressed; rerunning keeps valid WAVs.",
        tts_command,
        "",
    ]
    format_custom_index = custom_index.resolve()
    format_special_index = special_index.resolve()
    if validation_enabled:
        if not asr_gpus:
            raise ValueError("tts_validation.gpus must not be empty")
        lines.extend([
            "# Strict post-TTS ASR validation. Worker output is kept in log files.",
            f"mkdir -p {shlex.quote(str(validation_dir / 'logs'))}",
            "pids=()",
        ])
        for shard_index, (gpu, result_path) in enumerate(zip(asr_gpus, asr_result_paths)):
            log_path = validation_dir / "logs" / f"asr_worker_{shard_index:02d}_gpu{gpu}.log"
            lines.extend([
                (
                    f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} python scripts/34_validate_tts_asr.py "
                    f"--tasks {shlex.quote(str(task_file))} --results {shlex.quote(str(result_path))} "
                    f"--model {shlex.quote(str(validation.get('model', 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')))} "
                    f"--batch_size {int(validation.get('batch_size', 64))} "
                    f"--bucket_size {int(validation.get('bucket_size', 1024))} "
                    f"--max_cer {float(validation.get('max_cer', 0.65))} "
                    f"--min_coverage {float(validation.get('min_coverage', 0.25))} "
                    f"--min_target_chars {int(validation.get('min_target_chars', 4))} "
                    f"--num_shards {len(asr_gpus)} --shard_index {shard_index} "
                    f"--progress_every {int(validation.get('progress_every', 1000))} "
                    f"> {shlex.quote(str(log_path))} 2>&1 &"
                ),
                "pids+=($!)",
            ])
        result_args = " ".join(
            f"--asr_results {shlex.quote(str(path))}" for path in asr_result_paths
        )
        lines.extend([
            'for pid in "${pids[@]}"; do wait "$pid"; done',
            (
                f"python scripts/35_filter_tts_validated_index.py "
                f"--index {shlex.quote(str((fingerprinted / 'scenario_index.jsonl').resolve()))} "
                f"{result_args} --customized_out {shlex.quote(str(validated_custom_index))} "
                f"--special_out {shlex.quote(str(validated_special_index))} "
                f"--rejected_out {shlex.quote(str(rejected_index))}"
            ),
            "",
        ])
        format_custom_index = validated_custom_index
        format_special_index = validated_special_index
    format_args = (
        f"--workers 100 --sample_rate {int(fmt.get('sample_rate', 24000))} "
        f"--chunk_ms {int(fmt.get('chunk_ms', 180))} "
        f"--tokenizer_json {shlex.quote(str(fmt.get('tokenizer_json', 'tokenizers/qwen3_8b/tokenizer.json')))} "
        f"--vad_mode {shlex.quote(str(fmt.get('vad_mode', 'silero')))}"
    )
    lines.extend([
        "# Format to an intermediate manifest, then append action tokens and publish absolute WAV paths.",
        (
            f"python scripts/22_format_duplex_manifest_parallel.py "
            f"--index {shlex.quote(str(format_custom_index))} "
            f"--out_dir {shlex.quote(str(base_customized))} {format_args}"
        ),
        (
            f"python scripts/22_format_duplex_manifest_parallel.py "
            f"--index {shlex.quote(str(format_special_index))} "
            f"--out_dir {shlex.quote(str(base_special))} {format_args}"
        ),
        (
            f"python scripts/24_apply_multimodal_enrichment.py "
            f"--manifest {shlex.quote(str(base_customized / 'manifest.jsonl'))} "
            f"--filled {shlex.quote(str(custom_enrichment))} "
            f"--out {shlex.quote(str(final_customized / 'manifest.jsonl'))} "
            f"--wav_dir {shlex.quote(str(final_customized / 'wav'))} "
            f"--tokenizer_json {shlex.quote(str(fmt.get('tokenizer_json', 'tokenizers/qwen3_8b/tokenizer.json')))} "
            f"--normalize_voice_prefix --progress_every 0"
        ),
        (
            f"python scripts/24_apply_multimodal_enrichment.py "
            f"--manifest {shlex.quote(str(base_special / 'manifest.jsonl'))} "
            f"--filled {shlex.quote(str(special_enrichment))} "
            f"--out {shlex.quote(str(final_special / 'manifest.jsonl'))} "
            f"--wav_dir {shlex.quote(str(final_special / 'wav'))} "
            f"--tokenizer_json {shlex.quote(str(fmt.get('tokenizer_json', 'tokenizers/qwen3_8b/tokenizer.json')))} "
            f"--normalize_voice_prefix --progress_every 0"
        ),
        "",
        f"python scripts/duplex_pipeline.py release --config {shlex.quote(str(config['config_path']))}",
    ])
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    result = {
        "raw_tasks": task_rows,
        "scenario_rows": index_rows,
        "customized_rows": custom_count,
        "special_rows": special_count,
        "fingerprint": fingerprint_stats,
        "commands": str(commands_path),
    }
    atomic_write_json(run_dir / "05_tts" / "stats.json", result)
    return result


def format_outputs(config: Dict[str, Any], run_dir: Path, workers: int) -> Dict[str, Any]:
    fingerprinted = run_dir / "05_tts" / "fingerprinted"
    validation = dict(config.get("tts_validation") or {})
    validated = run_dir / "05_tts" / "validation"
    release = dict(config["release"])
    fmt = dict(config.get("format") or {})
    base_root = run_dir / "06_manifest" / "base"
    enrichment_root = run_dir / "04_scenarios"
    outputs: Dict[str, Any] = {}
    for name, original_index, root_key, enrichment_name in (
        ("customized", "customized_index.jsonl", "customized_root", "multimodal_customized.jsonl"),
        ("special", "special_index.jsonl", "special_root", "multimodal_special.jsonl"),
    ):
        index = (
            validated / f"{name}_index.jsonl"
            if bool(validation.get("enabled", False))
            else fingerprinted / original_index
        )
        if not index.is_file():
            raise FileNotFoundError(index)
        base_dir = base_root / name
        format_command = [
            sys.executable, "scripts/22_format_duplex_manifest_parallel.py",
            "--index", str(index), "--out_dir", str(base_dir),
            "--workers", str(workers), "--sample_rate", str(int(fmt.get("sample_rate", 24000))),
            "--chunk_ms", str(int(fmt.get("chunk_ms", 180))),
            "--tokenizer_json", str(fmt.get("tokenizer_json", "tokenizers/qwen3_8b/tokenizer.json")),
            "--vad_mode", str(fmt.get("vad_mode", "silero")),
        ]
        _run(format_command, run_dir / "06_manifest" / f"format_{name}.log")
        final_dir = Path(str(release[root_key]))
        enrichment_file = enrichment_root / enrichment_name
        format_stats = json.loads(
            (base_dir / "parallel_format_stats.json").read_text(encoding="utf-8")
        )
        if enrichment_file.is_file() and enrichment_file.stat().st_size > 0:
            enrich_command = [
                sys.executable, "scripts/24_apply_multimodal_enrichment.py",
                "--manifest", str(base_dir / "manifest.jsonl"),
                "--filled", str(enrichment_file),
                "--out", str(final_dir / "manifest.jsonl"),
                "--wav_dir", str(final_dir / "wav"),
                "--tokenizer_json", str(fmt.get("tokenizer_json", "tokenizers/qwen3_8b/tokenizer.json")),
                "--normalize_voice_prefix",
                "--progress_every", "0",
            ]
            _run(enrich_command, run_dir / "06_manifest" / f"enrich_{name}.log")
            publication = {
                "mode": "multimodal_enrichment",
                **json.loads(
                    (final_dir / "manifest.jsonl.stats.json").read_text(encoding="utf-8")
                ),
            }
        else:
            publication = publish_base_manifest(
                base_dir / "manifest.jsonl",
                final_dir,
                copy_wav=bool(release.get("copy_base_wav", False)),
                absolute_wav_paths=bool(release.get("absolute_wav_paths", True)),
                workers=workers,
            )
        outputs[name] = {"format": format_stats, "publication": publication}
    return outputs


def _max_rebalanced_targets(available: Counter, ratios: Dict[str, float]) -> Dict[str, int]:
    positive = [
        int(available[name] / ratios[name])
        for name in SCENARIOS
        if ratios[name] > 0
    ]
    upper = min(sum(available.values()), min(positive, default=0))
    for total in range(upper, 0, -1):
        targets = largest_remainder(total, ratios)
        if all(targets[name] <= available[name] for name in SCENARIOS):
            return targets
    return {name: 0 for name in SCENARIOS}


def _release_rebalanced(
    config: Dict[str, Any],
    run_dir: Path,
    inputs: List[Path],
    out_root: Path,
) -> Dict[str, Any]:
    release = dict(config["release"])
    ratios = {name: float(config["ratios"][name]) for name in SCENARIOS}
    seed = int(release.get("rebalance_seed", config.get("planning", {}).get("seed", 20260818)))
    candidates: Dict[str, List[tuple[str, str]]] = {name: [] for name in SCENARIOS}
    seen: Dict[str, str] = {}
    input_counts: Dict[str, int] = {}
    available: Counter = Counter()
    for source in inputs:
        count = 0
        for row in iter_jsonl(source):
            sample_id = str(row.get("id") or "")
            if not sample_id:
                raise ValueError(f"missing id in {source}")
            previous = seen.get(sample_id)
            if previous is not None:
                raise ValueError(f"duplicate id {sample_id}: {previous} and {source}")
            seen[sample_id] = str(source)
            scenario = canonical_scenario(row)
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown scenario {scenario!r} in {source}")
            audio = Path(str(row.get("audio") or ""))
            if bool(release.get("absolute_wav_paths", True)) and not audio.is_absolute():
                raise ValueError(f"non-absolute audio path for {sample_id}: {audio}")
            rank = stable_hash({"id": sample_id, "scenario": scenario, "seed": seed})
            candidates[scenario].append((rank, sample_id))
            available[scenario] += 1
            count += 1
        input_counts[str(source)] = count
    targets = _max_rebalanced_targets(available, ratios)
    selected = set()
    for scenario in SCENARIOS:
        candidates[scenario].sort()
        selected.update(sample_id for _, sample_id in candidates[scenario][:targets[scenario]])
    output = out_root / "manifest.jsonl"
    written: Counter = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for source in inputs:
            for row in iter_jsonl(source):
                if str(row["id"]) not in selected:
                    continue
                handle.write(canonical_json(row) + "\n")
                written[canonical_scenario(row)] += 1
    result = {
        "manifest": str(output.resolve()),
        "rows": sum(written.values()),
        "inputs": input_counts,
        "available_counts": {name: available[name] for name in SCENARIOS},
        "scenario_counts": {name: written[name] for name in SCENARIOS},
        "target_counts": targets,
        "ratios": ratios,
        "rebalance_after_filter": True,
        "rebalance_seed": seed,
        "dropped_for_balance": sum(available.values()) - sum(written.values()),
        "audio_paths": "absolute",
        "copied_base_wav": False,
    }
    atomic_write_json(out_root / "release_stats.json", result)
    atomic_write_json(run_dir / "08_release" / "stats.json", result)
    return result

def release_balanced(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    release = dict(config["release"])
    inputs = [Path(path) for path in config.get("base_manifests", [])]
    inputs.extend([
        Path(str(release["customized_root"])) / "manifest.jsonl",
        Path(str(release["special_root"])) / "manifest.jsonl",
    ])
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    out_root = Path(str(release["balanced_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    if bool(release.get("rebalance_after_filter", False)):
        return _release_rebalanced(config, run_dir, inputs, out_root)
    output = out_root / "manifest.jsonl"
    db_path = run_dir / "08_release" / "ids.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE ids (id TEXT PRIMARY KEY,source TEXT NOT NULL)")
    scenario_counts: Counter = Counter()
    input_counts: Dict[str, int] = {}
    total = 0
    with output.open("w", encoding="utf-8") as handle:
        for path in inputs:
            current = 0
            for row in iter_jsonl(path):
                sample_id = str(row.get("id") or "")
                if not sample_id:
                    raise ValueError(f"missing id in {path}")
                try:
                    connection.execute("INSERT INTO ids VALUES (?,?)", (sample_id, str(path)))
                except sqlite3.IntegrityError as exc:
                    previous = connection.execute("SELECT source FROM ids WHERE id=?", (sample_id,)).fetchone()[0]
                    raise ValueError(f"duplicate id {sample_id}: {previous} and {path}") from exc
                audio = Path(str(row.get("audio") or ""))
                if bool(release.get("absolute_wav_paths", True)) and not audio.is_absolute():
                    raise ValueError(f"non-absolute audio path for {sample_id}: {audio}")
                handle.write(canonical_json(row) + "\n")
                scenario_counts[canonical_scenario(row)] += 1
                total += 1
                current += 1
                if total % 10000 == 0:
                    connection.commit()
            input_counts[str(path)] = current
    connection.commit()
    connection.close()
    db_path.unlink(missing_ok=True)
    result = {
        "manifest": str(output.resolve()), "rows": total,
        "inputs": input_counts, "scenario_counts": dict(scenario_counts),
        "audio_paths": "absolute", "copied_base_wav": False,
    }
    atomic_write_json(out_root / "release_stats.json", result)
    atomic_write_json(run_dir / "08_release" / "stats.json", result)
    return result
