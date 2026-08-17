#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VALID_JUDGMENTS = {"complete", "incomplete", "uncertain"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"expected JSON object at {path}:{line_no}")
            yield row


def parse_dataset(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError("--dataset must be NAME=SOURCE_DIR:OUTPUT_DIR")
    name, paths = parts
    path_parts = paths.rsplit(":", 1)
    if len(path_parts) != 2 or not all(part.strip() for part in path_parts):
        raise argparse.ArgumentTypeError("--dataset must be NAME=SOURCE_DIR:OUTPUT_DIR")
    return name.strip(), Path(path_parts[0]), Path(path_parts[1])


def load_decisions(
    filled_path: Path,
    auto_path: Path,
    expected_model_path: Path,
    uncertain_policy: str,
) -> tuple[dict[str, set[str]], dict[str, Counter], dict[str, Any]]:
    expected = {str(row.get("request_id") or "") for row in iter_jsonl(expected_model_path)}
    if "" in expected:
        raise ValueError("expected model requests contain an empty request_id")

    drops: dict[str, set[str]] = defaultdict(set)
    stats: dict[str, Counter] = defaultdict(Counter)
    filled_ids: set[str] = set()
    for row in iter_jsonl(filled_path):
        request_id = str(row.get("request_id") or "")
        dataset = str(row.get("dataset") or "")
        source_id = str(row.get("source_id") or "")
        judgment = str(row.get("judgment") or "").strip().lower()
        if not request_id or not dataset or not source_id:
            raise ValueError("filled result has an empty request_id, dataset, or source_id")
        if request_id in filled_ids:
            raise ValueError(f"duplicate filled request_id: {request_id}")
        if judgment not in VALID_JUDGMENTS:
            raise ValueError(f"invalid judgment for {request_id}: {judgment!r}")
        filled_ids.add(request_id)
        stats[dataset][f"model_{judgment}"] += 1
        if judgment == "complete" or (judgment == "uncertain" and uncertain_policy == "drop"):
            drops[dataset].add(source_id)

    if filled_ids != expected:
        missing = sorted(expected - filled_ids)[:20]
        extra = sorted(filled_ids - expected)[:20]
        raise ValueError(
            f"filled request coverage mismatch: expected={len(expected)} filled={len(filled_ids)} "
            f"missing={missing} extra={extra}"
        )

    auto_ids: set[str] = set()
    for row in iter_jsonl(auto_path):
        request_id = str(row.get("request_id") or "")
        dataset = str(row.get("dataset") or "")
        source_id = str(row.get("source_id") or "")
        if not request_id or not dataset or not source_id:
            raise ValueError("automatic decision has an empty request_id, dataset, or source_id")
        if request_id in auto_ids:
            raise ValueError(f"duplicate automatic request_id: {request_id}")
        if request_id in filled_ids:
            raise ValueError(f"request appears in both model and automatic decisions: {request_id}")
        if row.get("decision") != "drop":
            raise ValueError(f"automatic decision is not drop: {request_id}")
        auto_ids.add(request_id)
        drops[dataset].add(source_id)
        stats[dataset]["auto_drop_length"] += 1

    return drops, stats, {
        "expected_model_requests": len(expected),
        "filled_model_requests": len(filled_ids),
        "automatic_decisions": len(auto_ids),
        "uncertain_policy": uncertain_policy,
    }


def filter_manifest(source: Path, output: Path, drop_ids: set[str]) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    input_rows = output_rows = 0
    found: Counter[str] = Counter()
    with source.open("r", encoding="utf-8", errors="ignore") as handle, tmp.open("w", encoding="utf-8") as target:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {source}:{line_no}") from exc
            source_id = str(row.get("id") or "")
            if not source_id:
                raise ValueError(f"empty id at {source}:{line_no}")
            input_rows += 1
            if source_id in drop_ids:
                found[source_id] += 1
                continue
            target.write(line if line.endswith("\n") else line + "\n")
            output_rows += 1

    missing = sorted(drop_ids - set(found))
    repeated = sorted(source_id for source_id, count in found.items() if count != 1)
    if missing or repeated:
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"drop ID coverage failed for {source}: missing={missing[:20]} repeated={repeated[:20]}"
        )
    tmp.replace(output)
    return {
        "source": str(source),
        "output": str(output),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "dropped_rows": input_rows - output_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply filled F_WAIT completeness judgments to new manifest versions.")
    parser.add_argument("--filled", required=True)
    parser.add_argument("--auto_drop", required=True)
    parser.add_argument("--expected_model_requests", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset,
        metavar="NAME=SOURCE_DIR:OUTPUT_DIR",
    )
    parser.add_argument("--uncertain_policy", choices=["keep", "drop"], default="keep")
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()

    drops, decision_stats, decision_summary = load_decisions(
        Path(args.filled),
        Path(args.auto_drop),
        Path(args.expected_model_requests),
        args.uncertain_policy,
    )
    configured = {name for name, _, _ in args.dataset}
    unknown = sorted(set(drops) - configured)
    if unknown:
        raise ValueError(f"decisions contain unconfigured datasets: {unknown}")

    dataset_results: dict[str, Any] = {}
    for name, source_dir, output_dir in args.dataset:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        source_manifest = source_dir / "manifest.jsonl"
        if not source_manifest.is_file():
            raise FileNotFoundError(source_manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifests = ["manifest.jsonl"]
        if (source_dir / "manifest_relative.jsonl").is_file():
            manifests.append("manifest_relative.jsonl")
        manifest_results = [
            filter_manifest(source_dir / filename, output_dir / filename, drops.get(name, set()))
            for filename in manifests
        ]
        base = manifest_results[0]
        if any(result["input_rows"] != base["input_rows"] or result["output_rows"] != base["output_rows"] for result in manifest_results[1:]):
            raise ValueError(f"absolute and relative manifest counts differ for {name}")
        dataset_results[name] = {
            "source_dir": str(source_dir),
            "output_dir": str(output_dir),
            "drop_ids": len(drops.get(name, set())),
            "decisions": dict(decision_stats.get(name, Counter())),
            "manifests": manifest_results,
        }
        readme = (
            "F_WAIT completeness-filtered manifest version.\n"
            f"Source version: {source_dir}\n"
            "WAV files are reused through the unchanged audio paths in retained rows.\n"
            "Dropped rows: model judgment complete, plus the >=15-character automatic rule.\n"
            f"Uncertain policy: {args.uncertain_policy}\n"
        )
        (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    result = {
        "schema_version": "fwait_completeness_filter_apply_v1",
        "filled": str(Path(args.filled)),
        "auto_drop": str(Path(args.auto_drop)),
        "decision_summary": decision_summary,
        "datasets": dataset_results,
        "total_dropped": sum(row["drop_ids"] for row in dataset_results.values()),
    }
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in dataset_results.values():
        (Path(row["output_dir"]) / "filter_stats.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
