#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "fwait_completeness_v1"
FWAIT = "<FD_F_WAIT>"
FWAIT_SCENARIOS = {"incomplete_query", "incomplete_query_clarification"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at {path}:{line_no}") from exc


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--dataset must be NAME=/path/to/manifest.jsonl")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("--dataset requires a non-empty name and path")
    return name, Path(raw_path)


def has_fwait(row: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("label") == FWAIT
        for item in (row.get("timeline") or [])
    )


def extraction_fields(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    source_row = row.get("source_row") if isinstance(row.get("source_row"), dict) else {}
    source_request = (
        source_row.get("source_request")
        if isinstance(source_row.get("source_request"), dict)
        else {}
    )
    partial = str(
        source_row.get("query_part1_text")
        or source_row.get("partial_question_text")
        or source_request.get("query_part1_text")
        or source_request.get("partial_question_text")
        or ""
    ).strip()
    split = source_row.get("split") if isinstance(source_row.get("split"), dict) else {}
    if not split and isinstance(source_request.get("split"), dict):
        split = source_request["split"]
    return source_row, source_request, partial, split


def special_turn_index(source_row: dict[str, Any], turns: list[dict[str, Any]]) -> int:
    wanted = source_row.get("incomplete_turn_id") or source_row.get("inserted_turn_id")
    if wanted is not None:
        for idx, turn in enumerate(turns):
            if turn.get("turn_id") == wanted:
                return idx
    for idx, turn in enumerate(turns):
        if turn.get("source") == "inserted_incomplete_query":
            return idx
    return max(0, len(turns) - 1)


def prior_context(source_row: dict[str, Any], limit: int) -> list[dict[str, str]]:
    turns = [turn for turn in (source_row.get("turns") or []) if isinstance(turn, dict)]
    if not turns or limit <= 0:
        return []
    idx = special_turn_index(source_row, turns)
    output: list[dict[str, str]] = []
    for turn in turns[max(0, idx - limit):idx]:
        question = str(turn.get("question_text") or "").strip()
        answer = str(turn.get("answer_text") or "").strip()
        if question or answer:
            output.append({"user": question, "assistant": answer})
    return output


def prompt_text(partial: str, context: list[dict[str, str]]) -> str:
    if context:
        context_text = "\n".join(
            f"玩家：{turn['user']}\nAI：{turn['assistant']}"
            for turn in context
        )
    else:
        context_text = "无"
    return (
        "判断下面玩家已经说出的内容，单独看是否已经足以作为一条自然、完整、可以直接回复的语音。"
        "这些候选由程序随机截断产生，不能因为数据名是不完整 query 就默认它不完整。\n\n"
        "标签定义：\n"
        "- complete：语法或语义已经闭合，是自然的陈述、问题、指令或感叹；即使后面还能补充细节，也算 complete。\n"
        "- incomplete：明显停在词语中间、连接词/虚词后、未完成的句法成分或强依赖后文，当前无法自然作答。\n"
        "- uncertain：仅凭现有内容确实难以稳定判断。不要为了避免判断而滥用 uncertain。\n\n"
        "只根据下面展示的已说内容判断，不猜测隐藏的后半句。reason 用一句简短中文说明。"
        "只返回合法 JSON，不要 Markdown："
        '{"judgment":"complete|incomplete|uncertain","confidence":"high|medium|low","reason":"..."}'
        f"\n\n此前对话：\n{context_text}\n\n玩家已经说出的内容：\n{partial}"
    )


def common_record(
    dataset: str,
    manifest: Path,
    row: dict[str, Any],
    partial: str,
    cut_char_index: int,
    context: list[dict[str, str]],
) -> dict[str, Any]:
    sample_id = str(row.get("id") or "")
    if not sample_id:
        raise ValueError(f"{dataset}: empty sample id")
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": f"{dataset}::{sample_id}",
        "dataset": dataset,
        "source_manifest": str(manifest),
        "source_id": sample_id,
        "scenario": str(row.get("scenario") or ""),
        "partial_query": partial,
        "partial_query_char_count": len(partial),
        "cut_char_index": cut_char_index,
        "prior_dialogue": context,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export text-only judgments for whether F_WAIT prefixes are actually complete utterances."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset,
        metavar="NAME=MANIFEST",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--auto_drop_chars", type=int, default=15)
    parser.add_argument("--context_turns", type=int, default=2)
    args = parser.parse_args()
    if args.auto_drop_chars <= 0:
        raise SystemExit("--auto_drop_chars must be > 0")
    if args.context_turns < 0:
        raise SystemExit("--context_turns must be >= 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "llm_requests.jsonl"
    auto_drop_path = out_dir / "auto_drop_length_ge15.jsonl"
    index_path = out_dir / "candidate_index.jsonl"
    stats: Counter = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    by_scenario: dict[str, Counter] = defaultdict(Counter)
    seen_request_ids: set[str] = set()

    with (
        requests_path.open("w", encoding="utf-8") as request_file,
        auto_drop_path.open("w", encoding="utf-8") as auto_drop_file,
        index_path.open("w", encoding="utf-8") as index_file,
    ):
        for dataset, manifest in args.dataset:
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            for row in iter_jsonl(manifest):
                scenario = str(row.get("scenario") or "")
                if scenario not in FWAIT_SCENARIOS:
                    continue
                if not has_fwait(row):
                    raise ValueError(f"{manifest}: {row.get('id')}: F_WAIT scenario has no {FWAIT}")
                source_row, _, partial, split = extraction_fields(row)
                if not partial:
                    raise ValueError(f"{manifest}: {row.get('id')}: missing partial query text")
                if split.get("cut_char_index") is None:
                    raise ValueError(f"{manifest}: {row.get('id')}: missing cut_char_index")
                cut_char_index = int(split["cut_char_index"])
                context = prior_context(source_row, args.context_turns)
                common = common_record(dataset, manifest, row, partial, cut_char_index, context)
                request_id = common["request_id"]
                if request_id in seen_request_ids:
                    raise ValueError(f"duplicate request_id: {request_id}")
                seen_request_ids.add(request_id)

                stats["fwait_total"] += 1
                by_dataset[dataset]["fwait_total"] += 1
                by_scenario[scenario]["fwait_total"] += 1
                if len(partial) >= args.auto_drop_chars:
                    decision = {
                        **common,
                        "status": "auto_drop_length",
                        "judgment": "complete",
                        "confidence": "rule",
                        "reason": f"partial_query_char_count >= {args.auto_drop_chars}",
                        "decision": "drop",
                    }
                    auto_drop_file.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
                    index_file.write(json.dumps(decision, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["auto_drop_length"] += 1
                    by_dataset[dataset]["auto_drop_length"] += 1
                    by_scenario[scenario]["auto_drop_length"] += 1
                    continue

                request = {
                    **common,
                    "status": "needs_model",
                    "llm_messages": [{"role": "user", "content": prompt_text(partial, context)}],
                    "judgment": "",
                    "confidence": "",
                    "reason": "",
                }
                request_file.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
                index_file.write(json.dumps(common | {"status": "needs_model"}, ensure_ascii=False, separators=(",", ":")) + "\n")
                stats["needs_model"] += 1
                by_dataset[dataset]["needs_model"] += 1
                by_scenario[scenario]["needs_model"] += 1

    result = {
        "schema_version": SCHEMA_VERSION,
        "datasets": [{"name": name, "manifest": str(path)} for name, path in args.dataset],
        "auto_drop_chars": args.auto_drop_chars,
        "auto_drop_rule": "len(partial_query.strip()) >= auto_drop_chars",
        "context_turns": args.context_turns,
        "outputs": {
            "llm_requests": str(requests_path),
            "auto_drop_length": str(auto_drop_path),
            "candidate_index": str(index_path),
        },
        "counts": dict(stats),
        "by_dataset": {name: dict(counts) for name, counts in by_dataset.items()},
        "by_scenario": {name: dict(counts) for name, counts in by_scenario.items()},
    }
    (out_dir / "stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
