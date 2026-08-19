#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def prompt(partial: str) -> str:
    return (
        "判断下面玩家在 F_WAIT 时刻已经说出的内容，是否已经足以作为一条自然、完整、可以直接回复的话。"
        "不要假设还有未展示的后半句。\n"
        "complete：语法或语义已经闭合，可以自然回复。\n"
        "incomplete：明显停在词中间、连接成分后或未完成的句法结构，不能自然回复。\n"
        "uncertain：仅凭文本确实无法稳定判断。\n"
        "只返回 JSON："
        '{"judgment":"complete|incomplete|uncertain","confidence":"high|medium|low","reason":"一句简短中文"}'
        f"\n\n玩家已经说出的内容：\n{partial}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export aligned external F_WAIT prefixes for local LLM judgment.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejected", default="")
    parser.add_argument("--include_uncertain_alignment", action="store_true")
    args = parser.parse_args()
    source = Path(args.results)
    out = Path(args.out)
    rejected = Path(args.rejected) if args.rejected else out.with_suffix(out.suffix + ".rejected.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with source.open("r", encoding="utf-8", errors="ignore") as handle, out.open("w", encoding="utf-8") as accepted, rejected.open("w", encoding="utf-8") as bad:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            status = str(row.get("status") or "error")
            counts[status] += 1
            if status != "ok" and not (args.include_uncertain_alignment and status == "uncertain"):
                bad.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["rejected"] += 1
                continue
            request = {
                "schema_version": "external_fwait_completeness_v1",
                "request_id": row["task_id"],
                "dataset": row["dataset"],
                "source_id": row["source_id"],
                "partial_query": row["partial_query"],
                "partial_query_char_count": row["partial_query_char_count"],
                "alignment_status": status,
                "alignment": row.get("alignment") or {},
                "llm_messages": [{"role": "user", "content": prompt(row["partial_query"])}],
                "judgment": "",
                "confidence": "",
                "reason": "",
            }
            accepted.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["exported"] += 1
    print(json.dumps({"counts": dict(counts), "out": str(out.resolve()), "rejected": str(rejected.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
