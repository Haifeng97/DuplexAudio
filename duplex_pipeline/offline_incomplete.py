from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .io import atomic_write_json, canonical_json, iter_jsonl
from .text import effective_char_count


TERMINAL_PUNCTUATION = set("。！？!?；;")
TRAILING_PUNCTUATION = TERMINAL_PUNCTUATION | set("，,、：:…")

# These endings leave a required object, complement, clause, or continuation.
# The offline mode intentionally favors precision over coverage.
HANGING_ENDINGS: Tuple[str, ...] = tuple(sorted({
    "我想", "我想要", "我想问", "我想去", "我想让", "我想把", "我想跟",
    "我要", "我需要", "我有个", "我有点", "我刚", "我刚才", "我准备", "我发现",
    "我听说", "我记得", "我觉得", "我说", "我在", "我去", "我让", "我把",
    "我们想", "我们要", "我们去", "我们能", "我们在", "咱们想", "咱们要", "咱们去",
    "你能", "你能不能", "能不能", "你可不可以", "可不可以", "你有没有", "有没有",
    "你是不是", "是不是", "你为什么", "为什么", "你怎么", "怎么", "你觉得", "你知道",
    "你说", "你看", "你帮我", "帮我", "你告诉我", "告诉我", "麻烦", "麻烦你",
    "如果", "要是", "假如", "万一", "因为", "虽然", "但是", "可是", "不过", "然后", "所以",
    "关于", "至于", "那个", "这个", "就是", "比如", "比如说", "其实", "刚才", "现在",
    "后来", "还有", "除了", "为了", "按照", "根据", "作为", "对于", "等会", "待会",
    "跟你", "和你", "给我", "让我", "把我", "在那", "去那", "说一下", "问一下", "看一下",
}, key=lambda item: (len(item), item), reverse=True))


def conservative_split(
    question: str,
    min_effective_chars: int = 3,
    max_effective_chars: int = 14,
) -> Optional[Dict[str, Any]]:
    text = str(question or "").strip()
    if effective_char_count(text) <= min_effective_chars:
        return None
    candidates = []
    for cut in range(1, len(text)):
        prefix = text[:cut].rstrip()
        suffix = text[cut:].lstrip()
        prefix_chars = effective_char_count(prefix)
        if not min_effective_chars <= prefix_chars <= max_effective_chars:
            continue
        if effective_char_count(suffix) < 2:
            continue
        if any(char in TERMINAL_PUNCTUATION for char in prefix):
            continue
        if not prefix or prefix[-1] in TRAILING_PUNCTUATION or suffix[0] in TRAILING_PUNCTUATION:
            continue
        endings = [ending for ending in HANGING_ENDINGS if prefix.endswith(ending)]
        if not endings:
            continue
        ending = endings[0]
        score = 100 + len(ending) * 4 + prefix_chars
        candidates.append((score, cut, ending, prefix, suffix))
    if not candidates:
        return None
    _, cut, ending, prefix, suffix = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "question_text": text,
        "cut_char_index": cut,
        "query_part1_text": prefix,
        "query_part2_text": suffix,
        "method": "offline_conservative_hanging_ending_v1",
        "matched_ending": ending,
        "prefix_effective_chars": effective_char_count(prefix),
    }


def apply_offline_splits(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    llm = dict(config.get("llm") or {})
    mode = str(llm.get("offline_split_mode") or "")
    if mode != "conservative":
        raise ValueError("llm.offline_split_mode must be 'conservative'")
    min_chars = int(llm.get("min_incomplete_prefix_effective_chars", 3))
    max_chars = int(llm.get("max_incomplete_prefix_effective_chars", 14))
    source = run_dir / "02_plan" / "customized_selected.jsonl"
    out_dir = run_dir / "03_llm" / "incomplete_rank"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "selected_splits.jsonl"
    counts: Counter = Counter()
    with output.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(source):
            if row.get("primary_scenario") not in {"incomplete_query", "incomplete_query_clarification"}:
                continue
            turns = [turn for turn in (row.get("turns") or []) if isinstance(turn, dict)]
            question = str((turns[-1] if turns else {}).get("question_text") or "")
            split = conservative_split(question, min_chars, max_chars)
            if split is None:
                counts["rejected"] += 1
                continue
            handle.write(canonical_json({
                "sample_id": row["id"],
                "primary_scenario": row["primary_scenario"],
                **split,
            }) + "\n")
            counts["selected"] += 1
    result = {
        "source": str(source),
        "output": str(output),
        "mode": mode,
        "min_effective_chars": min_chars,
        "max_effective_chars": max_chars,
        "counts": dict(counts),
    }
    atomic_write_json(out_dir / "offline_apply_stats.json", result)
    return result
