#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pyarrow.parquet as pq  # type: ignore


EMOTIONS = ["neutral", "happiness", "anger", "sadness", "surprise"]


def maybe_fix_text(text: str) -> str:
    for encoding in ("latin1", "cp1252"):
        try:
            fixed = text.encode(encoding).decode("utf-8")
        except Exception:
            continue
        if re.search(r"[\u4e00-\u9fff]", fixed):
            return fixed
    return text


def zh_char_count(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def good_text(text: str, min_chars: int, max_chars: int) -> bool:
    n = zh_char_count(maybe_fix_text(text).strip())
    return min_chars <= n <= max_chars


def parse_speakers(value: str) -> List[str]:
    speakers: List[str] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            speakers.extend(f"{i:04d}" for i in range(int(start), int(end) + 1))
        else:
            speakers.append(f"{int(part):04d}")
    return speakers


def iter_rows(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(
            batch_size=512,
            columns=["audio", "transcript", "emotion", "speaker_id", "gender", "language"],
        ):
            yield from batch.to_pylist()


def write_index(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    (out_dir / "index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    lines = [
        "# ESD Samples",
        "",
        "| speaker | gender | emotion | transcript | wav |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        wav_name = Path(row["wav"]).name
        lines.append(
            f"| {row['speaker_id']} | {row['gender']} | {row['emotion']} | "
            f"{row['transcript']} | [wav/{wav_name}](wav/{wav_name}) |"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_rows = []
    for row in rows:
        wav_name = Path(row["wav"]).name
        html_rows.append(
            "<tr>"
            f"<td>{row['speaker_id']}</td>"
            f"<td>{row['gender']}</td>"
            f"<td>{row['emotion']}</td>"
            f"<td>{row['transcript']}</td>"
            f"<td><audio controls preload=\"none\" src=\"wav/{wav_name}\"></audio></td>"
            "</tr>"
        )
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>ESD Samples</title>
<style>
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;background:#f6f7f9;color:#111827}
h1{font-size:22px;margin:0 0 16px}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #d8dee8}
th,td{border-bottom:1px solid #e5e7eb;padding:10px 12px;text-align:left;vertical-align:middle;font-size:14px}
th{position:sticky;top:0;background:#eef2f7;z-index:1}
tr:hover{background:#f9fafb}
audio{width:260px}
.wrap{max-width:1280px;margin:auto}
</style>
</head>
<body>
<div class="wrap">
<h1>ESD zh samples</h1>
<table>
<thead><tr><th>speaker</th><th>gender</th><th>emotion</th><th>transcript</th><th>audio</th></tr></thead>
<tbody>
""" + "\n".join(html_rows) + """
</tbody>
</table>
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract small ESD wav samples by speaker/emotion.")
    ap.add_argument("--parquet_glob", default="/data/haifengjia/datasets/esd/train-*-of-00007.parquet")
    ap.add_argument("--out_dir", default="outputs/esd_samples_zh")
    ap.add_argument("--speakers", default="1-10")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--emotions", default=",".join(EMOTIONS))
    ap.add_argument("--per_pair", type=int, default=1)
    ap.add_argument("--min_zh_chars", type=int, default=6)
    ap.add_argument("--max_zh_chars", type=int, default=30)
    args = ap.parse_args()

    paths = sorted(Path().glob(args.parquet_glob) if not args.parquet_glob.startswith("/") else Path("/").glob(args.parquet_glob.lstrip("/")))
    if not paths:
        raise FileNotFoundError(args.parquet_glob)

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    speakers = parse_speakers(args.speakers)
    emotions = [x.strip() for x in args.emotions.split(",") if x.strip()]
    want = {(speaker, emotion): args.per_pair for speaker in speakers for emotion in emotions}
    selected_counts: Dict[Tuple[str, str], int] = {}
    index_rows: List[Dict[str, Any]] = []

    for row in iter_rows(paths):
        if row.get("language") != args.language:
            continue
        key = (str(row.get("speaker_id")), str(row.get("emotion")))
        if key not in want or selected_counts.get(key, 0) >= want[key]:
            continue
        transcript = maybe_fix_text(str(row.get("transcript") or ""))
        if args.language == "zh" and not good_text(transcript, args.min_zh_chars, args.max_zh_chars):
            continue

        audio = row["audio"]
        source_name = Path(str(audio.get("path") or "sample.wav")).name
        sample_idx = selected_counts.get(key, 0)
        wav_name = f"{key[0]}_{key[1]}_{sample_idx:02d}_{source_name}"
        wav_path = wav_dir / wav_name
        wav_path.write_bytes(audio["bytes"])
        selected_counts[key] = sample_idx + 1
        index_rows.append(
            {
                "speaker_id": key[0],
                "emotion": key[1],
                "gender": row.get("gender"),
                "language": row.get("language"),
                "transcript": transcript,
                "source_path": audio.get("path"),
                "wav": str(wav_path),
            }
        )
        if all(selected_counts.get(key, 0) >= count for key, count in want.items()):
            break

    index_rows.sort(key=lambda x: (x["speaker_id"], EMOTIONS.index(x["emotion"]) if x["emotion"] in EMOTIONS else x["emotion"]))
    write_index(out_dir, index_rows)
    missing = [
        {"speaker_id": speaker, "emotion": emotion, "needed": count, "found": selected_counts.get((speaker, emotion), 0)}
        for (speaker, emotion), count in want.items()
        if selected_counts.get((speaker, emotion), 0) < count
    ]
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "wav_dir": str(wav_dir),
                "count": len(index_rows),
                "missing": missing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
