#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate duplex manifest basics.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    n = 0
    bad = []
    scenarios = Counter()
    labels = Counter()
    examples = []
    for line_no, line in enumerate(open(args.manifest, encoding="utf-8", errors="ignore"), start=1):
        if not line.strip():
            continue
        n += 1
        try:
            row = json.loads(line)
        except Exception as exc:
            bad.append({"line": line_no, "error": repr(exc)})
            continue
        scenarios[row.get("scenario", "")] += 1
        timeline = row.get("timeline") or []
        audio = Path(str(row.get("audio", "")))
        if not audio.exists():
            bad.append({"line": line_no, "id": row.get("id"), "error": "missing_audio", "audio": str(audio)})
            continue
        chunk_n = int(round(row.get("sample_rate", 16000) * row.get("chunk_ms", 180) / 1000.0))
        frames = wav_frames(audio)
        if len(timeline) * chunk_n != frames:
            bad.append({"line": line_no, "id": row.get("id"), "error": "timeline_audio_mismatch", "timeline": len(timeline), "frames": frames})
        for i, ent in enumerate(timeline):
            if ent.get("idx") != i:
                bad.append({"line": line_no, "id": row.get("id"), "error": "bad_idx", "at": i, "idx": ent.get("idx")})
                break
            labels[str(ent.get("label"))] += 1
        if not any(ent.get("label") == "ANSWER" for ent in timeline):
            bad.append({"line": line_no, "id": row.get("id"), "error": "missing_answer"})
        if len(examples) < args.show:
            examples.append({
                "id": row.get("id"),
                "scenario": row.get("scenario"),
                "audio": row.get("audio"),
                "timeline_head": timeline[:12],
            })

    result = {
        "manifest": args.manifest,
        "n": n,
        "bad": len(bad),
        "bad_examples": bad[:20],
        "scenario_counts": dict(scenarios),
        "label_counts": dict(labels.most_common(20)),
        "examples": examples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
