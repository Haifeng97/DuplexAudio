#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import wave
from collections import Counter
from pathlib import Path

from duplex_label_protocol import (
    ACTIVE_CONTROL_LABELS,
    EOR,
    FD_A_ANSWER,
    FD_D_WAIT,
    FD_F_WAIT,
    FD_G_INTERRUPT,
    FD_H_CONTINUE,
    FD_IDLE,
    FD_J_ACTIVE,
    LEGACY_LABEL_MAP,
    PROTOCOL_NAME,
    RESERVED_CONTROL_LABELS,
)


def wav_frames(path: Path) -> int:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes()


def protocol_errors(row: dict) -> list[str]:
    timeline = row.get("timeline") if isinstance(row.get("timeline"), list) else []
    labels = [str(item.get("label")) for item in timeline]
    errors: list[str] = []

    if row.get("label_protocol") != PROTOCOL_NAME:
        errors.append(f"label_protocol={row.get('label_protocol')!r}, expected={PROTOCOL_NAME!r}")

    legacy = sorted({label for label in labels if label in LEGACY_LABEL_MAP and label != EOR})
    if legacy:
        errors.append(f"legacy_labels={legacy}")

    reserved = sorted(set(labels) & RESERVED_CONTROL_LABELS)
    if reserved:
        errors.append(f"reserved_labels={reserved}")

    for item in timeline:
        label = str(item.get("label"))
        if item.get("label_type") != "text" and label not in ACTIVE_CONTROL_LABELS:
            errors.append(f"unknown_control_label={label!r}")
            break

    if FD_A_ANSWER not in labels:
        errors.append("missing_answer")

    scenario = str(row.get("scenario") or "")
    expected_f = 1 if scenario in {"incomplete_query", "incomplete_query_candidate", "incomplete_query_clarification"} else 0
    expected_g = 1 if scenario in {"player_interrupts_ai", "player_backchannel"} else 0
    expected_h = 1 if scenario == "player_backchannel" else 0
    expected_j = 1 if scenario == "incomplete_query_clarification" else 0

    for label, expected in (
        (FD_F_WAIT, expected_f),
        (FD_G_INTERRUPT, expected_g),
        (FD_H_CONTINUE, expected_h),
        (FD_J_ACTIVE, expected_j),
    ):
        actual = labels.count(label)
        if actual != expected:
            errors.append(f"{label}_count={actual}, expected={expected}")

    if expected_f and labels.count(FD_F_WAIT) == 1:
        f_idx = labels.index(FD_F_WAIT)
        if f_idx + 1 >= len(labels) or labels[f_idx + 1] != FD_IDLE:
            errors.append("F_WAIT_not_followed_by_IDLE")

    pause_kinds = {"incomplete_pause_wait", "clarification_wait"}
    bad_pause = [
        item.get("idx")
        for item in timeline
        if item.get("kind") in pause_kinds and item.get("label") != FD_IDLE
    ]
    if bad_pause:
        errors.append(f"pause_not_IDLE_at={bad_pause[:5]}")

    if expected_j and labels.count(FD_J_ACTIVE) == 1:
        j_idx = labels.index(FD_J_ACTIVE)
        if j_idx + 1 >= len(labels) or labels[j_idx + 1] != FD_A_ANSWER:
            errors.append("J_ACTIVE_not_followed_by_A_ANSWER")
        f_idx = labels.index(FD_F_WAIT) if FD_F_WAIT in labels else len(labels)
        if f_idx >= j_idx:
            errors.append("F_WAIT_not_before_J_ACTIVE")
        elif any(label != FD_IDLE for label in labels[f_idx + 1:j_idx]):
            errors.append("non_IDLE_between_F_WAIT_and_J_ACTIVE")
        if timeline[j_idx].get("kind") != "clarification_active":
            errors.append("J_ACTIVE_bad_kind")

    if expected_h and labels.count(FD_H_CONTINUE) == 1:
        h_idx = labels.index(FD_H_CONTINUE)
        if h_idx + 1 >= len(labels) or labels[h_idx + 1] != FD_A_ANSWER:
            errors.append("H_CONTINUE_not_followed_by_A_ANSWER")
        g_idx = labels.index(FD_G_INTERRUPT) if FD_G_INTERRUPT in labels else len(labels)
        if g_idx >= h_idx:
            errors.append("G_INTERRUPT_not_before_H_CONTINUE")

    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate duplex manifest basics and fd_control_v1 labels.")
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
        for error in protocol_errors(row):
            bad.append({"line": line_no, "id": row.get("id"), "error": error})
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
