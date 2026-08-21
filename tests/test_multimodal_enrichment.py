from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "apply_multimodal_enrichment",
    ROOT / "scripts" / "24_apply_multimodal_enrichment.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def item(label: str, kind: str, source: str, turn_id: int = 1) -> dict:
    return {
        "label": label,
        "kind": kind,
        "audio_source": source,
        "turn_id": turn_id,
    }


class ActionTimelineTests(unittest.TestCase):
    def test_interrupted_action_does_not_delay_interrupt(self) -> None:
        timeline = [
            item("回答", "text_token", "base_answer_prefix_gn"),
            item("<FD_G_INTERRUPT>", "interrupt", "donor_query_audio", 2),
            item("<FD_D_WAIT>", "wait", "donor_query_audio", 2),
        ]
        chunks = [b"answer", b"user-1", b"user-2"]
        action = [item("（", "text_token", "action_expression_gn")]

        mode, counts = MODULE.apply_action_tokens(
            timeline, chunks, 1, "before_interrupt", action, [b"noise"]
        )

        self.assertEqual(mode, "metadata_only_interrupted")
        self.assertEqual(timeline[1]["label"], "<FD_G_INTERRUPT>")
        self.assertEqual(chunks, [b"answer", b"user-1", b"user-2"])
        self.assertEqual(counts["omitted_interrupted_action_tokens"], 1)

    def test_intervene_action_preserves_contiguous_user_audio(self) -> None:
        timeline = [
            item("回复", "text_token", "intervene_query_suffix_audio"),
            item("<EOR>", "eor", "intervene_query_suffix_audio"),
            item("<FD_D_WAIT>", "wait", "intervene_query_suffix_audio"),
            item("<FD_IDLE>", "final_idle", "gn_after", 0),
        ]
        chunks = [b"user-0", b"user-1", b"user-2", b"idle"]
        action = [
            item("（", "text_token", "action_expression_gn"),
            item("点头", "text_token", "action_expression_gn"),
            item("）", "text_token", "action_expression_gn"),
        ]

        mode, counts = MODULE.apply_action_tokens(
            timeline,
            chunks,
            1,
            "before_eor",
            action,
            [b"noise-0", b"noise-1", b"noise-2"],
        )

        self.assertEqual(mode, "overlap_reused_before_eor")
        self.assertEqual([x["label"] for x in timeline[1:5]], ["（", "点头", "）", "<EOR>"])
        self.assertEqual(chunks[:4], [b"user-0", b"user-1", b"user-2", b"idle"])
        self.assertEqual(chunks[4], b"noise-2")
        self.assertEqual(counts["inserted_action_audio_chunks"], 1)


if __name__ == "__main__":
    unittest.main()
