# Multimodal Enrichment Pipeline

This pipeline adds three annotations to selected duplex samples while preserving the original reply text:

- `scene_description`: one visual scene description per sample, used as input.
- `voice_description`: one fictional AI voice description per sample, used as input.
- `action_expression_descriptions`: one visible action/expression description per AI reply.

Each action description is wrapped as `（action_expression）`, tokenized, and inserted into the output timeline immediately before that reply's `<EOR>`. A truly interrupted answer has no `<EOR>`; its description is inserted immediately before `<FD_G_INTERRUPT>`. Backchannel answers continue to completion and therefore use their final `<EOR>`.

Every inserted text token adds one same-width Gaussian-noise WAV chunk. With the current protocol, one token adds 180 ms.

## Export Requests

The exporter uses per-scenario reservoir sampling and balances the requested demo count across all scenarios found in the manifest.

```bash
python scripts/23_export_multimodal_enrichment_requests.py \
  --manifest /nfs/shared_data/customized_duplex_0806/v3/manifest.jsonl \
  --out outputs/customized_0806_multimodal_demo/llm_requests.jsonl \
  --limit 100 \
  --seed 20260812
```

Fill these fields in every JSONL row without changing IDs or turn order:

```json
{
  "scene_description": "...",
  "voice_description": "说话时的声音特征：...",
  "turn_descriptions": [
    {"turn_id": 1, "action_expression": "..."}
  ]
}
```

The action field must not include parentheses. The apply step adds Chinese full-width parentheses.

## Apply Filled Results

```bash
python scripts/24_apply_multimodal_enrichment.py \
  --manifest /nfs/shared_data/customized_duplex_0806/v3/manifest.jsonl \
  --filled outputs/customized_0806_multimodal_demo/llm_requests.filled.jsonl \
  --out outputs/customized_0806_multimodal_demo/final/manifest.jsonl \
  --wav_dir outputs/customized_0806_multimodal_demo/final/wav \
  --tokenizer_json tokenizers/qwen3_8b/tokenizer.json \
  --noise_rms 0.003 \
  --seed 20260812
```

The apply step fails on missing descriptions, duplicate IDs, changed turn IDs, absent source rows, invalid audio geometry, or replies without a valid EOR/interrupt insertion point.
