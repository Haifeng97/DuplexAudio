# Unified Duplex Pipeline

`scripts/duplex_pipeline.py` is the config-driven entry point for new datasets. Each run writes immutable stage artifacts under `runs/<run_name>/` and records completion in `run_state.json`.

## Current 0818 Run

Configuration: `configs/customized_0811_0817_0818.json`

Inputs:

- Customized 0811 and 0817 are both retained under separate namespaces.
- Special uses only 0817.
- The five existing internal manifests provide the balancing baseline.
- Multimodal and external GCP/QA/Pure datasets are intentionally excluded.

The target primary-scenario distribution is 70% normal, 15% interrupt, 4% ordinary incomplete query, 1% clarification, 5% backchannel, and 5% other.

## Stages

```bash
python scripts/duplex_pipeline.py prepare \
  --config configs/customized_0811_0817_0818.json \
  --resume
```

`prepare` runs normalization/deduplication and deterministic quota planning. It does not call an LLM or TTS model.

Export incomplete-query split requests:

```bash
python scripts/duplex_pipeline.py llm-export-splits \
  --config configs/customized_0811_0817_0818.json \
  --resume
```

With the DSV4 backend, set credentials through the environment and run a request file:

```bash
export DSV4_API_KEY=...
export DSV4_WSID=...
python scripts/duplex_pipeline.py llm-run \
  --config configs/customized_0811_0817_0818.json \
  --input runs/customized_0811_0817_balanced_0818/03_llm/incomplete_split/requests.jsonl \
  --output runs/customized_0811_0817_balanced_0818/03_llm/incomplete_split/results.jsonl \
  --concurrency 8 \
  --resume
```

For offline filling, leave `llm.backend` as `offline`, fill each request's `response_text`, and pass the filled file to the next command. Running `llm-run` is itself the explicit opt-in to DSV4; no config edit is needed.

```bash
python scripts/duplex_pipeline.py llm-export-rank --config configs/customized_0811_0817_0818.json --input SPLIT_FILLED.jsonl
python scripts/duplex_pipeline.py llm-apply-rank --config configs/customized_0811_0817_0818.json --input RANK_FILLED.jsonl
python scripts/duplex_pipeline.py llm-export-clarification --config configs/customized_0811_0817_0818.json
python scripts/duplex_pipeline.py llm-apply-clarification --config configs/customized_0811_0817_0818.json --input CLARIFICATION_FILLED.jsonl
python scripts/duplex_pipeline.py materialize --config configs/customized_0811_0817_0818.json --resume
```

The split workflow enforces an exact source-text split and 3-14 effective prefix characters. Whitespace and punctuation do not count. Invalid or missing split/clarification results are explicitly counted and downgraded to normal during materialization.


## Role Card And Opening Run

Configuration: `configs/rolecard_opening_110k_0818.json`.

The deterministic plan contains 110,000 candidates: 11,000 for each of XiaoTian, Jili,
Paraboy, Amei, and Hua Aotian, plus 55,000 random assistant roles. Turn counts 1-5 each
have 22,000 rows. Scene and voice descriptions are role-level reusable assets, while
actions are generated for every non-silent assistant turn.

Run the 5,100 reusable description requests first:

```bash
python scripts/duplex_pipeline.py llm-run   --config configs/rolecard_opening_110k_0818.json   --input runs/rolecard_opening_110k_0818/00_rolecard_generation/description_requests.jsonl   --output runs/rolecard_opening_110k_0818/00_rolecard_generation/description_results.jsonl   --concurrency 8   --resume
```

Apply descriptions and export both the 110,000-row dialogue request file and a
deterministic 500-row pilot:

```bash
python scripts/duplex_pipeline.py rolecard-apply-descriptions   --config configs/rolecard_opening_110k_0818.json   --input runs/rolecard_opening_110k_0818/00_rolecard_generation/description_results.jsonl
python scripts/duplex_pipeline.py rolecard-export-dialogues   --config configs/rolecard_opening_110k_0818.json
```

After dialogue generation, apply the full result file and then use the normal
`prepare -> incomplete split/rank -> clarification -> materialize -> tts-prepare`
workflow. The generated `05_tts/NEXT_COMMANDS.sh` runs resumable Qwen3-TTS,
four-shard CTC verification, whole-scenario rejection, format, multimodal action-token
insertion, and deterministic post-filter ratio balancing.

## TTS Cache Contract

After `02_make_turn_tts_tasks.py` creates raw tasks and `scenario_index.jsonl`, rewrite them to content-addressed tasks:

```bash
python scripts/duplex_pipeline.py tts-fingerprint \
  --config configs/customized_0811_0817_0818.json \
  --input RAW_DIR/tts_tasks.jsonl \
  --index RAW_DIR/scenario_index.jsonl \
  --output FINGERPRINTED_DIR
```

The fingerprint includes TTS text, reference WAV, reference transcript, model/version, language, sample rate, and generation settings. A legacy WAV is reused only when all fields match. Legacy task ID equality alone is ignored.

For a materialized full run, build all raw tasks in streaming mode, import matching legacy WAVs, and produce one deduplicated TTS task file:

```bash
python scripts/duplex_pipeline.py tts-prepare \
  --config configs/customized_0811_0817_0818.json \
  --resume
```

This writes `runs/<run_name>/05_tts/NEXT_COMMANDS.sh`. It contains the resumable four-GPU Qwen3-TTS command, separate 100-worker Customized/Special format commands, and the final release command.

The equivalent pipeline stages are:

```bash
python scripts/duplex_pipeline.py format \
  --config configs/customized_0811_0817_0818.json \
  --workers 100

python scripts/duplex_pipeline.py release \
  --config configs/customized_0811_0817_0818.json
```

`release` streams the five base manifests plus the new Customized and Special manifests, rejects duplicate IDs or relative audio paths, and writes the balanced manifest without copying old WAV files.
