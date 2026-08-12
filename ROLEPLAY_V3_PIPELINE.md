# Roleplay v3: Qwen3-TTS + Backchannel

## Inputs

- Roleplay intermediate:
  `/data/haifengjia/datasets/roleplay_zh/converted/duplex_intermediate/combined/selected.jsonl`
- MagicData RAMC:
  `/data/haifengjia/datasets/Easy-Turn-Trainset/trainset/magicdata_ramc`
- Qwen3-TTS model:
  `/data/haifengjia/models/Qwen3-TTS-12Hz-1.7B-Base`
- ESD voice bank:
  `outputs/esd_voice_bank_zh_neutral60/voice_bank.jsonl`

The intermediate currently has 17,441 rows and 41,191 turns. Candidate generation
keeps the previous limits: current query at most 240 characters, current answer at
most 360 characters, and the latest 8 valid turns.

## Scenarios

The four core scenarios are selected before TTS:

- `normal_qa`
- `incomplete_query_candidate`
- `player_interrupts_ai`
- `player_backchannel`

`incomplete_query_clarification` remains a separately generated fifth scenario.
Pass its candidate file to `17_select_roleplay_scenario_mix.py --exclude_manifest`
so its original source groups cannot be selected again for the four core scenarios.

Backchannel uses real RAMC player audio restricted to the curated high-frequency
acknowledgement list in `16_extract_magicdata_backchannels.py`. Its first audio
chunk is labeled `<FD_G_INTERRUPT>`; any remaining chunks are `<FD_D_WAIT>`.
The assistant answer is:

```text
<FD_A_ANSWER> + answer prefix
<FD_G_INTERRUPT> + <FD_D_WAIT>...
<FD_H_CONTINUE> + <FD_A_ANSWER> + answer continuation + <EOR>
```

For original multi-turn rows, complete turn boundaries still receive 1-3 seconds
of random-noise `<FD_IDLE>`. No inter-turn idle is inserted inside the special
backchannel, interrupt, or split incomplete-query event.

The inserted partial-query turn in `incomplete_query_clarification` emits one
`<FD_F_WAIT>` on the final voiced chunk, then waits for a random 3-5 seconds.
The final wait chunk emits `<FD_J_ACTIVE>`, immediately followed by `<FD_A_ANSWER>`.

## Environment

The environment created on this host is:

```text
/data/haifengjia/miniforge3/envs/qwen3-tts
```

Equivalent setup:

```bash
conda create -n qwen3-tts python=3.12 pip -y
conda activate qwen3-tts
pip install qwen-tts==0.1.1
pip install ninja
pip install --force-reinstall \
  torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
FLASH_ATTN_CUDA_ARCHS=90 MAX_JOBS=4 \
  pip install flash-attn --no-build-isolation
```

FlashAttention 2 is the normal production setting. `sdpa` is an explicit slower
mode useful for smoke tests; the worker does not silently change attention modes.

## 1. Extract Backchannels

```bash
python scripts/16_extract_magicdata_backchannels.py \
  --input_dir /data/haifengjia/datasets/Easy-Turn-Trainset/trainset/magicdata_ramc \
  --out_dir outputs/roleplay_zh_v3/backchannel_corpus \
  --min_duration_sec 0.2 \
  --max_duration_sec 2.0 \
  --max_text_chars 8 \
  --max_per_text 200 \
  --workers 100 \
  --seed 20260730
```

Only the curated high-frequency acknowledgement texts are retained. Audio is
16 kHz mono and is resampled to 24 kHz during formatting.

## 2. Build Candidate Pools

```bash
python scripts/01_make_scenario_candidate_pools.py \
  --input /data/haifengjia/datasets/roleplay_zh/converted/duplex_intermediate/combined/selected.jsonl \
  --out_dir outputs/roleplay_zh_v3/scenario_candidates \
  --limit_each 0 \
  --seed 20260730 \
  --interrupt_pair_mode same_row_previous \
  --backchannel_manifest outputs/roleplay_zh_v3/backchannel_corpus/backchannels.jsonl
```

## 3. Build Clarification Requests

Choose the clarification count before selecting the four core scenarios:

```bash
python scripts/09_export_incomplete_clarification_requests.py \
  --input outputs/roleplay_zh_v3/scenario_candidates/normal_qa_candidates.jsonl \
  --out outputs/roleplay_zh_v3/incomplete_clarification/llm_requests.jsonl \
  --limit CLARIFICATION_COUNT \
  --seed 20260730
```

After filling `clarification_answer_text`:

```bash
python scripts/10_build_incomplete_clarification_scenarios.py \
  --input outputs/roleplay_zh_v3/incomplete_clarification/llm_requests.filled.jsonl \
  --out outputs/roleplay_zh_v3/incomplete_clarification/candidates.jsonl
```

## 4. Select Core Ratios Before TTS

The ratios below are only an example and must be explicitly chosen:

```bash
python scripts/17_select_roleplay_scenario_mix.py \
  --normal outputs/roleplay_zh_v3/scenario_candidates/normal_qa_candidates.jsonl \
  --incomplete outputs/roleplay_zh_v3/scenario_candidates/incomplete_query_candidates.jsonl \
  --interrupt outputs/roleplay_zh_v3/scenario_candidates/player_interrupt_candidates.jsonl \
  --backchannel outputs/roleplay_zh_v3/scenario_candidates/player_backchannel_candidates.jsonl \
  --exclude_manifest outputs/roleplay_zh_v3/incomplete_clarification/candidates.jsonl \
  --out_dir outputs/roleplay_zh_v3/selected_core \
  --ratios normal_qa=0.60,incomplete_query_candidate=0.15,player_interrupts_ai=0.15,player_backchannel=0.10 \
  --total 0 \
  --seed 20260730
```

`--total 0` selects the largest feasible set after source-group deduplication.
Interrupt is allocated first because it requires genuine multi-turn data.

## 5. Create and Run TTS Tasks

Run `02_make_turn_tts_tasks.py` once per selected category and once for
clarification:

```bash
python scripts/02_make_turn_tts_tasks.py \
  --input outputs/roleplay_zh_v3/selected_core/player_backchannel_candidates.jsonl \
  --out_dir outputs/roleplay_zh_v3/tts_backchannel
```

Qwen3-TTS multi-GPU command:

```bash
PROJECT=Cgame_aimate_haifengjia \
python scripts/03_run_tts_multi_gpu.py \
  --engine qwen3_tts \
  --tasks outputs/roleplay_zh_v3/tts_backchannel/tts_tasks.jsonl \
  --work_dir outputs/roleplay_zh_v3/tts_backchannel \
  --gpus 4,5,6,7 \
  --procs_per_gpu 1 \
  --python /data/haifengjia/miniforge3/envs/qwen3-tts/bin/python \
  --model_dir /data/haifengjia/models/Qwen3-TTS-12Hz-1.7B-Base \
  --batch_size 128 \
  --language Chinese \
  --dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --monitor_every 0.5 \
  --progress_every 50
```

Each worker sorts its pending tasks by estimated synthesis length, then sends
up to `--batch_size` texts and their per-sample voice-clone prompts through one
native Qwen3-TTS batch call. Batch errors are reported directly; the worker does
not silently fall back to scalar inference.

The worker resumes from valid output WAV files. Restarting with a different
`--procs_per_gpu` reshards all tasks, records existing WAVs as `cached`, and only
synthesizes missing or invalid files.

## 6. Format and Validate

Use the `duplexaudio` environment:

```bash
python scripts/14_format_duplex_manifest_parallel.py \
  --index outputs/roleplay_zh_v3/tts_backchannel/scenario_index.jsonl \
  --out outputs/roleplay_zh_v3/final_backchannel/manifest.jsonl \
  --wav_dir outputs/roleplay_zh_v3/final_backchannel/wav \
  --workers 8 \
  --sample_rate 24000 \
  --chunk_ms 180 \
  --tokenizer_json tokenizers/qwen3_8b/tokenizer.json \
  --vad_mode silero \
  --backchannel_vad_mode energy \
  --min_query_audio_sec 1.0 \
  --min_backchannel_audio_sec 0.08

python scripts/05_validate_duplex_manifest.py \
  --manifest outputs/roleplay_zh_v3/final_backchannel/manifest.jsonl \
  --show 0
```

Normal TTS query audio uses Silero VAD. Short recorded backchannel audio uses the
explicit energy VAD mode because Silero can miss sub-second fillers. Both modes
are recorded in each manifest row's `query_vad` metadata.

## 7. Final No-Drop Concatenation

After all five manifests pass validation:

```bash
python scripts/18_concat_duplex_manifests.py \
  --manifest outputs/roleplay_zh_v3/final_normal/manifest.jsonl \
  --manifest outputs/roleplay_zh_v3/final_incomplete/manifest.jsonl \
  --manifest outputs/roleplay_zh_v3/final_interrupt/manifest.jsonl \
  --manifest outputs/roleplay_zh_v3/final_backchannel/manifest.jsonl \
  --manifest outputs/roleplay_zh_v3/final_incomplete_clarification/manifest.jsonl \
  --out outputs/roleplay_zh_v3/final_all/manifest.jsonl \
  --seed 20260730
```

This final step only shuffles and concatenates complete manifests. It does not
resample categories or drop rows.
