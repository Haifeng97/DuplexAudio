#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf
from transformers import AutoTokenizer


def norm(x: Any) -> str:
    return str(x or '').strip()


def read_wav_mono(path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype='float32', always_2d=False)
    if getattr(wav, 'ndim', 1) == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype='float32').reshape(-1)
    if int(sr) != int(target_sr):
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), int(target_sr))
        wav = resample_poly(wav, int(target_sr)//g, int(sr)//g).astype('float32')
        sr = int(target_sr)
    return np.clip(wav, -1.0, 1.0).astype('float32'), int(sr)


def write_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype('float32'), sr)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype('float32') ** 2) + 1e-12))


def rms_dbfs(x: np.ndarray) -> float:
    return 20.0 * math.log10(max(rms(x), 1e-12))


def gaussian_noise(n: int, rms_min: float, rms_max: float, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(int(n)).astype('float32')
    x = x / (float(np.sqrt(np.mean(x * x) + 1e-12)) + 1e-8)
    target = float(rng.uniform(float(rms_min), float(rms_max)))
    return np.clip(x * target, -1.0, 1.0).astype('float32')


def state_entry(idx: int, label: str, kind: str, chunk_n: int, chunk_ms: int, audio_source: str, turn_id: int = 1, r: float | None = None) -> Dict[str, Any]:
    ent: Dict[str, Any] = {
        'idx': int(idx),
        'kind': kind,
        'label_type': 'state',
        'label': label,
        'start_sec': round(idx * chunk_ms / 1000.0, 6),
        'end_sec': round((idx + 1) * chunk_ms / 1000.0, 6),
        'start_sample': int(idx * chunk_n),
        'end_sample': int((idx + 1) * chunk_n),
        'audio_source': audio_source,
        'turn_id': int(turn_id),
    }
    if r is not None:
        ent['rms'] = float(r)
    return ent


def text_entry(idx: int, token_id: int, token_text: str, text_token_idx: int, chunk_n: int, chunk_ms: int, turn_id: int, r: float) -> Dict[str, Any]:
    return {
        'idx': int(idx),
        'kind': 'text_token',
        'label_type': 'text',
        'label': token_text,
        'token_id': int(token_id),
        'token_text': token_text,
        'text_token_idx': int(text_token_idx),
        'turn_id': int(turn_id),
        'start_sec': round(idx * chunk_ms / 1000.0, 6),
        'end_sec': round((idx + 1) * chunk_ms / 1000.0, 6),
        'start_sample': int(idx * chunk_n),
        'end_sample': int((idx + 1) * chunk_n),
        'audio_source': 'answer_gaussian',
        'rms': float(r),
    }


def eor_entry(idx: int, chunk_n: int, chunk_ms: int, turn_id: int, r: float) -> Dict[str, Any]:
    return {
        'idx': int(idx),
        'kind': 'eor',
        'label_type': 'text',
        'label': '<EOR>',
        'token_id': None,
        'token_text': '<EOR>',
        'turn_id': int(turn_id),
        'start_sec': round(idx * chunk_ms / 1000.0, 6),
        'end_sec': round((idx + 1) * chunk_ms / 1000.0, 6),
        'start_sample': int(idx * chunk_n),
        'end_sample': int((idx + 1) * chunk_n),
        'audio_source': 'answer_gaussian',
        'rms': float(r),
    }


def build_formatted_example(obj: Dict[str, Any], out_wav: Path, tok, args) -> Dict[str, Any]:
    sid = norm(obj['id'])
    q = norm(obj['question_text'])
    a = norm(obj['answer_text'])
    q_wav_path = Path(norm(obj['question_audio']))
    q_wav, sr = read_wav_mono(q_wav_path, args.sample_rate)
    chunk_n = max(1, int(round(args.sample_rate * args.chunk_ms / 1000.0)))
    rng = np.random.default_rng(abs(hash(sid)) % (2**32))

    q_chunks = int(math.ceil(len(q_wav) / chunk_n)) if len(q_wav) else 0
    padded_q_len = q_chunks * chunk_n
    if len(q_wav) < padded_q_len:
        q_wav = np.pad(q_wav, (0, padded_q_len - len(q_wav))).astype('float32')

    # VAD over question chunks. first_active..last_active are WAIT; outside are IDLE.
    rms_vals = []
    active = []
    for i in range(q_chunks):
        seg = q_wav[i*chunk_n:(i+1)*chunk_n]
        db = rms_dbfs(seg)
        rms_vals.append(rms(seg))
        active.append(db >= float(args.vad_threshold_dbfs))
    active_idxs = [i for i, x in enumerate(active) if x]
    first_active = min(active_idxs) if active_idxs else -1
    last_active = max(active_idxs) if active_idxs else -1

    timeline: List[Dict[str, Any]] = []
    chunk_states: List[str] = []
    for i in range(q_chunks):
        if first_active >= 0 and first_active <= i <= last_active:
            label, kind = 'WAIT', 'wait'
        else:
            label, kind = 'IDLE', 'idle'
        chunk_states.append(label)
        timeline.append(state_entry(i, label, kind, chunk_n, args.chunk_ms, 'question_audio', 1, rms_vals[i]))

    ids = [int(x) for x in tok(a, add_special_tokens=False)['input_ids'][:max(0, args.max_answer_tokens)]]
    answer_min = 1 + len(ids) + 1
    answer_by_char = max(1, int(math.ceil((len(''.join(a.split())) + 3) * 1.1)))
    answer_region_chunks = max(answer_min, answer_by_char, int(args.min_answer_chunks))
    answer_audio = gaussian_noise(answer_region_chunks * chunk_n, args.noise_rms_min, args.noise_rms_max, rng)

    idx = q_chunks
    timeline.append(state_entry(idx, 'ANSWER', 'answer_trigger', chunk_n, args.chunk_ms, 'answer_gaussian', 1, rms(answer_audio[:chunk_n])))
    idx += 1
    for j, tid in enumerate(ids):
        seg = answer_audio[(idx-q_chunks)*chunk_n:(idx-q_chunks+1)*chunk_n]
        timeline.append(text_entry(idx, tid, tok.decode([tid], skip_special_tokens=False), j, chunk_n, args.chunk_ms, 1, rms(seg)))
        idx += 1
    seg = answer_audio[(idx-q_chunks)*chunk_n:(idx-q_chunks+1)*chunk_n]
    timeline.append(eor_entry(idx, chunk_n, args.chunk_ms, 1, rms(seg)))
    idx += 1
    while idx < q_chunks + answer_region_chunks:
        seg = answer_audio[(idx-q_chunks)*chunk_n:(idx-q_chunks+1)*chunk_n]
        timeline.append(state_entry(idx, 'IDLE', 'answer_tail_idle', chunk_n, args.chunk_ms, 'answer_tail_gaussian', 1, rms(seg)))
        idx += 1

    final_audio = gaussian_noise(args.final_idle_chunks * chunk_n, args.noise_rms_min, args.noise_rms_max, rng)
    for k in range(args.final_idle_chunks):
        seg = final_audio[k*chunk_n:(k+1)*chunk_n]
        timeline.append(state_entry(idx, 'IDLE', 'final_idle', chunk_n, args.chunk_ms, 'final_idle_gaussian', 1, rms(seg)))
        idx += 1

    full_wav = np.concatenate([q_wav, answer_audio, final_audio]).astype('float32')
    write_wav(out_wav, full_wav, args.sample_rate)

    return {
        'id': sid,
        'source': args.source,
        'task': 'qa_answer',
        'audio': str(out_wav),
        'source_audio': str(q_wav_path),
        'source_question_audios': [str(q_wav_path)],
        'source_answer_audios': [],
        'sample_rate': int(args.sample_rate),
        'chunk_ms': float(args.chunk_ms),
        'text_query': q,
        'question_text': q,
        'answer_text': a,
        'text': a,
        'target_text': a,
        'output': a,
        'asr_text': q,
        'turns': [{'turn_id': 1, 'question_text': q, 'answer_text': a}],
        'timeline': timeline,
        'chunk_states': chunk_states,
        'label_policy': 'textqa_tts_v3_timeline_chunk_rms_answer_tail',
        'training_mode': 'streaming_teacher_forcing_qa',
        'stats': {
            'mode': 'single_turn_textqa_tts',
            'num_turns': 1,
            'chunk_samples': int(chunk_n),
            'timeline_chunks': len(timeline),
            'question_audio': str(q_wav_path),
            'question_audio_sec': round(len(q_wav) / args.sample_rate, 6),
            'first_wait_idx': int(first_active),
            'last_wait_idx': int(last_active),
            'answer_idx': int(q_chunks),
            'answer_tokens': len(ids),
            'answer_region_chunks_final': int(answer_region_chunks),
            'final_idle_chunks': int(args.final_idle_chunks),
            'synthetic_audio_mode': 'gaussian_noise',
            'noise_rms_min': float(args.noise_rms_min),
            'noise_rms_max': float(args.noise_rms_max),
            'th_rms_dbfs': float(args.vad_threshold_dbfs),
        },
        'tts': {'engine': 'cosyvoice', 'question_audio': str(q_wav_path)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Format question TTS index into v3-style QA manifest with full per-chunk timeline.')
    ap.add_argument('--index', required=True, help='qa_index.jsonl from 01_make_question_tts_tasks.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--base_model', default='/nfs/shared_models/Qwen3-8B')
    ap.add_argument('--source', default='textqa_tts')
    ap.add_argument('--sample_rate', type=int, default=16000)
    ap.add_argument('--chunk_ms', type=int, default=180)
    ap.add_argument('--vad_threshold_dbfs', type=float, default=-40.0)
    ap.add_argument('--noise_rms_min', type=float, default=0.0005)
    ap.add_argument('--noise_rms_max', type=float, default=0.003)
    ap.add_argument('--final_idle_chunks', type=int, default=4)
    ap.add_argument('--min_answer_chunks', type=int, default=0)
    ap.add_argument('--max_answer_tokens', type=int, default=512)
    ap.add_argument('--allow_missing_audio', action='store_true')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wav_dir = out.parent / 'wav'
    wav_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    n = skipped = 0
    missing = []
    with open(args.index, 'r', encoding='utf-8', errors='ignore') as fin, out.open('w', encoding='utf-8') as fw:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = norm(obj.get('id'))
            q = norm(obj.get('question_text'))
            a = norm(obj.get('answer_text'))
            wav = norm(obj.get('question_audio'))
            if not sid or not q or not a or not wav:
                skipped += 1
                continue
            if not Path(wav).exists() or Path(wav).stat().st_size <= 1000:
                missing.append({'id': sid, 'question_audio': wav})
                if not args.allow_missing_audio:
                    skipped += 1
                    continue
            ex = build_formatted_example(obj, wav_dir / f'{sid}.wav', tok, args)
            if obj.get('source_id') is not None:
                ex['source_id'] = obj.get('source_id')
            fw.write(json.dumps(ex, ensure_ascii=False) + '\n')
            n += 1
    print(json.dumps({'n': n, 'skipped': skipped, 'missing_audio': missing[:20], 'out': str(out), 'wav_dir': str(wav_dir)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
