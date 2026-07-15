#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

QUESTION_KEYS = ['question', 'query', 'input', 'instruction', 'text_query', 'question_text']
ANSWER_KEYS = ['answer', 'response', 'output', 'target', 'target_text', 'answer_text', 'text']


def norm_text(x: Any) -> str:
    return re.sub(r'\s+', ' ', str(x or '')).strip()


def safe_id(x: str) -> str:
    x = re.sub(r'[^0-9A-Za-z_.-]+', '_', str(x or ''))
    return x[:160] or 'sample'


def pick(obj: Dict[str, Any], keys) -> str:
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return norm_text(v)
    return ''


def main() -> None:
    ap = argparse.ArgumentParser(description='Build CosyVoice TTS tasks for text QA questions.')
    ap.add_argument('--input', required=True, help='Input text QA jsonl')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--ref_wav', required=True, help='Zero-shot speaker prompt wav')
    ap.add_argument('--ref_text', required=True, help='Transcript of --ref_wav')
    ap.add_argument('--id_prefix', default='qa')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / 'question_wav'
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    tasks_path = out_dir / 'tts_tasks.jsonl'
    index_path = out_dir / 'qa_index.jsonl'
    n = 0
    skipped = 0
    seen = set()

    with open(args.input, 'r', encoding='utf-8', errors='ignore') as fin, \
         tasks_path.open('w', encoding='utf-8') as ft, \
         index_path.open('w', encoding='utf-8') as fi:
        for line_no, line in enumerate(fin, start=1):
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue
            q = pick(obj, QUESTION_KEYS)
            a = pick(obj, ANSWER_KEYS)
            if not q or not a:
                skipped += 1
                continue
            sid = safe_id(obj.get('id') or obj.get('qid') or f'{args.id_prefix}_{line_no:08d}')
            base = sid
            k = 1
            while sid in seen:
                k += 1
                sid = f'{base}_{k}'
            seen.add(sid)
            out_wav = wav_dir / f'{sid}.wav'
            task = {
                'id': sid,
                'text': q,
                'out': str(out_wav),
                'ref_wav': str(args.ref_wav),
                'ref_text': str(args.ref_text),
            }
            idx = {
                'id': sid,
                'source_id': obj.get('id') or obj.get('qid') or line_no,
                'question_text': q,
                'answer_text': a,
                'question_audio': str(out_wav),
                'tts_task': task,
                'source': obj.get('source', 'textqa'),
            }
            ft.write(json.dumps(task, ensure_ascii=False) + '\n')
            fi.write(json.dumps(idx, ensure_ascii=False) + '\n')
            n += 1

    print(json.dumps({
        'n': n,
        'skipped': skipped,
        'tts_tasks': str(tasks_path),
        'qa_index': str(index_path),
        'wav_dir': str(wav_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
