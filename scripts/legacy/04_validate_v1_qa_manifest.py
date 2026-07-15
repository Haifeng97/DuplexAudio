#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--show', type=int, default=3)
    args = ap.parse_args()
    n = 0
    bad = []
    examples = []
    for line_no, line in enumerate(open(args.manifest, encoding='utf-8', errors='ignore'), start=1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except Exception as e:
            bad.append({'line': line_no, 'err': repr(e)})
            continue
        n += 1
        required = ['id', 'audio', 'text', 'target_text', 'answer_text', 'question_text', 'text_query', 'asr_text']
        miss = [k for k in required if not o.get(k)]
        audio = Path(str(o.get('audio', '')))
        if not audio.exists() or audio.stat().st_size <= 1000:
            miss.append('audio_exists')
        if miss:
            bad.append({'line': line_no, 'id': o.get('id'), 'missing_or_bad': miss, 'audio': o.get('audio')})
        if len(examples) < args.show:
            examples.append(o)
    print(json.dumps({'n': n, 'bad': len(bad), 'bad_examples': bad[:20], 'examples': examples}, ensure_ascii=False, indent=2))
    if bad:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
