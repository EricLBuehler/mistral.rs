#!/usr/bin/env python3
"""One finite closed-loop request phase with exact monotonic request timestamps."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from clock_anchor import anchor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'dependencies'))
import bench_concurrency  # noqa: E402
import bench_serving  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:1234')
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--concurrency', required=True, type=int)
    parser.add_argument('--requests', required=True, type=int)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.requests < args.concurrency or args.requests % len(bench_serving.PROMPTS):
        parser.error('Use balanced eight-prompt cycles with at least concurrency requests')
    if min(args.concurrency, args.requests, args.max_tokens, args.timeout) <= 0:
        parser.error('Counts and timeout must be positive')
    if args.output.exists():
        parser.error('Output already exists')
    tokenizer = bench_serving.Tokenizer.from_file(str(args.tokenizer))
    args.expected_prompt_tokens = {
        name: len(tokenizer.encode(prompt, add_special_tokens=False).ids)
        for name, prompt in bench_serving.PROMPTS.items()
    }
    result = {
        'complete': False, 'scope': 'One profiled or warmup diagnostic phase, not a normal benchmark.',
        'client_sha256': sha(Path(__file__)),
        'shared_harness_sha256': sha(Path(bench_serving.__file__)),
        'timestamped_harness_sha256': sha(Path(bench_concurrency.__file__)),
        'tokenizer_sha256': sha(args.tokenizer),
        'prompts_sha256': hashlib.sha256(json.dumps(bench_serving.PROMPTS, sort_keys=True).encode()).hexdigest(),
        'settings': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'clock_before': anchor(),
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    save()
    try:
        trial = bench_concurrency.run_trial(args, 'closed-loop', args.concurrency)
        result['clock_after'] = anchor()
        result['trial'] = trial
        result['request_window_perf_counter_ns'] = {
            'start': min(row['started_perf_counter_ns'] for row in trial['requests']),
            'end': max(row['finished_perf_counter_ns'] for row in trial['requests']),
            'definition': 'First HTTP request begin through last complete response read; includes prefill/startup/drain and client overhead.',
        }
        if not trial['complete']:
            raise ValueError('Incomplete requests; see saved trial errors')
        result['complete'] = True
    finally:
        save()
    print(json.dumps({'complete': True, 'requests': len(trial['requests']),
                      'output_tokens': trial['completed_output_tokens'],
                      'profiled_client_wall_seconds': trial['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
