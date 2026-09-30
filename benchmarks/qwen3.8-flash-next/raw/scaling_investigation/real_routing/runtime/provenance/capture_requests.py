#!/usr/bin/env python3
"""Submit diagnostic capture requests; recorded wall times are not benchmarks."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'benchmarks/qwen3.8-flash-next'))
import bench_serving


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--base-url', default='http://127.0.0.1:1234')
    parser.add_argument('--max-tokens', type=int, default=128, choices=[128, 256])
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve prior request result'
    report = {
        'scope': 'diagnostic eager capture with fixed MTP depth6; synchronized capture and wall times are not throughput measurements',
        'requested_concurrencies': [1, 6, 8], 'requests_per_concurrency': 8,
        'max_tokens': args.max_tokens, 'fixed_mtp_depth': 6, 'logprobs_requested': False,
        'validation': 'Exact output counts/nonempty text here; finite captured operands/output checked by validate_capture.py',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'prompts_source_sha256': hashlib.sha256(Path(bench_serving.__file__).read_bytes()).hexdigest(),
        'controls': [], 'results': [], 'complete': False,
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    def arm(case, rows, max_per_shape, every_n):
        subprocess.run([sys.executable, str(WORK / 'control_capture.py'), 'arm',
                        '--directory', str(args.directory), '--case', case,
                        '--rows', *map(str, rows), '--max-per-shape', str(max_per_shape),
                        '--every-n', str(every_n), '--skip-per-shape', '0'], check=True)
        control = json.loads((args.directory / 'control.json').read_text())
        report['controls'].append(dict(control=control, armed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        save()

    def request(index):
        name, prompt = list(bench_serving.PROMPTS.items())[index]
        result = bench_serving.request(args.base_url, prompt, args.max_tokens)
        response = result['response']
        assert len(response['choices']) == 1
        assert isinstance(response['choices'][0]['text'], str) and response['choices'][0]['text']
        assert response['usage']['prompt_tokens'] > 0
        return dict(index=index, prompt_name=name, prompt=prompt, logprobs_requested=False,
                    expected_output_count_met=True, semantic_correctness='not automatically asserted', **result)

    try:
        for concurrency in [1, 6, 8]:
            if concurrency == 1:
                for index in range(8):
                    case = f'c1_r{index:02}'
                    arm(case, [1, 7], 1, 1)
                    result = request(index)
                    report['results'].append(dict(case=case, requested_concurrency=concurrency, **result))
                    save()
            else:
                case = f'c{concurrency}'
                arm(case, [concurrency, concurrency * 7], 8, 4)
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(request, index) for index in range(8)]
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        report['results'].append(dict(case=case, requested_concurrency=concurrency, **result))
                        save()
        assert len(report['results']) == 24
        report['complete'] = True
    finally:
        (args.directory / 'control.json').unlink(missing_ok=True)
        save()
    print('Capture requests complete; this is not a throughput benchmark.')


if __name__ == '__main__':
    main()
