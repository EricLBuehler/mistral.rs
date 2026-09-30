#!/usr/bin/env python3
"""Compare two completed adaptive serving stages using the frozen raw validators."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
DEFAULT_BEFORE = Path('/home/ericbuehler/qwen4exp_work/moe_optimization_20260930/final_serving/optimized_final')
BEFORE_SHA = 'd2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d'
HELPERS = ROOT / 'validation_helpers'
sys.path.insert(0, str(HELPERS))
import compare_final as validator  # noqa: E402


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def change(before, after):
    return {'ratio_of_means': after / before, 'percent_change': (after / before - 1) * 100}


def direct_comparison(before, after):
    serving = []
    assert [row['workload'] for row in before['serving_changes']] == [row['workload'] for row in after['serving_changes']]
    for a, b in zip(before['serving_changes'], after['serving_changes'], strict=True):
        old, new = a['optimized_candidate'], b['optimized_candidate']
        serving.append({'workload': a['workload'], 'before': old, 'after': new,
                        **change(old['mean_tokens_per_second'], new['mean_tokens_per_second'])})
    concurrency = []
    assert [row['concurrency'] for row in before['concurrency_changes']] == [row['concurrency'] for row in after['concurrency_changes']]
    for a, b in zip(before['concurrency_changes'], after['concurrency_changes'], strict=True):
        old, new = a['optimized_candidate'], b['optimized_candidate']
        for key in ('mode', 'concurrency', 'requests_per_trial', 'output_tokens_per_request', 'measured_trials', 'warmup_trials'):
            assert old[key] == new[key], key
        concurrency.append({'concurrency': a['concurrency'], 'before': old, 'after': new,
                            'metric_changes': {name: change(old['metrics'][name]['mean'], value['mean'])
                                               for name, value in new['metrics'].items()}})
    ordinary_old = before['ordinary_prompt_mean']['optimized_candidate']
    ordinary_new = after['ordinary_prompt_mean']['optimized_candidate']
    return {'serving_changes': serving, 'concurrency_changes': concurrency,
            'ordinary_prompt_mean': {'before': ordinary_old, 'after': ordinary_new,
                                     **change(ordinary_old, ordinary_new)},
            'scaling': {'before': before['scaling']['optimized_candidate'],
                        'after': after['scaling']['optimized_candidate']},
            'counters': {name: {'before': rows['optimized_candidate'],
                                'after': after['counters'][name]['optimized_candidate']}
                         for name, rows in before['counters'].items()},
            'memory': {name: {'before': rows['optimized_candidate'],
                              'after': after['memory'][name]['optimized_candidate']}
                       for name, rows in before['memory'].items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--expected-binary-sha256', required=True)
    parser.add_argument('--before', type=Path, default=DEFAULT_BEFORE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for folder in (args.before, args.candidate):
        assert load(folder / 'metadata.json')['complete'], f'Wait for clean completion: {folder}'
    before_meta, after_meta = (load(folder / 'metadata.json') for folder in (args.before, args.candidate))
    assert validator.normalized_server_command(before_meta['command']) == validator.normalized_server_command(after_meta['command'])
    assert before_meta['env'] == after_meta['env'], 'Recorded execution environment differs'
    topology_before = load(args.before / 'startup_validation.json')
    topology_after = load(args.candidate / 'startup_validation.json')
    assert topology_before['complete'] and topology_before['checks']['graphs_34']
    assert topology_after['complete'] and topology_after['checks']['graphs_36']
    assert {key: value for key, value in topology_before['checks'].items() if key != 'graphs_34'} == {
        key: value for key, value in topology_after['checks'].items() if key != 'graphs_36'}
    args.output.mkdir(parents=True, exist_ok=False)
    validated = []
    validation_commands = []
    for name, folder, digest in [('before', args.before, BEFORE_SHA),
                                 ('after', args.candidate, args.expected_binary_sha256)]:
        output = args.output / f'{name}_validation_against_common_ancestor'
        command = [sys.executable, str(HELPERS / 'compare_final.py'), str(folder),
                   '--output', str(output), '--expected-binary-sha256', digest]
        with (args.output / f'{name}.validation.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        validation_commands.append(command)
        validated.append(load(output / 'comparison.json'))
    result = direct_comparison(*validated)
    text_smoke = validator.check_text_smokes(load(args.before / 'text.json'), load(args.candidate / 'text.json'))
    text_smoke['before'] = text_smoke.pop('pre_moe_optimization')
    text_smoke['after'] = text_smoke.pop('optimized_candidate')
    result.update({
        'validation': {'complete': True, 'method': 'Both raw stages independently revalidated against the same frozen ancestor with matching workload identities; final differences compare the two stages directly.'},
        'provenance': {'before_directory': str(args.before.resolve()), 'after_directory': str(args.candidate.resolve()),
                       'before_binary_sha256': BEFORE_SHA, 'after_binary_sha256': args.expected_binary_sha256,
                       'expected_graphs': {'before': 34, 'after': 36}, 'validation_commands': validation_commands,
                       'script_sha256': sha(Path(__file__))},
        'text_smoke': text_smoke,
        'mixed_context_smoke': validator.check_mixed_smokes(load(args.before / 'mixed_context.json'), load(args.candidate / 'mixed_context.json')),
        'limitations': [
            'Both stages run adaptive MTP; this is not a fixed-depth-four A/B experiment.',
            'Sequential runs are not a randomized causal estimate; inspect acceptance, mean depth, graph dispatches, generated text, and memory/swap differences.',
            'The intended policy difference adds two eligible depth-four graph shapes, with 36 startup graphs versus 34. Other startup checks must match.',
            'Counter windows include all warmups and measured trials; the C6/C8 command combines both concurrencies.',
            'Graph dispatch counts do not measure kernel-time coverage; global swap counters cannot identify model page-ins or timing impact.',
            'Full raw before/after outputs remain in the run archives. Smoke answers require human semantic inspection; finite logprobs do not prove MTP equivalence.',
            'Intermediate validation reports compare each stage with the historical common ancestor; use this file for the direct graph-policy stage comparison.',
        ],
    })
    save(args.output / 'comparison.json', result)
    lines = ['# Adaptive graph-policy stage comparison', '',
             '| C | Before aggregate tok/s | After aggregate tok/s | Change | After per-active tok/s |',
             '|---:|---:|---:|---:|---:|']
    for row in result['concurrency_changes']:
        a, b = row['before']['metrics'], row['after']['metrics']
        lines.append(f"| {row['concurrency']} | {a['aggregate_output_tokens_per_second']['mean']:.3f} | {b['aggregate_output_tokens_per_second']['mean']:.3f} | {row['metric_changes']['aggregate_output_tokens_per_second']['percent_change']:+.2f}% | {b['latency_weighted_request_output_tokens_per_second']['mean']:.3f} |")
    lines.extend(['', 'Raw samples and sample standard deviations are in comparison.json.', ''])
    lines.extend('- ' + limit for limit in result['limitations'])
    (args.output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    save(args.output / 'SHA256SUMS.json', {str(path.relative_to(args.output)): sha(path)
                                         for path in sorted(args.output.rglob('*')) if path.is_file()})
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
