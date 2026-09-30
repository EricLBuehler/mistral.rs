#!/usr/bin/env python3
import collections
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def occupancy(rows, correlated):
    counts = collections.Counter()
    for row in range(rows):
        group = row // 7 if correlated else row
        for rank in range(10):
            counts[(group * 131 + rank * 53 + 17) % 512] += 1
    histogram = collections.Counter(counts.values())
    histogram[0] = 512 - len(counts)
    return {
        'experts_by_assigned_rows': dict(sorted(histogram.items())),
        'active_experts': len(counts),
        'max_expert_rows': max(counts.values()),
        'mean_rows_per_active_expert': sum(counts.values()) / len(counts),
    }


results = []
for scheme in ('gguf', 'isq'):
    raw = json.loads((ROOT / f'{scheme}.json').read_text())
    for case in raw['results']:
        result = {key: case[key] for key in ('scheme', 'routing', 'rows', 'gate_dtype', 'down_dtype', 'numerical')}
        result['occupancy'] = occupancy(case['rows'], case['routing'] == 'groups_of_7')
        assert result['occupancy']['active_experts'] == case['active_experts']
        assert result['occupancy']['max_expert_rows'] == case['max_expert_rows']
        for path in ('gemv', 'grouped'):
            values = [sample['stream_elapsed_ms_per_layer'] for sample in case[path]]
            result[path] = {'median_ms': statistics.median(values), 'min_ms': min(values), 'max_ms': max(values)}
        result['gemv_over_grouped_median_ratio'] = result['gemv']['median_ms'] / result['grouped']['median_ms']
        results.append(result)

scaling = []
for scheme in ('gguf', 'isq'):
    for routing in ('independent', 'groups_of_7'):
        cases = {case['rows']: case for case in results if case['scheme'] == scheme and case['routing'] == routing}
        one, eight = cases[1]['gemv']['median_ms'], cases[8]['gemv']['median_ms']
        scaling.append({'scheme': scheme, 'routing': routing, 'gemv_one_row_ms': one, 'gemv_eight_rows_ms': eight,
                        'latency_ratio_eight_over_one': eight / one,
                        'isolated_row_throughput_ratio_eight_over_one': 8 * one / eight})

summary = {
    'scope': 'Real checkpoint layer8 expert weights; BF16 synthetic activations; deterministic synthetic routes. Full eager expert pipeline, repeated warmed single layer. No graphs or server requests.',
    'routing_labels': {
        'independent': 'Deterministic spread control, different affine expert set per row; not independent random samples.',
        'groups_of_7': 'Seven adjacent rows share all10 experts. Groups use different affine expert sets.',
        'expert_formula': '(group * 131 + rank * 53 + 17) % 512; group=row or row//7',
    },
    'timing': 'Median of7 paired alternating rounds,10 invocations per round,3 warmups per path. CUDA event elapsed spans the full pipeline and host launch gaps; does not isolate summed kernel duration.',
    'caveats': [
        'Repeated single-layer timing has different L2/cache residency and weight reuse from a48-layer model; do not extrapolate serving throughput.',
        'Synthetic routing brackets overlap patterns and does not establish actual model routing distributions.',
        'Cross-path finite-output, relative-RMS and cosine checks passed; neither path is a high-precision mathematical oracle.',
        'No evidence here supports raising the32-row grouped dispatch threshold.',
        'A smaller tile remains untested; lowering ncols_max also reduces grid coverage and is unsafe if an expert receives more rows than that bound.',
    ],
    'gemv_one_to_eight_scaling': scaling,
    'results': results,
    'raw_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('gguf.json', 'isq.json')},
}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['Actual Flash-Next layer8 expert dispatch microbenchmark', '', summary['scope'], '',
         'Scheme Routing Rows Active MaxRows GEMV_ms Grouped_ms GEMV/Grouped']
for case in results:
    lines.append(f"{case['scheme']} {case['routing']} {case['rows']} {case['occupancy']['active_experts']} {case['occupancy']['max_expert_rows']} {case['gemv']['median_ms']:.4f} {case['grouped']['median_ms']:.4f} {case['gemv_over_grouped_median_ratio']:.3f}")
lines.extend(['', summary['timing'], '', *summary['caveats']])
(ROOT / 'findings.txt').write_text('\n'.join(lines) + '\n')
