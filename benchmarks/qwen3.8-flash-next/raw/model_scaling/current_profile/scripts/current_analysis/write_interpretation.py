#!/usr/bin/env python3
"""Summarize saved exact-window analysis without querying SQLite or CUDA."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = ROOT.parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = json.loads((RESULTS / 'compact_summary.json').read_text())
metadata = json.loads((MODEL / 'current_capture/metadata.json').read_text())
rows = {}
for phase in summary:
    name = phase['phase']
    base_path = MODEL / 'analysis' / f'{name}.analysis.json'
    base = json.loads(base_path.read_text())['windows']['client_request_envelope']
    request_path = MODEL / 'current_capture' / f'{name}.requests.json'
    trial = json.loads(request_path.read_text())['trial']
    norm = phase['normalization']
    category = {row['category']: row['envelope_ms_per_completed_output_token'] for row in norm['categories']}
    groups = {
        'Expert projections': category['moe_gemv'] + category['moe_grouped_mmq_context'],
        'Vocabulary projection': category['vocabulary_head_projection'],
        'GDN projections': category['gdn_projection'],
        'Hyper-connection projections': sum(category[k] for k in ['hyper_up_projection', 'hyper_down_projection', 'hyper_inject_projection']),
        'Attention projections': category['attention_projection'],
        'Shared expert projections': category['shared_expert_projection'],
        'GDN recurrence and convolution': category['gdn'],
        'Attention kernels and cache': category['attention'],
    }
    groups['Other kernels'] = norm['summed_kernel_ms_per_completed_output_token'] - sum(groups.values())
    groups['All kernels'] = norm['summed_kernel_ms_per_completed_output_token']
    groups['All except expert projections'] = groups['All kernels'] - groups['Expert projections']
    assert abs(sum(v for k, v in groups.items() if k not in ['All kernels', 'All except expert projections']) - groups['All kernels']) < 1e-10
    busy = base['gpu_activity_busy_union_ns']
    elapsed = base['duration_ns']
    stages = {}
    for kind, stats in phase['request_window']['forward_kind_totals'].items():
        stages[kind] = {
            **stats,
            'kernel_share_pct': 100 * stats['kernel_time_ns'] / base['summed_kernel_time_ns'],
            'graph_kernel_time_pct': 100 * stats['graph_kernel_time_ns'] / stats['kernel_time_ns'],
            'head_fraction_of_stage_kernel_time_pct': 100 * stats['head_kernel_time_ns'] / stats['kernel_time_ns'],
        }
    counts = phase['phase_counters']['counts']
    rows[name] = {
        'output_tokens': norm['completed_output_tokens'],
        'client_trial_wall_seconds': trial['wall_seconds'],
        'instrumented_finite_phase_output_tokens_per_second': trial['completed_output_tokens'] / trial['wall_seconds'],
        'request_envelope_ns': elapsed,
        'observed_gpu_activity_union_ns': busy,
        'observed_kernel_union_ns': base['kernel_busy_union_ns'],
        'observed_gpu_activity_busy_pct': 100 * busy / elapsed,
        'observed_no_gpu_activity_ns': elapsed - busy,
        'observed_no_gpu_activity_internal_ns': base['gpu_internal_idle_gap_ns']['sum'],
        'observed_no_gpu_activity_boundary_ns': base['boundary_idle_ns'],
        'fixed_observed_gpu_work_remove_all_observed_gaps_speedup': elapsed / busy,
        'groups_ms_per_output': groups,
        'stages': stages,
        'counters': phase['phase_counters'],
        'proposed_plus_sequence_rows_per_output': (counts['proposed_draft_tokens'] + counts['sequence_proposals']) / norm['completed_output_tokens'],
        'api_call_kinds': base['api_call_kinds'],
        'draft': {k: v for k, v in norm.items() if 'draft' in k or 'projected_row' in k or k == 'proposal_limits'},
        'input_hashes': {str(base_path): sha(base_path), str(request_path): sha(request_path)},
    }

c1, c8 = rows['profile_c1'], rows['profile_c8']
ratios = {name: c1['groups_ms_per_output'][name] / c8['groups_ms_per_output'][name] for name in c1['groups_ms_per_output']}
limits = [
    'This is the current d2c85e instrumented snapshot with masked small MMQ tiles, scoped MMQ at rows 24-31, and the autotuner fixes. It is not the historical pre-fix trace and does not include the later graph-cap candidate.',
    'All per-output and per-proposal normalization uses the whole finite request envelope, including prefill and startup/drain. These are summed recorded kernel durations, not disjoint wall-time components or pure-decode measurements.',
    'Observed GPU activity union includes recovered kernels and transfers. Nsight warns that some CUDA and OS runtime events may be missing. Observed idle can therefore include unrecorded activity.',
    'The gap-removal ratio is conditional on fixed recorded GPU work and removal of every observed gap. It is not a bound on kernel redesigns, changes in accepted work, or a prediction for a graph-cap change.',
    'C1 has 8 requests and C8 has 24, each requesting 128 outputs with the repeated canonical prompt mix. Different batching, outputs and routing prevent this from being a paired same-route scaling experiment.',
    'Target/draft attribution follows 97 versus 3 hyper-connection mixes between vocabulary heads. Mixed 100-mix intervals remain unknown; post-head sampling belongs to the next interval. Both target interval counts and draft projected rows independently match phase telemetry.',
    'Graph time fraction is the duration of kernels carrying graph IDs, not graph hit rate or the fraction of wall time spent launching graphs. API durations overlap GPU execution and must not be added to the wall-time decomposition.',
    'The same-server capture-off/profiled phases are diagnostic controls, not a pure profiler-overhead experiment: only 4/8 and 14/24 outputs were text-identical.',
]
result = {
    'snapshot_label': summary[0]['snapshot_label'],
    'binary_sha256': metadata['plan']['binary_sha256'],
    'normalization': 'Whole finite client request envelope; summed recovered kernel ms per completed output token.',
    'phases': rows,
    'c1_over_c8_kernel_cost_ratios': ratios,
    'instrumented_finite_throughput_ratio': c8['instrumented_finite_phase_output_tokens_per_second'] / c1['instrumented_finite_phase_output_tokens_per_second'],
    'limitations': limits,
    'source_hashes': {str(RESULTS / 'compact_summary.json'): sha(RESULTS / 'compact_summary.json'), str(MODEL / 'current_capture/metadata.json'): sha(MODEL / 'current_capture/metadata.json'), str(Path(__file__).resolve()): sha(Path(__file__).resolve())},
}
(RESULTS / 'interpretation.json').write_text(json.dumps(result, indent=2) + '\n')

lines = [
    '# Current C1/C8 kernel accounting', '',
    f"Current binary SHA256: `{result['binary_sha256']}`. Snapshot: `{result['snapshot_label']}`.", '',
    'The current trace localizes the weak batching gain to expert projections. Their recorded cost per output improves only 1.54x; all other kernels together improve 4.13x. At C8, expert projections occupy 69.19% of summed recovered kernel time. Acceptance remains similar, so the observed scaling loss is not explained by an acceptance collapse.', '',
    'The table uses the full finite request envelope, including prefill and drain. It reports kernel duration per completed output, not wall-time shares or pure decode. C1 completed 1,024 output tokens and C8 completed 3,072.', '',
    '| Recorded kernel group | C1 ms/output | C8 ms/output | C1/C8 cost ratio |',
    '| --- | ---: | ---: | ---: |',
]
for name, c1val in c1['groups_ms_per_output'].items():
    lines.append(f"| {name} | {c1val:.3f} | {c8['groups_ms_per_output'][name]:.3f} | {ratios[name]:.2f}x |")
lines += [
    '',
    'The vocabulary head, GDN projections, hyper-connection projections, and attention projections each amortize roughly 4.8-5.2x. GDN recurrence/convolution amortizes only 1.16x, but costs 0.308 ms/output at C8, compared with 5.016 ms/output for experts. Expert routing, reduction, and grouped activation quantization together are 0.074 ms/output at C8. They are not a hidden large share comparable with the three expert projections.', '',
    '| Current phase measure | C1 | C8 |',
    '| --- | ---: | ---: |',
    '| Draft token acceptance | 77.75% | 76.44% |',
    '| Mean proposed depth per sequence proposal | 3.359 | 3.789 |',
    '| Accepted draft tokens per sequence proposal | 2.612 | 2.896 |',
    '| Target graph replays / eager dispatches | 289 / 0 | 59 / 65 |',
    '| Target-shaped graph kernel time coverage | 94.79% | 28.76% |',
    '| Draft-shaped graph kernel time coverage | 0% | 0% |',
    '| Draft-shaped share of summed kernel time | 13.68% | 8.77% |',
    '| Draft kernel ms per proposed token | 2.514 | 0.660 |',
    '| Draft vocabulary projection ms per proposed token | 1.644 | 0.309 |',
    '| Proposed draft + sequence rows per output | 1.196 | 1.217 |',
    '',
    'All 65 C8 eager target dispatches report `batch_unsupported`; prefill is counted separately (8 C1 and 23 C8). The 289/124 target-shaped intervals exactly match the target dispatch totals. Draft vocabulary projections contain exactly 944/2,959 rows, matching the proposed-token counters. Mixed target/draft intervals remain unassigned: 1.46% of C1 and 10.43% of C8 summed kernel time. The graph-cap expansion is therefore a concrete next control, but these counts alone do not predict its gain.', '',
    '| Whole request envelope | C1 | C8 |',
    '| --- | ---: | ---: |',
]
for label, key in [('Envelope seconds', 'request_envelope_ns'), ('Observed GPU activity union seconds', 'observed_gpu_activity_union_ns'), ('Observed no-GPU-activity seconds', 'observed_no_gpu_activity_ns')]:
    lines.append(f"| {label} | {c1[key] / 1e9:.6f} | {c8[key] / 1e9:.6f} |")
lines += [
    f"| Observed GPU busy fraction | {c1['observed_gpu_activity_busy_pct']:.2f}% | {c8['observed_gpu_activity_busy_pct']:.2f}% |",
    f"| Remove all observed gaps, fixed GPU work | {c1['fixed_observed_gpu_work_remove_all_observed_gaps_speedup']:.3f}x | {c8['fixed_observed_gpu_work_remove_all_observed_gaps_speedup']:.3f}x |", '',
    'At C8, the recovered trace has 22.303 seconds of GPU activity in a 23.442-second envelope. Removing its entire 1.140 seconds of observed gaps would give at most 1.051x under fixed recorded GPU work. This conditional calculation is not an overall hardware ceiling: kernels, accepted work, or the GPU schedule could change, and missing events can inflate apparent idle. It does show that merely removing the observed launch gaps cannot explain a several-fold gain on this run.', '',
    'The next supported check is the narrow depth-4 graph expansion, measured with matched homogeneous prompts and explicit acceptance/graph counters. It tests the 65 observed batch-unsupported dispatches without assuming they cost 65 full forward passes of CPU idle. The draft path remains eager, but eliminating all of its recorded kernels would remove only 8.77% of the summed kernel work; optimizing it alone cannot recover near-linear C8 scaling. The small Q6K hyper-injection projection is a possible focused follow-up (0.268 seconds total, including 0.185 seconds of MMQ fixup), but its entire category is only 1.21% of C8 summed kernels.', '',
    'C1/C8 instrumented finite-phase throughput was 53.139/131.031 output tokens/s (2.466x). This is diagnostic trace context, not a new unprofiled benchmark. Capture-off/profiled wall times were 19.393/19.270 seconds for C1 and 23.393/23.445 seconds for C8, with only 4/8 and 14/24 text-identical outputs.', '',
    'The sampled clock-offset envelopes are 2.992 us for C1 and 2.704 us for C8. Switching from the whole request envelope to its trimmed interior changes recorded kernel duration by only 2,416 ns and 0 ns respectively. No full-phase token counts are assigned to the separately exported completion-boundary interior.', '',
    'Nsight retained these warnings: not all CUDA events might have been collected; not all OS runtime events might have been collected; no NVTX events; absent scheduling data; unified-memory tracing unavailable; cuBLAS symbol lookup failures. This report describes recovered kernel timelines, not guaranteed complete hardware activity.', '',
    'Reproduce the arithmetic with `python3 current_analysis/write_interpretation.py` after running the gated adapter. The exact source/trace/export/input hashes are in `results/provenance.json`; `results/manifest.json` and the parent `SHA256SUMS` cover the derived artifacts. No GPU execution or model requests are involved in this report generation.',
]
(RESULTS / 'interpretation.md').write_text('\n'.join(lines) + '\n')
manifest = {file.name: sha(file) for file in sorted(RESULTS.iterdir()) if file.is_file() and file.name != 'manifest.json'}
(RESULTS / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'binary_sha256': result['binary_sha256'], 'ratios': ratios, 'output': str(RESULTS / 'interpretation.md')}, indent=2))
