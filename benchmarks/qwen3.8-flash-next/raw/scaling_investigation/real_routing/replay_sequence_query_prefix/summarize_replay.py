#!/usr/bin/env python3
"""Summarize isolated real-route kernel timings; never infer model throughput."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


EXPERT_WEIGHT_BYTES = 2867200


def range_stats(values):
    assert values and all(math.isfinite(value) for value in values)
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--oracle', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve prior summary'
    raw = json.loads(args.input.read_text())
    oracle = json.loads(args.oracle.read_text())
    oracle_lookup = {(row['source_metadata'], row['selection'], tuple(row['source_row_indices'])): row
                     for row in oracle['results']}
    groups = collections.defaultdict(list)
    samples = []
    for row in raw['results']:
        cohort = row['capture']['case'].split('_', 1)[0]
        selection = row['selection']
        assert row['derived'] == (selection != 'native')
        assert [row['capture'][name] for name in ['gate_dtype', 'up_dtype', 'down_dtype']] == ['Q4K', 'Q4K', 'Q4_1']
        assert len(row['source_row_indices']) == row['rows']
        assert len(set(row['source_row_indices'])) == row['rows']
        assert sum(row['expert_occupancy']) == row['rows'] * raw['topk']
        assert sum(count > 0 for count in row['expert_occupancy']) == row['active_experts']
        assert max(row['expert_occupancy']) == row['max_expert_rows']
        sample = dict(source_metadata=row['source_metadata'], cohort=cohort,
                      stage=row['capture']['stage'], selection=selection, rows=row['rows'],
                      derived=row['derived'], source_row_indices=row['source_row_indices'],
                      active_experts=row['active_experts'], max_expert_rows=row['max_expert_rows'])
        sample['cross_kernel_guard_passed'] = row['cross_kernel_guard_passed']
        sample['native_replay_guard_passed'] = row['native_replay_guard_passed']
        oracle_key = (row['source_metadata'], selection, tuple(row['source_row_indices']))
        sample['fp32_oracle'] = oracle_lookup.get(oracle_key)
        if not row['derived'] or not row['cross_kernel_guard_passed'] or selection == 'sequence_query_prefix':
            assert sample['fp32_oracle'] is not None
        sample['mean_assignments_per_selected_expert'] = row['rows'] * raw['topk'] / row['active_experts']
        sample['selected_distinct_expert_weight_bytes'] = row['active_experts'] * EXPERT_WEIGHT_BYTES
        for backend in ['gemv', 'grouped']:
            rounds = row[backend]
            assert len(rounds) == raw['rounds']
            for timing in rounds:
                assert all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                           for value in timing.values())
            sample[backend] = {key: range_stats([timing[key] for timing in rounds])
                               for key in ['stream_elapsed_ms_per_layer', 'host_elapsed_ms_per_layer']}
        sample['gemv_over_grouped_stream_ratio'] = (sample['gemv']['stream_elapsed_ms_per_layer']['median'] /
                                                    sample['grouped']['stream_elapsed_ms_per_layer']['median'])
        sample['gemv_over_grouped_host_ratio'] = (sample['gemv']['host_elapsed_ms_per_layer']['median'] /
                                                  sample['grouped']['host_elapsed_ms_per_layer']['median'])
        sample['worst_relative_rms'] = max(row[name]['relative_rms']
            for name in ['numerical', 'captured_vs_gemv', 'captured_vs_grouped'])
        sample['lowest_cosine'] = min(row[name]['cosine']
            for name in ['numerical', 'captured_vs_gemv', 'captured_vs_grouped'])
        assert math.isfinite(sample['worst_relative_rms']) and math.isfinite(sample['lowest_cosine'])
        samples.append(sample)
        groups[(cohort, sample['stage'], selection, row['rows'])].append(sample)
    assert samples
    summary = []
    for (cohort, stage, selection, rows), values in sorted(groups.items()):
        record = dict(cohort=cohort, stage=stage, selection=selection, rows=rows, samples=len(values),
                      derived=selection != 'native',
                      active_experts=range_stats([value['active_experts'] for value in values]),
                      max_expert_rows=range_stats([value['max_expert_rows'] for value in values]),
                      gemv_over_grouped_stream_ratio=range_stats([value['gemv_over_grouped_stream_ratio'] for value in values]),
                      gemv_over_grouped_host_ratio=range_stats([value['gemv_over_grouped_host_ratio'] for value in values]),
                      worst_relative_rms=max(value['worst_relative_rms'] for value in values),
                      lowest_cosine=min(value['lowest_cosine'] for value in values))
        record['cross_kernel_guard_failures'] = sum(not value['cross_kernel_guard_passed'] for value in values)
        oracles = [value['fp32_oracle'] for value in values if value['fp32_oracle'] is not None]
        record['fp32_oracle_samples'] = len(oracles)
        record['fp32_oracle'] = {backend: {metric: range_stats([value['reference_vs_' + backend][metric] for value in oracles])
            for metric in ['relative_rms', 'cosine']} for backend in ['gemv', 'grouped']} if oracles else None
        record['mean_assignments_per_selected_expert'] = range_stats([value['mean_assignments_per_selected_expert'] for value in values])
        record['selected_distinct_expert_weight_bytes'] = range_stats([value['selected_distinct_expert_weight_bytes'] for value in values])
        for backend in ['gemv', 'grouped']:
            record[backend] = {key: range_stats([value[backend][key]['median'] for value in values])
                              for key in ['stream_elapsed_ms_per_layer', 'host_elapsed_ms_per_layer']}
        summary.append(record)
        print(f"{cohort} {stage} {selection} rows={rows} n={len(values)} "
              f"GEMV={record['gemv']['stream_elapsed_ms_per_layer']['median']:.4f}ms "
              f"MMQ={record['grouped']['stream_elapsed_ms_per_layer']['median']:.4f}ms "
              f"GEMV/MMQ={record['gemv_over_grouped_stream_ratio']['median']:.3f} "
              f"active_experts={record['active_experts']['median']} max_rows={record['max_expert_rows']['median']} "
              f"cross_guard_failures={record['cross_kernel_guard_failures']}")
    report = dict(oracle_sha256=hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        accuracy_scope='Cross-kernel guard failures are reported, not waived as safe replacements. All native source-kernel equivalence checks remain strict. Independent FP32 reference errors are measurements, with no invented tolerance or model-quality guarantee.',
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        source_layer=raw['layer'], groups=summary, samples=samples,
        expert_weight_bytes=EXPERT_WEIGHT_BYTES,
        reuse_definition='10*M/U assignments per selected expert measures potential intra-call reuse. U*2867200 is distinct selected logical expert weight payload for these Q4K/Q4K/Q4_1 weights, not actual DRAM traffic or a bandwidth lower bound; caches may supply weights.',
        statistics='Per input: median of seven alternating-order timing rounds, each averaging ten forwards. Group: median/min/max of per-input medians; ratio summaries use per-input paired ratios.',
        ratio_definition='GEMV milliseconds divided by grouped-MMQ milliseconds; greater than one means grouped-MMQ was faster for this isolated sample.',
        scope='Exact captured target-layer8 expert inputs, routes, weights and output, plus explicitly derived controls. Timings cover warmed repeated eager expert work only.',
        limitations=[
            'This is not full-model throughput, decode TPS, CUDA graph performance, or a natural MTP-depth/route distribution.',
            'Native captures use fixed-depth6 MTP, disabled graphs and synchronized diagnostics on ordinary diverse prompts.',
            'A repeated isolated layer can reuse selected expert weights in cache differently from a full model.',
            'CUDA-event intervals include host launch gaps; both host and stream medians are retained.',
            'Derived prefix, per-sequence query-zero and per-sequence query-prefix operands are real captured rows, not independently observed smaller decode or lower-depth verification batches.',
            'The hook excludes the shared expert and all attention/GDN/hyperconnection work.',
        ])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
