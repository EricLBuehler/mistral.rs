#!/usr/bin/env python3
"""Compare isolated MMQ tile timings only for strictly validated matching operands."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    assert values and all(math.isfinite(x) and x > 0 for x in values)
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values))


def key(row):
    return (row['source_metadata'], row['selection'], tuple(row['source_row_indices']))


def load(directory):
    meta = json.loads((directory / 'lifecycle.json').read_text())
    assert meta['complete']
    assert sha(directory / 'replay.json') == meta['replay_sha256']
    raw = json.loads((directory / 'replay.json').read_text())
    return meta, raw, {key(row): row for row in raw['results']}


def median(row, backend, metric):
    values = [timing[metric] for timing in row[backend]]
    return stats(values)['median']


def monitor_summary(directory, metadata):
    record = metadata.get('gpu_monitor')
    if not record:
        return None
    path = directory / record['csv']
    assert sha(path) == record['csv_sha256']
    rows = list(csv.reader(path.read_text().splitlines()))
    header = [name.strip() for name in rows[0]]
    values = {name: [] for name in header[1:]}
    unavailable = {name: 0 for name in header[1:]}
    timestamps = []
    for row in rows[1:]:
        if [cell.strip() for cell in row] == header:
            continue
        assert len(row) == len(header)
        timestamps.append(row[0].strip())
        for name, cell in zip(header[1:], row[1:]):
            try:
                value = float(cell.strip())
            except ValueError:
                unavailable[name] += 1
                continue
            assert math.isfinite(value)
            values[name].append(value)
    return dict(csv_sha256=record['csv_sha256'], samples=len(timestamps),
                first_timestamp=timestamps[0], last_timestamp=timestamps[-1],
                scope=record['scope'],
                fields={name: dict(samples=len(items), unavailable=unavailable[name],
                                  median=statistics.median(items), minimum=min(items), maximum=max(items),
                                  mean=statistics.mean(items)) if items else dict(samples=0, unavailable=unavailable[name])
                        for name, items in values.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--variants', required=True, nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve prior summary'
    base_meta, base_raw, base = load(args.baseline)
    phases = []
    for directory in args.variants:
        meta = json.loads((directory / 'lifecycle.json').read_text())
        phase = dict(path=str(directory), variant=meta['variant'], complete=meta['complete'],
                     lifecycle_sha256=sha(directory / 'lifecycle.json'),
                     gpu_monitor=monitor_summary(directory, meta))
        if not meta['complete']:
            phase['error'] = meta.get('error')
            phase['interpretation'] = 'Failed phase is not a validated candidate; inspect raw log and partial outputs.'
            phases.append(phase)
            continue
        meta, raw, rows = load(directory)
        assert set(rows) == set(base)
        assert meta['baseline_replay_sha256'] == base_meta['replay_sha256']
        assert meta['test_source_sha256'] == base_meta['test_source_sha256']
        assert [raw[name] for name in ('rounds', 'warmups', 'repetitions')] == [base_raw[name] for name in ('rounds', 'warmups', 'repetitions')]
        grouped = defaultdict(list)
        samples = []
        for identity, row in rows.items():
            original = base[identity]
            difference = row['baseline_grouped_comparison']
            assert difference['relative_rms'] < 1e-3 and difference['cosine'] > 0.999999
            sample = dict(source_metadata=row['source_metadata'], selection=row['selection'],
                          source_row_indices=row['source_row_indices'], rows=row['rows'], derived=row['derived'],
                          baseline_grouped_comparison=difference,
                          active_experts=row['active_experts'], max_expert_rows=row['max_expert_rows'])
            for metric in ('stream_elapsed_ms_per_layer', 'host_elapsed_ms_per_layer'):
                before = median(original, 'grouped', metric)
                after = median(row, 'grouped', metric)
                sample[metric] = dict(baseline=before, variant=after, baseline_over_variant=before / after,
                                     baseline_gemv_over_variant_gemv=median(original, 'gemv', metric) / median(row, 'gemv', metric))
            samples.append(sample)
            grouped[(row['selection'], row['rows'])].append(sample)
        groups = []
        for (selection, width), samples_in_group in sorted(grouped.items()):
            group = dict(selection=selection, rows=width, samples=len(samples_in_group), derived=selection != 'native',
                         worst_relative_rms=max(s['baseline_grouped_comparison']['relative_rms'] for s in samples_in_group),
                         lowest_cosine=min(s['baseline_grouped_comparison']['cosine'] for s in samples_in_group))
            for metric in ('stream_elapsed_ms_per_layer', 'host_elapsed_ms_per_layer'):
                group[metric] = {name: stats([s[metric][name] for s in samples_in_group])
                    for name in ('baseline', 'variant', 'baseline_over_variant', 'baseline_gemv_over_variant_gemv')}
            groups.append(group)
            timing = group['stream_elapsed_ms_per_layer']
            print(f"tile={meta['variant']} {selection} rows={width} n={len(samples_in_group)} "
                  f"base={timing['baseline']['median']:.4f}ms variant={timing['variant']['median']:.4f}ms "
                  f"paired_ratio={timing['baseline_over_variant']['median']:.3f} "
                  f"range={timing['baseline_over_variant']['minimum']:.3f}..{timing['baseline_over_variant']['maximum']:.3f}")
        phase.update(groups=groups, samples=samples, binary=meta['binary'])
        phases.append(phase)
    report = dict(baseline=str(args.baseline), baseline_lifecycle_sha256=sha(args.baseline / 'lifecycle.json'),
                  phases=phases, baseline_gpu_monitor=monitor_summary(args.baseline, base_meta),
                  ratio='Baseline grouped-MMQ milliseconds divided by variant grouped-MMQ milliseconds for the SAME source rows. Greater than one means variant is faster.',
                  statistics='Each input uses median of seven rounds averaging ten forwards. Group ratio is median of per-input paired ratios, not ratio of group time medians.',
                  gemv_control='The unchanged GEMV path is timed in both binaries; its baseline/variant time ratio is an observational drift control, not a correction factor.',
                  scope='Warm repeated eager target-layer8 expert pipeline only. Native42/56 and explicitly derived B8xQ3/4/5 layouts. No full-model throughput or deployment claim.',
                  correctness='Every successful variant sample must reproduce separately saved baseline grouped output with relative RMS<1e-3 and cosine>.999999, in addition to finite and native-capture checks. This is kernel equivalence, not a model-quality guarantee.')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
