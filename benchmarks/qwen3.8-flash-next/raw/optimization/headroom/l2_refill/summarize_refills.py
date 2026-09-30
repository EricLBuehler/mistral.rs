#!/usr/bin/env python3
"""Compare measured L2 refill byte equivalents with selected quantized weights."""

import argparse
import hashlib
import json
from pathlib import Path

SECTOR_BYTES = 32
EXPERTS = 512
HIDDEN = 2560
INTERMEDIATE = 640
FORMATS = [('gate', 12, 256, 144), ('up', 12, 256, 144), ('down', 3, 32, 20)]


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'metadata.json').read_text())
    counters = json.loads((args.run / 'kernels.json').read_text())
    replay = json.loads((args.run / 'profiled_test_output_not_benchmark.json').read_text())
    assert metadata['complete']
    assert (replay['experts'], replay['hidden'], replay['intermediate']) == (EXPERTS, HIDDEN, INTERMEDIATE)
    native = [row for row in replay['results'] if not row['derived']]
    assert len(native) == 1 and native[0]['native_replay_guard_passed']
    source_match = native[0]['captured_vs_grouped']
    assert source_match['max_abs'] == 0 and source_match['relative_rms'] == 0
    assert len(counters['kernels']) == len(FORMATS)
    occupancy = metadata['sample']['metadata']['expert_occupancy']
    unique = sum(count > 0 for count in occupancy)
    assert unique == native[0]['active_experts']
    rows = []
    for kernel, (name, dtype, block_elements, block_bytes) in zip(counters['kernels'], FORMATS, strict=True):
        assert f'mul_mat_q<{dtype}, 64, 0>' in kernel['kernel']
        raw = kernel['metrics']
        def metric(key):
            return float(raw[key]['value'])
        duration_ns = metric('gpu__time_duration.sum')
        assert raw['gpu__time_duration.sum']['unit'] in ('ns', 'nsecond') and duration_ns > 0
        sysmem = SECTOR_BYTES * metric('lts__d_sectors_fill_sysmem.sum')
        device = SECTOR_BYTES * metric('lts__d_sectors_fill_device.sum')
        selected_bytes = unique * HIDDEN * INTERMEDIATE // block_elements * block_bytes
        refill = sysmem + device
        assert refill == kernel['derived']['l2_refill_total_byte_equivalent']
        rows.append({
            'projection': name, 'kernel_id': kernel['id'],
            'quantized_block_elements': block_elements, 'quantized_block_bytes': block_bytes,
            'selected_quantized_weight_payload_bytes': selected_bytes,
            'system_memory_l2_refill_byte_equivalent': sysmem,
            'device_memory_l2_refill_byte_equivalent': device,
            'l2_refill_byte_equivalent': refill,
            'refill_to_selected_weight_payload_ratio': refill / selected_bytes,
            'profiled_duration_ns': duration_ns,
            'l2_refill_byte_equivalent_per_profiled_second': refill * 1e9 / duration_ns,
            'profiler_replay_passes': metric('profiler__replayer_passes'),
            'overall_l2_sector_hit_rate_percent': metric('lts__t_sector_hit_rate.pct'),
        })
    selected = sum(row['selected_quantized_weight_payload_bytes'] for row in rows)
    refill = sum(row['l2_refill_byte_equivalent'] for row in rows)
    duration_ns = sum(row['profiled_duration_ns'] for row in rows)
    summary = {
        'complete': True, 'scope': 'Three warmed isolated-layer projections from one actual B8xQ7 target capture; NCU counter replay.',
        'sample': metadata['plan']['sample_selection'],
        'unique_selected_experts': unique, 'expert_route_assignments': sum(occupancy),
        'frozen_binary': metadata['binary'],
        'native_captured_output_comparison': source_match,
        'weight_payload_formula': 'unique experts * hidden 2560 * intermediate 640 / block elements * block bytes; Q4K=144/256 and Q4_1=20/32 including scales/minima',
        'rows': rows,
        'totals': {
            'selected_quantized_weight_payload_bytes': selected,
            'l2_refill_byte_equivalent': refill,
            'refill_to_selected_weight_payload_ratio': refill / selected,
            'refill_excess_over_selected_weight_payload_percent': (refill / selected - 1) * 100,
            'sum_profiled_projection_duration_ns': duration_ns,
            'l2_refill_byte_equivalent_per_summed_profiled_second': refill * 1e9 / duration_ns,
        },
        'limits': counters['limits'] + [
            'The selected-weight payload is logical storage including quantization metadata. It is not a physical-DRAM measurement.',
            'Read-hit, read-miss, and total counters can differ slightly across replay passes; they are not asserted to be an exact same-pass partition.',
            'Refill totals include activation/metadata reads and any repeated fills. Near-weight-payload totals do not identify exact per-buffer traffic.',
        ],
        'source_sha256': {name: sha(args.run / name) for name in
                          ('metadata.json', 'kernels.json', 'raw.log', 'profiled_test_output_not_benchmark.json')},
        'summarizer_sha256': sha(Path(__file__)),
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(f'{refill / 1e6:.3f} MB L2 refill equivalents / {selected / 1e6:.3f} MB selected weights = {refill / selected:.6f}')


if __name__ == '__main__':
    main()
