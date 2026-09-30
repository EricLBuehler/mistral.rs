#!/usr/bin/env python3
"""Count cross-sequence expert reuse within each exact saved verification batch."""
import hashlib
import json
from pathlib import Path
import statistics
import struct

CAPTURES = Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/runtime/captures')
OUT = Path(__file__).resolve().parent
QUERY_POSITIONS = 7
TOPK = 10
WEIGHT_BYTES = {'gate_q4k': 921600, 'up_q4k': 921600, 'down_q4_1': 1024000}
TOTAL_WEIGHT_BYTES = sum(WEIGHT_BYTES.values())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        tensor = header['ids']
        assert tensor['dtype'] == 'U32' and tensor['shape'][1] == TOPK
        start, end = tensor['data_offsets']
        stream.seek(8 + size + start)
        data = stream.read(end - start)
    return tensor['shape'], struct.unpack('<' + 'I' * (len(data) // 4), data)


cases = []
for batch in (6, 8):
    paths = sorted(CAPTURES.glob(f'c{batch}_target_verify_b{batch}_q7_*.json'))
    assert len(paths) == 5
    for path in paths:
        metadata = json.loads(path.read_text())
        assert metadata['batch'] == batch and metadata['query_len'] == QUERY_POSITIONS
        assert metadata['rows'] == batch * QUERY_POSITIONS and metadata['topk'] == TOPK
        assert not metadata['derived']
        tensor_path = CAPTURES / metadata['tensor_file']
        shape, ids = read_ids(tensor_path)
        assert shape == [batch * QUERY_POSITIONS, TOPK]
        individual = []
        sets = []
        for sequence in range(batch):
            row_start = sequence * QUERY_POSITIONS
            row_end = row_start + QUERY_POSITIONS
            experts = set(ids[row_start * TOPK:row_end * TOPK])
            sets.append(experts)
            individual.append({
                'sequence_index': sequence,
                'source_row_indices': list(range(row_start, row_end)),
                'expert_ids': sorted(experts),
                'unique_experts': len(experts),
                'selected_payload_bytes': len(experts) * TOTAL_WEIGHT_BYTES,
            })
        union = set().union(*sets)
        assert union == set(ids)
        assert len(union) == sum(count > 0 for count in metadata['expert_occupancy'])
        separate_count = sum(len(experts) for experts in sets)
        cases.append({
            'capture': path.name, 'batch': batch, 'query_positions_per_sequence': QUERY_POSITIONS,
            'per_sequence': individual,
            'sum_individual_unique_experts': separate_count,
            'batch_union_expert_ids': sorted(union), 'batch_unique_experts': len(union),
            'sum_individual_selected_payload_bytes': separate_count * TOTAL_WEIGHT_BYTES,
            'batch_selected_payload_bytes': len(union) * TOTAL_WEIGHT_BYTES,
            'logical_payload_saved_by_cross_sequence_reuse_bytes': (separate_count - len(union)) * TOTAL_WEIGHT_BYTES,
            'cross_sequence_reuse_factor': separate_count / len(union),
            'batch_payload_fraction_of_separate_sequence_sum': len(union) / separate_count,
            'logical_payload_reduction_fraction': 1 - len(union) / separate_count,
            'batch_payload_over_mean_individual_sequence_payload': batch * len(union) / separate_count,
            'provenance': [
                {'path': str(source), 'bytes': source.stat().st_size, 'sha256': sha(source)}
                for source in (path, tensor_path)
            ],
        })
summaries = []
for batch in (6, 8):
    selected = [case for case in cases if case['batch'] == batch]
    summary = {'batch': batch, 'captures': len(selected), 'query_positions_per_sequence': QUERY_POSITIONS}
    for key in ('cross_sequence_reuse_factor', 'logical_payload_reduction_fraction',
                'batch_payload_over_mean_individual_sequence_payload'):
        values = [case[key] for case in selected]
        summary[key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    summaries.append(summary)
result = {
    'scope': 'Matched logical selected-weight reuse in native layer8 B6xQ7/B8xQ7 captures; no GPU measurement.',
    'layout': 'Rows are sequence-major: sequence s owns rows [s*7, (s+1)*7), with 10 expert IDs per row. This is the same row selection used by the validated balanced-query replay.',
    'definitions': {
        'cross_sequence_reuse_factor': 'Sum of distinct experts within each sequence divided by the distinct expert union across the whole batch.',
        'separate_sequence_payload': 'Each sequence can already reuse an expert across its own seven query positions. The sum counts an expert once for every separate sequence selecting it.',
        'batch_payload': 'An expert is counted once if any sequence in the batch selects it.',
        'payload': 'Compressed gate/up/down weights for selected experts only. Not measured traffic, a throughput prediction, a hardware ceiling, or a claim about another layer or natural adaptive-depth workload.',
    },
    'compressed_weight_bytes_per_expert': WEIGHT_BYTES,
    'compressed_weight_bytes_per_expert_total': TOTAL_WEIGHT_BYTES,
    'summaries': summaries, 'cases': cases,
}
(OUT / 'paired_sequence_reuse.json').write_text(json.dumps(result, indent=2) + '\n')
lines = [
    '# Matched cross-sequence selected-weight reuse', '',
    'Each native capture is split into its own constituent sequences, retaining all seven query positions and ten routes per position. Each separate sequence already reuses weights across its own seven positions. We compare the sum of those sequence-specific expert unions with the union for that exact batch. This avoids comparing unrelated C1 and C8 requests.', '',
    '| Capture | Unique experts per sequence | Sum | Batch union | Reuse factor | Logical payload reduction |',
    '| --- | --- | ---: | ---: | ---: | ---: |',
]
for case in cases:
    counts = ', '.join(str(value['unique_experts']) for value in case['per_sequence'])
    lines.append(f"| {case['capture']} | {counts} | {case['sum_individual_unique_experts']} | {case['batch_unique_experts']} | {case['cross_sequence_reuse_factor']:.3f}x | {case['logical_payload_reduction_fraction']:.2%} |")
lines += ['',
    'Across five native captures per batch size:', '',
    '| Batch | Median reuse | Range | Median logical payload reduction | Batch payload / mean sequence payload |',
    '| --- | ---: | ---: | ---: | ---: |',
]
for summary in summaries:
    factor = summary['cross_sequence_reuse_factor']
    saving = summary['logical_payload_reduction_fraction']['median']
    growth = summary['batch_payload_over_mean_individual_sequence_payload']['median']
    lines.append(f"| B{summary['batch']}xQ7 | {factor['median']:.3f}x | {factor['min']:.3f}x to {factor['max']:.3f}x | {saving:.2%} | {growth:.3f}x |")
lines += ['',
    'For the selected native B8 sample04, the eight sequences individually select313 expert payloads in total, while their union selects213. The logical selected-weight payload is897,433,600 bytes separately versus610,713,600 bytes batched:286,720,000 bytes of cross-sequence reuse, or31.95%. The union is5.444 times the mean individual sequence payload. This illustrates why eight sequences do not automatically share nearly all expert weights.', '',
    'The calculation is a property of these exact saved routes and compressed weight geometry. It is not actual memory traffic, a serving speedup prediction, a physical throughput ceiling, or a model-quality result. It does not account for cache persistence across steps, different routing in other layers, or the final adaptive-depth workload.', '',
    'Reproduce with `python3 build_paired_sequence_reuse.py`. The JSON includes sequence row indices, exact expert sets, per-case byte counts, source hashes and summary statistics. No model weights are loaded and no CUDA calls are made.',
]
text = '\n'.join(lines) + '\n'
for old, new in [('select313', 'select 313'), ('selects213', 'selects 213'),
                 ('is897,433,600', 'is 897,433,600'), ('versus610,713,600', 'versus 610,713,600'),
                 (':286,720,000', ': 286,720,000'), ('or31.95%', 'or 31.95%'), ('is5.444', 'is 5.444')]:
    text = text.replace(old, new)
(OUT / 'paired_sequence_reuse.md').write_text(text)
print(json.dumps(summaries, indent=2))
