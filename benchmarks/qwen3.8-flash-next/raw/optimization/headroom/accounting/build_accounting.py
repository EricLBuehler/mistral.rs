#!/usr/bin/env python3
"""Recompute route/tile arithmetic using saved CPU-readable capture IDs only."""
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct

REPO = Path('/home/ericbuehler/mistral.rs')
WORK = Path('/home/ericbuehler/qwen4exp_work')
CAPTURES = WORK / 'real_routing_20260930/runtime/captures'
REPLAY = WORK / 'moe_optimization_20260930/grouped_gemv/all25/results.json'
PROFILE = WORK / 'batched_draft_20260929/profile_c8.analysis.json'
NCU = WORK / 'moe_optimization_20260930/masked_y/ncu_comparison.json'
OUT = Path(__file__).resolve().parent
HIDDEN = 2560
INTERMEDIATE = 640
EXPERTS = 512
TOPK = 10
TILE_Y = 128
ITER_K = 256
WEIGHT_BYTES = {'gate': HIDDEN * INTERMEDIATE // 256 * 144,
                'up': HIDDEN * INTERMEDIATE // 256 * 144,
                'down': HIDDEN * INTERMEDIATE // 32 * 20}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}

def read_ids(path):
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
        tensor = header['ids']
        assert tensor['dtype'] == 'U32' and tensor['shape'][1] == TOPK
        start, end = tensor['data_offsets']
        stream.seek(8 + size + start)
        data = stream.read(end - start)
    return struct.unpack('<' + 'I' * (len(data) // 4), data)

cases = []
inputs = {REPLAY, PROFILE, NCU}
for run in json.loads(REPLAY.read_text())['results']:
    rows = run['rows']
    if rows not in (24, 42, 56):
        continue
    meta_path = CAPTURES / run['source_metadata']
    tensor_path = CAPTURES / run['capture']['tensor_file']
    inputs.update((meta_path, tensor_path))
    ids = read_ids(tensor_path)
    selected = [ids[row * TOPK + j] for row in run['source_row_indices'] for j in range(TOPK)]
    counts = collections.Counter(selected)
    unique = len(counts)
    assert unique == run['active_experts']
    assert max(counts.values()) == run['max_expert_rows']
    tile_x = {24: 24, 42: 48, 56: 64}[rows]
    active_tiles = sum(math.ceil(value / tile_x) for value in counts.values())
    assert active_tiles == unique
    projection = {}
    for name, n, k in [('gate', INTERMEDIATE, HIDDEN), ('up', INTERMEDIATE, HIDDEN),
                       ('down', HIDDEN, INTERMEDIATE)]:
        executed_k = math.ceil(k / ITER_K) * ITER_K
        useful_mac = len(selected) * n * k
        executed_mac = active_tiles * tile_x * n * executed_k
        projection[name] = {
            'output_features': n, 'input_features': k, 'executed_k': executed_k,
            'tile_x': tile_x, 'tile_y': TILE_Y,
            'launched_ctas': EXPERTS * math.ceil(rows / tile_x) * (n // TILE_Y),
            'nonempty_ctas': active_tiles * (n // TILE_Y),
            'useful_integer_mac': useful_mac, 'executed_integer_mac': executed_mac,
            'executed_over_useful': executed_mac / useful_mac,
            'selected_distinct_compressed_weight_bytes': unique * WEIGHT_BYTES[name],
            'ideal_activation_shared_load_bytes_current_tiles': len(selected) * (executed_k // 128) * 144 * (n // TILE_Y),
        }
    useful = sum(value['useful_integer_mac'] for value in projection.values())
    executed = sum(value['executed_integer_mac'] for value in projection.values())
    tiles8 = sum(math.ceil(value / 8) for value in counts.values())
    cases.append({
        'source_metadata': str(meta_path), 'tensor_file': str(tensor_path),
        'selection': run['selection'], 'derived': run['derived'],
        'source_row_indices': run['source_row_indices'], 'rows': rows,
        'assignments': len(selected), 'active_experts': unique,
        'max_expert_rows': max(counts.values()),
        'expert_occupancy': [counts.get(expert, 0) for expert in range(EXPERTS)],
        'tile_x': tile_x, 'computed_columns': active_tiles * tile_x,
        'column_mac_amplification': active_tiles * tile_x / len(selected),
        'projection': projection,
        'total_useful_integer_mac': useful, 'total_executed_integer_mac': executed,
        'total_integer_mac_amplification': executed / useful,
        'total_selected_distinct_compressed_weight_bytes': unique * sum(WEIGHT_BYTES.values()),
        'warp_private_8_column_design_arithmetic_only': {
            'query_tiles': tiles8,
            'column_mac_amplification': tiles8 * 8 / len(selected),
            'weight_replication_if_each_8_column_tile_rereads_weights': tiles8 / unique,
            'experts_with_at_most_8_rows': sum(value <= 8 for value in counts.values()),
            'assignments_in_those_experts': sum(value for value in counts.values() if value <= 8),
            'activation_shared_load_multiplier_if_16_output_features_replace_128_without_reuse': 8,
        },
    })
summary = []
for rows in (24, 42, 56):
    selected = [case for case in cases if case['rows'] == rows]
    item = {'rows': rows, 'samples': len(selected)}
    for key in ('active_experts', 'max_expert_rows', 'column_mac_amplification',
                'total_integer_mac_amplification', 'total_useful_integer_mac',
                'total_executed_integer_mac', 'total_selected_distinct_compressed_weight_bytes'):
        values = [case[key] for case in selected]
        item[key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    summary.append(item)
profile = json.loads(PROFILE.read_text())
window = profile['windows']['recorded_kernel_span']
categories = window['categories']
regions = profile['classification']['completed_dispatch_reduce_regions']
router_ns = sum(kernel['duration_ns']['sum'] for kernel in window['top_kernels']
                if kernel['name'].startswith('void moe_router_topk_kernel'))
projections_ns = categories['moe_grouped_mmq_context']['sum']
quant_ns = categories['grouped_moe_quantization_context']['sum']
dispatch_reduce_ns = categories['moe_dispatch_routing_reduce']['sum'] - router_ns
historical = {
    'scope': 'Historical pre-mask recovered C8 Nsight Systems kernel timeline; not final-serving time or a current isolated benchmark.',
    'warnings': [text for text in profile['diagnostics'] if 'Not all CUDA events' in text],
    'grouped_regions': regions,
    'grouped_projection_share_of_summed_kernel_time_pct': categories['moe_grouped_mmq_context']['share_of_summed_kernel_time_pct'],
    'expert_gemv_share_of_summed_kernel_time_pct': categories['moe_gemv']['share_of_summed_kernel_time_pct'],
    'grouped_projection_mean_us_per_region': projections_ns / regions / 1000,
    'both_quantizers_mean_us_per_region': quant_ns / regions / 1000,
    'dispatch_count_prefix_scatter_and_weighted_reduce_mean_us_per_region': dispatch_reduce_ns / regions / 1000,
    'those_overheads_relative_to_projection_time': (quant_ns + dispatch_reduce_ns) / projections_ns,
    'exclusions': 'Routing-logit dense projection, allocation/API latency and inter-kernel gaps are outside these kernel-duration totals. Top-k router is excluded from dispatch/reduce mean because it also covers non-grouped FFNs.',
}
selected_case = next(case for case in cases if case['rows'] == 56 and case['source_metadata'].endswith('c8_target_verify_b8_q7_04.json'))
isolated = []
for entry in json.loads(NCU.read_text())['results']:
    name = entry['projection']
    metric = entry['candidate_metrics']
    read_bytes = int(metric['lts__t_sectors_op_read.sum']['value']) * 32
    weight_bytes = selected_case['projection'][name]['selected_distinct_compressed_weight_bytes']
    isolated.append({'projection': name, 'masked_l2_read_request_bytes': read_bytes,
                     'selected_distinct_compressed_weight_bytes': weight_bytes,
                     'request_bytes_over_selected_weight_bytes': read_bytes / weight_bytes,
                     'registers_per_thread': metric['launch__registers_per_thread'],
                     'shared_bytes_per_block': metric['launch__shared_mem_per_block'],
                     'l2_hit_rate': metric['lts__t_sector_hit_rate.pct'],
                     'tensor_pipe_active_pct': metric['sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'],
                     'long_scoreboard_stall_ratio': metric['smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio']})
read_total = sum(entry['masked_l2_read_request_bytes'] for entry in isolated)
weight_total = selected_case['total_selected_distinct_compressed_weight_bytes']
source_files = [
    'mistralrs-quant/kernels/mmq_gguf/mmq_instance_q4_k.cu',
    'mistralrs-quant/kernels/mmq_gguf/mmq_instance_q4_1.cu',
    'mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh',
    'mistralrs-quant/kernels/mmq_gguf/mmq_mma.cuh',
    'mistralrs-quant/kernels/mmq_gguf/mmq_quantize.cu',
    'mistralrs-quant/src/gguf/fast_mmq.rs',
    'mistralrs-core/src/moe/experts/backends.rs',
]
inputs.update(REPO / name for name in source_files)
result = {
    'scope': 'Read-only analytical accounting of exact captured layer8 routes. No new GPU measurement.',
    'definitions': {
        'mac': 'One integer multiply-accumulate lane. Counts exclude scale/min F32 corrections, load/decode, routing and output reduction.',
        'executed_mac': 'Nonempty expert tiles only. X is24/48/64, Y128, K iterations256; down K640 executes768 lanes with zero-padded activation tail.',
        'derived': 'M24 selects first3 query positions from each of8 original sequences; no claim it reproduces natural adaptive-depth routing.',
        'bytes': 'Compressed selected-weight payload is a logical unique-byte count. L2 request sectors are not actual DRAM bytes or bandwidth.',
        'weight_tiling': 'All these counts are <=X, so no selected expert weight tile repeats across query tiles. N640 and2560 are exact multiples ofY128.',
    },
    'geometry': {'experts': EXPERTS, 'topk': TOPK, 'hidden': HIDDEN, 'intermediate': INTERMEDIATE,
                 'compressed_weight_bytes_per_expert': WEIGHT_BYTES},
    'summaries': summary, 'cases': cases, 'historical_profile': historical,
    'isolated_masked_ncu': {
        'scope': 'Same median-U nativeM56 capture; masked-Y isolated counter runs. Profiling durations are deliberately omitted as benchmark claims.',
        'projections': isolated, 'l2_read_request_bytes_total': read_total,
        'selected_distinct_compressed_weight_bytes_total': weight_total,
        'request_bytes_over_selected_weight_bytes': read_total / weight_total,
        'dram_metrics_available': False,
    },
    'sources': [source(path) for path in sorted(inputs)],
}
(OUT / 'accounting.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'cases':len(cases), 'source_files':len(inputs), 'output':str(OUT/'accounting.json')}, indent=2))
