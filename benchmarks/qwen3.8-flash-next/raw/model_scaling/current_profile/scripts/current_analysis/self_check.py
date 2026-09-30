#!/usr/bin/env python3
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import analyze_current as analysis


def kernel(name, start, end, graph=False):
    return SimpleNamespace(name=name, start=start, end=end, graph=graph,
                           pid=1, context=1, stream=1, category='other', grid=(1, 1, 1), block=(32, 1, 1))


checks = []
k = kernel('probe', 10, 20)
assert analysis.clipped_ns(k, 0, 10) == 0
assert analysis.clipped_ns(k, 15, 19) == 4
assert analysis.clipped_ns(k, 0, 100) == 10
checks.append('Boundary clipping preserves only interval overlap')

kernels = []
semantic = {}
for hc, gdn in ((3, 0), (97, 36), (100, 0)):
    for _ in range(hc):
        t = len(kernels) * 10
        kernels.append(kernel('q4_hc_mix_kernel', t, t + 5))
    for _ in range(gdn):
        t = len(kernels) * 10
        kernels.append(kernel('gdn_speculative_recurrence_rmsnorm_gate', t, t + 5))
    t = len(kernels) * 10
    semantic[len(kernels)] = 'vocabulary_head_projection'
    kernels.append(kernel('mmvq_gguf_q4_k_bf16_plain_cuda8', t, t + 5))
segments = analysis.forward_segments(kernels, semantic)
assert [segment['kind'] for segment in segments] == ['draft', 'target', 'unknown']
assert all(segment['projected_vocabulary_rows'] == 8 for segment in segments)
checks.append('97/3 HC topology separates target/draft and retains mixed intervals as unknown')

window = {'start_ns': 0, 'end_ns': kernels[-1].end + 1}
summary = analysis.summarize_window(kernels, semantic, segments, window)
assert summary['summed_kernel_time_ns'] == len(kernels) * 5
assert sum(row['duration_ns'] for row in summary['categories']) == summary['summed_kernel_time_ns']
counts = {'sequence_proposals': 2, 'proposed_draft_tokens': 8, 'accepted_draft_tokens': 4}
normalized = analysis.normalize(summary, summary, 16, counts)
assert normalized['alignment_boundary_kernel_time_difference_ns'] == 0
assert normalized['known_draft_head_projected_rows'] == 8
assert normalized['projected_row_count_minus_proposed_counter'] == 0
assert normalized['draft_head_ms_per_recorded_proposed_draft_token'] == 5 / 8 / 1e6
checks.append('Per-phase denominators, exact head rows and zero alignment sensitivity are consistent')

try:
    analysis.normalize(summary, summary, 0, counts)
except ValueError:
    pass
else:
    raise AssertionError('Empty output token denominator accepted')
checks.append('Empty phase output is rejected')

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    run = root / 'run'
    run.mkdir()
    (run / 'metadata.json').write_text(json.dumps({'complete': False, 'cleanup': {}}))
    try:
        analysis.require_finished(run, root / 'missing_analysis')
    except ValueError as error:
        assert 'completed captures' in str(error)
    else:
        raise AssertionError('Incomplete run passed completion gate')
checks.append('Incomplete run refuses before reading any export')

payload = {'complete': True, 'scope': 'Synthetic CPU-only adapter checks; no capture export or real trace analysis.', 'checks': checks}
(Path(__file__).resolve().parent / 'self_check.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))
