#!/usr/bin/env python3
"""Reconcile observed MoE launch geometry with BF16 graph and token counters."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/ericbuehler/mistral.rs')


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    analysis = ROOT / 'analysis'
    for name, expected in json.loads((analysis / 'manifest.json').read_text()).items():
        assert sha(analysis / name) == expected, name
    proof = json.loads((analysis / 'kernel_proof.json').read_text())
    assert proof['complete']
    capture = ROOT / 'capture'
    metadata = json.loads((capture / 'metadata.json').read_text())
    assert metadata['complete']
    log = (capture / 'server.log').read_text()
    assert 'Captured 3 CUDA decode graphs' in log
    assert 'Deferred 5 CUDA decode graph shapes' in log
    assert '<=32 bk=128,bm=16,bn=64,' in log
    assert 'split_k=1' in log
    config = json.loads(Path('/home/ericbuehler/hf_models/qwen3.5_35b_a3b/config.json').read_text())['text_config']
    assert (config['num_hidden_layers'], config['hidden_size'], config['moe_intermediate_size'], config['num_experts_per_tok']) == (40, 2048, 512, 8)
    phase = next(p for p in proof['phases'] if p['name'] == 'profile_c8')
    w = phase['exact_window']
    with sqlite3.connect(f'file:{analysis}/profile_c8.sqlite?mode=ro', uri=True) as connection:
        rows = connection.execute('''SELECT k.start,k.graphId,k.gridX FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.demangledName WHERE k.start<? AND k.end>?
        AND INSTR(s.value,'fused_moe_kernel')>0 ORDER BY k.start''', (w['end_ns'], w['start_ns'])).fetchall()
    assert len(rows) == 34 * 80
    steps = []
    for offset in range(0, len(rows), 80):
        block = rows[offset:offset+80]
        groups = collections.Counter((row[1] or 0, row[2]) for row in block)
        assert len(groups) == 2 and set(groups.values()) == {40}, groups
        (graph, low), (graph2, high) = sorted(groups)
        assert graph == graph2 and high == 2 * low
        decode_batch = low // 128 if low <= 8 * 128 and low % 128 == 0 else None
        steps.append({'step_index': offset // 80, 'first_moe_start_relative_to_envelope_ns': block[0][0]-w['start_ns'],
                      'graph_id': graph or None, 'gate_up_grid_x': low, 'down_grid_x': high,
                      'launches_per_projection': 40, 'decode_batch_inferred_from_grid': decode_batch})
    assert [s['decode_batch_inferred_from_grid'] for s in steps] == [None, 1, None] + [8] * 30 + [7]
    assert [s['graph_id'] for s in steps] == [None, 2, None] + [11] * 30 + [None]
    def delta(name, counter, labels=None):
        doc = json.loads((capture / f'{name}.metrics.delta.json').read_text())
        return sum(r['delta'] for r in doc['all_counter_deltas'] if r['name'] == counter and (labels is None or r['labels'] == labels))
    decode = delta('profile_c8', 'mistralrs_decode_tokens_processed_total')
    prefill = delta('profile_c8', 'mistralrs_prefill_tokens_processed_total')
    captures = delta('c8.warmup0', 'mistralrs_cuda_graph_events_total', {'component':'target','event':'capture','outcome':'success'})
    assert decode == 1 + 30 * 8 + 7 == 248
    assert prefill == 129 and captures == 1
    assert delta('profile_c8', 'mistralrs_cuda_graph_events_total', {'component':'target','event':'capture','outcome':'success'}) == 0
    source_paths = ['mistralrs-core/src/pipeline/multimodal.rs', 'mistralrs-core/src/pipeline/cuda_graph.rs',
                    'mistralrs-quant/src/cutile/fused_moe.rs', 'mistralrs-quant/src/moe/cuda.rs']
    inputs = ['analysis/kernel_proof.json','analysis/manifest.json','capture/metadata.json','capture/server.log',
              'capture/c8.warmup0.metrics.delta.json','capture/profile_c8.metrics.delta.json',
              'capture/profile_c8.requests.json']
    result = {'complete': True, 'scope': 'Post-warmup diagnostic trace only; positive geometry and counter evidence, not a serving benchmark.',
              'startup': {'captured_batches_from_ascending_exact_policy': [1,2,3], 'deferred_batches': [4,5,6,7,8],
                          'c8_warmup_successful_lazy_captures': captures,
                          'configured_max_log_is_not_largest_captured_batch': True},
              'decode_geometry': {'recorded_bm':16,'recorded_bn':64,'recorded_split_k':1,'experts':256,'top_k':8,
                'gate_up_n':1024,'down_n':2048,
                'formula_for_batches_1_through_8':'EM=M*8*16; gridX=ceil(EM/16)*ceil(N/64), giving gate/up128*M and down256*M.',
                'source_refs':['mistralrs-quant/src/moe/cuda.rs:146','mistralrs-quant/src/cutile/fused_moe.rs:534',
                               'mistralrs-core/src/pipeline/multimodal.rs:2309','mistralrs-core/src/pipeline/cuda_graph.rs:452']},
              'c8': {'batch8_graph_id':11,'batch8_graph_replays':30,'batch8_moe_launches':2400,
                     'batch1_graph_id':2,'batch1_graph_replays':1,'batch1_moe_launches':80,
                     'batch7_eager_steps':1,'batch7_eager_moe_launches':80,
                     'prefill_steps':2,'prefill_moe_launches':160,'prefill_tokens':prefill,'decode_tokens':decode,
                     'token_reconciliation':'1 + 30*8 + 7 = 248 decode tokens; 129 prompt tokens; 8*32=256 completed output tokens.',
                     'prefill_interpretation':'Launch grids3840/7680 and2120/4240 are consistent with separate30-token and99-token prompt batches; only total129 is directly confirmed by phase counters.',
                     'ordered_model_steps':steps},
              'limitations':['Startup captured3 shapes versus8 in the unprofiled throughput run; one B7 trace tail was eager under memory pressure.',
                             'Any nonzero graphId alone would not prove B8; this conclusion additionally uses exact launch geometry and phase counters.',
                             'The profiler warns that not all CUDA events might have been collected. This is observed positive evidence, not guaranteed exhaustive coverage.',
                             'Recorded phases include prefill and drain. Kernel durations and profiled request walls are not used as official throughput results.'],
              'input_sha256':{p:sha(ROOT/p) for p in inputs},'source_sha256':{p:sha(REPO/p) for p in source_paths},
              'sqlite_sha256':phase['sqlite_sha256'],'script_sha256':sha(Path(__file__))}
    target = analysis / 'graph_shape_proof.json'
    assert not target.exists(), target
    target.write_text(json.dumps(result,indent=2)+'\n')
    (analysis/'graph_shape_proof.manifest.json').write_text(json.dumps({'graph_shape_proof.json':sha(target),'../prove_graph_shapes.py':sha(Path(__file__))},indent=2)+'\n')
    print(json.dumps({k:v for k,v in result['c8'].items() if k != 'ordered_model_steps'},indent=2))


if __name__ == '__main__':
    main()
