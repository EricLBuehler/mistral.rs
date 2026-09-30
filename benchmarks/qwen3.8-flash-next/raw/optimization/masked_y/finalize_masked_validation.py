#!/usr/bin/env python3
from pathlib import Path
import datetime, hashlib, json, subprocess
R=Path('/home/ericbuehler/mistral.rs')
W=Path(__file__).resolve().parent/'masked_y_bounded'
def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def record(path):
    path=Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
static=json.loads((W/'static_validation.json').read_text())
assert len(static)==5 and all(r.get('returncode')==0 for r in static)
gpu=json.loads((W/'validation_final/metadata.json').read_text())
assert gpu['complete'] and all(r['returncode']==0 for r in gpu['commands'])
baseline=json.loads((W/'oracle_baseline_corrected/provenance.json').read_text())
assert baseline['test_status']==0
sources={name:record(R/name) for name in ['mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh','mistralrs-quant/tests/grouped_mmq_tail_cuda_tests.rs','mistralrs-quant/tests/grouped_mmq_packed_cuda_tests.rs']}
assert sources['mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh']['sha256']==gpu['header_sha256']
assert (W/'oracle_baseline_corrected/grouped_mmq_tail_cuda_tests.rs').read_text().replace('(row + block) % 2 == 0', '(row + block).is_multiple_of(2)')==(R/'mistralrs-quant/tests/grouped_mmq_tail_cuda_tests.rs').read_text()
commands=[]
for log in ['production_build.log','production_build_corrected.log']:
    parsed=[]
    for line in (W/log).read_text().splitlines():
        try: parsed.append(json.loads(line))
        except json.JSONDecodeError: pass
    assert any(r.get('reason')=='build-finished' and r.get('success') for r in parsed)
    commands.append(dict(command=['cargo','test','--release','-p','mistralrs-quant','--features','cuda,cutile','--test','grouped_mmq_tail_cuda_tests','--test','grouped_mmq_packed_cuda_tests','--no-run','--message-format=json-render-diagnostics'],returncode=0,log=record(W/log)))
for r in gpu['commands']:
    r=dict(r);r['log']=record(W/'validation_final'/(r['name']+'.log'));commands.append(r)
commands.extend(static)
cleanup_command=['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']
cleanup=subprocess.check_output(cleanup_command,text=True)
assert not cleanup.strip(),cleanup
(W/'gpu_cleanup.txt').write_text(cleanup)
refs=['validation/metadata.json','validation/grouped_mmq_tail_cuda_tests.test.log','oracle_baseline/provenance.json','oracle_baseline/test.log','oracle_baseline/grouped_mmq_tail_cuda_tests.rs','oracle_baseline_corrected/provenance.json','oracle_baseline_corrected/test.log','oracle_baseline_corrected/grouped_mmq_tail_cuda_tests.rs','validation_corrected/metadata.json','validation_final/metadata.json','initial_static_validation.json','initial_cargo_clippy.log','static_validation.json','generic_resources/comparison.json','code_equivalence.json','masked_y.patch','replay/lifecycle.json']
refs=[p for p in refs if (W/p).is_file()]
result=dict(complete=True,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,binaries=gpu['binaries'],commands=commands,corrected_unmasked_baseline=baseline,fixture_diagnosis='Original test failed identically in masked and unmasked Q4K/BF16 kernels (0.61914825 versus CPU 0.64160025). CUDA half2 scale/min products round before accumulation, unlike CPU f32 dequantization. The revised test retains quantized bits and integer scales, assigns exact power-of-two block scales to both CPU and GPU tensors, and keeps the original 5e-3 numerical bound. Both implementations pass the revised independent CPU oracle.',evidence={p:record(W/p) for p in refs},gpu_cleanup=dict(command=cleanup_command,output=record(W/'gpu_cleanup.txt'),no_compute_processes=True),baseline_fixture_note='The final source differs from the corrected unmasked baseline test only by the equivalent is_multiple_of syntax; final production binaries were rebuilt and rerun under tests and memcheck.',scope='Production bounded mask, four GGUF quantization formats and BF16/F16/F32 sparse tail/stream-K CPU oracles, packed route ordering and CUDA graph dynamic-routing replay, focused memcheck. Serving validation remains a separate parent-owned phase.')
(W/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(W/'validation.json')
