from pathlib import Path
import hashlib,json,shutil
ROOT=Path('/home/ericbuehler/qwen4exp_work/moe_optimization_20260930')
R=Path('/home/ericbuehler/mistral.rs')
W=ROOT/'compact_schedule'; B=W/'build'; S=B/'shadow'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
S.mkdir(parents=True,exist_ok=False)
base=ROOT/'masked_y_bounded/build'
d=json.loads((base/'provenance.json').read_text())
for key in ['build_started_at','build_finished_at','build_script_sha256','variant_binary','variant_binary_sha256','shadow_archive_sha256','compiled_objects','originals_unchanged','archive_command']:
 d.pop(key,None)
d['complete']=False
d['parent_baseline_binary_sha256']=d.pop('baseline_binary_sha256')
d['masked_reference_binary']=str(base/'moe_dispatch_bench.variant')
d['masked_reference_binary_sha256']=sha(base/'moe_dispatch_bench.variant')
for original, expected in json.loads((W/'sources/sources.json').read_text())['original_sources'].items():
 assert sha(original)==expected, original
d['method']='External compact full-K Q4K/Q4_1 scheduler using finalized masked-Y kernel. Other quantization dispatch is unchanged.'
d['original_sources']={str(p):sha(p) for p in [R/'mistralrs-quant/kernels/mmq_gguf'/n for n in ['mmq_gguf.cuh','mmq_instance_q4_k.cu','mmq_instance_q4_1.cu']]}
d['native_archive_sha256']=sha(d['native_archive'])
d['production_cli_sha256']=sha(R/'target/release/mistralrs')
d['patched_sources']={}
for p in (W/'sources').iterdir():
 if p.suffix in ('.cu','.cuh'):
  dest=B/p.name;shutil.copy2(p,dest);d['patched_sources'][str(dest)]=sha(dest)
d['patch_sha256']=sha(W/'sources/compact_schedule.patch')
d.pop('grid_invariant',None)
for command in d['nvcc_commands']:
 for i,arg in enumerate(command):
  if arg.startswith(str(base)):command[i]=str(B)+arg[len(str(base)):]
d['rustc_command']=[str(S) if arg==str(base/'shadow') else arg for arg in d['rustc_command']]
src=base/'replay.source.rs'
shutil.copy2(src,B/'replay.source.rs')
d['rustc_command']=[str(B/'replay.source.rs') if arg=='mistralrs-quant/tests/moe_dispatch_bench.rs' else arg for arg in d['rustc_command']]
d['test_source_sha256']=sha(B/'replay.source.rs')
shutil.copy2(ROOT/'masked_y_bounded/build_shadow.py',W/'build_shadow.py')
(B/'provenance.json').write_text(json.dumps(d,indent=2)+'\n')
print(B)
