"""Build and validate the final small-group MMQ candidate with frozen artifacts."""
import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path('/home/ericbuehler/mistral.rs')
OUT=Path(__file__).resolve().parent/'build'
OUT.mkdir(exist_ok=False)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
meta=dict(complete=False,started_at=now(),commands=[],binaries={},sources={})
def save():(OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
def run(name,command):
 record=dict(name=name,command=command,started_at=now());meta['commands'].append(record);save()
 print('START',name,flush=True)
 with (OUT/(name+'.log')).open('wb') as log:
  result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 record.update(returncode=result.returncode,finished_at=now(),log_sha256=sha(OUT/(name+'.log')));save()
 if result.returncode:raise RuntimeError(name+' failed; see '+str(OUT/(name+'.log')))
 print('PASS',name,flush=True)
try:
 run('format',['cargo','fmt','--all'])
 files=subprocess.check_output(['git','ls-files','-co','--exclude-standard','-z','--','mistralrs-core','mistralrs-quant','mistralrs-cli','Cargo.toml','Cargo.lock','.cargo'],cwd=ROOT).decode().split('\0')
 for f in sorted(set(files)-{''}):
  p=ROOT/f
  if p.is_file():meta['sources'][f]=sha(p)
 meta['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 (OUT/'source.diff').write_bytes(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT))
 for source in ['mistralrs-core/src/moe/experts/backends.rs','mistralrs-quant/src/utils/log.rs','mistralrs-quant/src/cutile/mod.rs','mistralrs-quant/tests/support/gguf_moe.rs','mistralrs-quant/tests/support/cutile_gguf_replay.rs','mistralrs-quant/tests/cutile_gguf_moe_tests.rs','mistralrs-quant/tests/grouped_mmq_packed_cuda_tests.rs','mistralrs-quant/tests/moe_dispatch_bench.rs','mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh']:
  dst=OUT/'sources'/source;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dst)
 shutil.copy2(__file__,OUT/'build_candidate.py');save()
 run('gpu_tests_build',['cargo','test','--release','-p','mistralrs-quant','--features','cuda,cutile','--test','cutile_gguf_moe_tests','--test','grouped_mmq_packed_cuda_tests','--test','moe_dispatch_bench','--no-run','--message-format=json'])
 for line in (OUT/'gpu_tests_build.log').read_text().splitlines():
  if not line.startswith('{'):continue
  record=json.loads(line)
  if record.get('reason')=='compiler-artifact' and record.get('executable'):
   source=Path(record['executable']);dest=OUT/source.name
   os.link(source,dest);meta['binaries'][record['target']['name']]=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size)
 save()
 run('projection_tests',[meta['binaries']['cutile_gguf_moe_tests']['path'],'--nocapture','--test-threads=1'])
 run('packed_tests',[meta['binaries']['grouped_mmq_packed_cuda_tests']['path'],'--nocapture','--test-threads=1'])
 run('projection_memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','99',meta['binaries']['cutile_gguf_moe_tests']['path'],'--nocapture','--test-threads=1'])
 run('packed_memcheck',['compute-sanitizer','--tool','memcheck','--error-exitcode','99',meta['binaries']['grouped_mmq_packed_cuda_tests']['path'],'--nocapture','--test-threads=1'])
 for name in ['projection_memcheck','packed_memcheck']:assert 'ERROR SUMMARY: 0 errors' in (OUT/(name+'.log')).read_text()
 run('logging_regression',['cargo','test','--release','-p','mistralrs-quant','--features','cuda,cutile','--lib','repeated_messages_are_cached_once','--','--nocapture','--test-threads=1'])
 run('cli_release',['cargo','build','--release','-p','mistralrs-cli','--features','cuda,flash-attn,cutile'])
 source=ROOT/'target/release/mistralrs';dest=OUT/'mistralrs';os.link(source,dest)
 meta['binaries']['mistralrs']=dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size);save()
 run('cargo_check',['cargo','check','--release','-p','mistralrs-cli','--features','cuda,flash-attn,cutile'])
 run('quant_clippy',['cargo','clippy','--release','-p','mistralrs-quant','--features','cuda,cutile','--lib','--test','cutile_gguf_moe_tests','--test','grouped_mmq_packed_cuda_tests','--test','moe_dispatch_bench','--','-D','warnings'])
 run('core_clippy',['cargo','clippy','--release','-p','mistralrs-core','--features','cuda,flash-attn,cutile','--lib','--','-D','warnings'])
 run('format_check',['cargo','fmt','--all','--','--check'])
 for name,source_hash in meta['sources'].items():assert sha(ROOT/name)==source_hash,name
 for binary in meta['binaries'].values():assert sha(binary['path'])==binary['sha256']
 assert sha(OUT.parent/'baseline.mistralrs')=='d245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588'
 meta.update(complete=True,finished_at=now());save();print('COMPLETE',OUT,flush=True)
except BaseException as error:
 meta.update(error=repr(error),finished_at=now());save();raise
