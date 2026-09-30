"""Run a cuTile GGUF full-FFN replay with provenance and GPU monitor."""
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,sys
from pathlib import Path
REPO=Path('/home/ericbuehler/mistral.rs')
CAPTURE_ROOT=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930')
sys.path.insert(0,str(CAPTURE_ROOT))
from build_diagnostic import isolate
from run_diagnostic import stop_child
CAPTURES=CAPTURE_ROOT/'runtime/captures'
TEST='cutile_gguf_replay::flash_next_cutile_gguf_real_routing'
SOURCES=['mistralrs-quant/src/cutile/gguf_moe.rs','mistralrs-quant/src/cutile/mod.rs','mistralrs-quant/tests/moe_dispatch_bench.rs','mistralrs-quant/tests/support/cutile_gguf_replay.rs','mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh','mistralrs-quant/src/gguf/fast_mmq.rs','mistralrs-quant/src/gguf/cuda.rs']
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--tile',default='16,64,128')
p.add_argument('--sample')
p.add_argument('--source-root',type=Path,default=REPO)
p.add_argument('--released',action='store_true',required=True)
a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
metadata=dict(complete=False,started_at=now(),binary=str(a.binary),binary_sha256=sha(a.binary),tile=a.tile,sample=a.sample,commands=[],sources={},source_root=str(a.source_root))
meta=a.output/'metadata.json'
def save():meta.write_text(json.dumps(metadata,indent=2)+'\n')
def interrupted(signum,frame):raise InterruptedError(signum)
for signum in [signal.SIGTERM,signal.SIGINT]:signal.signal(signum,interrupted)
monitor=None;child=None
try:
 metadata['isolation_before']=isolate()
 for source in SOURCES:
  dest=a.output/'sources'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a.source_root/source,dest);metadata['sources'][source]=sha(dest)
 for helper in ['build_diagnostic.py','run_diagnostic.py']:
  dest=a.output/'helpers'/helper;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CAPTURE_ROOT/helper,dest)
  metadata['sources']['external_helpers/'+helper]=sha(dest)
 capture_manifest=CAPTURE_ROOT/'runtime/capture.validation.json'
 shutil.copy2(capture_manifest,a.output/'capture.validation.json')
 metadata['capture_validation_sha256']=sha(capture_manifest)
 for record in json.loads(capture_manifest.read_text())['files']:
  captured=CAPTURES/record['path'];assert captured.stat().st_size==record['bytes'];assert sha(captured)==record['sha256']
 metadata['all_capture_files_verified']=True
 metadata['captured_metadata_sha256']={p.name:sha(p) for p in sorted(CAPTURES.glob('*.json'))}
 shutil.copy2(__file__,a.output/'run_replay.py')
 metadata['runner_sha256']=sha(__file__)
 metadata['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 metadata['git_diff_sha256']=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=REPO)).hexdigest()
 env=dict(os.environ)
 overrides={'MISTRALRS_MOE_REPLAY_DIR':str(CAPTURES),'MISTRALRS_MOE_REPLAY_OUTPUT_DIR':str(a.output/'outputs'),'MISTRALRS_MOE_BENCH_OUTPUT':str(a.output/'replay.json'),'MISTRALRS_GGUF_MOE_TILE':a.tile}
 for name in ['MISTRALRS_MOE_REPLAY_SAMPLE','MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE','MISTRALRS_MOE_REPLAY_BASELINE_DIR']:
  env.pop(name,None)
 if a.sample:overrides['MISTRALRS_MOE_REPLAY_SAMPLE']=a.sample
 env.update(overrides);metadata['environment_overrides']=overrides
 command=[str(a.binary),'--exact',TEST,'--ignored','--nocapture','--test-threads=1'];metadata['command']=command
 monitor_command=['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv,nounits','--loop-ms=1000']
 metadata['monitor_command']=monitor_command;save()
 with (a.output/'gpu_monitor.csv').open('wb') as out,(a.output/'gpu_monitor.stderr.log').open('wb') as err,(a.output/'replay.log').open('wb') as log:
  monitor=subprocess.Popen(monitor_command,stdout=out,stderr=err,start_new_session=True)
  child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  metadata['child_pid']=child.pid;metadata['monitor_pid']=monitor.pid;save()
  child.wait(timeout=1800);metadata['returncode']=child.returncode
  metadata['monitor_returncode_before_stop']=monitor.poll();assert monitor.poll() is None
  stop_child(monitor,timeout=5);metadata['monitor_returncode']=monitor.returncode
 assert child.returncode==0
 d=json.loads((a.output/'replay.json').read_text());rows=d['results'];assert len(rows)==(1 if a.sample else 41),len(rows)
 metadata['replay_sha256']=sha(a.output/'replay.json')
 metadata['saved_outputs']=[dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted((a.output/'outputs').iterdir())]
 assert len(metadata['saved_outputs'])==len(rows)
 metadata['isolation_after']=isolate();assert sha(a.binary)==metadata['binary_sha256']
 metadata.update(complete=True,finished_at=now());save();print('COMPLETE',a.output,flush=True)
except BaseException as error:
 for process in [child,monitor]:
  if process is not None:stop_child(process,timeout=5)
 metadata.update(complete=False,error=repr(error),finished_at=now());save();raise
