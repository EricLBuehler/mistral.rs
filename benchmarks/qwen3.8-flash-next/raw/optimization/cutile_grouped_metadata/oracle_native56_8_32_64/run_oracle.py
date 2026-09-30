"""Validate saved cuTile full-FFN outputs independently of timing."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
REPO=Path('/home/ericbuehler/mistral.rs')
PRIOR=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930')
sys.path.insert(0,str(PRIOR))
from build_diagnostic import isolate
from run_diagnostic import stop_child
TEST='cutile_gguf_replay::flash_next_cutile_gguf_oracle'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True,type=Path)
p.add_argument('--replay',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--released',required=True,action='store_true')
a=p.parse_args();a.output.mkdir(exist_ok=False)
replay_meta=json.loads((a.replay/'metadata.json').read_text());assert replay_meta['complete']
assert sha(a.replay/'replay.json')==replay_meta['replay_sha256']
for output in replay_meta['saved_outputs']:assert sha(a.replay/output['path'])==output['sha256']
meta=dict(complete=False,started_at=now(),binary=str(a.binary),binary_sha256=sha(a.binary),replay=str(a.replay),replay_metadata_sha256=sha(a.replay/'metadata.json'),saved_outputs=replay_meta['saved_outputs'],isolation_before=isolate())
assert meta['binary_sha256']==replay_meta['binary_sha256']
shutil.copy2(__file__,a.output/'run_oracle.py')
shutil.copy2(a.replay/'sources/mistralrs-quant/tests/support/cutile_gguf_replay.rs',a.output/'cutile_gguf_replay.rs')
shutil.copy2(a.replay/'sources/mistralrs-quant/tests/moe_dispatch_bench.rs',a.output/'moe_dispatch_bench.rs')
meta['source_hashes']={p.name:sha(p) for p in a.output.iterdir()}
def save():(a.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
env=dict(os.environ)
overrides=dict(MISTRALRS_MOE_REPLAY_DIR=str(PRIOR/'runtime/captures'),MISTRALRS_MOE_REPLAY_RESULTS=str(a.replay/'replay.json'),MISTRALRS_MOE_ORACLE_OUTPUT=str(a.output/'oracle.json'))
env.update(overrides)
cmd=[str(a.binary),'--exact',TEST,'--ignored','--nocapture','--test-threads=1']
meta.update(command=cmd,environment_overrides=overrides);save()
child=None
try:
 with (a.output/'oracle.log').open('wb') as log:
  child=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  meta['child_pid']=child.pid;save();child.wait(timeout=1200)
 assert child.returncode==0,child.returncode
 d=json.loads((a.output/'oracle.json').read_text());r=json.loads((a.replay/'replay.json').read_text());assert len(d['results'])==len(r['results'])
 meta.update(complete=True,returncode=child.returncode,finished_at=now(),oracle_sha256=sha(a.output/'oracle.json'),log_sha256=sha(a.output/'oracle.log'),isolation_after=isolate());save();print('COMPLETE',a.output,flush=True)
except BaseException as error:
 if child:stop_child(child,timeout=5)
 meta.update(error=repr(error),finished_at=now());save();raise
