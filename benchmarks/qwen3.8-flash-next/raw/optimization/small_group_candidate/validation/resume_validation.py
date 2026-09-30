"""Finish linting using the same feature union as the production CLI."""
import datetime,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path('/home/ericbuehler/mistral.rs')
OUT=Path(__file__).resolve().parent/'build'
meta=json.loads((OUT/'metadata.json').read_text());assert not meta['complete']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
meta['interrupted_attempt']=dict(error=meta.pop('error'),reason='Standalone core Clippy selected a different feature union and started rebuilding unchanged GDN CUDA code. That owned process tree was terminated; core and CLI are linted together below using the production feature union.')
shutil.copy2(__file__,OUT/'resume_validation.py')
def save():(OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
save()
try:
 for name,cmd in [('core_cli_clippy',['cargo','clippy','--release','-p','mistralrs-cli','-p','mistralrs-core','--features','cuda,flash-attn,cutile','--','-D','warnings']),('format_check',['cargo','fmt','--all','--','--check'])]:
  r=dict(name=name,command=cmd,started_at=now());meta['commands'].append(r);save();print('START',name,flush=True)
  with (OUT/(name+'.log')).open('wb') as log:res=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  r.update(returncode=res.returncode,finished_at=now(),log_sha256=sha(OUT/(name+'.log')));save();assert res.returncode==0,name;print('PASS',name,flush=True)
 for path,h in meta['sources'].items():assert sha(ROOT/path)==h,path
 for r in meta['binaries'].values():assert sha(r['path'])==r['sha256']
 assert sha(OUT.parent/'baseline.mistralrs')=='d245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588'
 meta.update(complete=True,finished_at=now());save();print('COMPLETE',OUT,flush=True)
except BaseException as e:meta.update(error=repr(e),finished_at=now());save();raise
