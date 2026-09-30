#!/usr/bin/env python3
from pathlib import Path
import datetime,hashlib,json,os,signal,subprocess
R=Path('/home/ericbuehler/mistral.rs');W=Path(__file__).resolve().parent/'masked_y_bounded';V=W/'validation';V.mkdir(exist_ok=False)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
executables={}
for line in (W/'production_build.log').read_text().splitlines():
 try:d=json.loads(line)
 except:continue
 if d.get('reason')=='compiler-artifact' and d.get('target',{}).get('name') in ['grouped_mmq_tail_cuda_tests','grouped_mmq_packed_cuda_tests']:
  path=Path(d['executable']);name=d['target']['name'];dest=V/name;os.link(path,dest);executables[name]=dest
assert len(executables)==2
meta=dict(complete=False,started_at=now(),header_sha256=sha(R/'mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh'),binaries={n:dict(path=str(p),sha256=sha(p))for n,p in executables.items()},commands=[])
def save():(V/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
def interrupted(n,f):raise InterruptedError(n)
for s in [signal.SIGTERM,signal.SIGINT]:signal.signal(s,interrupted)
try:
 for sanitizer in [False,True]:
  for name,path in executables.items():
   label=name+('.memcheck'if sanitizer else'.test');command=([str(Path('/usr/local/cuda/bin/compute-sanitizer')),'--tool','memcheck','--error-exitcode','99','--target-processes','all','--print-limit','20']if sanitizer else[])+[str(path),'--test-threads=1']
   record=dict(name=label,command=command,started_at=now());meta['commands'].append(record);save();print(now(),label,flush=True)
   with (V/(label+'.log')).open('wb') as log:
    p=subprocess.Popen(command,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);record['pid']=p.pid;save()
    try:code=p.wait(timeout=240)
    except BaseException:
     os.killpg(p.pid,signal.SIGTERM)
     try:p.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
     raise
   record.update(returncode=code,finished_at=now(),log_sha256=sha(V/(label+'.log')));save();assert code==0,label
   if sanitizer:assert 'ERROR SUMMARY: 0 errors' in (V/(label+'.log')).read_text()
 meta.update(complete=True,finished_at=now());save();print('GPU validation complete',flush=True)
except BaseException as e:meta['error']=repr(e);save();raise
