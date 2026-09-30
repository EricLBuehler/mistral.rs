#!/usr/bin/env python3
import concurrent.futures,datetime,hashlib,json,os,shutil,signal,subprocess
from pathlib import Path
W=Path(__file__).resolve().parent; B=W/'build'; S=B/'shadow'; R=Path('/home/ericbuehler/mistral.rs')
P=B/'provenance.json'; meta=json.loads(P.read_text()); children=[]
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():P.write_text(json.dumps(meta,indent=2)+'\n')
def interrupted(n,f):raise InterruptedError(n)
for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
def run(command,name,env=None):
 with (B/(name+'.log')).open('wb') as out:
  p=subprocess.Popen(command,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True); children.append(p)
  rc=p.wait(); assert rc==0,(name,rc)
try:
 meta['build_started_at']=now(); meta['build_script_sha256']=sha(__file__);save()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run,c,'nvcc_'+str(i)) for i,c in enumerate(meta['nvcc_commands'])]
  for f in futures:f.result()
 archive=S/'libmistralrsquant.a'; shutil.copy2(meta['native_archive'],archive)
 command=['ar','r',str(archive),*[c[-1] for c in meta['nvcc_commands']]];meta['archive_command']=command;save();run(command,'archive')
 env=dict(os.environ,**meta['rustc_environment']);run(meta['rustc_command'],'rustc',env)
 binary=S/'moe_dispatch_bench-0dfba70e2949feab';dest=B/'moe_dispatch_bench.variant';os.link(binary,dest)
 meta['variant_binary']=str(dest);meta['variant_binary_sha256']=sha(dest);meta['shadow_archive_sha256']=sha(archive)
 meta['compiled_objects']={c[-1]:sha(c[-1]) for c in meta['nvcc_commands']}
 assert sha(meta['native_archive'])==meta['native_archive_sha256']
 assert sha(R/'target/release/mistralrs')==meta['production_cli_sha256']
 for p,h in meta['original_sources'].items():assert sha(p)==h
 meta['originals_unchanged']=True;meta['complete']=True;meta['build_finished_at']=now();save();print('Build complete',flush=True)
except BaseException as e:
 for p in children:
  if p.poll() is None:
   os.killpg(p.pid,signal.SIGTERM)
 for p in children:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 meta['error']=repr(e);meta['failed_at']=now();save();raise
