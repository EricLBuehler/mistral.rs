import argparse,datetime,hashlib,json,os,pathlib,signal,subprocess,sys
ROOT=pathlib.Path('/home/ericbuehler/mistral.rs')
WORK=pathlib.Path(__file__).resolve().parent
BINARY=pathlib.Path('/home/ericbuehler/qwen4exp_work/moe_optimization_20260930/final_serving/build/moe_dispatch_bench-0dfba70e2949feab')
CAPTURES=pathlib.Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/runtime/captures')
EXPECTED='eaf7d867dadfd56433f0f8a093e0f6dd320b1bcfde76ee76c58e32027443f7f1'
TEST='flash_next_real_routing_replay'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stop(p):
 if p is not None and p.poll() is None:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('output',type=pathlib.Path);parser.add_argument('--profile',action='store_true');args=parser.parse_args()
 assert sha(BINARY)==EXPECTED
 args.output.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ)
 for k in list(env):
  if k.startswith('MISTRALRS_'):del env[k]
 overrides={'MISTRALRS_MOE_REPLAY_DIR':str(CAPTURES),'MISTRALRS_MOE_BENCH_OUTPUT':str(args.output/'replay.json')};env.update(overrides)
 command=[str(BINARY),'--exact',TEST,'--ignored','--nocapture','--test-threads=1']
 if args.profile:
  command=['/opt/nvidia/nsight-systems/2026.1.3/bin/nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--force-overwrite=false','--output='+str(args.output/'trace')]+command
 meta={'complete':False,'started':now(),'command':command,'binary_sha256':EXPECTED,'profiled':args.profile,'environment_overrides':overrides,'runner_sha256':sha(__file__),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'limits':['Fixed-depth captured routes at one layer, not current adaptive full-model workload.','FFN CUDA-event timings include routing, quantization, activation and reduction; profiler durations are diagnostic.','Scan comparison across processes does not isolate a production optimization or establish a physical bandwidth ceiling.']}
 def save():(args.output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 def interrupt(signum,frame):raise InterruptedError(signum)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupt)
 child=None;monitor=None
 try:
  save()
  with (args.output/'gpu.csv').open('w') as gpu,(args.output/'gpu.stderr').open('w') as err,(args.output/'replay.log').open('w') as log:
   monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv,nounits','--loop-ms=1000'],stdout=gpu,stderr=err)
   child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   meta['child_pid']=child.pid;save();meta['returncode']=child.wait(timeout=900)
   assert meta['returncode']==0,meta['returncode']
  data=json.loads((args.output/'replay.json').read_text());meta['replay_sha256']=sha(args.output/'replay.json');meta['result_count']=len(data['results'])
  assert sha(BINARY)==EXPECTED
  meta['complete']=True
 finally:
  stop(child);stop(monitor);meta['finished']=now();save()
 print(json.dumps({k:meta.get(k) for k in ['complete','returncode','result_count','started','finished']}))
if __name__=='__main__':main()
