#!/usr/bin/env python3
import argparse,csv,io,json,os,signal,time,sys
from pathlib import Path
sys.path.insert(0, '/home/ericbuehler/qwen4exp_work/real_routing_20260930')
from profile_selected_replay import NCU,SECTIONS,command,parse_metrics
from build_diagnostic import WORK,atomic_json,isolate,sha
parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['baseline','candidate']);args=parser.parse_args()
ROOT=Path('/home/ericbuehler/qwen4exp_work/moe_optimization_20260930');W=ROOT/'masked_y';out=W/('ncu_'+args.kind);phase=ROOT/'baseline' if args.kind=='baseline' else W/'replay';prior=WORK/'ncu_native56' 
assert not out.exists();out.mkdir()
meta={'complete':False,'started_unix':time.time(),'commands':[],'scope':'Three grouped projection launches for the same median-U native56 input as baseline NCU; counter replay, not benchmark timings.','script_sha256':sha(Path(__file__))}
def save():atomic_json(out/'metadata.json',meta)
def run(name,argv,env=None):
 r={'name':name,'command':argv,'started_unix':time.time()};meta['commands'].append(r);save();text=command(argv,out/(name+'.log'),env);r.update(returncode=0,finished_unix=time.time(),log_sha256=sha(out/(name+'.log')));save();return text
try:
 lifecycle=json.loads((phase/'lifecycle.json').read_text());assert lifecycle['complete'];binary=Path(lifecycle['binary']['path']);assert sha(binary)==lifecycle['binary']['sha256']
 meta['validated_replay']={'path':str(phase/'lifecycle.json'),'sha256':sha(phase/'lifecycle.json')};meta['binary']=lifecycle['binary'];meta['isolation_before']=isolate()
 original=json.loads((prior/'metadata.json').read_text());meta['baseline_profile_metadata']={'path':str(prior/'metadata.json'),'sha256':sha(prior/'metadata.json')};meta['sample']=original['sample'];meta['sample_selection']=original['sample_selection'];meta['limits']=original['limits'];meta['metrics_available']=original['metrics_available'];meta['metrics_unavailable']=original['metrics_unavailable'];meta['dram_counters_available']=False
 selected=out/'selected_sample';selected.mkdir()
 for link in original['sample']['links']:
  p=Path(link);(selected/p.name).symlink_to(p)
 env=dict(os.environ,MISTRALRS_MOE_REPLAY_DIR=str(selected),MISTRALRS_MOE_BENCH_OUTPUT=str(out/'profiled_test_output_not_benchmark.json'))
 env.pop('MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE',None);env.pop('MISTRALRS_MOE_REPLAY_OUTPUT_DIR',None);env.pop('MISTRALRS_MOE_REPLAY_BASELINE_DIR',None)
 meta['environment_overrides']={k:v for k,v in env.items() if k.startswith('MISTRALRS_MOE_') or k=='MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE'}
 argv=[str(NCU),'--config-file','0','--section-folder',str(SECTIONS),'--target-processes','application-only','--replay-mode','kernel','--cache-control','none','--clock-control','none','--import-sass','no','--kernel-name-base','function','--kernel-name','regex:^mul_mat_q$','--launch-skip','12','--launch-count','3','--metrics',','.join(original['metrics_available']),'--disable-extra-suffixes','--export',str(out/'mmq'),str(binary),'--exact','flash_next_real_routing_replay','--ignored','--nocapture','--test-threads=1']
 run('mmq',argv,env)
 raw=run('mmq_raw',[str(NCU),'--import',str(out/'mmq.ncu-rep'),'--page','raw','--csv','--print-units','base'])
 parsed=parse_metrics(raw);assert len(parsed)==3
 rows=csv.DictReader(io.StringIO(raw[raw.index('"ID",'):]))
 units=next(rows)
 for result,row in zip(parsed,rows):
  result['launch_resources']={k:{'value':v,'unit':units[k]} for k,v in row.items() if k.startswith('launch__') and any(s in k for s in ['shared','occupancy_limit','register','block_size'])}
 assert all('ELi64E' in k['kernel'] or ', 64,' in k['kernel'] for k in parsed),[k['kernel'] for k in parsed]
 atomic_json(out/'kernels.json',{'scope':meta['scope'],'limits':meta['limits'],'results':parsed});meta.update(complete=True,finished_unix=time.time(),isolation_after=isolate());save();print('NCU complete',flush=True)
except BaseException as e:meta['error']=repr(e);save();raise
