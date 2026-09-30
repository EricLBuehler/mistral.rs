#!/usr/bin/env python3
"""Bounded same-input MMQ/cuTile projection profiling after ordinary replay."""
import argparse, csv, io, json, os, signal, sys, time
from pathlib import Path

PRIOR=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930')
sys.path.insert(0,str(PRIOR))
from profile_selected_replay import NCU, SECTIONS, command, parse_metrics
from build_diagnostic import atomic_json, isolate, sha

WORK=Path(__file__).resolve().parent
BINARY=WORK/'moe_dispatch_bench.validated'
BINARY_SHA256='6b9424feab37eeb64ba39495bb536001abd8457a919a63ef5d88efd8617ca03b'
SAMPLE='c8_target_verify_b8_q7_04.json'
TEST='cutile_gguf_replay::flash_next_cutile_gguf_real_routing'
LAUNCH_SKIP=27
LAUNCH_COUNT=3
LOCAL_METRICS=['l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released',action='store_true')
    parser.add_argument('--tile',default='16,64,128')
    parser.add_argument('--output-dir',type=Path,default=WORK/'ncu_native56')
    parser.add_argument('--replay-dir',type=Path,default=WORK/'native56_default')
    parser.add_argument('--backend',choices=['both','cutile'],default='both')
    parser.add_argument('--launch-count',type=int,default=LAUNCH_COUNT)
    args=parser.parse_args()
    assert args.released,'Coordinator must release GPU slot'
    out=args.output_dir;out.mkdir(exist_ok=False)
    meta=dict(complete=False,started_unix=time.time(),commands=[],script_sha256=sha(Path(__file__)),
              scope='Three warmed projections (gate, up, down) from each backend on one native56 input; counter replay diagnostics, not unprofiled FFN performance.',
              source_test=str(WORK/'build/cutile_gguf_replay.rs'),source_test_sha256=sha(WORK/'build/cutile_gguf_replay.rs'),
              launch_skip=LAUNCH_SKIP,launch_count=args.launch_count,tile=args.tile,
              skip_derivation='Three matched projection launches per forward. One initial forward + one capture-prewarm forward + one initial graph replay + three warmups each with eager and graph forwards = nine forwards = 27 matched launches; profile the following eager forward.',
              graph_profiling='node',cache_control='none',clock_control='none',terminate_after_profile=True,
              limits=['Kernel replay serializes launches and changes subsequent replay cache state.',
                      'Per-kernel profiled duration is not an ordinary forward benchmark.',
                      'L2 read/write sectors are not DRAM bytes or system-memory bandwidth.',
                      'One captured input, native B8xQ7 layer8; no whole-model speed claim.',
                      'NCU kills the target after the selected launches; profiled test output is intentionally incomplete.'])
    def save():atomic_json(out/'metadata.json',meta)
    def run(name,argv,env=None):
        r=dict(name=name,command=argv,started_unix=time.time());meta['commands'].append(r);save()
        result=command(argv,out/(name+'.log'),env)
        r.update(returncode=0,finished_unix=time.time(),log_sha256=sha(out/(name+'.log')));save()
        return result
    def interrupted(number,_frame):raise InterruptedError(number)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,interrupted)
    try:
        assert sha(BINARY)==BINARY_SHA256
        prior=json.loads((PRIOR/'ncu_native56/metadata.json').read_text())
        replay=args.replay_dir
        validation=json.loads((replay/'metadata.json').read_text())
        assert validation['complete'] and validation['binary_sha256']==BINARY_SHA256
        assert sha(replay/'replay.json')==validation['replay_sha256']
        meta.update(binary=dict(path=str(BINARY),sha256=BINARY_SHA256),ordinary_replay=dict(path=str(replay/'metadata.json'),sha256=sha(replay/'metadata.json')),
                    sample=prior['sample'],sample_selection=prior['sample_selection'],metrics_available=prior['metrics_available'],
                    metrics_unavailable=prior['metrics_unavailable'],dram_counters_available=False,isolation_before=isolate(),
                    prior_metric_query=dict(path=str(PRIOR/'ncu_native56/metadata.json'),sha256=sha(PRIOR/'ncu_native56/metadata.json')))
        assert Path(prior['sample']['path']).name==SAMPLE
        query_path=PRIOR/'ncu_native56/query_metrics.log'
        query=query_path.read_text()
        assert all(name+' ' in query for name in LOCAL_METRICS)
        meta['metrics_available']=prior['metrics_available']+LOCAL_METRICS
        meta['local_metric_support']=dict(path=str(query_path),sha256=sha(query_path),metrics=LOCAL_METRICS)
        selected=out/'selected_sample';selected.mkdir()
        for path in prior['sample']['links']:
            source=Path(path);(selected/source.name).symlink_to(source)
        summary=[]
        filters=[('cutile','regex:^projection(_entry)?$')]
        if args.backend=='both':filters.append(('mmq','regex:^mul_mat_q$'))
        for name,regex in filters:
            env=dict(os.environ,MISTRALRS_MOE_REPLAY_DIR=str(selected),MISTRALRS_MOE_REPLAY_SAMPLE=SAMPLE,
                     MISTRALRS_MOE_REPLAY_OUTPUT_DIR=str(out/(name+'.outputs')),
                     MISTRALRS_MOE_BENCH_OUTPUT=str(out/(name+'.profiled_test_output_not_benchmark.json')),
                     MISTRALRS_GGUF_MOE_TILE=args.tile)
            for key in ['MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE','MISTRALRS_MOE_REPLAY_BASELINE_DIR']:env.pop(key,None)
            meta.setdefault('environment_overrides',{})[name]={k:v for k,v in env.items() if k.startswith('MISTRALRS_MOE_') or k=='MISTRALRS_GGUF_MOE_TILE'}
            argv=[str(NCU),'--config-file','0','--section-folder',str(SECTIONS),'--target-processes','application-only',
                  '--replay-mode','kernel','--graph-profiling','node','--cache-control','none','--clock-control','none','--import-sass','no',
                  '--kernel-name-base','function','--kernel-name',regex,'--launch-skip',str(LAUNCH_SKIP),'--launch-count',str(args.launch_count),'--kill','yes',
                  '--metrics',','.join(meta['metrics_available']),'--disable-extra-suffixes','--export',str(out/name),
                  str(BINARY),'--exact',TEST,'--ignored','--nocapture','--test-threads=1']
            run(name,argv,env)
            raw=run(name+'_raw',[str(NCU),'--import',str(out/(name+'.ncu-rep')),'--page','raw','--csv','--print-units','base'])
            parsed=parse_metrics(raw);assert len(parsed)==args.launch_count,(name,len(parsed))
            rows=csv.DictReader(io.StringIO(raw[raw.index('"ID",'):]))
            units=next(rows)
            for result,row in zip(parsed,rows):
                result['metrics'].update({k:dict(value=row[k],unit=units[k]) for k in LOCAL_METRICS if k in row})
                result['launch_resources']={k:dict(value=v,unit=units[k]) for k,v in row.items()
                                           if k.startswith('launch__') and any(s in k for s in ['shared','occupancy_limit','register','block_size'])}
            summary.append(dict(backend=name,results=parsed))
        atomic_json(out/'kernels.json',dict(scope=meta['scope'],limits=meta['limits'],results=summary))
        meta.update(complete=True,finished_unix=time.time(),isolation_after=isolate());save()
        (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
        print('cuTile projection profiling complete',flush=True)
    except BaseException as error:meta.update(error=repr(error),failed_unix=time.time());save();raise

if __name__=='__main__':main()
