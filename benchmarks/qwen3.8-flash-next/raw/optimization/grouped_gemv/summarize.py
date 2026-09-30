#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics as stats

WORK = Path(__file__).resolve().parent
BACKENDS = ['default', 'indexed_gemv', 'grouped_2', 'grouped_4']

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def values(case, backend, mode):
    return [row['stream_elapsed_ms_per_layer'] for row in case['timings'][backend][mode]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all25')
    parser.add_argument('--oracle', default='oracle25')
    parser.add_argument('--output', default='summary.json')
    args = parser.parse_args()
    phase = WORK / args.phase
    meta = json.loads((phase / 'metadata.json').read_text())
    assert meta['complete']
    report = json.loads((phase / 'results.json').read_text())
    assert len(report['results']) == 25
    summary = dict(scope='Same captured layer8 FFN, synthetic balanced row subsets of real native captures. Not end-to-end serving performance.',
        limits=['Same-layer repeated weights can have different cache behavior from a full model.',
                'Candidate retains GEMV arithmetic, while MMQ uses a different activation quantization and rounding path.',
                'Candidate preallocates scratch/output; routing construction, quantization, output zeroing and conversion are timed.',
                'GPU monitor covers whole child including loading, guards and warmups. No measured DRAM traffic or bandwidth claim.'],
        replay_sha256=sha(phase / 'results.json'), metadata_sha256=sha(phase / 'metadata.json'),
        benchmark=dict(warmups=report['warmups'], rounds=report['rounds'], repetitions=report['repetitions']),
        comparisons=[], cases=[], numeric={})
    for case in report['results']:
        summary['cases'].append(dict(source_metadata=case['source_metadata'], selection=case['selection'], rows=case['rows'],
            active_experts=case['active_experts'], max_expert_rows=case['max_expert_rows'],
            median_ms={mode: {b: stats.median(values(case,b,mode)) for b in BACKENDS} for mode in ['eager','graph']}))
    for rows in [24,32,40,42,56]:
        cases = [x for x in report['results'] if x['rows']==rows]
        assert len(cases)==5
        for mode in ['eager','graph']:
            data=dict(rows=rows, selection=cases[0]['selection'], mode=mode,
                median_ms={b:stats.median([stats.median(values(c,b,mode)) for c in cases]) for b in BACKENDS}, candidates={})
            for candidate in ['grouped_2','grouped_4']:
                item={}
                for baseline in ['default','indexed_gemv']:
                    ratios=[stats.median(values(c,baseline,mode))/stats.median(values(c,candidate,mode)) for c in cases]
                    item[baseline]=dict(median_speedup=stats.median(ratios), min_speedup=min(ratios), max_speedup=max(ratios), wins=sum(r>1 for r in ratios), samples=len(ratios))
                data['candidates'][candidate]=item
            summary['comparisons'].append(data)
    summary['numeric']['max_relative_rms_vs_gemv'] = max(m['relative_rms'] for c in report['results'] for m in c['candidate_vs_gemv'])
    summary['numeric']['max_relative_rms_graph_vs_eager'] = max(m['relative_rms'] for c in report['results'] for m in c['graph_checks'])
    summary['numeric']['changed_route_checks'] = [m for c in report['results'] for m in c['changed_route_checks']]
    summary['numeric']['strict_guard'] = dict(relative_rms_max=1e-3, cosine_min=0.999999)
    oracle_path=WORK / args.oracle / 'oracle.json'
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text())
        summary['oracle']=dict(sha256=sha(oracle_path), scope=oracle.get('reference'), results={})
        for rows in [24,32,40,42,56]:
            cases=[x for x in oracle['results'] if len(x['source_row_indices'])==rows]
            assert len(cases)==5
            summary['oracle']['results'][rows]={b:dict(median_relative_rms=stats.median([c['metrics'][b]['relative_rms'] for c in cases]), max_relative_rms=max(c['metrics'][b]['relative_rms'] for c in cases)) for b in BACKENDS}
    with (phase/'gpu_monitor.csv').open() as stream:
        monitor=list(csv.DictReader(stream,skipinitialspace=True))
    summary['gpu_monitor']={}
    for key in monitor[0]:
        numbers=[]
        for row in monitor:
            try:numbers.append(float(row[key]))
            except (ValueError,TypeError):pass
        if numbers:summary['gpu_monitor'][key]=dict(count=len(numbers),min=min(numbers),median=stats.median(numbers),max=max(numbers))
    target=WORK / args.output
    assert not target.exists(),target
    target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(target)

if __name__=='__main__':main()
