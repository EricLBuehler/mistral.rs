from pathlib import Path
import hashlib,json,statistics,collections
W=Path(__file__).resolve().parent
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def key(r):return r['source_metadata'],r['selection'],tuple(r['source_row_indices'])
def load(name):
 p=W/name;meta=json.loads((p/'lifecycle.json').read_text());assert meta['complete'];assert sha(p/'replay.json')==meta['replay_sha256'];d=json.loads((p/'replay.json').read_text());return {key(r):r for r in d['results']}
def med(r,path):return statistics.median(t['stream_elapsed_ms_per_layer'] for t in r[path])
def stats(xs):return dict(median=statistics.median(xs),minimum=min(xs),maximum=max(xs))
names=['baseline','masked_y/replay','masked_y/replay_reverse','baseline_reverse'];data=[load(n) for n in names];assert all(set(d)==set(data[0]) for d in data)
rows=[]
for k in data[0]:
 rr=[d[k] for d in data]
 for r in rr[1:]:assert r['baseline_grouped_comparison']['relative_rms']<1e-3 and r['baseline_grouped_comparison']['cosine']>.999999
 result=dict(source_metadata=k[0],selection=k[1],source_rows=list(k[2]),rows=rr[0]['rows'],derived=rr[0]['derived'],timings=[dict(phase=n,grouped_ms=med(r,'grouped'),gemv_ms=med(r,'gemv')) for n,r in zip(names,rr)])
 result['forward_ratio']=med(rr[0],'grouped')/med(rr[1],'grouped');result['reverse_ratio']=med(rr[3],'grouped')/med(rr[2],'grouped');result['mean_pair_ratio']=statistics.mean([result['forward_ratio'],result['reverse_ratio']]);result['forward_gemv_control']=med(rr[0],'gemv')/med(rr[1],'gemv');result['reverse_gemv_control']=med(rr[3],'gemv')/med(rr[2],'gemv');rows.append(result)
groups=[]
for count in [24,32,40,42,56]:
 subset=[r for r in rows if r['rows']==count];group=dict(rows=count,selection=subset[0]['selection'],samples=len(subset),**{field:stats([r[field] for r in subset]) for field in ['forward_ratio','reverse_ratio','mean_pair_ratio','forward_gemv_control','reverse_gemv_control']});groups.append(group);print(count,json.dumps(group))
report=dict(scope='Warm repeated exact layer-8 expert operands only; not full model throughput.',phases=[dict(path=str(W/n),lifecycle_sha256=sha(W/n/'lifecycle.json')) for n in names],ordering='Baseline,candidate,candidate,baseline. Two full passes; each pass has seven alternating GEMV/grouped rounds of ten forwards per input.',statistic='Ratios compare per-input median CUDA event times. mean_pair_ratio is arithmetic mean of forward and reverse ratios for each input; group reports median/range of those per-input means.',groups=groups,samples=rows,strict_outputs='Every candidate output is bit-exact to saved grouped baseline for all25 cases in both passes.',caveats=['Derived layouts select first3/4/5 queries per each of eight captured sequences; they are not actual independently observed model requests.','Unchanged GEMV is a drift observation, not a normalization correction.','NCU runs preceded the reversed pair; they are not used as benchmark timings.'])
(W/'masked_y/abba_summary.json').write_text(json.dumps(report,indent=2)+'\n')
