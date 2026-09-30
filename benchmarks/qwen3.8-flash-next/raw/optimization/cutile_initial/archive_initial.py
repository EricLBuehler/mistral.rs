"""Freeze the validated direct cuTile decoder and its first three real-route probes."""
import datetime,hashlib,json,shutil,statistics
from pathlib import Path
WORK=Path(__file__).resolve().parent
REPO=Path('/home/ericbuehler/mistral.rs')
DEST=REPO/'benchmarks/qwen3.8-flash-next/raw/optimization/cutile_initial'
PHASES=['native56_default','native56_8_64_256','native56_8_32_64']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not DEST.exists()
validation=json.loads((WORK/'projection.validation.json').read_text());assert validation['complete']
for p,h in validation['source_hashes'].items():assert sha(REPO/p)==h,p
for p,h in validation['log_hashes'].items():assert sha(WORK/p)==h,p
results=[]
for phase in PHASES:
 folder=WORK/phase;meta=json.loads((folder/'metadata.json').read_text());assert meta['complete'] and meta['all_capture_files_verified']
 assert sha(folder/'replay.json')==meta['replay_sha256']
 for out in meta['saved_outputs']:assert sha(folder/out['path'])==out['sha256']
 replay=json.loads((folder/'replay.json').read_text());assert len(replay['results'])==1
 row=replay['results'][0]
 assert row['source_metadata']=='c8_target_verify_b8_q7_04.json' and row['rows']==56 and not row['derived']
 timings={k:statistics.median(t['stream_elapsed_ms_per_layer'] for t in row[k]) for k in ['baseline_eager','candidate_eager','baseline_graph','candidate_graph']}
 results.append(dict(phase=phase,config=replay['config'],median_ms=timings,candidate_slowdown_eager=timings['candidate_eager']/timings['baseline_eager'],candidate_slowdown_graph=timings['candidate_graph']/timings['baseline_graph'],numerical=row['numerical']))
files=[]
for p in WORK.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(WORK)
 if len(rel.parts)>1 and rel.parts[0] not in PHASES+['build']:continue
 if p.suffix not in ['.rs','.py','.json','.log','.csv','.md','.cuh','.safetensors']:continue
 if '__pycache__' in rel.parts:continue
 files.append(p)
manifest=[]
for p in sorted(files):
 rel=p.relative_to(WORK);dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 manifest.append(dict(path=str(rel),bytes=dest.stat().st_size,sha256=sha(dest)))
summary=dict(scope='One native56 capture, three tile configurations, independent per-shape warmup and7 alternating rounds of10 complete routed FFN forwards. Eager and CUDAgraph latencies measured separately. No end-to-end serving measurement.',results=results,default_dispatch_changed=False,oracle_scope='4 dedicated CUDA projection/decode/sentinel/graph tests pass against independent CPU reference. Complete FFN comparison is against existing MMQ/captured output; separate full-FP32/BF16-weight oracle harness is implemented but not run at this checkpoint.')
(DEST/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest.append(dict(path='summary.json',bytes=(DEST/'summary.json').stat().st_size,sha256=sha(DEST/'summary.json')))
(DEST/'manifest.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(WORK),files=manifest,excluded='Compiled binaries; model weights; subsequent profiling/blocked-decoder work.'),indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(p['bytes'] for p in manifest))))
