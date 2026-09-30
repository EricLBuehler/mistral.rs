"""Freeze selected textual evidence, preserving raw bytes and recording output identities."""
from pathlib import Path
import datetime,hashlib,json,shutil
WORK=Path(__file__).resolve().parent
REPO=Path('/home/ericbuehler/mistral.rs')
DEST=REPO/'benchmarks/qwen3.8-flash-next/raw/optimization/masked_y'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert not DEST.exists()
files=[]
for p in WORK.rglob('*'):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(WORK)
 if any(part in rel.parts for part in ('compact_schedule','cutile_gguf','outputs','selected_sample','__pycache__','shadow')):continue
 if p.suffix not in ('.py','.json','.log','.txt','.csv','.patch','.rs'):continue
 if 'cuTile' in str(rel):continue
 files.append(p)
manifest=[]
for p in sorted(files):
 rel=p.relative_to(WORK); dest=DEST/rel
 dest.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,dest)
 manifest.append(dict(path=str(rel),bytes=dest.stat().st_size,sha256=sha(dest)))
outputs=[]
for name in ['baseline','masked_y/replay','masked_y/replay_reverse','baseline_reverse','masked_y_bounded/replay']:
 folder=WORK/name
 lifecycle=json.loads((folder/'lifecycle.json').read_text())
 assert lifecycle['complete']
 assert sha(folder/'replay.json')==lifecycle['replay_sha256']
 for row in lifecycle['saved_outputs']:
  assert sha(folder/row['path'])==row['sha256']
 outputs.append(dict(phase=name,files=lifecycle['saved_outputs']))
(DEST/'manifest.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(WORK),files=manifest,excluded='Compiled binaries/native archives, duplicate output tensors, model weights, and selected-sample copies. Exact saved output identities retained below and in lifecycle files.',saved_outputs=outputs),indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest),destination=str(DEST))))
