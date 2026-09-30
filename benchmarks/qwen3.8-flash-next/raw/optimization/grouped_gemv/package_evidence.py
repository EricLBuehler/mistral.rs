#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil

WORK=Path(__file__).resolve().parent
DEST=Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/optimization/grouped_gemv')

def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    DEST.mkdir(exist_ok=False)
    paths=[WORK/p for p in ['grouped_gemv.cu','grouped_gemv_replay.rs','replay.source.rs','build_external.py','run_external.py','summarize.py','package_evidence.py','findings.txt','summary.json','more_features_summary.json','final_provenance.json','harness_sources.json','more_features/grouped_gemv.cu','more_features/features_per_warp.patch','validation_harness/grouped_gemv_replay.rs','validation_harness/replay.source.rs']]
    for build in ['build_final','build_validation','build_more_features','build_more_features_validation']:
        paths.extend(WORK/build/name for name in ['provenance.json','nvcc.log','rustc.log'] if (WORK/build/name).is_file())
    for phase in ['smoke_native56','all25','oracle25','memcheck_native56','memcheck_focused','memcheck_validation','more_features_all25','more_features_oracle25','more_features_memcheck_validation']:
        paths.extend(p for p in (WORK/phase).iterdir() if p.is_file() and p.suffix in ['.json','.log','.csv','.py'])
    for path in paths:
        destination=DEST/path.relative_to(WORK)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    base=WORK.parent/'compact_schedule/build/provenance.json'
    shutil.copy2(base,DEST/'build_base.provenance.json')
    files=[dict(path=str(p.relative_to(DEST)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.rglob('*')) if p.is_file()]
    manifest=dict(scope='External grouped small-row GEMV experiment, both output-feature layouts; no production adoption.',external_source_root=str(WORK),files=files,
        excluded='Executable/object/native archives, full SASS, model weights and output tensors. Exact executable/object/source/archive hashes are retained in build and final provenance; saved output hashes are retained in phase metadata.',
        validation='Two initial sanitizer attempts were stopped and are incomplete. Validation-only runs pass with zero errors while reusing the exact benchmark CUDA objects. Sanitizer timings are not benchmark results.')
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (DEST/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='SHA256SUMS'))
    for entry in files:
        assert sha(DEST/entry['path'])==entry['sha256']
    print(f'{len(files)} files, {sum(f["bytes"] for f in files)} bytes: {DEST}')

if __name__=='__main__':main()
