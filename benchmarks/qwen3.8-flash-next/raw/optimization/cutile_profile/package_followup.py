#!/usr/bin/env python3
"""Archive small immutable cuTile profile and compact-scheduler evidence."""
from pathlib import Path
import datetime,hashlib,json,shutil

WORK=Path(__file__).resolve().parent
REPO=Path('/home/ericbuehler/mistral.rs')
DEST=REPO/'benchmarks/qwen3.8-flash-next/raw/optimization'
PRIOR=WORK.parent/'real_routing_20260930'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def package(name, source, paths, extras, note):
    out=DEST/name
    out.mkdir(exist_ok=False)
    copied=[]
    for relative in paths:
        original=source/relative
        assert original.is_file() and not original.is_symlink(),original
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        copied.append(dict(path=str(relative),source=str(original),bytes=target.stat().st_size,sha256=sha(target)))
    for relative,original in extras.items():
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
        copied.append(dict(path=str(relative),source=str(original),bytes=target.stat().st_size,sha256=sha(target)))
    manifest=dict(complete=True,created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),external_original_root=str(source),scope=note,files=copied,excluded='No executable, native archive, object, model weight, saved tensor output, selected-sample symlink, or duplicate output metadata was copied. External lifecycle/build records retain their hashes and paths.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.relative_to(out)}\n' for path in sorted(out.rglob('*')) if path.is_file() and path.name!='SHA256SUMS'))
    for record in copied:assert sha(out/record['path'])==record['sha256']
    print(name,len(copied),'files',sum(record['bytes'] for record in copied),'bytes')

source=WORK/'cutile_gguf'
assert json.loads((source/'ncu_native56/metadata.json').read_text())['complete']
assert json.loads((source/'ncu_native56_wide/metadata.json').read_text())['complete']
paths=[Path('profile_projection.py'),Path('ncu_comparison.json')]
for phase in ['ncu_native56','ncu_native56_wide']:
    paths += [p.relative_to(source) for p in (source/phase).iterdir() if p.is_file() and p.suffix in ['.json','.log','.ncu-rep']]
package('cutile_profile',source,paths,{
    Path('helpers/profile_selected_replay.py'):PRIOR/'profile_selected_replay.py',
    Path('helpers/build_diagnostic.py'):PRIOR/'build_diagnostic.py',
    Path('package_followup.py'):Path(__file__),
},'Same native56 layer8 input; default cuTile and masked MMQ three projections each, plus one wide cuTile gate projection. NCU kernel replay, caches/clocks unchanged, ten passes. Profile completion means requested records were exported; --kill yes intentionally stops the test before its replay summary. Counter durations are not benchmark timings. Source/prototype correctness and ordinary timing evidence remain in sibling cutile_initial; metric-query support is referenced externally by path/hash.')

source=WORK/'compact_schedule'
assert json.loads((source/'final_provenance.json').read_text())['complete']
paths=[Path(p) for p in ['compact.snippet.cuh','prepare_sources.py','prepare_build.py','check_schedule.py','mapping_check.json','build_shadow.py','validate_compact.py','run_comparison.py','comparison_commands.json','summary.json','findings.txt','final_provenance.json','resources.json','q4_k.resources.txt','q4_1.resources.txt','sources/compact_schedule.patch','sources/sources.json','build/provenance.json','build/replay.source.rs','build/nvcc_0.log','build/nvcc_1.log','build/archive.log','build/rustc.log']]
paths += [p.relative_to(source) for p in (source/'validation').rglob('*') if p.is_file() and p.suffix in ['.json','.log','.rs']]
for phase in ['baseline','compact']:
    paths += [source.joinpath(phase,p).relative_to(source) for p in ['lifecycle.json','replay.json','replay.log','gpu_monitor.csv','gpu_monitor.stderr.log']]
package('compact_schedule',source,paths,{
    Path('helpers/run_variant.py'):PRIOR/'tile_variants/run_variant.py',
    Path('helpers/summarize_variants.py'):PRIOR/'tile_variants/summarize_variants.py',
    Path('helpers/build_diagnostic.py'):PRIOR/'build_diagnostic.py',
    Path('helpers/run_diagnostic.py'):PRIOR/'run_diagnostic.py',
    Path('package_followup.py'):Path(__file__),
},'Rejected external fixed-grid compact full-K Q4K/Q4_1 scheduler, built against committed bounded mask. Freshly linked baseline and candidate use identical replay source and Rust dependencies. All25 outputs are bit-exact; CPU-oracle tail and changed-route graph tests plus memcheck pass. Ordinary paired FFN timings regress; no production adoption or tile16 followup. Unchanged GEMV and whole-child clock/power/temperature samples are observational drift controls. Lifecycle records hash excluded output tensors. Source snapshots describe preparation; validation/final_provenance record completed build/tests.')
