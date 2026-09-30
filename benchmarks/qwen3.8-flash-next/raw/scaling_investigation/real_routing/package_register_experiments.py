#!/usr/bin/env python3
"""Archive completed register experiments without their large binaries or tensor outputs."""
import hashlib
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parent
DEST = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/real_routing')
EXPERIMENTS = {'register_streaming': 'monitored_streaming', 'register_streaming_tile16': 'monitored_streaming16'}
SUFFIXES = ('.py', '.json', '.patch', '.cu', '.cuh', '.rs', '.txt', '.log', '.csv', '.ncu-rep', '.base')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for experiment, phase in EXPERIMENTS.items():
        source = WORK / experiment
        dest = DEST / experiment
        assert not dest.exists()
        final = json.loads((source / 'final_provenance.json').read_text())
        assert all(final['originals_unchanged'].values())
        assert sha(source / 'summary.json') == final['summary_sha256']
        lifecycle = json.loads((source / phase / 'lifecycle.json').read_text())
        assert lifecycle['complete']
        assert sha(source / phase / 'replay.json') == lifecycle['replay_sha256']
        for row in json.loads((source / phase / 'replay.json').read_text())['results']:
            assert row['baseline_grouped_comparison']['relative_rms'] == 0
        paths = [p for p in source.rglob('*') if p.is_file() and not p.is_symlink()
                 and p.suffix in SUFFIXES and 'shadow' not in p.relative_to(source).parts
                 and 'outputs' not in p.relative_to(source).parts]
        dest.mkdir()
        for path in sorted(paths):
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        build = json.loads((source / 'build/provenance.json').read_text())
        external = dict(reason='Large binary, native archive and output tensors remain external, identified by exact build/lifecycle hashes.',
                        binary=lifecycle['binary'], saved_outputs_directory=str(source / phase),
                        saved_outputs=lifecycle['saved_outputs'],
                        shadow_archive=dict(path=str(source / 'build/shadow/libmistralrsquant.a'), sha256=build['shadow_archive_sha256']))
        (dest / 'external_artifacts.json').write_text(json.dumps(external, indent=2) + '\n')
        helpers = ['tile_variants/run_variant.py', 'tile_variants/summarize_variants.py', 'build_diagnostic.py', 'run_diagnostic.py']
        if experiment.endswith('tile16'):
            helpers += ['profile_streaming16.py', 'profile_selected_replay.py']
            ncu = json.loads((source / 'ncu_native56/metadata.json').read_text())
            assert ncu['complete']
            assert sha(WORK / 'profile_streaming16.py') == ncu['script_sha256']
            assert sha(source / 'ncu_native56/metadata.json') == final['ncu_metadata_sha256']
        dependencies = []
        for helper in helpers:
            original = WORK / helper
            target = dest / 'helper_sources' / helper
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            dependencies.append(dict(original_path=str(original), archive_path=str(target.relative_to(dest)), sha256=sha(target)))
        (dest / 'dependency_sources.json').write_text(json.dumps(dependencies, indent=2) + '\n')
        (dest / 'README.md').write_text((source / 'results.README.md').read_text())
        manifest = [dict(path=str(p.relative_to(dest)), bytes=p.stat().st_size, sha256=sha(p))
                    for p in sorted(dest.rglob('*')) if p.is_file()]
        (dest / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for record in manifest:
            assert sha(dest / record['path']) == record['sha256']
        print(json.dumps(dict(experiment=experiment, files=len(manifest), bytes=sum(r['bytes'] for r in manifest))))
    original = json.loads((DEST / 'SHA256.json').read_text())
    for record in original:
        assert sha(DEST / record['path']) == record['sha256']
    manifest = [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(DEST.rglob('*')) if p.is_file() and p != DEST / 'SHA256.json']
    (DEST / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for record in manifest:
        assert sha(DEST / record['path']) == record['sha256']
    print(json.dumps(dict(total_files=len(manifest), bytes=sum(r['bytes'] for r in manifest), previous_artifacts_unchanged=len(original))))


if __name__ == '__main__':
    main()
