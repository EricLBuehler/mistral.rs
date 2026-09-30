#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parent
DEST = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/real_routing/tile_variants')
PHASES = ('baseline', 'patched_unset', 'monitored_baseline', 'monitored_unset', 'monitored_16', 'monitored_32')
SUFFIXES = ('.py', '.json', '.patch', '.cu', '.rs', '.txt', '.log', '.csv')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    final = json.loads((WORK / 'final_provenance.json').read_text())
    assert final['check_passed'] and all(final['runtime_sources_and_production_cli_unchanged'].values())
    assert sha(WORK / 'summary.json') == final['summary_sha256']
    paths = [p for p in WORK.iterdir() if p.is_file() and p.suffix in SUFFIXES]
    for name in (*PHASES, 'build'):
        paths += [p for p in (WORK / name).rglob('*') if p.is_file() and p.suffix in SUFFIXES
                  and 'outputs' not in p.relative_to(WORK).parts and 'shadow' not in p.relative_to(WORK).parts]
    external = []
    for phase in PHASES:
        metadata = json.loads((WORK / phase / 'lifecycle.json').read_text())
        assert metadata['complete']
        assert sha(WORK / phase / 'replay.json') == metadata['replay_sha256']
        external.append(dict(phase=phase, binary=metadata['binary'],
                             saved_outputs_directory=str(WORK / phase),
                             saved_outputs=metadata['saved_outputs']))
    DEST.mkdir(parents=True)
    for path in sorted(set(paths)):
        target = DEST / path.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (DEST / 'external_artifacts.json').write_text(json.dumps(dict(
        reason='Executables, native archives and tensor outputs remain external; copied lifecycle/build provenance records exact hashes and paths.',
        phases=external), indent=2) + '\n')
    (DEST / 'README.md').write_text((WORK / 'results.README.md').read_text())
    manifest = [dict(path=str(p.relative_to(DEST)), bytes=p.stat().st_size, sha256=sha(p))
                for p in sorted(DEST.rglob('*')) if p.is_file()]
    (DEST / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for record in manifest:
        assert sha(DEST / record['path']) == record['sha256']
    print(json.dumps(dict(files=len(manifest), bytes=sum(r['bytes'] for r in manifest), destination=str(DEST))))


if __name__ == '__main__':
    main()
