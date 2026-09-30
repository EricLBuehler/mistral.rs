#!/usr/bin/env python3
"""Archive completed L2-refill evidence without following weight/binary symlinks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DEFAULT_DESTINATION = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/optimization/headroom/l2_refill')


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    run = ROOT / 'masked_mmq'
    metadata = json.loads((run / 'metadata.json').read_text())
    assert metadata['complete']
    original_manifest = {}
    for line in (run / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(run / name) == digest, name
        original_manifest[name] = digest
    source_files = {Path(name): ROOT / name for name in (
        'profile_l2_refill.py', 'summarize_refills.py', 'archive_l2_refill.py',
        'README.md', 'plan.json', 'self_check.json', 'summary.json')}
    for name in [*original_manifest, 'SHA256SUMS']:
        source_files[Path('masked_mmq') / name] = run / name
    sample = Path(metadata['sample']['path'])
    assert sha(sample) == metadata['sample']['sha256']
    source_files[Path('capture') / sample.name] = sample
    args.destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        'scope': 'Exact compact L2-refill profiling evidence; no weights or binaries copied.',
        'source_root': str(ROOT),
        'original_manifest_verified': True,
        'omissions': ['Large input GGUF weights and capture tensor symlinks remain referenced by original metadata.',
                      'Frozen replay executable is identified by SHA256 and build metadata.',
                      'The 89 MB metric-discovery log is referenced by SHA256; relevant lines are included in plan.json.'],
        'files': {},
    }
    for target, source in source_files.items():
        assert source.is_file() and not source.is_symlink(), source
        destination = args.destination / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = sha(source)
        assert sha(destination) == digest
        provenance['files'][str(target)] = {'source_path': str(source), 'sha256': digest,
                                            'bytes': destination.stat().st_size}
    (args.destination / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    files = sorted(path for path in args.destination.rglob('*') if path.is_file())
    manifest = {str(path.relative_to(args.destination)): sha(path) for path in files}
    (args.destination / 'SHA256SUMS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name, digest in manifest.items():
        assert sha(args.destination / name) == digest
    print(json.dumps({'complete': True, 'destination': str(args.destination),
                      'hashed_files': len(manifest), 'bytes': sum(path.stat().st_size for path in files)}))


if __name__ == '__main__':
    main()
