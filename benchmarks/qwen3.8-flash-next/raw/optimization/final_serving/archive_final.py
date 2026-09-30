#!/usr/bin/env python3
"""Archive completed serving/comparison evidence without model or executable copies."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
DEFAULT_RUN = WORK / 'optimized_final'
DEFAULT_DEST = ROOT / 'benchmarks/qwen3.8-flash-next/raw/optimization/final_serving'
MAX_COMPACT_BYTES = 64 * 1024 * 1024
COPY_SUFFIXES = {'.json', '.jsonl', '.py', '.md', '.txt', '.diff', '.patch', '.log', '.csv', '.prom', '.svg', '.png'}
BINARY_SUFFIXES = {'.o', '.a', '.so', '.bin', '.safetensors', '.gguf', '.pt', '.pth', '.cubin', '.nsys-rep', '.sqlite'}
SUPPORT_NAMES = ('README.md', 'self_check.py', 'self_check.json', 'build_candidate.py', 'resume_validation.py', 'prepare_relocation.py', 'baseline.binary.json', 'test_archive_final.py', 'archive_self_check.json')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe_path(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, f'Unsafe manifest path: {name}')
    path = root / relative
    require(not path.is_symlink(), f'Unexpected symlink: {path}')
    require(path.is_file(), f'Missing manifest file: {path}')
    return path


def verify_manifest(root, manifest):
    require(isinstance(manifest, dict) and manifest, f'Empty manifest: {root}')
    for name, digest in manifest.items():
        require(sha(safe_path(root, name)) == digest, f'Hash mismatch: {root / name}')


def completion_gate(run, comparison):
    metadata = load(run / 'metadata.json')
    require(metadata.get('complete') is True, 'Serving run is incomplete; no archiving or result hashing was performed')
    require(metadata.get('binary_sha256') == metadata.get('binary_sha256_after') and metadata.get('binary_sha256'), 'Executable identity changed during run')
    require(metadata.get('profiled') is False, 'Expected ordinary unprofiled serving run')
    shutdown = metadata['server_shutdown']
    require(shutdown.get('forced_kill') is False and shutdown.get('returncode') in (0, -15), 'Server did not shut down cleanly')
    require(not metadata.get('monitor_errors'), 'Run records monitor errors')
    require(metadata['phases'] and all(item.get('complete') is True for item in metadata['phases']), 'Incomplete measurement phase')
    pid = metadata['model_pid']
    stat = Path('/proc') / str(pid) / 'stat'
    try:
        current_start = int(stat.read_text().rsplit(')', 1)[1].split()[19])
    except FileNotFoundError:
        pass
    else:
        original_start = load(run / 'serving.memory.before.json')['model']['starttime_ticks']
        require(current_start != original_start, 'Original server PID is still alive')
    result = load(comparison / 'comparison.json')
    require(result['validation'].get('complete') is True, 'Comparison is incomplete')
    provenance = result['provenance']
    require(provenance['candidate_binary_sha256'] == metadata['binary_sha256'], 'Comparison uses another executable')
    require(Path(provenance['candidate_directory']).resolve() == run.resolve(), 'Comparison uses another run directory')
    require(provenance['candidate_metadata_sha256'] == sha(run / 'metadata.json'), 'Comparison metadata hash mismatch')
    require(provenance['candidate_artifact_manifest_sha256'] == sha(run / 'SHA256SUMS.json'), 'Comparison run-manifest hash mismatch')
    return metadata, result


def omission_reason(path):
    if path.name == 'tokenizer.json':
        return 'Tokenizer omitted; original checkpoint path and SHA256 retained.'
    if path.suffix in BINARY_SUFFIXES:
        return 'Model/executable/native/profile artifact omitted; source path, size and SHA256 retained.'
    with path.open('rb') as source:
        magic = source.read(8)
    if magic.startswith(b'\x7fELF') or magic.startswith(b'!<arch>\n'):
        return 'Executable/native archive omitted; source path, size and SHA256 retained.'
    return None


def collect(root, prefix, authoritative=None):
    included, omitted = [], []
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.is_dir():
            continue
        require(not path.is_symlink() and path.is_file(), f'Unexpected file type: {path}')
        relative = path.relative_to(root)
        size = path.stat().st_size
        recorded = authoritative.get(str(relative)) if authoritative else None
        reason = omission_reason(path)
        digest = recorded or sha(path)
        if reason:
            omitted.append(dict(source=str(path), relative_path=str(Path(prefix) / relative), bytes=size, sha256=digest, reason=reason))
            continue
        require(path.suffix in COPY_SUFFIXES, f'Unreviewed evidence extension: {path}')
        require(size <= MAX_COMPACT_BYTES, f'File exceeds compact archive limit: {path} ({size} bytes)')
        included.append(dict(source=str(path), destination=str(Path(prefix) / relative), bytes=size, sha256=digest))
    return included, omitted


def prepare(run, comparison, postprocess, support):
    metadata, result = completion_gate(run, comparison)
    run_manifest = load(run / 'SHA256SUMS.json')
    comparison_manifest = load(comparison / 'SHA256SUMS.json')
    verify_manifest(run, run_manifest)
    verify_manifest(comparison, comparison_manifest)
    require(sha(postprocess / 'compare_final.py') == result['provenance']['script_sha256'], 'Postprocessor source no longer matches comparison')
    dependencies = result['provenance']['dependencies']
    require(load(postprocess / 'dependencies.json') == dependencies, 'Postprocessor dependency metadata changed')
    verify_manifest(postprocess, {name: item['sha256'] for name, item in dependencies.items()})
    included, omitted = collect(run, 'run', run_manifest)
    entries, skipped = collect(comparison, 'comparison', comparison_manifest)
    included.extend(entries)
    omitted.extend(skipped)
    entries, skipped = collect(postprocess, 'postprocess')
    included.extend(entries)
    omitted.extend(skipped)
    for name in SUPPORT_NAMES:
        path = support / name
        if path.is_file():
            require(path.stat().st_size <= MAX_COMPACT_BYTES, f'Large support file: {path}')
            included.append(dict(source=str(path), destination='runner_support/' + name, bytes=path.stat().st_size, sha256=sha(path)))
    included.append(dict(source=str(Path(__file__).resolve()), destination='archive_final.py', bytes=Path(__file__).stat().st_size, sha256=sha(Path(__file__))))
    omitted.append(dict(source=metadata['binary'], bytes=metadata['binary_bytes'], sha256=metadata['binary_sha256'], reason='Executable omitted; before/after SHA256 comes from completed runner metadata. Archive script does not re-read the executable.'))
    require(len({entry['destination'] for entry in included}) == len(included), 'Duplicate archive destination')
    return included, omitted, metadata, result


def archive(run, comparison, postprocess, support, destination, execute):
    require(not destination.exists(), f'Archive destination already exists: {destination}')
    included, omitted, metadata, result = prepare(run, comparison, postprocess, support)
    report = dict(complete=True, prepared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Completed unprofiled serving run and validated historical-baseline comparison. Compact evidence only; original files remain unchanged.',
        source_run=str(run.resolve()), source_comparison=str(comparison.resolve()), source_postprocessor=str(postprocess.resolve()),
        completion_checks=dict(run_complete=True, comparison_complete=True, clean_server_shutdown=True, original_model_process_exited=True, full_source_manifests_verified=True),
        binary_sha256=metadata['binary_sha256'], comparison_baseline_directory=result['provenance']['baseline_directory'],
        files=[dict(path=entry['destination'], bytes=entry['bytes'], sha256=entry['sha256'], external_source=entry['source']) for entry in included],
        excluded=omitted,
        limits=['Historical baseline artifacts remain in the existing repository archive; they are not duplicated here.',
                'GPU and memory monitor data covers loading, warmups and timed requests; boundary counters include the whole command.',
                'Global swap counters do not establish model-specific paging or throughput impact.',
                'Excluded model/executable data is identified by recorded SHA256 and external path, not copied.',
                'Re-running the postprocessor from this compact archive requires restoring the omitted tokenizer from its recorded checkpoint snapshot and verifying its SHA256.'])
    if not execute:
        print(json.dumps(dict(plan_only=True, destination=str(destination), files=len(included), bytes=sum(entry['bytes'] for entry in included), excluded=omitted), indent=2))
        return report
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.' + destination.name + '.archive-', dir=destination.parent))
    try:
        for entry in included:
            source = Path(entry['source'])
            target = temporary / entry['destination']
            target.parent.mkdir(parents=True, exist_ok=True)
            require(source.stat().st_size == entry['bytes'] and sha(source) == entry['sha256'], f'Source changed before copy: {source}')
            shutil.copy2(source, target)
            require(sha(target) == entry['sha256'], f'Copy checksum mismatch: {target}')
        require(sha(run / 'metadata.json') == result['provenance']['candidate_metadata_sha256'], 'Run changed while archiving')
        (temporary / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        (temporary / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.relative_to(temporary)}\n' for path in sorted(temporary.rglob('*')) if path.is_file() and path.name != 'SHA256SUMS'))
        require(not destination.exists(), 'Destination appeared during archive')
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    print(json.dumps(dict(archived=str(destination), files=len(included), bytes=sum(entry['bytes'] for entry in included), manifest_sha256=sha(destination / 'manifest.json')), indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--comparison', type=Path)
    parser.add_argument('--postprocess', type=Path, default=WORK / 'postprocess')
    parser.add_argument('--support', type=Path, default=WORK)
    parser.add_argument('--destination', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--execute', action='store_true', help='Copy only after all completion and provenance checks pass; otherwise print plan.')
    args = parser.parse_args()
    comparison = args.comparison or args.run.with_name(args.run.name + '.comparison')
    archive(args.run, comparison, args.postprocess, args.support, args.destination, args.execute)


if __name__ == '__main__':
    main()
