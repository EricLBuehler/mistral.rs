#!/usr/bin/env python3
"""Package bounded capture/replay evidence without model weights or executables."""
import hashlib
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parent
DEST = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/real_routing')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not DEST.exists(), 'Preserve previous evidence package'
    for phase in ('replay_adjusted', 'replay_sequence_query_prefix'):
        metadata = json.loads((WORK / phase / 'lifecycle.json').read_text())
        assert metadata['complete'] and metadata['ordinary_replay_complete'] and metadata['oracle_complete']
        assert sha(Path(metadata['replay_binary']['path'])) == metadata['replay_binary']['sha256']
    original = json.loads((WORK / 'runtime/runtime.metadata.json').read_text())
    assert original['complete'] is False and original['phase'] == 'replay' and original['server_pid_gone']
    assert not Path(f"/proc/{original['server_pid']}").exists()
    paths = [p for p in WORK.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.patch', '.md')
             and p.name not in ('profile_selected_replay.py', 'ncu_plan.json')]
    for directory in ('base_tree', 'patch_tree', 'runtime', 'build', 'replay_adjusted', 'replay_sequence_query_prefix'):
        paths.extend(p for p in (WORK / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts
                     and (p.suffix in ('.py', '.json', '.patch', '.md', '.rs', '.log', '.jsonl', '.safetensors', '.txt')
                          or p.name in ('build-complete', 'complete')))
    DEST.mkdir(parents=True)
    for source in sorted(set(paths)):
        target = DEST / source.relative_to(WORK)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    weight = WORK / 'runtime/captures/layer8.gguf'
    external = {'reason': 'Model-derived quantized weights and executables remain outside the repository; exact identity and original local path are retained.',
                'artifacts': [{'path': str(weight), 'bytes': weight.stat().st_size, 'sha256': sha(weight)}]}
    build = json.loads((WORK / 'build/lifecycle.json').read_text())
    for key in ('diagnostic_binary', 'replay_test_binary'):
        if key in build:
            external['artifacts'].append(build[key])
    for phase in ('replay_adjusted', 'replay_sequence_query_prefix'):
        external['artifacts'].append(json.loads((WORK / phase / 'lifecycle.json').read_text())['replay_binary'])
    (DEST / 'external_artifacts.json').write_text(json.dumps(external, indent=2) + '\n')
    (DEST / 'README.md').write_text((WORK / 'results.README.md').read_text())
    manifest = [{'path': str(p.relative_to(DEST)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(DEST.rglob('*')) if p.is_file()]
    (DEST / 'SHA256.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for record in manifest:
        assert sha(DEST / record['path']) == record['sha256']
    print(json.dumps({'destination': str(DEST), 'files': len(manifest), 'bytes': sum(r['bytes'] for r in manifest)}))


if __name__ == '__main__':
    main()
