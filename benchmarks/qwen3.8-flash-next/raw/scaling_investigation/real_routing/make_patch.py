#!/usr/bin/env python3
import difflib
import hashlib
import json
from pathlib import Path

root = Path('/home/ericbuehler/mistral.rs')
work = Path(__file__).resolve().parent
patches = {'capture.patch': [], 'replay.patch': []}
manifest = {}
for revised in sorted((work / 'patch_tree').rglob('*.rs')):
    relative = revised.relative_to(work / 'patch_tree')
    original = root / relative
    before = original.read_text() if original.exists() else ''
    baseline = work / 'base_tree' / relative
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(before)
    after = revised.read_text()
    manifest[str(relative)] = {
        'before_sha256': hashlib.sha256(before.encode()).hexdigest() if original.exists() else None,
        'after_sha256': hashlib.sha256(after.encode()).hexdigest(),
    }
    name = 'replay.patch' if str(relative).startswith('mistralrs-quant/') else 'capture.patch'
    patches[name].extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
        fromfile='a/' + str(relative) if original.exists() else '/dev/null', tofile='b/' + str(relative)))
for name, diff in patches.items():
    (work / name).write_text(''.join(diff))
(work / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
