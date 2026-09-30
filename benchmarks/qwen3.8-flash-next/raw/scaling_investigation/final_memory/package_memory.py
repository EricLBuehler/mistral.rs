#!/usr/bin/env python3
"""Package saved benchmark memory boundaries without querying the server."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil


SOURCE = Path(__file__).resolve().parent
DEST = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/final_memory')
PAGE_BYTES = os.sysconf('SC_PAGE_SIZE')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    orchestration = json.loads((SOURCE / 'orchestration.metadata.json').read_text())
    assert orchestration['complete'] is True
    assert (SOURCE / 'measurements-complete').exists()
    commands = json.loads((SOURCE / 'measurement_commands.json').read_text())
    assert len(commands) == 6 and all(command['returncode'] == 0 for command in commands)
    rows = []
    paths = [SOURCE / 'startup_ready.swap.baseline.json']
    for command in commands:
        name = command['name']
        pair = {}
        for phase in ('before', 'after'):
            path = SOURCE / f'{name}.swap.{phase}.json'
            paths.append(path)
            pair[phase] = json.loads(path.read_text())
        before, after = pair['before'], pair['after']
        assert before['model_pid'] == after['model_pid'] == orchestration['model_pid']
        assert before['captured_monotonic'] <= command['started_monotonic']
        assert command['finished_monotonic'] <= after['captured_monotonic']
        swap = {}
        for field in ('pswpin_pages', 'pswpout_pages'):
            delta = after[field] - before[field]
            assert delta >= 0
            swap[field + '_delta'] = delta
            swap[field.replace('_pages', '_MiB') + '_delta'] = delta * PAGE_BYTES / (1024 ** 2)
        rows.append({
            'name': name,
            'command_started_utc': command['started_utc'],
            'command_finished_utc': command['finished_utc'],
            'command_seconds': command['finished_monotonic'] - command['started_monotonic'],
            'snapshot_window_seconds': after['captured_monotonic'] - before['captured_monotonic'],
            'system_swap': swap,
            'before_process': before['process'],
            'after_process': after['process'],
            'before_system': before['system'],
            'after_system': after['system'],
        })
    extra = ['measurement_commands.json', 'measurement_script.provenance.json',
             'pre_measurement_provenance.json', 'editor_retry_cleanup.json', 'package_memory.py']
    paths.extend(SOURCE / name for name in extra)
    assert not DEST.exists(), f'Preserve existing package: {DEST}'
    DEST.mkdir(parents=True)
    provenance = {}
    for path in paths:
        shutil.copy2(path, DEST / path.name)
        provenance[path.name] = {'source': str(path), 'sha256': sha(path)}
        assert sha(DEST / path.name) == provenance[path.name]['sha256']
    summary = {
        'packaged_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_directory': str(SOURCE),
        'model_pid': orchestration['model_pid'],
        'binary_sha256': orchestration['binary_sha256'],
        'system_page_bytes': PAGE_BYTES,
        'startup_configuration_checks': orchestration['startup_configuration_checks'],
        'startup_memory_line': orchestration['startup_memory_line'],
        'startup_ready': json.loads(paths[0].read_text()),
        'limitations': [
            'Each pair brackets a whole benchmark subprocess, including warmups and all trials, not individual timed requests.',
            'pswpin and pswpout are system-wide counters; they do not identify which process paged or when within the command.',
            'Before/after process VmSwap is swapped virtual memory at each boundary, not a count of process page faults.',
            'Process VmRSS does not account for all CUDA allocations on this unified-memory device.',
            'The C6/C8 command combines both concurrencies; its swap counters cannot be split between them.',
            'No additional memory polling or server requests were inserted into timed request intervals by this script.',
        ],
        'subprocesses': rows,
        'files': provenance,
    }
    (DEST / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (SOURCE / 'memory_boundary_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    manifest = ''.join(f'{sha(path)}  {path.name}\n' for path in sorted(DEST.iterdir()))
    (DEST / 'SHA256SUMS').write_text(manifest)
    print(json.dumps({'destination': str(DEST), 'subprocesses': len(rows), 'files': len(paths),
                      'system_swap_deltas': [{ 'name': row['name'], **row['system_swap']} for row in rows]}, indent=2))


if __name__ == '__main__':
    main()
