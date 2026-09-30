#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import time
from profile_selected_replay import NCU, SECTIONS, TEST, command, parse_metrics
from build_diagnostic import WORK, atomic_json, isolate, sha

out = WORK / 'ncu_native56'
metadata_file = out / 'metadata.json'
metadata = json.loads(metadata_file.read_text())
assert not metadata['complete']
assert metadata['error']['type'] == 'CalledProcessError'
assert (out / 'gemv.ncu-rep').is_file() and not (out / 'mmq.ncu-rep').exists()
if not (out / 'metadata.first_import_failure.json').exists():
    shutil.copy2(metadata_file, out / 'metadata.first_import_failure.json')
shutil.copy2(WORK / 'profile_selected_replay.py', out / 'profile_selected_replay.measured.py')
shutil.copy2(__file__, out / 'resume_selected_profile.py')
metadata['resumption'] = {'script_sha256': sha(Path(__file__)), 'reason': 'Remove details-only print-metric-name from raw CSV import; existing successful GEMV capture retained.', 'isolation_before': isolate()}
base = [str(NCU), '--config-file', '0', '--section-folder', str(SECTIONS)]

def run(name, argv, env=None):
    record = {'name': name, 'command': argv, 'started_unix': time.time()}
    metadata['commands'].append(record)
    atomic_json(metadata_file, metadata)
    result = command(argv, out / (name + '.log'), env)
    record.update(returncode=0, finished_unix=time.time(), log_sha256=sha(out / (name + '.log')))
    atomic_json(metadata_file, metadata)
    return result

try:
    gemv = run('gemv_raw_final', [str(NCU), '--import', str(out / 'gemv.ncu-rep'), '--page', 'raw', '--csv', '--print-units', 'base'])
    gemv_kernels = parse_metrics(gemv)
    assert len(gemv_kernels) == 2
    env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(out / 'selected_sample'),
               MISTRALRS_MOE_BENCH_OUTPUT=str(out / 'mmq.profiled_test_output_not_benchmark.json'))
    argv = base + ['--target-processes', 'application-only', '--replay-mode', 'kernel',
        '--cache-control', 'none', '--clock-control', 'none', '--import-sass', 'no',
        '--kernel-name-base', 'function', '--kernel-name', 'regex:^mul_mat_q$',
        '--launch-skip', '12', '--launch-count', '3', '--metrics', ','.join(metadata['metrics_available']),
        '--disable-extra-suffixes', '--export', str(out / 'mmq'), metadata['binary']['path'],
        '--exact', TEST, '--ignored', '--nocapture', '--test-threads=1']
    run('mmq', argv, env)
    mmq = run('mmq_raw', [str(NCU), '--import', str(out / 'mmq.ncu-rep'), '--page', 'raw', '--csv', '--print-units', 'base'])
    mmq_kernels = parse_metrics(mmq)
    assert len(mmq_kernels) == 3
    atomic_json(out / 'kernels.json', {'scope': metadata['scope'], 'limits': metadata['limits'], 'results': [
        {'backend':'gemv','launch_skip':8,'kernels':gemv_kernels}, {'backend':'mmq','launch_skip':12,'kernels':mmq_kernels}]})
    metadata.update(complete=True, finished_unix=time.time(), isolation_after=isolate())
    metadata['initial_import_error'] = metadata.pop('error')
    atomic_json(metadata_file, metadata)
    (out / 'SHA256SUMS').write_text(''.join(f'{sha(item)}  {item.name}\n' for item in sorted(out.iterdir()) if item.is_file() and item.name != 'SHA256SUMS'))
    print('NCU_COMPLETE five kernels collected; GPU free')
except BaseException as error:
    metadata['resumption_error'] = {'type':type(error).__name__,'message':str(error)}
    atomic_json(metadata_file, metadata)
    raise
