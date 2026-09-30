#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

WORK = Path(__file__).resolve().parent
PRIOR = Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930')
sys.path.insert(0, str(PRIOR))
from build_diagnostic import atomic_json, isolate
from run_diagnostic import stop_child

TIMEOUT = 1800
QUERY = 'timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu'

def sha(p):
    with open(p, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--released', action='store_true')
    parser.add_argument('--build', type=Path, default=WORK / 'build/provenance.json')
    parser.add_argument('--test', required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--sample')
    parser.add_argument('--results', type=Path)
    parser.add_argument('--archive-dir', type=Path)
    parser.add_argument('--memcheck', action='store_true')
    args = parser.parse_args()
    assert args.released
    out = WORK / args.phase
    out.mkdir(exist_ok=False)
    provenance = json.loads(args.build.read_text())
    assert provenance['complete']
    binary = Path(provenance['binary'])
    assert sha(binary) == provenance['binary_sha256']
    env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(PRIOR / 'runtime/captures'),
               MISTRALRS_MOE_BENCH_OUTPUT=str(out / 'results.json'),
               MISTRALRS_MOE_REPLAY_OUTPUT_DIR=str(out / 'outputs'),
               MISTRALRS_MOE_ORACLE_OUTPUT=str(out / 'oracle.json'))
    for name in ['MISTRALRS_MOE_REPLAY_SAMPLE', 'MISTRALRS_MOE_REPLAY_BASELINE_DIR', 'MISTRALRS_MOE_REPLAY_RESULTS', 'MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE']:
        env.pop(name, None)
    if args.sample:
        env['MISTRALRS_MOE_REPLAY_SAMPLE'] = args.sample
    if args.results:
        env['MISTRALRS_MOE_REPLAY_RESULTS'] = str(args.results.resolve())
    if args.archive_dir:
        env['MISTRALRS_MOE_REPLAY_OUTPUT_DIR'] = str(args.archive_dir.resolve())
    command = [str(binary), '--exact', args.test, '--ignored', '--nocapture', '--test-threads=1']
    if args.memcheck:
        command = ['/usr/local/cuda/bin/compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '42'] + command
    metadata = dict(complete=False, started_at=now(), build_provenance=str(args.build.resolve()),
                    build_provenance_sha256=sha(args.build), binary_sha256=sha(binary),
                    command=command, script_sha256=sha(__file__),
                    source_sha256=provenance['source_sha256'],
                    environment_overrides={key: value for key, value in env.items() if key.startswith('MISTRALRS_MOE_')},
                    scope='Isolated captured layer8 FFN, not whole-model serving throughput.', memcheck=args.memcheck)
    path = out / 'metadata.json'
    def save():
        atomic_json(path, metadata)
    def interrupted(sig, frame):
        raise InterruptedError(sig)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    monitor = child = None
    try:
        metadata['isolation_before'] = isolate()
        save()
        monitor_command = ['nvidia-smi', '--query-gpu=' + QUERY, '--format=csv,nounits', '--loop-ms=1000']
        metadata['monitor'] = dict(command=monitor_command, scope='Whole child, including loading, correctness checks, warmups and timing; not kernel-local counters.')
        with (out / 'gpu_monitor.csv').open('wb') as csv, (out / 'gpu_monitor.stderr.log').open('wb') as err, (out / 'run.log').open('wb') as log:
            monitor = subprocess.Popen(monitor_command, stdout=csv, stderr=err, start_new_session=True)
            child = subprocess.Popen(command, cwd='/home/ericbuehler/mistral.rs', env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            metadata.update(pid=child.pid, launched_at=now())
            save()
            metadata['returncode'] = child.wait(timeout=TIMEOUT)
            metadata['finished_at'] = now()
            metadata['monitor']['returncode_before_stop'] = monitor.poll()
            stop_child(monitor, timeout=5)
        metadata['log_sha256'] = sha(out / 'run.log')
        metadata['monitor']['csv_sha256'] = sha(out / 'gpu_monitor.csv')
        metadata['monitor']['stderr_sha256'] = sha(out / 'gpu_monitor.stderr.log')
        save()
        if metadata['returncode']:
            raise subprocess.CalledProcessError(metadata['returncode'], command)
        if args.memcheck:
            assert 'ERROR SUMMARY: 0 errors' in (out / 'run.log').read_text()
        artifacts = [p for p in out.rglob('*') if p.is_file() and p != path]
        metadata['artifact_sha256'] = {str(p.relative_to(out)): sha(p) for p in artifacts}
        metadata.update(complete=True, isolation_after=isolate())
        save()
        print(str(out), flush=True)
    except BaseException as exc:
        if child and child.poll() is None:
            stop_child(child)
        if monitor and monitor.poll() is None:
            stop_child(monitor, timeout=5)
        metadata.update(error=repr(exc), failed_at=now(), returncode=child.poll() if child else None)
        save()
        raise

if __name__ == '__main__':
    main()
