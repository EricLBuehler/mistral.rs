#!/usr/bin/env python3
"""Capture bounded current-model C1/C8 traces after startup; launch only with --released."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from clock_anchor import anchor

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/ericbuehler/mistral.rs')
BUILD = Path('/home/ericbuehler/qwen4exp_work/moe_optimization_20260930/final_serving/build')
BINARY = BUILD / 'mistralrs'
BINARY_SHA = 'd2c85e943f4eafbfc2a9515339671f0fd3eb59e2684ad20dece0552ac855487d'
MODEL = Path('/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540')
NSYS = Path('/opt/nvidia/nsight-systems/2026.1.3/bin/nsys')
MIN_START_FREE_BYTES = 2 * 1024**3
MIN_RUNNING_FREE_BYTES = 1024**3
STARTUP_TIMEOUT_SECONDS = 2400
COMMAND_TIMEOUT_SECONDS = 900
STOP_TIMEOUT_SECONDS = 30
MONITOR_SECONDS = 2
EDITOR_PREFIXES = ('rust-analyzer', 'cpptools')
sys.path.insert(0, str(ROOT / 'dependencies'))
import counter_summary  # noqa: E402
import observe  # noqa: E402


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def same_process(record):
    try:
        now = observe.process_record(record['pid'])
    except (FileNotFoundError, ProcessLookupError):
        return False
    return (now['starttime_ticks'] == record['starttime_ticks'] and now['comm'] == record['comm']
            and now['state'] not in ('Z', 'X'))


def restore_editors(path):
    result = []
    for entry in json.loads(path.read_text()):
        assert entry['comm'].startswith(EDITOR_PREFIXES)
        record = {'pid': entry['pid'], 'comm': entry['comm'],
                  'starttime_ticks': int(entry.get('starttime', entry.get('starttime_ticks')))}
        row = {**record, 'restored': False}
        if entry.get('prior_state') in ('T', 't'):
            row['reason'] = 'Already stopped before the coordinator pause'
        elif not same_process(record):
            row['reason'] = 'Original process exited or identity changed'
        else:
            os.kill(record['pid'], signal.SIGCONT)
            row['restored'] = True
        result.append(row)
    return result


def terminate_group(child):
    if child is None or child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'current_capture')
    parser.add_argument('--port', type=int, default=1234)
    parser.add_argument('--session', default='qwen_model_scaling_20260930')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--c1-requests', type=int, default=8)
    parser.add_argument('--c8-requests', type=int, default=24)
    parser.add_argument('--restore-editor-record', type=Path,
                        help='Explicitly delegate SIGCONT cleanup for the coordinator-created pause record; the runner never pauses editors.')
    args = parser.parse_args()
    if args.warmups < 1 or any(n < 8 or n % 8 for n in (args.c1_requests, args.c8_requests)):
        parser.error('Use at least one warmup and balanced eight-prompt cycles')
    out = args.output.resolve()
    base_url = f'http://127.0.0.1:{args.port}'
    command = [str(BINARY), 'serve', '--no-ui', '-p', str(args.port), '--max-model-len', '16384',
               '--max-seqs', '8', '--prefix-cache-n', '0', '-m', str(MODEL), '--isq', 'q4k', '--mtp']
    launcher_command = [str(NSYS), 'profile', f'--session-new={args.session}', '--start-later=true',
                        '--trace=cuda-sw,nvtx,osrt', '--cuda-graph-trace=node:host-only',
                        '--sample=none', '--cpuctxsw=none', '--kill=none', '--force-overwrite=false',
                        '--output=' + str(out / 'profile_c1'), *command]
    plan = {'binary_sha256': BINARY_SHA, 'launcher_command': launcher_command,
            'phases': [{'concurrency': c, 'requests': n, 'warmups': args.warmups,
                        'profiled_trials': 1, 'output_tokens_per_request': 128}
                       for c, n in [(1, args.c1_requests), (8, args.c8_requests)]],
            'disk_free_now_bytes': shutil.disk_usage(ROOT).free,
            'minimum_start_free_bytes': MIN_START_FREE_BYTES,
            'minimum_running_free_bytes': MIN_RUNNING_FREE_BYTES,
            'estimated_trace_pair_and_temporary_budget_bytes': 1024**3,
            'disk_estimate_basis': 'Historical C8 report was 46.5 MB, C1/C8 SQLite exports 110/121 MB; reserve 1 GiB for two short traces and temporary conversion. No live SQLite export.',
            'editor_cleanup': str(args.restore_editor_record) if args.restore_editor_record else 'Coordinator retains editor restoration responsibility',
            'scope': 'Instrumented diagnostic traces, not official throughput benchmarks; no full-model bandwidth inference.'}
    if not args.released:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'complete': False, 'clock_started': anchor(), 'plan': plan, 'commands': [], 'phases': [],
                'runner_sha256': sha(Path(__file__)), 'cleanup': {}}
    metadata_path = out / 'metadata.json'
    launcher = None
    model = None
    session_created = False
    capture_active = False
    stop_monitor = threading.Event()
    monitor_error = []
    monitor_thread = None
    log = None

    def persist():
        save(metadata_path, metadata)

    def fail_if_disk_low():
        free = shutil.disk_usage(out).free
        if free < MIN_RUNNING_FREE_BYTES:
            raise RuntimeError(f'Free space below 1 GiB guard: {free} bytes')
        if monitor_error:
            raise RuntimeError(f'Monitor failed: {monitor_error}')

    def run(name, argv, *, cleanup=False):
        record = {'name': name, 'argv': argv, 'clock_before': anchor()}
        metadata['commands'].append(record)
        persist()
        child = None
        try:
            with (out / f'{name}.log').open('w') as output:
                child = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + COMMAND_TIMEOUT_SECONDS
                while child.poll() is None:
                    if not cleanup:
                        fail_if_disk_low()
                    if time.monotonic() > deadline:
                        raise TimeoutError(name)
                    time.sleep(0.25)
            record['returncode'] = child.returncode
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, argv)
        finally:
            terminate_group(child)
            record['clock_after'] = anchor()
            persist()

    def monitor():
        try:
            with (out / 'memory.jsonl').open('w') as output:
                while not stop_monitor.is_set():
                    row = observe.memory_snapshot(model['pid'] if model else None)
                    row.update(clock=anchor(), free_disk_bytes=shutil.disk_usage(out).free)
                    output.write(json.dumps(row) + '\n')
                    output.flush()
                    if row['free_disk_bytes'] < MIN_RUNNING_FREE_BYTES:
                        raise RuntimeError('Free space below 1 GiB')
                    stop_monitor.wait(MONITOR_SECONDS)
        except BaseException as error:
            monitor_error.append(repr(error))

    def metrics(name, when):
        before = anchor()
        with urllib.request.urlopen(base_url + '/metrics', timeout=10) as response:
            data = response.read()
        path = out / f'{name}.metrics.{when}.prom'
        path.write_bytes(data)
        return {'path': path.name, 'clock_before': before, 'clock_after': anchor(),
                'memory': observe.memory_snapshot(model['pid'])}, data.decode()

    def requests(name, concurrency, count, profiled):
        before, before_text = metrics(name, 'before')
        argv = [sys.executable, str(ROOT / 'profile_requests.py'), str(out / f'{name}.requests.json'),
                '--base-url', base_url, '--tokenizer', str(MODEL / 'tokenizer.json'),
                '--concurrency', str(concurrency), '--requests', str(count)]
        phase = {'name': name, 'profiled': profiled, 'metrics_before': before, 'complete': False}
        metadata['phases'].append(phase)
        run(name, argv)
        after, after_text = metrics(name, 'after')
        phase['metrics_after'] = after
        counters = counter_summary.summarize(name, counter_summary.counter_deltas(before_text, after_text))
        counters['scope'] = 'Only this one request phase, between saved /metrics snapshots; warmups are separate phases.'
        save(out / f'{name}.metrics.delta.json', counters)
        artifact = json.loads((out / f'{name}.requests.json').read_text())
        assert artifact['complete'] and artifact['trial']['completed_output_tokens'] == count * 128
        phase.update(complete=True, output_tokens=count * 128,
                     request_window_perf_counter_ns=artifact['request_window_perf_counter_ns'])
        persist()

    def interrupted(signum, _frame):
        raise InterruptedError(f'Signal {signum}')

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        persist()
        if shutil.disk_usage(out).free < MIN_START_FREE_BYTES:
            raise RuntimeError('Need at least 2 GiB free before launch')
        isolation = observe.isolation_snapshot()
        assert not isolation['unexpected_active'], isolation['unexpected_active']
        metadata['isolation_before'] = isolation
        if args.restore_editor_record:
            metadata['editor_pause_record'] = json.loads(args.restore_editor_record.read_text())
            shutil.copy2(args.restore_editor_record, out / 'editor_pause_record.json')
            for editor in metadata['editor_pause_record']:
                actual = observe.process_record(editor['pid'])
                assert actual['comm'] == editor['comm'] and actual['starttime_ticks'] == int(editor['starttime'])
                assert actual['state'] in ('T', 't'), actual
        if sha(BINARY) != BINARY_SHA:
            raise ValueError('Frozen server binary changed')
        metadata['build_metadata'] = json.loads((BUILD / 'metadata.json').read_text())
        metadata['checkpoint_metadata_sha256'] = {name: sha(MODEL / name) for name in
                                                 ('config.json', 'tokenizer.json', 'model.safetensors.index.json')}
        metadata['dependencies'] = json.loads((ROOT / 'dependencies.json').read_text())
        for name, entry in metadata['dependencies'].items():
            assert sha(ROOT / 'dependencies' / name) == entry['sha256'], name
        metadata['support_sha256'] = {name: sha(ROOT / name) for name in ('profile_requests.py', 'clock_anchor.py', 'exact_timestamps.patch')}
        try:
            urllib.request.urlopen(base_url + '/v1/models', timeout=2).close()
        except OSError:
            pass
        else:
            raise RuntimeError('A server already answers on the selected port')
        run('nsys_version', [str(NSYS), '--version'])
        env = {key: value for key, value in os.environ.items() if not key.startswith('MISTRALRS_')}
        env.update(HF_HUB_OFFLINE='1', RUST_LOG='info')
        metadata['environment_overrides'] = {'HF_HUB_OFFLINE': '1', 'RUST_LOG': 'info'}
        metadata['removed_mistralrs_environment_keys'] = sorted(key for key in os.environ if key.startswith('MISTRALRS_'))
        metadata['inherited_execution_environment'] = {key: value for key, value in env.items()
                                                     if key.startswith(('CUDA_', 'NVIDIA_', 'CUBLAS_')) or key == 'LD_LIBRARY_PATH'}
        log = (out / 'server.log').open('w')
        launcher = subprocess.Popen(launcher_command, cwd=REPO, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        session_created = True
        metadata['launcher_pid'] = launcher.pid
        metadata['clock_server_launched'] = anchor()
        persist()
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while True:
            fail_if_disk_low()
            if launcher.poll() is not None:
                raise RuntimeError(f'Profiler launcher exited during startup: {launcher.returncode}')
            candidates = []
            for entry in Path('/proc').iterdir():
                if not entry.name.isdecimal():
                    continue
                try:
                    if (entry / 'cmdline').read_bytes().split(b'\0')[:-1] == [arg.encode() for arg in command]:
                        candidates.append(observe.process_record(int(entry.name)))
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            if len(candidates) == 1:
                model = candidates[0]
                metadata['model_process'] = model
            try:
                with urllib.request.urlopen(base_url + '/v1/models', timeout=2) as response:
                    if response.status == 200 and model:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Model startup exceeded 2400 seconds')
            stop_monitor.wait(2)
        metadata['clock_ready'] = anchor()
        persist()
        print('Model ready; starting unrecorded warmups.', flush=True)
        for concurrency, count in [(1, args.c1_requests), (8, args.c8_requests)]:
            for warmup in range(args.warmups):
                requests(f'c{concurrency}.warmup{warmup}', concurrency, count, False)
            name = f'profile_c{concurrency}'
            run(name + '.start', [str(NSYS), 'start', f'--session={args.session}',
                                 '--sample=none', '--cpuctxsw=none', '--output=' + str(out / name)])
            capture_active = True
            requests(name, concurrency, count, True)
            run(name + '.stop', [str(NSYS), 'stop', f'--session={args.session}'])
            capture_active = False
            assert (out / f'{name}.nsys-rep').is_file(), f'Missing named {name} report'
            metadata.setdefault('reports', []).append({'name': name, 'bytes': (out / f'{name}.nsys-rep').stat().st_size})
            persist()
        metadata['measurements_complete'] = True
    except BaseException as error:
        metadata['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        cleanup_errors = []
        if session_created:
            if capture_active:
                try:
                    run('cleanup_stop', [str(NSYS), 'stop', f'--session={args.session}'], cleanup=True)
                except BaseException as error:
                    cleanup_errors.append(repr(error))
            try:
                run('profiler_shutdown', [str(NSYS), 'shutdown', f'--session={args.session}', '--kill=none'], cleanup=True)
            except BaseException as error:
                cleanup_errors.append(repr(error))
        if model and same_process(model):
            os.kill(model['pid'], signal.SIGTERM)
            deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
            while same_process(model) and time.monotonic() < deadline:
                time.sleep(0.25)
            if same_process(model):
                os.kill(model['pid'], signal.SIGKILL)
                metadata['cleanup']['model_forced_kill'] = True
        terminate_group(launcher)
        stop_monitor.set()
        if monitor_thread:
            monitor_thread.join(timeout=5)
        if log:
            log.close()
        metadata['cleanup']['model_exited'] = model is None or not same_process(model)
        if args.restore_editor_record:
            try:
                metadata['cleanup']['editor_restoration'] = restore_editors(args.restore_editor_record)
            except BaseException as error:
                cleanup_errors.append(repr(error))
        metadata['cleanup']['errors'] = cleanup_errors
        metadata['clock_finished'] = anchor()
        metadata['disk_free_finished_bytes'] = shutil.disk_usage(out).free
        metadata['complete'] = bool(metadata.get('measurements_complete') and not cleanup_errors
                                    and metadata['cleanup']['model_exited'] and not monitor_error)
        persist()
        files = sorted(path for path in out.iterdir() if path.is_file())
        save(out / 'manifest.json', {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size} for path in files})
        print(json.dumps({'complete': metadata['complete'], 'output': str(out), 'cleanup_errors': cleanup_errors}), flush=True)


if __name__ == '__main__':
    main()
