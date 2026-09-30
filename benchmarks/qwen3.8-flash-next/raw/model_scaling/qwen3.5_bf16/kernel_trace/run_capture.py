#!/usr/bin/env python3
"""Capture bounded Qwen3.5 BF16 MoE C1/C8 kernel-proof traces after startup; launch only with --released."""

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
BUILD = Path('/home/ericbuehler/qwen4exp_work/verify_graph_depth4_20260930/build')
BINARY = BUILD / 'mistralrs'
BINARY_SHA = '92c050e15bbfe15fc99ef11077366a8fb40db5af96223c67e0f9299fce23f2fa'
MODEL = Path('/home/ericbuehler/hf_models/qwen3.5_35b_a3b')
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
observe.MEMORY_FIELDS.update({'MemFree', 'Buffers', 'Cached', 'SReclaimable', 'Shmem'})


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
    parser.add_argument('--output', type=Path, default=ROOT / 'capture')
    parser.add_argument('--port', type=int, default=1234)
    parser.add_argument('--session', default='qwen35_bf16_kernel_proof_20260930')
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--c1-requests', type=int, default=8)
    parser.add_argument('--c8-requests', type=int, default=8)
    parser.add_argument('--max-tokens', type=int, default=32)
    parser.add_argument('--after-run', action='append', type=Path, default=[],
                        help='Require completed throughput metadata before launch; pass both native and vLLM run directories.')
    parser.add_argument('--restore-editor-record', type=Path,
                        help='Explicitly delegate SIGCONT cleanup for the coordinator-created pause record; the runner never pauses editors.')
    args = parser.parse_args()
    if args.max_tokens < 2 or args.max_tokens > 128 or args.warmups < 1 or any(n < 8 or n % 8 for n in (args.c1_requests, args.c8_requests)):
        parser.error('Use at least one warmup and balanced eight-prompt cycles')
    out = args.output.resolve()
    base_url = f'http://127.0.0.1:{args.port}'
    command = [str(BINARY), 'serve', '--no-ui', '--host', '127.0.0.1', '-p', str(args.port),
               '--disable-access-log', '--max-model-len', '16384', '--pa-context-len', '16384',
               '--max-seqs', '8', '--max-num-batched-tokens', '4096', '--max-prefill-chunk-tokens', '512',
               '--prefix-cache-n', '0', '--pa-cache-type', 'auto', '--dtype', 'bf16', '-m', str(MODEL)]
    launcher_command = [str(NSYS), 'profile', f'--session-new={args.session}', '--start-later=true',
                        '--trace=cuda-sw', '--cuda-graph-trace=node:host-only',
                        '--sample=none', '--cpuctxsw=none', '--kill=none', '--force-overwrite=false',
                        '--output=' + str(out / 'profile_c1'), *command]
    plan = {'binary_sha256': BINARY_SHA, 'launcher_command': launcher_command,
            'phases': [{'concurrency': c, 'requests': n, 'warmups': args.warmups,
                        'profiled_trials': 1, 'output_tokens_per_request': args.max_tokens}
                       for c, n in [(1, args.c1_requests), (8, args.c8_requests)]],
            'disk_free_now_bytes': shutil.disk_usage(ROOT).free,
            'minimum_start_free_bytes': MIN_START_FREE_BYTES,
            'minimum_running_free_bytes': MIN_RUNNING_FREE_BYTES,
            'estimated_trace_pair_and_temporary_budget_bytes': 512 * 1024**2,
            'disk_estimate_basis': 'Only two 8-request x32-output phases by default; CUDA software trace only, no CPU/GPU metrics, no live SQLite export. Keep2GiB starting and1GiB running free-space guards.',
            'editor_cleanup': str(args.restore_editor_record) if args.restore_editor_record else 'Coordinator retains editor restoration responsibility',
            'scope': 'Kernel-presence proof for actual cuTile BF16 MoE at C1/C8, not throughput or bandwidth measurement.',
            'after_runs': [str(path) for path in args.after_run],
            'execution_gate': 'Requires --released and two completed throughput run metadata files; never auto-launches.'}
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
                '--concurrency', str(concurrency), '--requests', str(count), '--max-tokens', str(args.max_tokens)]
        phase = {'name': name, 'profiled': profiled, 'metrics_before': before, 'complete': False}
        metadata['phases'].append(phase)
        run(name, argv)
        after, after_text = metrics(name, 'after')
        phase['metrics_after'] = after
        deltas = counter_summary.counter_deltas(before_text, after_text, require_speculative=False)
        if any(metric.startswith('mistralrs_speculative_') and delta for (metric, _), delta in deltas.items()):
            raise ValueError('Unexpected speculative work in target-only trace')
        counters = {
            'name': name, 'target_only': True,
            'graph_dispatches': [{'labels': dict(labels), 'dispatches': delta}
                                 for (metric, labels), delta in sorted(deltas.items())
                                 if metric == counter_summary.GRAPH_DISPATCH],
            'all_counter_deltas': [{'name': metric, 'labels': dict(labels), 'delta': delta}
                                  for (metric, labels), delta in sorted(deltas.items())],
        }
        counters['scope'] = 'Only this one request phase, between saved /metrics snapshots; warmups are separate phases.'
        save(out / f'{name}.metrics.delta.json', counters)
        artifact = json.loads((out / f'{name}.requests.json').read_text())
        assert artifact['complete'] and artifact['trial']['completed_output_tokens'] == count * args.max_tokens
        phase.update(complete=True, output_tokens=count * args.max_tokens,
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
        if len(args.after_run) < 2:
            raise ValueError('Pass both completed native and vLLM throughput directories with --after-run')
        metadata['preceding_runs'] = []
        for prior in args.after_run:
            prior_path = prior / 'metadata.json'
            prior_metadata = json.loads(prior_path.read_text())
            if not prior_metadata.get('complete'):
                raise ValueError(f'Throughput run not complete: {prior}')
            metadata['preceding_runs'].append({'path': str(prior_path), 'sha256': sha(prior_path)})
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
        env.update(HF_HUB_OFFLINE='1', RUST_LOG='info', MISTRALRS_MOE_BACKEND='cutile',
                   CUTILE_TILEIRAS_PATH='/usr/local/cuda-13.2/bin/tileiras', CUDA_TOOLKIT_PATH='/usr/local/cuda',
                   CUDA_HOME='/usr/local/cuda', LD_LIBRARY_PATH='/usr/local/cuda/compat')
        metadata['environment_overrides'] = {key: env[key] for key in ('HF_HUB_OFFLINE', 'RUST_LOG', 'MISTRALRS_MOE_BACKEND', 'CUTILE_TILEIRAS_PATH', 'CUDA_TOOLKIT_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH')}
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
        log.flush()
        startup = (out / 'server.log').read_text()
        required = ['DType selected is BF16', 'Layers 0-39: cuda[0]', 'PagedAttention KV cache type is BF16',
                    'available context length is 16384 tokens', 'max_num_seqs=8 max_num_batched_tokens=4096',
                    'CUDA decode graphs through batch bucket 8', 'cuTile MoE kernels.']
        metadata['startup_checks'] = {text: text in startup for text in required}
        if not all(metadata['startup_checks'].values()) or 'cuTile MoE warmup failed' in startup:
            raise ValueError('Startup differs from the BF16 throughput baseline')
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
