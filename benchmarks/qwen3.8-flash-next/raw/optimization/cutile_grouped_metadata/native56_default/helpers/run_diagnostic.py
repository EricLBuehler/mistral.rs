#!/usr/bin/env python3
"""Own one capture server, stop it, then replay saved expert tensors exclusively."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from build_diagnostic import EXPECTED_PRODUCTION_SHA256, REPO, REPLAY_SOURCE, WORK, atomic_json, isolate, sha

SNAPSHOT = Path('/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540')
STARTUP_SECONDS = 2400
CAPTURE_SECONDS = 1800
VALIDATE_SECONDS = 300
REPLAY_SECONDS = 1800
STOP_SECONDS = 30
POLL_SECONDS = 2
MIN_CAPTURE_FREE_BYTES = 2 * 1024**3
TEST_NAME = 'flash_next_real_routing_replay'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stop_child(child, timeout=STOP_SECONDS):
    if child is None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
    assert not Path(f'/proc/{child.pid}').exists(), f'Owned child PID still exists: {child.pid}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true', required=True)
    parser.add_argument('--build-dir', type=Path, default=WORK / 'build')
    parser.add_argument('--output-dir', type=Path, default=WORK / 'runtime')
    parser.add_argument('--port', type=int, default=1234)
    args = parser.parse_args()
    build_dir = args.build_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_file = out / 'runtime.metadata.json'
    assert not metadata_file.exists(), 'Preserve prior diagnostic run; use a new output directory'
    captures = out / 'captures'
    assert not captures.exists(), 'Capture directory must be fresh'
    captures.mkdir()
    scripts = ['run_diagnostic.py', 'build_diagnostic.py', 'capture_requests.py', 'control_capture.py',
               'validate_capture.py', 'summarize_replay.py', 'capture.patch', 'replay.patch',
               'source_manifest.json', 'README.md', 'test_request_runner.py', 'request_runner.selftest.json']
    report = dict(started_at=now(), complete=False, phase='preflight', commands=[],
                  scope='Diagnostic capture and isolated kernel replay; no end-to-end throughput claim',
                  scripts_sha256={name: sha(WORK / name) for name in scripts})
    server = None
    server_log = None
    memory_thread = None
    memory_stop = threading.Event()
    managed_artifacts = set()

    def save():
        atomic_json(metadata_file, report)

    def phase(name):
        report['phase'] = name
        save()
        print(f'{now()} PHASE {name}', flush=True)

    def interrupted(signum, _frame):
        raise InterruptedError(f'Diagnostic interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def child_command(name, command, timeout, env=None):
        log_path = out / f'{name}.log'
        managed_artifacts.add(log_path)
        record = dict(name=name, command=command, started_at=now(), log=log_path.name)
        report['commands'].append(record)
        save()
        with log_path.open('wb') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            save()
            try:
                child.wait(timeout=timeout)
            except BaseException:
                stop_child(child)
                raise
            finally:
                record.update(finished_at=now(), returncode=child.poll())
                save()
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
        return child.returncode

    def stop_server():
        nonlocal server_log
        if server is not None:
            stop_child(server)
            report['server_returncode'] = server.returncode
            report['server_stopped_at'] = now()
            report['server_pid_gone'] = not Path(f'/proc/{server.pid}').exists()
        memory_stop.set()
        if memory_thread is not None:
            memory_thread.join(timeout=STOP_SECONDS)
            assert not memory_thread.is_alive(), 'Memory monitor did not stop'
        if server_log is not None:
            server_log.close()
            server_log = None
        save()

    def monitor():
        with (out / 'server.memory.jsonl').open('w') as handle:
            while not memory_stop.is_set():
                row = dict(at=now())
                for line in Path('/proc/meminfo').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in ('MemAvailable', 'SwapFree'):
                        row[key + '_KiB'] = int(value.split()[0])
                vmstat = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
                row.update(pswpin_pages=int(vmstat['pswpin']), pswpout_pages=int(vmstat['pswpout']))
                if server is not None:
                    try:
                        for line in Path(f'/proc/{server.pid}/status').read_text().splitlines():
                            key, _, value = line.partition(':')
                            if key in ('VmRSS', 'VmSwap', 'VmHWM'):
                                row[key + '_KiB'] = int(value.split()[0])
                    except FileNotFoundError:
                        pass
                handle.write(json.dumps(row) + '\n')
                handle.flush()
                memory_stop.wait(POLL_SECONDS)

    try:
        build = json.loads((build_dir / 'lifecycle.json').read_text())
        assert build['complete'] is True and (build_dir / 'build-complete').exists()
        assert not build['restoration_errors'] and build['replay_extension_retained']
        source_manifest = json.loads((WORK / 'source_manifest.json').read_text())
        for name, expected in source_manifest.items():
            path = REPO / name
            required = expected['after_sha256'] if name == REPLAY_SOURCE else expected['before_sha256']
            assert (sha(path) if path.exists() else None) == required, f'Unexpected live source: {name}'
            assert sha(WORK / 'patch_tree' / name) == expected['after_sha256']
        cli = Path(build['diagnostic_binary']['path'])
        replay = Path(build['replay_test_binary']['path'])
        assert sha(cli) == build['diagnostic_binary']['sha256']
        assert sha(replay) == build['replay_test_binary']['sha256']
        assert sha(REPO / 'target/release/mistralrs') == EXPECTED_PRODUCTION_SHA256
        assert shutil.disk_usage(out).free >= MIN_CAPTURE_FREE_BYTES, 'Insufficient capture storage'
        report['isolation_before_server'] = isolate()
        with socket.socket() as connection:
            connection.settimeout(0.3)
            assert connection.connect_ex(('127.0.0.1', args.port)) != 0, 'Capture port already in use'
        index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())
        assert all((SNAPSHOT / filename).is_file() for filename in set(index['weight_map'].values()))
        report.update(
            build_metadata_sha256=sha(build_dir / 'lifecycle.json'),
            source_manifest_sha256=sha(WORK / 'source_manifest.json'),
            diagnostic_binary=dict(path=str(cli), sha256=sha(cli)),
            replay_binary=dict(path=str(replay), sha256=sha(replay)),
            snapshot=str(SNAPSHOT),
            checkpoint_metadata_sha256={name: sha(SNAPSHOT / name) for name in
                ['config.json', 'model.safetensors.index.json', 'tokenizer.json']},
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
            memory_scope='Global swap counters and process RSS across startup/capture only; not a kernel bandwidth measurement',
        )
        provenance = out / 'provenance'
        provenance.mkdir()
        for name in scripts:
            destination = provenance / name
            shutil.copy2(WORK / name, destination)
            managed_artifacts.add(destination)
        serving_source = REPO / 'benchmarks/qwen3.8-flash-next/bench_serving.py'
        serving_copy = provenance / 'bench_serving.py'
        shutil.copy2(serving_source, serving_copy)
        managed_artifacts.add(serving_copy)
        report['prompts_source_sha256'] = sha(serving_source)
        for name in ['lifecycle.json', 'source_before.diff']:
            destination = provenance / f'build.{name}'
            shutil.copy2(build_dir / name, destination)
            managed_artifacts.add(destination)
        for name in source_manifest:
            for tree in ['base_tree', 'patch_tree']:
                destination = provenance / tree / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(WORK / tree / name, destination)
                managed_artifacts.add(destination)
        command = [str(cli), 'serve', '--no-ui', '--host', '127.0.0.1', '-p', str(args.port),
                   '--max-model-len', '16384', '--max-seqs', '8', '--prefix-cache-n', '0',
                   '-m', str(SNAPSHOT), '--isq', 'q4k', '--mtp', '--mtp-n-predict', '6']
        env = dict(os.environ)
        overrides = dict(HF_HUB_OFFLINE='1', RUST_LOG='info', MISTRALRS_CUDA_GRAPHS='0',
                         MISTRALRS_MOE_CAPTURE_DIR=str(captures))
        env.update(overrides)
        assert env.get('CUDA_LAUNCH_BLOCKING') in (None, '0'), 'Debug CUDA synchronization enabled'
        report.update(server_command=command, environment_overrides=overrides,
                      performance_environment={name: env.get(name) for name in
                          ['CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'MISTRALRS_GGUF_AFFINE_BACKEND',
                           'MISTRALRS_IGPU_MEMORY_FRACTION', 'OMP_NUM_THREADS', 'RAYON_NUM_THREADS']})
        phase('loading')
        server_log = (out / 'server.log').open('wb')
        managed_artifacts.update([out / 'server.log', out / 'server.memory.jsonl'])
        started = time.monotonic()
        server = subprocess.Popen(command, cwd=REPO, env=env, stdout=server_log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        report['server_pid'] = server.pid
        save()
        memory_thread = threading.Thread(target=monitor, daemon=True)
        memory_thread.start()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Capture server exited during startup: {server.returncode}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v1/models', timeout=2) as response:
                    if response.status == 200:
                        report['models_response'] = json.load(response)
                        break
            except OSError:
                pass
            if time.monotonic() - started > STARTUP_SECONDS:
                raise TimeoutError(f'Capture startup exceeded {STARTUP_SECONDS}s')
            time.sleep(POLL_SECONDS)
        report['startup_seconds'] = time.monotonic() - started
        expected_cmdline = b'\0'.join(part.encode() for part in command) + b'\0'
        assert Path(f'/proc/{server.pid}/cmdline').read_bytes() == expected_cmdline
        report['loaded_executable_sha256'] = sha(Path(f'/proc/{server.pid}/exe'))
        assert report['loaded_executable_sha256'] == build['diagnostic_binary']['sha256']
        startup = (out / 'server.log').read_text()
        patterns = {
            'all_48_layers_cuda0': r'Layers 0-47: cuda\[0\]',
            'ple_q4': r'PLE table resident in device memory format=Q4(?: |\n)',
            'kv_bf16': r'PagedAttention KV cache type is BF16',
            'kv_513_blocks': r'block size 32 and 513 GPU blocks: available context length is 16384 tokens',
            'mtp_six': r'Speculative decoding enabled: MTP assistant `built-in` with n_predict=6',
            'max_sequences_eight': r'Configured PagedAttention scheduler max_num_seqs=8(?: |\n)',
        }
        checks = {name: bool(re.search(pattern, startup)) for name, pattern in patterns.items()}
        checks['no_decode_graph_capture_log'] = re.search(r'Captured [1-9][0-9]* CUDA decode graphs', startup) is None
        report['startup_configuration_checks'] = checks
        report['startup_configuration_lines'] = [line for line in startup.splitlines()
            if any(re.search(pattern, line) for pattern in patterns.values())]
        save()
        assert all(checks.values()), f'Diagnostic configuration mismatch: {checks}'
        phase('capture_requests')
        child_command('capture_requests', [sys.executable, str(WORK / 'capture_requests.py'),
                      '--directory', str(captures), '--output', str(out / 'requests.json'),
                      '--base-url', f'http://127.0.0.1:{args.port}', '--max-tokens', '128'], CAPTURE_SECONDS)
        managed_artifacts.add(out / 'requests.json')
        assert json.loads((out / 'requests.json').read_text())['complete'] is True
        assert server.poll() is None, 'Server exited during capture'
        phase('validate_capture')
        child_command('validate_capture', [sys.executable, str(WORK / 'validate_capture.py'), str(captures),
                      '--requests', str(out / 'requests.json'), '--output', str(out / 'capture.validation.json')], VALIDATE_SECONDS)
        managed_artifacts.add(out / 'capture.validation.json')
        phase('stopping_server')
        stop_server()
        report['isolation_before_replay'] = isolate()
        with socket.socket() as connection:
            connection.settimeout(0.3)
            assert connection.connect_ex(('127.0.0.1', args.port)) != 0, 'Server port remains occupied after stop'
        phase('replay')
        replay_env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(captures),
                          MISTRALRS_MOE_BENCH_OUTPUT=str(out / 'replay.json'))
        replay_env.pop('MISTRALRS_MOE_CAPTURE_DIR', None)
        report['replay_environment_overrides'] = {name: replay_env[name] for name in
            ['MISTRALRS_MOE_REPLAY_DIR', 'MISTRALRS_MOE_BENCH_OUTPUT']}
        save()
        child_command('replay', [str(replay), '--exact', TEST_NAME, '--ignored', '--nocapture',
                                '--test-threads=1'], REPLAY_SECONDS, replay_env)
        managed_artifacts.add(out / 'replay.json')
        report['isolation_after_replay'] = isolate()
        phase('summarizing')
        child_command('summarize_replay', [sys.executable, str(WORK / 'summarize_replay.py'), str(out / 'replay.json'),
                      '--output', str(out / 'summary.json')], VALIDATE_SECONDS)
        managed_artifacts.add(out / 'summary.json')
        assert sha(cli) == build['diagnostic_binary']['sha256']
        assert sha(replay) == build['replay_test_binary']['sha256']
        assert sha(REPO / 'target/release/mistralrs') == EXPECTED_PRODUCTION_SHA256
        for name, expected in source_manifest.items():
            path = REPO / name
            required = expected['after_sha256'] if name == REPLAY_SOURCE else expected['before_sha256']
            assert (sha(path) if path.exists() else None) == required
        report.update(complete=True, phase='complete', finished_at=now(),
                      interpretation='Native and derived isolated expert kernel results only; not end-to-end throughput or a natural route/depth histogram')
        save()
        managed_artifacts.add(metadata_file)
        managed_artifacts.update(path for path in captures.iterdir() if path.is_file())
        validation = json.loads((out / 'capture.validation.json').read_text())
        captured_hashes = {captures / record['path']: record['sha256'] for record in validation['files']}
        manifest = [dict(path=str(path.relative_to(out)), bytes=path.stat().st_size,
                         sha256=captured_hashes.get(path) or sha(path)) for path in sorted(managed_artifacts)]
        atomic_json(out / 'manifest.json', dict(files=manifest, binary_hashes={
            'diagnostic': build['diagnostic_binary']['sha256'], 'replay': build['replay_test_binary']['sha256'],
            'production': EXPECTED_PRODUCTION_SHA256}))
        (out / 'complete').write_text(now() + '\n')
        print(f'{now()} DIAGNOSTIC_COMPLETE no model throughput claim; results at {out}', flush=True)
    except BaseException as error:
        report.update(complete=False, error=dict(type=type(error).__name__, message=str(error)), failed_at=now())
        save()
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        (captures / 'control.json').unlink(missing_ok=True)
        if server is not None and (server.poll() is None or server_log is not None):
            stop_server()
        else:
            memory_stop.set()
            if memory_thread is not None:
                memory_thread.join(timeout=STOP_SECONDS)


if __name__ == '__main__':
    main()
