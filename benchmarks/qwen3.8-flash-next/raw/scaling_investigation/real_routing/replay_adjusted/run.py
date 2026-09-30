#!/usr/bin/env python3
"""Preserve failed replay, build the corrected diagnostic and run timings then FP32 oracle."""
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
from build_diagnostic import EXPECTED_PRODUCTION_SHA256, REPO, REPLAY_SOURCE, atomic_json, isolate, sha
from run_diagnostic import stop_child

OUT = Path(__file__).resolve().parent
CAPTURE = WORK / 'runtime/captures'
SOURCE = REPO / REPLAY_SOURCE
TIMEOUT = 1800


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    metadata_file = OUT / 'lifecycle.json'
    assert not metadata_file.exists(), 'Preserve prior adjusted replay attempt'
    capture_metadata = json.loads((WORK / 'runtime/runtime.metadata.json').read_text())
    assert capture_metadata['complete'] is False and capture_metadata['phase'] == 'replay'
    assert capture_metadata['server_pid_gone'] and not Path(f"/proc/{capture_metadata['server_pid']}").exists()
    build = json.loads((WORK / 'build/lifecycle.json').read_text())
    assert sha(Path(build['replay_test_binary']['path'])) == build['replay_test_binary']['sha256']
    assert sha(REPO / 'target/release/mistralrs') == EXPECTED_PRODUCTION_SHA256
    report = dict(complete=False, started_at=now(), phase='preflight', commands=[],
        original_capture_runtime=str(WORK / 'runtime'), captures=str(CAPTURE),
        original_failed_runtime_metadata_sha256=sha(WORK / 'runtime/runtime.metadata.json'),
        original_failed_replay_log_sha256=sha(WORK / 'runtime/replay.log'),
        original_failed_probe=build['replay_test_binary'],
        original_failed_source_sha256=sha(OUT / 'replay.failed_source.rs'),
        source_sha256=sha(SOURCE), source_archive=str(OUT / 'replay.adjusted_source.rs'),
        script_sha256=sha(Path(__file__)), summary_script_sha256=sha(OUT / 'summarize_replay.py'),
        method='Record old cross-kernel guard pass/fail; strict native equivalence and finite checks remain; independent FP32 oracle runs separately after all timing.')
    assert report['source_sha256'] == sha(OUT / 'replay.adjusted_source.rs')
    report['isolation_before'] = isolate()

    def save():
        atomic_json(metadata_file, report)

    def phase(name):
        report['phase'] = name
        save()
        print(now(), name, flush=True)

    def interrupted(number, _frame):
        raise InterruptedError(f'Interrupted by signal {number}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def command(name, argv, env=None):
        phase(name)
        log = OUT / f'{name}.log'
        record = dict(name=name, command=argv, started_at=now(), log=str(log))
        report['commands'].append(record)
        save()
        with log.open('wb') as output:
            child = subprocess.Popen(argv, cwd=REPO, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record['pid'] = child.pid
            save()
            try:
                child.wait(timeout=TIMEOUT)
            except BaseException:
                stop_child(child)
                raise
            finally:
                record.update(returncode=child.poll(), finished_at=now())
                save()
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, argv)
        return log

    try:
        base = ['--locked', '--release', '-p', 'mistralrs-quant', '--features', 'cuda,cutile', '--test', 'moe_dispatch_bench']
        build_log = command('build', ['cargo', 'test', *base, '--no-run', '--message-format=json-render-diagnostics'])
        executables = set()
        for line in build_log.read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (message.get('reason') == 'compiler-artifact' and message.get('target', {}).get('name') == 'moe_dispatch_bench'
                    and message.get('profile', {}).get('test') is True and message.get('executable')):
                executables.add(Path(message['executable']))
        assert len(executables) == 1
        binary = OUT / 'moe_dispatch_bench.adjusted'
        os.link(executables.pop(), binary)
        report['replay_binary'] = dict(path=str(binary), sha256=sha(binary), bytes=binary.stat().st_size)
        save()
        assert sha(Path(build['replay_test_binary']['path'])) == build['replay_test_binary']['sha256']
        command('check', ['cargo', 'check', *base])
        command('clippy', ['cargo', 'clippy', *base, '--', '-D', 'warnings'])
        report['isolation_before_timing'] = isolate()
        env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(CAPTURE), MISTRALRS_MOE_BENCH_OUTPUT=str(OUT / 'replay.json'))
        command('replay', [str(binary), '--exact', 'flash_next_real_routing_replay', '--ignored', '--nocapture', '--test-threads=1'], env)
        report['ordinary_replay_complete'] = True
        report['replay_results_sha256'] = sha(OUT / 'replay.json')
        save()
        env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(CAPTURE),
                   MISTRALRS_MOE_REPLAY_RESULTS=str(OUT / 'replay.json'), MISTRALRS_MOE_ORACLE_OUTPUT=str(OUT / 'oracle.json'))
        command('oracle', [str(binary), '--exact', 'flash_next_real_routing_oracle', '--ignored', '--nocapture', '--test-threads=1'], env)
        report['oracle_complete'] = True
        report['oracle_results_sha256'] = sha(OUT / 'oracle.json')
        save()
        command('summary', [sys.executable, str(OUT / 'summarize_replay.py'), str(OUT / 'replay.json'),
                            '--oracle', str(OUT / 'oracle.json'), '--output', str(OUT / 'summary.json')])
        report['isolation_after'] = isolate()
        assert sha(SOURCE) == report['source_sha256']
        assert sha(REPO / 'target/release/mistralrs') == EXPECTED_PRODUCTION_SHA256
        manifest = json.loads((WORK / 'source_manifest.json').read_text())
        for name, hashes in manifest.items():
            if name == REPLAY_SOURCE:
                continue
            path = REPO / name
            assert (sha(path) if path.exists() else None) == hashes['before_sha256']
        report.update(complete=True, phase='complete', finished_at=now(),
                      summary_sha256=sha(OUT / 'summary.json'), production_binary_sha256=EXPECTED_PRODUCTION_SHA256)
        save()
        (OUT / 'complete').write_text(now() + '\n')
        print(now(), 'ADJUSTED_REPLAY_AND_ORACLE_COMPLETE GPU_FREE', flush=True)
    except BaseException as error:
        report.update(complete=False, failed_at=now(), error=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
