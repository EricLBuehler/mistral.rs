#!/usr/bin/env python3
"""Build diagnostic artifacts and restore production sources/binary; never run CUDA tests."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import tempfile

REPO = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
PRODUCTION_BINARY = REPO / 'target/release/mistralrs'
EXPECTED_PRODUCTION_SHA256 = 'd245efbb543fa8131f71aee34fef3a868cb1f3c0cfba8eaafd5b4fd7d4c64588'
REPLAY_SOURCE = 'mistralrs-quant/tests/moe_dispatch_bench.rs'
BUILD_TIMEOUT_SECONDS = 7200
TERMINATE_TIMEOUT_SECONDS = 10
COMPILERS = {'cargo', 'rustc', 'clippy-driver', 'cargo-clippy', 'nvcc', 'cicc', 'ptxas', 'cc1plus', 'g++', 'clang++', 'rust-analyzer'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def restore_file(path, baseline, mode):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.restore_capture_', delete=False) as file:
        temporary = Path(file.name)
        file.write(baseline.read_bytes())
    try:
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def isolate():
    observed = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            name = (directory / 'comm').read_text().strip()
            relevant = name in COMPILERS or name.startswith(('mistralrs', 'dense_backend', 'moe_dispatch'))
            if not relevant:
                continue
            state = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        process = dict(pid=int(directory.name), name=name, state=state)
        observed.append(process)
        assert state in ('T', 't', 'Z'), f'Active model/compiler/test before build: {process}'
    return dict(at=now(), processes=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true', required=True,
                        help='Coordinator has explicitly released the build slot.')
    parser.add_argument('--output-dir', type=Path, default=WORK / 'build')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lifecycle_file = out / 'lifecycle.json'
    assert not lifecycle_file.exists(), 'Preserve prior build attempt; use a fresh output directory'
    manifest = json.loads((WORK / 'source_manifest.json').read_text())
    capture_sources = [name for name in manifest if name != REPLAY_SOURCE]
    report = {
        'started_at': now(), 'complete': False, 'scope': 'build/check only; no model or CUDA test is executed',
        'script_sha256': sha(Path(__file__)),
        'patches': {name: sha(WORK / name) for name in ('capture.patch', 'replay.patch')},
        'source_manifest_sha256': sha(WORK / 'source_manifest.json'),
        'expected_production_sha256': EXPECTED_PRODUCTION_SHA256,
        'commands': [], 'restoration': [],
        'environment': {name: os.environ.get(name) for name in (
            'CARGO_BUILD_JOBS', 'CARGO_TARGET_DIR', 'RUSTFLAGS', 'CUDA_COMPUTE_CAP',
            'CUDACXX', 'NVCC_CCBIN', 'CC', 'CXX', 'CUDA_HOME', 'CUDA_VISIBLE_DEVICES')},
    }
    production_archive = out / 'mistralrs.production'
    diagnostic_archive = out / 'mistralrs.capture'
    replay_archive = out / 'moe_dispatch_bench.capture'
    applied = False
    preserved = False
    build_success = False
    source_modes = {}

    def save():
        atomic_json(lifecycle_file, report)

    def interrupted(signum, _frame):
        raise InterruptedError(f'Build interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def command(name, argv):
        log_path = out / f'{name}.log'
        record = dict(name=name, command=argv, started_at=now(), log=str(log_path))
        report['commands'].append(record)
        save()
        print(f'{now()} START {name}', flush=True)
        with log_path.open('wb') as log:
            child = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            record['pid'] = child.pid
            save()
            try:
                child.wait(timeout=BUILD_TIMEOUT_SECONDS)
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                raise
            finally:
                record.update(finished_at=now(), returncode=child.poll())
                save()
        record['log_sha256'] = sha(log_path)
        save()
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, argv)
        print(f'{now()} DONE {name}', flush=True)
        return log_path

    try:
        target_override = os.environ.get('CARGO_TARGET_DIR')
        assert target_override is None or Path(target_override).resolve() == REPO / 'target'
        report['isolation_before'] = isolate()
        for name, expected in manifest.items():
            source = REPO / name
            if expected['before_sha256'] is None:
                assert not source.exists(), f'Unexpected diagnostic source already present: {name}'
            else:
                assert sha(source) == expected['before_sha256'], f'Base source changed: {name}'
                baseline = WORK / 'base_tree' / name
                assert sha(baseline) == expected['before_sha256'], f'Baseline copy changed: {name}'
                source_modes[name] = stat.S_IMODE(source.stat().st_mode)
            assert sha(WORK / 'patch_tree' / name) == expected['after_sha256'], f'Revised source changed: {name}'
        assert sha(PRODUCTION_BINARY) == EXPECTED_PRODUCTION_SHA256, 'Production binary changed'
        os.link(PRODUCTION_BINARY, production_archive)
        preserved = True
        assert sha(production_archive) == EXPECTED_PRODUCTION_SHA256
        report['production_archive'] = dict(path=str(production_archive), sha256=EXPECTED_PRODUCTION_SHA256,
                                            inode=production_archive.stat().st_ino,
                                            bytes=production_archive.stat().st_size)
        report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        before_diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=REPO)
        (out / 'source_before.diff').write_bytes(before_diff)
        report['source_before_diff_sha256'] = hashlib.sha256(before_diff).hexdigest()
        save()
        command('patch_check', ['git', 'apply', '--check', str(WORK / 'capture.patch'), str(WORK / 'replay.patch')])
        applied = True
        command('apply_patch', ['git', 'apply', str(WORK / 'capture.patch'), str(WORK / 'replay.patch')])
        report['instrumented_source_sha256'] = {}
        for name, expected in manifest.items():
            actual = sha(REPO / name)
            assert actual == expected['after_sha256'], f'Applied source differs: {name}'
            report['instrumented_source_sha256'][name] = actual
        save()
        command('diagnostic_cli_build', ['cargo', 'build', '--locked', '--release', '-p', 'mistralrs-cli',
                                       '--features', 'cuda,flash-attn,cutile'])
        assert sha(production_archive) == EXPECTED_PRODUCTION_SHA256, 'Cargo mutated preserved production inode'
        instrumented_sha = sha(PRODUCTION_BINARY)
        assert instrumented_sha != EXPECTED_PRODUCTION_SHA256, 'Diagnostic binary unexpectedly equals production'
        os.link(PRODUCTION_BINARY, diagnostic_archive)
        report['diagnostic_binary'] = dict(path=str(diagnostic_archive), sha256=instrumented_sha,
                                           inode=diagnostic_archive.stat().st_ino,
                                           bytes=diagnostic_archive.stat().st_size)
        save()
        test_log = command('replay_test_build', ['cargo', 'test', '--locked', '--release', '-p', 'mistralrs-quant',
                           '--features', 'cuda,cutile', '--test', 'moe_dispatch_bench', '--no-run',
                           '--message-format=json-render-diagnostics'])
        executables = set()
        for line in test_log.read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (message.get('reason') == 'compiler-artifact'
                    and message.get('target', {}).get('name') == 'moe_dispatch_bench'
                    and message.get('profile', {}).get('test') is True and message.get('executable')):
                executables.add(Path(message['executable']))
        assert len(executables) == 1, f'Expected one replay test executable, got {executables}'
        replay_binary = executables.pop()
        os.link(replay_binary, replay_archive)
        report['replay_test_binary'] = dict(path=str(replay_archive), cargo_path=str(replay_binary),
                                           sha256=sha(replay_archive), bytes=replay_archive.stat().st_size)
        save()
        command('diagnostic_cli_check', ['cargo', 'check', '--locked', '--release', '-p', 'mistralrs-cli',
                                       '--features', 'cuda,flash-attn,cutile'])
        command('replay_test_check', ['cargo', 'check', '--locked', '--release', '-p', 'mistralrs-quant',
                                    '--features', 'cuda,cutile', '--test', 'moe_dispatch_bench'])
        assert sha(diagnostic_archive) == instrumented_sha
        assert sha(production_archive) == EXPECTED_PRODUCTION_SHA256
        build_success = True
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        restoration_errors = []
        if applied:
            for name in capture_sources:
                source = REPO / name
                expected = manifest[name]
                try:
                    current = sha(source) if source.exists() else None
                    assert current in (expected['before_sha256'], expected['after_sha256']), \
                        f'Concurrent source change; refusing to overwrite {name}'
                    if expected['before_sha256'] is None:
                        source.unlink(missing_ok=True)
                    else:
                        restore_file(source, WORK / 'base_tree' / name, source_modes[name])
                    restored = sha(source) if source.exists() else None
                    assert restored == expected['before_sha256'], f'Restored source mismatch: {name}'
                    report['restoration'].append(dict(path=name, restored_sha256=restored))
                except BaseException as error:
                    restoration_errors.append(dict(path=name, type=type(error).__name__, message=str(error)))
            try:
                report['replay_source_sha256'] = sha(REPO / REPLAY_SOURCE)
                report['replay_extension_retained'] = report['replay_source_sha256'] == manifest[REPLAY_SOURCE]['after_sha256']
                if build_success and not report['replay_extension_retained']:
                    restoration_errors.append(dict(path=REPLAY_SOURCE, message='Replay extension unexpectedly changed'))
            except BaseException as error:
                restoration_errors.append(dict(path=REPLAY_SOURCE, type=type(error).__name__, message=str(error)))
        if preserved:
            try:
                assert sha(production_archive) == EXPECTED_PRODUCTION_SHA256, 'Preserved production binary hash mismatch'
                replacement = PRODUCTION_BINARY.with_name('mistralrs.restore-capture-tmp')
                assert not replacement.exists(), 'Unexpected temporary production binary'
                os.link(production_archive, replacement)
                replacement.replace(PRODUCTION_BINARY)
                assert sha(PRODUCTION_BINARY) == EXPECTED_PRODUCTION_SHA256
                report['restored_production_binary_sha256'] = EXPECTED_PRODUCTION_SHA256
            except BaseException as error:
                restoration_errors.append(dict(path=str(PRODUCTION_BINARY), type=type(error).__name__, message=str(error)))
        report.update(finished_at=now(), restoration_errors=restoration_errors,
                      complete=build_success and not restoration_errors)
        save()
        if report['complete']:
            (out / 'build-complete').write_text(now() + '\n')
            print(f'{now()} BUILD_COMPLETE sources and production binary restored; no GPU tests ran', flush=True)
        elif restoration_errors:
            print(json.dumps(restoration_errors, indent=2), flush=True)
            if build_success:
                raise RuntimeError('Restoration failed; inspect lifecycle.json before proceeding')


if __name__ == '__main__':
    main()
