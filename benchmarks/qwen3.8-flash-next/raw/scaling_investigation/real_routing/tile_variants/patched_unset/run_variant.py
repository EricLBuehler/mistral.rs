#!/usr/bin/env python3
"""Run one archived tile diagnostic against saved exact grouped-MMQ baseline outputs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

WORK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORK))
from build_diagnostic import REPO, atomic_json, isolate
from run_diagnostic import stop_child

TIMEOUT = 1800
TEST_NAME = 'flash_next_real_routing_tile_replay'
CAPTURES = WORK / 'runtime/captures'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def identity(row):
    return (row['source_metadata'], row['selection'], tuple(row['source_row_indices']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--provenance', '--patch-provenance', required=True, type=Path)
    parser.add_argument('--variant', choices=('baseline', '16', '32'), required=True)
    parser.add_argument('--test-source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--baseline', '--baseline-dir', type=Path, help='Completed baseline phase directory')
    parser.add_argument('--released', action='store_true')
    args = parser.parse_args()
    assert args.released, 'Coordinator must explicitly release GPU slot'
    assert args.variant == 'baseline' or args.baseline is not None, 'Variants require exact baseline outputs'
    assert not args.output.exists(), 'Preserve prior variant attempt'
    assert args.binary.is_file() and args.provenance.is_file() and args.test_source.is_file()
    original = json.loads((WORK / 'runtime/runtime.metadata.json').read_text())
    assert original['server_pid_gone'] and not Path(f"/proc/{original['server_pid']}").exists()
    assert json.loads((WORK / 'replay_sequence_query_prefix/lifecycle.json').read_text())['complete']
    baseline = None
    if args.baseline:
        baseline = json.loads((args.baseline / 'lifecycle.json').read_text())
        assert baseline['complete']
        assert sha(args.baseline / 'replay.json') == baseline['replay_sha256']
        for record in baseline['saved_outputs']:
            assert sha(args.baseline / record['path']) == record['sha256']
    args.output.mkdir(parents=True)
    shutil.copy2(args.provenance, args.output / 'build.provenance.json')
    shutil.copy2(args.test_source, args.output / 'replay.source.rs')
    shutil.copy2(Path(__file__), args.output / 'run_variant.py')
    metadata = dict(complete=False, started_at=now(), variant=args.variant, scope='Isolated real-route expert tile diagnostic, not end-to-end throughput.',
                    binary=dict(path=str(args.binary.resolve()), bytes=args.binary.stat().st_size, sha256=sha(args.binary)),
                    build_provenance_sha256=sha(args.provenance), test_source_sha256=sha(args.test_source),
                    runner_sha256=sha(Path(__file__)), captures=str(CAPTURES),
                    capture_validation_sha256=sha(WORK / 'runtime/capture.validation.json'),
                    baseline=str(args.baseline.resolve()) if args.baseline else None,
                    baseline_metadata_sha256=sha(args.baseline / 'lifecycle.json') if args.baseline else None,
                    baseline_replay_sha256=baseline['replay_sha256'] if baseline else None,
                    guard=dict(relative_rms_max=1e-3, cosine_min=0.999999), commands=[])
    metadata_path = args.output / 'lifecycle.json'

    def save():
        atomic_json(metadata_path, metadata)

    def interrupted(number, _frame):
        raise InterruptedError(f'Interrupted by signal {number}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        metadata['isolation_before'] = isolate()
        save()
        env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(CAPTURES),
                   MISTRALRS_MOE_REPLAY_OUTPUT_DIR=str(args.output / 'outputs'),
                   MISTRALRS_MOE_BENCH_OUTPUT=str(args.output / 'replay.json'))
        env.pop('MISTRALRS_MOE_REPLAY_BASELINE_DIR', None)
        env.pop('MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE', None)
        if args.variant != 'baseline':
            env['MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE'] = args.variant
        if args.baseline:
            env['MISTRALRS_MOE_REPLAY_BASELINE_DIR'] = str(args.baseline / 'outputs')
        command = [str(args.binary), '--exact', TEST_NAME, '--ignored', '--nocapture', '--test-threads=1']
        metadata['environment_overrides'] = {k: v for k, v in env.items() if k.startswith('MISTRALRS_MOE_') or k == 'MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE'}
        record = dict(command=command, started_at=now())
        metadata['commands'].append(record)
        print(now(), args.output.name, flush=True)
        with (args.output / 'replay.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
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
            raise subprocess.CalledProcessError(child.returncode, command)
        replay = json.loads((args.output / 'replay.json').read_text())
        rows = replay['results']
        assert replay['tile_comparison'] and len(rows) == 25
        assert len({identity(r) for r in rows}) == 25
        for count in (24, 32, 40, 42, 56):
            assert sum(r['rows'] == count for r in rows) == 5
        for row in rows:
            if row['rows'] in (42, 56):
                assert row['selection'] == 'native' and row['native_replay_guard_passed']
            else:
                assert row['selection'] == 'sequence_query_prefix' and row['derived']
            metrics = row['baseline_grouped_comparison']
            if args.baseline:
                assert metrics['relative_rms'] < 1e-3 and metrics['cosine'] > 0.999999
            else:
                assert metrics is None
        if args.baseline:
            baseline_rows = json.loads((args.baseline / 'replay.json').read_text())['results']
            assert {identity(r) for r in rows} == {identity(r) for r in baseline_rows}
        saved = sorted((args.output / 'outputs').iterdir())
        assert len(saved) == 50
        metadata['saved_outputs'] = [dict(path=str(p.relative_to(args.output)), bytes=p.stat().st_size, sha256=sha(p)) for p in saved]
        metadata['replay_sha256'] = sha(args.output / 'replay.json')
        metadata['isolation_after'] = isolate()
        assert sha(args.binary) == metadata['binary']['sha256']
        metadata.update(complete=True, finished_at=now())
        save()
        (args.output / 'complete').write_text(now() + '\n')
        print(now(), args.output.name, 'COMPLETE', flush=True)
    except BaseException as error:
        metadata.update(complete=False, failed_at=now(), error=dict(type=type(error).__name__, message=str(error)))
        save()
        raise


if __name__ == '__main__':
    main()
