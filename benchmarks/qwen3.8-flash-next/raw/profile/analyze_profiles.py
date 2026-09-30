#!/usr/bin/env python3
"""Export and analyze only after the controller reaches its profiling shutdown gate."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
BASELINE = OUT.parent / 'concurrency_20260929'
NSYS = '/opt/nvidia/nsight-systems/2026.1.3/bin/nsys'
NANOSECONDS = 1_000_000_000
F32_VOCAB_BYTES = 248320 * 4
D2H_QUERY = '''SELECT COUNT(*) AS transfer_count, COALESCE(SUM(m.bytes),0) AS total_bytes,
       COALESCE(SUM(m.end-m.start),0) AS summed_transfer_duration_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m JOIN ENUM_CUDA_MEMCPY_OPER AS k ON k.id=m.copyKind
WHERE k.name='CUDA_MEMCPY_KIND_DTOH' AND m.bytes=?'''


def read_json(path):
    return json.loads(path.read_text())


def epoch_ns(value):
    value = datetime.datetime.fromisoformat(value)
    delta = value - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * NANOSECONDS + delta.microseconds * 1000


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def gate_ready():
    command_file = OUT / 'measurement_commands.json'
    if not command_file.exists() or not (OUT / 'measurements.log').exists():
        return False
    commands = read_json(command_file)
    complete = {record['name'] for record in commands if record.get('returncode') == 0}
    required = {'profile_c1.stop', 'profile_c8.stop', 'profiler_shutdown'}
    return required <= complete and 'Waiting for profile analysis' in (OUT / 'measurements.log').read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while not gate_ready():
        if not args.wait:
            raise SystemExit('Capture shutdown/analysis gate is not complete; no work performed.')
        time.sleep(2)
    if (OUT / 'analysis-complete').exists():
        raise SystemExit('The timing gate has already been released; refusing offline analysis.')
    commands = read_json(OUT / 'measurement_commands.json')
    provenance = {
        'analysis_started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'nsys_binary': NSYS,
        'nsys_version': subprocess.check_output([NSYS, '--version'], text=True).strip(),
        'analysis_script_sha256': sha256(OUT / 'analyze_nsys.py'),
        'driver_script_sha256': sha256(Path(__file__)),
        'server_metadata': read_json(OUT / 'isq_mtp_batched.metadata.json'),
        'steps': [], 'captures': {},
    }

    def run(command, log_path):
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log_path.open('w') as output:
            result = subprocess.run(command, cwd=OUT, stdout=output, stderr=subprocess.STDOUT)
        provenance['steps'].append(dict(command=command, log=str(log_path), started_utc=start,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), returncode=result.returncode))
        (OUT / 'profile_analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        result.check_returncode()

    compact = []
    transfer_cases = []
    for tag in ['c1', 'c8']:
        named_report = OUT / f'profile_{tag}.nsys-rep'
        report = named_report if named_report.exists() else OUT / 'mtp_concurrency.nsys-rep'
        if tag == 'c8' and report != named_report:
            raise SystemExit('Missing named C8 report; refusing to reuse C1.')
        database = OUT / f'profile_{tag}.sqlite'
        if not database.exists():
            run([NSYS, 'export', '--type=sqlite', '--output=' + str(database), str(report)],
                OUT / f'profile_{tag}.export.log')
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA cache_size=-2048')
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            capture_start = dict(connection.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
            export_metadata = dict(connection.execute('SELECT name,value FROM META_DATA_EXPORT'))
            epoch = capture_start['utcEpochNs']
            record = next(record for record in commands if record['name'] == f'profile_{tag}.requests')
            start_record = next(record for record in commands if record['name'] == f'profile_{tag}.start')
            if not epoch_ns(start_record['started_utc']) <= epoch <= epoch_ns(start_record['finished_utc']):
                raise SystemExit(f'{tag}: capture start does not match controller start command.')
            windows = []
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                start, end = connection.execute('SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0').fetchone()
                if start is not None:
                    windows.append(dict(name='recorded_kernel_span', start_ns=start, end_ns=end))
            requests_path = OUT / f'profile_{tag}.requests.json'
            data = read_json(requests_path)
            assert data['complete']
            cases = list(data['cases'].values())
            assert len(cases) == 1 and len(cases[0]['runs']) == 1
            trial = cases[0]['runs'][0]
            lower = epoch_ns(data['date'])
            upper = epoch_ns(record['finished_utc']) - round(trial['wall_seconds'] * NANOSECONDS)
            drift = ((epoch_ns(record['finished_utc']) - epoch_ns(record['started_utc']))
                     - round((record['finished_monotonic'] - record['started_monotonic']) * NANOSECONDS))
            margin = abs(drift) + 1000
            lower -= margin
            upper += margin
            assert lower <= upper, (tag, lower, upper)
            sustained = trial['sustained_window']
            if sustained:
                start = upper - epoch + round(sustained['start_seconds'] * NANOSECONDS)
                end = lower - epoch + round(sustained['end_seconds'] * NANOSECONDS)
                if start < end:
                    windows.append(dict(name='sustained_interior', start_ns=start, end_ns=end))
            if not windows:
                windows.append(dict(name='request_command_span', start_ns=epoch_ns(record['started_utc'])-epoch,
                                    end_ns=epoch_ns(record['finished_utc'])-epoch))
            alignment = {
                'capture_epoch_ns': epoch, 'controller_command': record,
                'trial_origin_epoch_lower_ns': lower, 'trial_origin_epoch_upper_ns': upper,
                'trial_origin_uncertainty_ns': upper-lower,
                'command_utc_minus_monotonic_duration_ns': drift,
                'method': 'Harness data.date precedes trial_start. Controller finish minus trial wall duration follows trial_start. Sustained interior trims both ends by the bounded origin uncertainty; kernel-span results require no request alignment.',
                'warning': 'Controller clock pairs and export systemClockNs are retained as provenance; systemClockNs is not assumed to be Python CLOCK_MONOTONIC_RAW.',
            }
            if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
                transfer = dict(connection.execute(D2H_QUERY, (F32_VOCAB_BYTES,)).fetchone())
            else:
                transfer = {'unavailable': True}
        (OUT / f'profile_{tag}.windows.json').write_text(json.dumps(windows, indent=2) + '\n')
        (OUT / f'profile_{tag}.alignment.json').write_text(json.dumps(alignment, indent=2) + '\n')
        provenance['captures'][tag] = dict(report=str(report), report_sha256=sha256(report),
            sqlite=str(database), sqlite_bytes=database.stat().st_size,
            request_artifact=str(requests_path), request_sha256=sha256(requests_path),
            capture_start=capture_start, export_metadata=export_metadata, alignment=alignment)
        output = OUT / f'profile_{tag}.analysis.json'
        run([sys.executable, str(OUT / 'analyze_nsys.py'), str(database), '--output=' + str(output),
             '--windows=' + str(OUT / f'profile_{tag}.windows.json')], OUT / f'profile_{tag}.analysis.log')
        analysis = read_json(output)
        transfer_cases.append(dict(capture=tag, source_sqlite=str(database), **transfer))
        compact.append(dict(capture=tag, kernel_timeline_status=analysis['kernel_timeline_status'],
            classification=analysis['classification'],
            diagnostics=[row['text'] for row in analysis['diagnostics'] if any(word in row['text'].lower() for word in ['cuda','cupti','hardware','software'])],
            windows={name: {key: value for key, value in window.items() if key in [
                'duration_ns','kernel_count','summed_kernel_time_ns','kernel_busy_union_ns','gpu_activity_busy_union_ns',
                'gpu_activity_busy_pct','categories','graph_kernel_count','graph_kernel_time_ns','api_call_kinds','gpu_metrics_status',
                'cpu_inter_launch_api_gap_ns','gpu_internal_idle_gap_ns']}
                     for name, window in analysis['windows'].items()}))
    transfer_evidence = dict(query=D2H_QUERY, query_parameters=[F32_VOCAB_BYTES], cases=transfer_cases,
        baseline=read_json(BASELINE / 'draft_sampling_transfer_evidence.json'),
        caveat='Recorded transfer durations exclude host synchronization, CPU argmax/softmax and temporary allocations. These profiles diagnose work; do not compare profiled throughput with unprofiled benchmarks.')
    (OUT / 'draft_sampling_transfer_evidence.json').write_text(json.dumps(transfer_evidence, indent=2) + '\n')
    (OUT / 'profile_compact_summary.json').write_text(json.dumps(compact, indent=2) + '\n')
    provenance['analysis_finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT / 'profile_analysis_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'offline-analysis-complete').write_text(provenance['analysis_finished_utc'] + '\n')
    print('Offline analysis complete; official timing gate remains unchanged.', flush=True)


if __name__ == '__main__':
    main()
