#!/usr/bin/env python3
"""Export only after shutdown, align exact client windows, and compare capture-off phases."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NSYS = '/opt/nvidia/nsight-systems/2026.1.3/bin/nsys'
MIN_FREE_BYTES = 1024**3


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clock_bounds(anchors):
    low = min(row['realtime_ns'] - row['realtime_perf_counter_upper_ns'] for row in anchors)
    high = max(row['realtime_ns'] - row['realtime_perf_counter_lower_ns'] for row in anchors)
    return low, high


def aligned_window(name, start, end, epoch, offset_lower, offset_upper):
    lo = start + offset_upper - epoch
    hi = end + offset_lower - epoch
    if hi <= lo:
        raise ValueError(f'Clock uncertainty consumes window {name}')
    return {'name': name, 'start_ns': lo, 'end_ns': hi,
            'client_start_perf_counter_ns': start, 'client_end_perf_counter_ns': end,
            'client_exact_monotonic_duration_ns': end - start,
            'definition': 'Interior trimmed by the observed realtime-minus-monotonic offset envelope; not streaming-token or decode-only timing.'}


def phase_comparison(warmup, profiled):
    before, after = warmup['trial'], profiled['trial']
    assert warmup['complete'] and profiled['complete']
    for key in ('tokenizer_sha256', 'prompts_sha256', 'shared_harness_sha256', 'timestamped_harness_sha256'):
        assert warmup[key] == profiled[key], key
    assert len(before['requests']) == len(after['requests'])
    identical = 0
    for a, b in zip(before['requests'], after['requests'], strict=True):
        for key in ('request_index', 'prompt_name', 'prompt_sha256', 'request'):
            assert a[key] == b[key], key
        assert a['response']['usage']['completion_tokens'] == b['response']['usage']['completion_tokens'] == 128
        assert a['response']['usage']['prompt_tokens'] == b['response']['usage']['prompt_tokens']
        identical += a['response']['choices'][0]['text'] == b['response']['choices'][0]['text']
    return {
        'requests': len(after['requests']),
        'capture_off_output_tokens': before['completed_output_tokens'],
        'profiled_output_tokens': after['completed_output_tokens'],
        'capture_off_client_wall_seconds': before['wall_seconds'],
        'profiled_client_wall_seconds': after['wall_seconds'],
        'profiled_over_capture_off_wall_ratio': after['wall_seconds'] / before['wall_seconds'],
        'identical_completion_text_count': identical,
        'limitations': [
            'Same server with profiler instrumentation attached in both phases; only recording is disabled in the warmup.',
            'Sequential warmup then recorded trial, not randomized or an estimate of pure profiler overhead.',
            'MTP adaptation, generated text, routes, cache state, and memory pressure may also differ; inspect separate counter and memory windows.',
            'One finite trial each, without a variability estimate; these elapsed rates do not replace official serving results.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'current_capture')
    parser.add_argument('--output', type=Path, default=ROOT / 'analysis')
    args = parser.parse_args()
    metadata = load(args.run / 'metadata.json')
    assert metadata['complete'] and metadata['cleanup']['model_exited'], 'Wait for completed capture and shutdown'
    model = metadata['model_process']
    proc = Path('/proc') / str(model['pid'])
    if proc.exists():
        fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
        assert int(fields[19]) != model['starttime_ticks'] or fields[0] in ('Z', 'X'), 'Original model process still alive'
    assert shutil.disk_usage(args.run).free >= MIN_FREE_BYTES
    for name, row in load(args.run / 'manifest.json').items():
        assert sha(args.run / name) == row['sha256'], name
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'complete': False, 'analyzer_sha256': sha(Path(__file__)),
                  'run_metadata_sha256': sha(args.run / 'metadata.json'), 'steps': [], 'phases': []}
    save(args.output / 'provenance.json', provenance)

    def run(name, command):
        assert shutil.disk_usage(args.output).free >= MIN_FREE_BYTES
        with (args.output / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
        provenance['steps'].append({'command': command, 'returncode': result.returncode,
                                    'log': f'{name}.log', 'log_sha256': sha(args.output / f'{name}.log')})
        save(args.output / 'provenance.json', provenance)
        result.check_returncode()

    memory = [json.loads(line) for line in (args.run / 'memory.jsonl').read_text().splitlines()]
    comparisons = []
    for concurrency in (1, 8):
        name = f'profile_c{concurrency}'
        report = args.run / f'{name}.nsys-rep'
        database = args.output / f'{name}.sqlite'
        run(name + '.export', [NSYS, 'export', '--type=sqlite', '--output=' + str(database), str(report)])
        client = load(args.run / f'{name}.requests.json')
        assert client['complete']
        request_window = client['request_window_perf_counter_ns']
        anchors = [client['clock_before'], client['clock_after']]
        anchors.extend(row['clock'] for row in memory
                       if request_window['start'] <= row['clock']['realtime_perf_counter_lower_ns'] <= request_window['end'])
        offset_lower, offset_upper = clock_bounds(anchors)
        with sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA cache_size=-2048')
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            capture = dict(connection.execute('SELECT * FROM TARGET_INFO_SESSION_START_TIME').fetchone())
            epoch = capture['utcEpochNs']
            control = next(row for row in metadata['commands'] if row['name'] == name + '.start')
            assert control['clock_before']['realtime_ns'] <= epoch <= control['clock_after']['realtime_ns'], 'Unexpected capture origin'
            windows = [
                {'name': 'client_request_envelope',
                 'start_ns': request_window['start'] + offset_lower - epoch,
                 'end_ns': request_window['end'] + offset_upper - epoch,
                 'client_exact_monotonic_duration_ns': request_window['end'] - request_window['start'],
                 'definition': 'Whole request-phase envelope including observed clock uncertainty; full-phase token/proposal denominators apply, not streaming or decode-only timing.'},
                aligned_window('client_request_interior', request_window['start'], request_window['end'],
                               epoch, offset_lower, offset_upper),
            ]
            boundary = client['trial']['sustained_window']
            if boundary:
                origin = client['trial']['trial_start_perf_counter_ns']
                windows.append(aligned_window('client_completion_boundary_interior',
                                               origin + round(boundary['start_seconds'] * 1e9),
                                               origin + round(boundary['end_seconds'] * 1e9),
                                               epoch, offset_lower, offset_upper))
            if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables:
                start, end = connection.execute('SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=0').fetchone()
                if start is None:
                    raise ValueError('GPU kernel table is empty')
                windows.append({'name': 'recorded_kernel_span', 'start_ns': start, 'end_ns': end})
            else:
                raise ValueError('Missing GPU kernel timeline; do not infer kernel attribution from APIs only')
            diagnostics = [dict(row) for row in connection.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else []
        alignment = {
            'capture_epoch_ns': epoch, 'capture_start': capture, 'clock_anchors': anchors,
            'observed_realtime_minus_perf_counter_offset_lower_ns': offset_lower,
            'observed_realtime_minus_perf_counter_offset_upper_ns': offset_upper,
            'observed_offset_envelope_width_ns': offset_upper - offset_lower,
            'exact_client_request_window': request_window, 'windows': windows,
            'limitations': [
                'Client monotonic timestamps are exact; trace alignment uses observed clock pairs, not an assumed identity with Nsight systemClockNs.',
                'The offset envelope covers sampled offsets before/during/after the trial, not arbitrary unobserved realtime clock steps.',
                'The request window includes prefill, final response delivery, and finite-trial startup/drain.',
            ],
            'profiler_diagnostics': diagnostics,
        }
        save(args.output / f'{name}.alignment.json', alignment)
        save(args.output / f'{name}.windows.json', windows)
        run(name + '.analysis', [sys.executable, str(ROOT / 'dependencies/analyze_nsys.py'), str(database),
                                '--output=' + str(args.output / f'{name}.analysis.json'),
                                '--windows=' + str(args.output / f'{name}.windows.json')])
        phase = {'name': name, 'report_sha256': sha(report), 'sqlite_sha256': sha(database),
                 'client_requests_sha256': sha(args.run / f'{name}.requests.json'),
                 'alignment_file': f'{name}.alignment.json', 'diagnostics': diagnostics}
        provenance['phases'].append(phase)
        warmups = [row for row in metadata['phases'] if row['name'].startswith(f'c{concurrency}.warmup')]
        for row in warmups:
            warmup_name = row['name']
            comparisons.append({'concurrency': concurrency, 'warmup_phase': warmup_name,
                                'recorded_phase': name,
                                **phase_comparison(load(args.run / f'{warmup_name}.requests.json'), client),
                                'capture_off_counters': load(args.run / f'{warmup_name}.metrics.delta.json'),
                                'recorded_counters': load(args.run / f'{name}.metrics.delta.json')})
    save(args.output / 'capture_off_vs_recorded.json', comparisons)
    provenance['complete'] = True
    save(args.output / 'provenance.json', provenance)
    save(args.output / 'manifest.json', {str(path.relative_to(args.output)): sha(path)
                                        for path in sorted(args.output.rglob('*')) if path.is_file()})
    print(json.dumps({'complete': True, 'output': str(args.output),
                      'capture_off_vs_recorded': [{key: value for key, value in row.items() if not key.endswith('_counters')}
                                                for row in comparisons]}, indent=2))


if __name__ == '__main__':
    main()
