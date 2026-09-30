#!/usr/bin/env python3
"""Profile five warmed native-56 expert kernels only after ordinary replay finishes."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

from build_diagnostic import WORK, atomic_json, isolate, sha

NCU = Path('/usr/local/cuda/bin/ncu')
SECTIONS = Path('/opt/nvidia/nsight-compute/2026.2.1/sections')
TEST = 'flash_next_real_routing_replay'
TIMEOUT_SECONDS = 180
METRICS = [
    'gpu__time_duration.sum',
    'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg.pct_of_peak_sustained_active',
    'smsp__issue_active.avg.per_cycle_active',
    'smsp__warps_eligible.avg.per_cycle_active',
    'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio',
    'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'lts__t_sector_hit_rate.pct',
    'lts__t_sectors_op_read.sum',
    'lts__t_sectors_op_write.sum',
    'dram__bytes_read.sum',
    'dram__bytes_write.sum',
]


def command(argv, log, env=None):
    with log.open('w') as out:
        child = subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=TIMEOUT_SECONDS)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            raise
    if result:
        raise subprocess.CalledProcessError(result, argv)
    return log.read_text()


def parse_metrics(text):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
    reader = csv.DictReader(io.StringIO('\n'.join(lines[start:])))
    units = next(reader)
    assert not units['ID']
    extra = ['profiler__replayer_passes', 'launch__registers_per_thread',
             'launch__shared_mem_per_block', 'launch__waves_per_multiprocessor']
    return [{'id': row['ID'], 'kernel': row['Kernel Name'], 'grid': row['Grid Size'],
             'block': row['Block Size'], 'compute_capability': row['CC'],
             'metrics': {name: {'value': row[name], 'unit': units[name]}
                         for name in METRICS + extra if name in row}}
            for row in reader if row.get('ID')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true', required=True)
    parser.add_argument('--runtime', type=Path, default=WORK / 'runtime')
    parser.add_argument('--build-dir', type=Path, default=WORK / 'build')
    parser.add_argument('--replay-dir', type=Path, default=WORK / 'replay_adjusted')
    parser.add_argument('--output-dir', type=Path, default=WORK / 'ncu_native56')
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise InterruptedError(f'Profiling interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'complete': False, 'started_unix': time.time(), 'commands': [],
                'script_sha256': sha(Path(__file__)), 'ncu': str(NCU.resolve()),
                'scope': 'Five projection kernel launches from one native56 sample; diagnostic counter replay, not benchmark timings',
                'cache_control': 'none', 'clock_control': 'none', 'replay_mode': 'kernel',
                'limits': ['NCU serializes launches and may replay kernels; it cannot measure original CPU launch gaps.',
                           'No cache flushing preserves preceding work, but later replay passes can change cache state.',
                           'NCU-reported peak bandwidth is not used to normalize results; no DRAM claim when its counters are absent.',
                           'Only one captured input is profiled; earlier unprofiled replay remains the performance evidence.']}
    path = out / 'metadata.json'
    def save():
        atomic_json(path, metadata)
    def run(name, argv, env=None):
        record = {'name': name, 'command': argv, 'started_unix': time.time()}
        metadata['commands'].append(record)
        save()
        result = command(argv, out / (name + '.log'), env)
        record.update(finished_unix=time.time(), returncode=0, log_sha256=sha(out / (name + '.log')))
        save()
        return result
    try:
        runtime = json.loads((args.runtime / 'runtime.metadata.json').read_text())
        assert runtime['server_pid_gone']
        adjusted = json.loads((args.replay_dir / 'lifecycle.json').read_text())
        assert adjusted['complete'] and adjusted['ordinary_replay_complete'] and adjusted['oracle_complete']
        assert (args.replay_dir / 'complete').is_file()
        assert sha(args.runtime / 'runtime.metadata.json') == adjusted['original_failed_runtime_metadata_sha256']
        assert sha(args.replay_dir / 'replay.json') == adjusted['replay_results_sha256']
        assert sha(args.replay_dir / 'oracle.json') == adjusted['oracle_results_sha256']
        metadata['authoritative_replay_lifecycle'] = {'path': str(args.replay_dir / 'lifecycle.json'),
                                                    'sha256': sha(args.replay_dir / 'lifecycle.json')}
        metadata['original_runtime_failed_numeric_guard_preserved'] = True
        assert not Path('/proc', str(runtime['server_pid'])).exists()
        metadata['isolation_before'] = isolate()
        build = json.loads((args.build_dir / 'lifecycle.json').read_text())
        assert build['complete'] and not build['restoration_errors']
        binary = Path(adjusted['replay_binary']['path'])
        assert sha(binary) == adjusted['replay_binary']['sha256']
        metadata['binary'] = adjusted['replay_binary']
        captures = args.runtime / 'captures'
        available = []
        for item in sorted(captures.glob('*.json')):
            data = json.loads(item.read_text())
            if data.get('rows') == 56 and data.get('batch') == 8 and data.get('query_len') == 7 and data.get('derived') is False:
                available.append((item, data))
        median_unique = statistics.median(sum(count > 0 for count in data['expert_occupancy'])
                                          for _, data in available)
        source, sample = min(available, key=lambda pair: (
            abs(sum(count > 0 for count in pair[1]['expert_occupancy']) - median_unique), pair[0].name))
        metadata['sample_selection'] = {
            'rule': 'Closest observed U to median unique-expert count among all native B8xQ7 captures; filename breaks equal-distance ties.',
            'median_unique_experts': median_unique,
            'candidates': [{'metadata': item.name, 'unique_experts': sum(count > 0 for count in data['expert_occupancy'])}
                           for item, data in available],
        }
        selected = out / 'selected_sample'
        selected.mkdir()
        links = [source, captures / sample['tensor_file'], *sorted(captures.glob('*.gguf'))]
        assert len(links) >= 3
        for item in links:
            (selected / item.name).symlink_to(item.resolve())
        metadata['sample'] = {'path': str(source), 'sha256': sha(source), 'metadata': sample,
                              'links': [str(item.resolve()) for item in links]}
        metadata['validation_manifest_sha256'] = sha(args.runtime / 'capture.validation.json')
        metadata['ncu_version'] = run('version', [str(NCU), '--version']).strip()
        base = [str(NCU), '--config-file', '0', '--section-folder', str(SECTIONS)]
        query = run('query_metrics', base + ['--query-metrics', '--devices', '0', '--query-metrics-mode', 'all'])
        supported = set(re.findall(r'\b[A-Za-z][A-Za-z0-9_]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\b', query))
        wanted = [metric for metric in METRICS if metric in supported]
        assert 'gpu__time_duration.sum' in wanted and 'sm__throughput.avg.pct_of_peak_sustained_elapsed' in wanted
        metadata['metrics_available'] = wanted
        metadata['metrics_unavailable'] = [metric for metric in METRICS if metric not in supported]
        metadata['dram_counters_available'] = all(metric in wanted for metric in METRICS[-2:])
        filters = [
            ('gemv', 'regex:^moe_gemv_(fused_gate_up_q4k_q8_1|down_aggregate_q4_1_q8_1)$', 8, 2),
            ('mmq', 'regex:^mul_mat_q$', 12, 3),
        ]
        summary = []
        for name, regex, skip, count in filters:
            env = dict(os.environ, MISTRALRS_MOE_REPLAY_DIR=str(selected),
                       MISTRALRS_MOE_BENCH_OUTPUT=str(out / f'{name}.profiled_test_output_not_benchmark.json'))
            argv = base + ['--target-processes', 'application-only', '--replay-mode', 'kernel',
                '--cache-control', 'none', '--clock-control', 'none', '--import-sass', 'no',
                '--kernel-name-base', 'function', '--kernel-name', regex,
                '--launch-skip', str(skip), '--launch-count', str(count),
                '--metrics', ','.join(wanted), '--disable-extra-suffixes',
                '--export', str(out / name), str(binary), '--exact', TEST,
                '--ignored', '--nocapture', '--test-threads=1']
            run(name, argv, env)
            raw = run(name + '_raw', [str(NCU), '--import', str(out / (name + '.ncu-rep')),
                       '--page', 'raw', '--csv', '--print-units', 'base'])
            kernels = parse_metrics(raw)
            assert len(kernels) == count, (name, len(kernels), count)
            summary.append({'backend': name, 'launch_skip': skip, 'kernels': kernels})
        atomic_json(out / 'kernels.json', {'scope': metadata['scope'], 'limits': metadata['limits'], 'results': summary})
        metadata.update(complete=True, finished_unix=time.time(), isolation_after=isolate())
        save()
        (out / 'SHA256SUMS').write_text(''.join(f'{sha(item)}  {item.name}\n' for item in sorted(out.iterdir()) if item.is_file() and item.name != 'SHA256SUMS'))
    except BaseException as error:
        metadata['error'] = {'type': type(error).__name__, 'message': str(error)}
        save()
        raise


if __name__ == '__main__':
    main()
