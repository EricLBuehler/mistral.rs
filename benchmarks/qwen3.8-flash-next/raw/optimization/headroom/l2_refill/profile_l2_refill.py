#!/usr/bin/env python3
"""Profile warmed captured expert projections; report L2 refills, not DRAM bandwidth."""

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

WORK = Path('/home/ericbuehler/qwen4exp_work')
ROOT = Path(__file__).resolve().parent
BUILD = WORK / 'moe_optimization_20260930/final_serving/build'
DEFAULT_BINARY = BUILD / 'moe_dispatch_bench-0dfba70e2949feab'
DEFAULT_BINARY_SHA = 'eaf7d867dadfd56433f0f8a093e0f6dd320b1bcfde76ee76c58e32027443f7f1'
CAPTURES = WORK / 'real_routing_20260930/runtime/captures'
QUERY = WORK / 'real_routing_20260930/ncu_native56/query_metrics.log'
QUERY_SHA = '52dc30eb05091c32c404e7fe2a1ff61d3b898aee0b0d89f3110fd41acdc45d4b'
NCU = Path('/opt/nvidia/nsight-compute/2026.2.1/ncu')
SECTIONS = NCU.parent / 'sections'
TIMEOUT_SECONDS = 300
TERMINATE_TIMEOUT_SECONDS = 10
SECTOR_BYTES = 32
NANOSECONDS_PER_SECOND = 1_000_000_000
METRICS = [
    'gpu__time_duration.sum',
    'lts__d_sectors_fill_sysmem.sum',
    'lts__d_sectors_fill_device.sum',
    'lts__t_sectors_op_read.sum',
    'lts__t_sectors_op_read_lookup_hit.sum',
    'lts__t_sectors_op_read_lookup_miss.sum',
    'lts__t_sectors_aperture_sysmem_op_read_lookup_miss.sum',
    'lts__t_sectors_aperture_device_op_read_lookup_miss.sum',
    'lts__t_sectors_op_write.sum',
    'lts__t_sector_hit_rate.pct',
]
EXTRA_METRICS = ['profiler__replayer_passes', 'launch__registers_per_thread',
                 'launch__shared_mem_per_block', 'launch__waves_per_multiprocessor']
COMPILERS = {'cargo', 'rustc', 'clippy-driver', 'cargo-clippy', 'nvcc', 'cicc',
             'ptxas', 'cc1plus', 'g++', 'clang++', 'rust-analyzer'}
LIMITS = [
    'L2 refill sectors are data delivered to GPU L2 from system/device apertures, not memory-controller transactions or total LPDDR traffic.',
    '32 times a sector count is an L2 interface byte equivalent. It does not establish physical DRAM bytes, CPU traffic, memory utilization, or bandwidth headroom.',
    'NCU serializes and replays kernels; cache-control none preserves preceding work but later passes may alter cache contents. These are diagnostic rates, not serving performance.',
    'One selected captured layer/input is reused. This warmed isolated-layer footprint differs from a full-model execution.',
    'Read misses count requests that missed; fills count returned sectors. They need not agree because of request merging, other requests, aperture semantics, and replay.',
    'The script does not measure external writes. L2 write-request sectors are not writeback traffic.',
]


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def process_snapshot():
    relevant = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            name = (directory / 'comm').read_text().strip()
            if name not in COMPILERS and not name.startswith(('mistralrs', 'dense_backend', 'moe_dispatch', 'selected_scan')):
                continue
            fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        process = {'pid': int(directory.name), 'name': name, 'state': fields[0],
                   'starttime_ticks': fields[19]}
        relevant.append(process)
        if fields[0] not in ('T', 't', 'Z'):
            raise RuntimeError(f'Active model/compiler/replay process: {process}')
    return {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'processes': relevant, 'scope': 'Boundary snapshots, not a continuous process monitor.'}


def select_sample(captures):
    candidates = []
    for path in sorted(captures.glob('*.json')):
        data = json.loads(path.read_text())
        if (data.get('rows'), data.get('batch'), data.get('query_len'), data.get('derived')) == (56, 8, 7, False):
            candidates.append((path, data, sum(count > 0 for count in data['expert_occupancy'])))
    if not candidates:
        raise ValueError('No native B8xQ7 captures')
    median_unique = statistics.median(candidate[2] for candidate in candidates)
    source, sample, unique = min(candidates, key=lambda row: (abs(row[2] - median_unique), row[0].name))
    return source, sample, {
        'rule': 'Closest observed unique-expert count to the median among native B8xQ7 captures; filename breaks ties.',
        'median_unique_experts': median_unique, 'selected_unique_experts': unique,
        'selected_filename': source.name,
        'candidates': [{'filename': row[0].name, 'unique_experts': row[2]} for row in candidates],
    }


def supported_metrics(path):
    found = {}
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if fields and fields[0] in METRICS:
                found[fields[0]] = {'line': line_number, 'query_line': line.strip()}
    missing = set(METRICS) - found.keys()
    if missing:
        raise ValueError(f'Metrics absent from saved device query: {sorted(missing)}')
    return found


def make_command(args, out):
    return [str(NCU), '--config-file', '0', '--section-folder', str(SECTIONS),
            '--target-processes', 'application-only', '--replay-mode', 'kernel',
            '--cache-control', 'none', '--clock-control', 'none', '--import-sass', 'no',
            '--kernel-name-base', 'function', '--kernel-name', args.kernel_regex,
            '--launch-skip', str(args.launch_skip), '--launch-count', str(args.launch_count),
            '--metrics', ','.join(METRICS), '--disable-extra-suffixes',
            '--export', str(out / 'profile'), str(args.binary.resolve()),
            '--exact', args.test, '--ignored', '--nocapture', '--test-threads=1']


def run_command(argv, log, env):
    with log.open('w') as output:
        child = subprocess.Popen(argv, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            result = child.wait(timeout=TIMEOUT_SECONDS)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
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
    if units['ID']:
        raise ValueError('Expected NCU CSV unit row')
    kernels = []
    for row in reader:
        if not row.get('ID'):
            continue
        values = {}
        for metric in METRICS:
            value = float(row[metric].replace(',', ''))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {metric}: {row[metric]}')
            values[metric] = value
        duration = values['gpu__time_duration.sum']
        if units['gpu__time_duration.sum'] not in ('ns', 'nsecond') or duration <= 0:
            raise ValueError(f'Unexpected duration/unit: {duration} {units["gpu__time_duration.sum"]}')
        for metric in METRICS:
            if metric.startswith('lts__') and metric.endswith('.sum') and units[metric] != 'sector':
                raise ValueError(f'Unexpected unit for {metric}: {units[metric]}')
        sectors = values['lts__d_sectors_fill_sysmem.sum'] + values['lts__d_sectors_fill_device.sum']
        kernels.append({
            'id': row['ID'], 'kernel': row['Kernel Name'], 'grid': row['Grid Size'],
            'block': row['Block Size'], 'compute_capability': row['CC'],
            'metrics': {metric: {'value': row[metric], 'unit': units[metric]}
                        for metric in METRICS + EXTRA_METRICS if metric in row},
            'derived': {
                'l2_refill_sysmem_byte_equivalent': SECTOR_BYTES * values['lts__d_sectors_fill_sysmem.sum'],
                'l2_refill_device_byte_equivalent': SECTOR_BYTES * values['lts__d_sectors_fill_device.sum'],
                'l2_refill_total_byte_equivalent': SECTOR_BYTES * sectors,
                'l2_refill_byte_equivalent_per_profiled_second': SECTOR_BYTES * sectors * NANOSECONDS_PER_SECOND / duration,
                'l2_read_request_byte_equivalent': SECTOR_BYTES * values['lts__t_sectors_op_read.sum'],
                'l2_read_miss_byte_equivalent': SECTOR_BYTES * values['lts__t_sectors_op_read_lookup_miss.sum'],
            },
        })
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true', help='Run only after the coordinator releases the GPU slot; otherwise print a read-only plan.')
    parser.add_argument('--binary', type=Path, default=DEFAULT_BINARY)
    parser.add_argument('--binary-sha256', default=DEFAULT_BINARY_SHA)
    parser.add_argument('--build-metadata', type=Path, default=BUILD / 'metadata.json')
    parser.add_argument('--captures', type=Path, default=CAPTURES)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'masked_mmq')
    parser.add_argument('--label', default='masked_mmq')
    parser.add_argument('--test', default='flash_next_real_routing_replay')
    parser.add_argument('--kernel-regex', default='regex:^mul_mat_q$')
    parser.add_argument('--launch-skip', type=int, default=12)
    parser.add_argument('--launch-count', type=int, default=3)
    args = parser.parse_args()
    if args.launch_skip < 0 or args.launch_count < 1:
        parser.error('launch-skip must be nonnegative and launch-count positive')
    source, sample, selection = select_sample(args.captures)
    metrics = supported_metrics(QUERY)
    out = args.output_dir.resolve()
    env_overrides = {
        'MISTRALRS_MOE_REPLAY_DIR': str(out / 'selected_sample'),
        'MISTRALRS_MOE_BENCH_OUTPUT': str(out / 'profiled_test_output_not_benchmark.json'),
    }
    argv = make_command(args, out)
    plan = {'label': args.label, 'output': str(out), 'command': argv, 'environment_overrides': env_overrides,
            'binary_sha256_expected': args.binary_sha256, 'sample_selection': selection,
            'metrics': metrics, 'limits': LIMITS,
            'launch_selection': 'Default MMQ skip=12 skips one reference forward and three warmups, each with gate/up/down; captures the next native56 gate/up/down. Custom kernels/tests must re-audit their launch order.'}
    if not args.released:
        print(json.dumps(plan, indent=2))
        return
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'complete': False, 'started_unix': time.time(), 'plan': plan,
                'script_sha256': sha(__file__), 'commands': []}
    metadata_path = out / 'metadata.json'
    save_json(metadata_path, metadata)

    def interrupted(signum, _frame):
        raise InterruptedError(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(name, command, env=None):
        record = {'name': name, 'command': command, 'started_unix': time.time()}
        metadata['commands'].append(record)
        save_json(metadata_path, metadata)
        result = run_command(command, out / f'{name}.log', env)
        record.update(finished_unix=time.time(), returncode=0, log_sha256=sha(out / f'{name}.log'))
        save_json(metadata_path, metadata)
        return result

    try:
        metadata['isolation_before'] = process_snapshot()
        actual_sha = sha(args.binary)
        if actual_sha != args.binary_sha256:
            raise ValueError(f'Binary SHA mismatch: {actual_sha}')
        metadata['binary'] = {'path': str(args.binary.resolve()), 'sha256': actual_sha}
        if sha(QUERY) != QUERY_SHA:
            raise ValueError('Saved metric-discovery output changed')
        metadata['metric_discovery'] = {'path': str(QUERY), 'sha256': QUERY_SHA, 'device': 'GB20B / NVIDIA GB10'}
        metadata['build_metadata'] = {'path': str(args.build_metadata.resolve()), 'sha256': sha(args.build_metadata)}
        build = json.loads(args.build_metadata.read_text())
        metadata['build_metadata_contents'] = build
        metadata['runtime_environment'] = {key: value for key, value in sorted(os.environ.items())
                                           if key.startswith(('MISTRALRS_', 'CUDA_', 'CUBLAS_', 'NVIDIA_')) or key == 'LD_LIBRARY_PATH'}
        metadata['sample'] = {'path': str(source), 'sha256': sha(source), 'metadata': sample}
        selected = out / 'selected_sample'
        selected.mkdir()
        links = [source, args.captures / sample['tensor_file'], args.captures / sample['weights_file']]
        metadata['sample_links'] = []
        for path in links:
            resolved = path.resolve(strict=True)
            (selected / path.name).symlink_to(resolved)
            metadata['sample_links'].append({'path': str(resolved), 'bytes': path.stat().st_size,
                                             'sha256': sha(path) if path.suffix != '.gguf' else None,
                                             'hash_note': 'Large unchanged weights are referenced by original capture provenance; no redundant multi-GB hash here.' if path.suffix == '.gguf' else None})
        validation = args.captures.parent / 'capture.validation.json'
        metadata['capture_validation'] = {'path': str(validation), 'sha256': sha(validation)}
        metadata['ncu_version'] = run('version', [str(NCU), '--version']).strip()
        env = {key: value for key, value in os.environ.items() if not key.startswith('MISTRALRS_')}
        env.update(env_overrides)
        metadata['removed_ambient_mistralrs_keys'] = sorted(key for key in os.environ if key.startswith('MISTRALRS_'))
        run('profile', argv, env)
        raw = run('raw', [str(NCU), '--import', str(out / 'profile.ncu-rep'),
                          '--page', 'raw', '--csv', '--print-units', 'base'])
        kernels = parse_metrics(raw)
        if len(kernels) != args.launch_count:
            raise ValueError(f'Expected {args.launch_count} kernels, got {len(kernels)}')
        save_json(out / 'kernels.json', {'scope': 'NCU kernel replay of warmed isolated-layer projections',
                                         'limits': LIMITS, 'sector_bytes': SECTOR_BYTES,
                                         'label': args.label, 'sample_selection': selection, 'kernels': kernels})
        metadata.update(complete=True, finished_unix=time.time(), isolation_after=process_snapshot())
        save_json(metadata_path, metadata)
        (out / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.name}\n'
                                              for path in sorted(out.iterdir())
                                              if path.is_file() and path.name != 'SHA256SUMS'))
    except BaseException as error:
        metadata['error'] = {'type': type(error).__name__, 'message': str(error)}
        save_json(metadata_path, metadata)
        raise


if __name__ == '__main__':
    main()
