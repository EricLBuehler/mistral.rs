#!/usr/bin/env python3
"""Measure a scan of real selected GGUF expert payloads, not a hardware bandwidth limit."""
import argparse
import ctypes as ct
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import time

import numpy as np

WORK = Path(__file__).resolve().parent
CAPTURES = Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/runtime/captures')
EXPERTS = 512
ROUNDS = 7
REPEATS = 10
WARMUPS = 3
CHUNK_KIB = (8, 32, 64, 256)
PROJECTIONS = ('gate', 'up', 'down', 'total')
EXPECTED = {
    'blk.8.ffn_gate_exps.weight': ('gate', (2560, 640, 512), 12, 921600),
    'blk.8.ffn_up_exps.weight': ('up', (2560, 640, 512), 12, 921600),
    'blk.8.ffn_down_exps.weight': ('down', (640, 2560, 512), 3, 1024000),
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def gguf_layout(path):
    tensors = []
    with path.open('rb') as source:
        assert source.read(4) == b'GGUF'
        version, count, metadata = struct.unpack('<IQQ', source.read(20))
        assert (version, count, metadata) == (2, 3, 0)
        for _ in range(count):
            length, = struct.unpack('<Q', source.read(8))
            name = source.read(length).decode('utf8')
            rank, = struct.unpack('<I', source.read(4))
            dims = struct.unpack('<' + 'Q' * rank, source.read(8 * rank))
            dtype, offset = struct.unpack('<IQ', source.read(12))
            projection, expected_dims, expected_dtype, expert_bytes = EXPECTED[name]
            assert dims == expected_dims and dtype == expected_dtype
            tensors.append(dict(name=name, projection=projection, dimensions=list(dims),
                                ggml_type=dtype, relative_offset=offset,
                                expert_bytes=expert_bytes, bytes=expert_bytes * EXPERTS))
        payload = (source.tell() + 31) // 32 * 32
    for tensor in tensors:
        tensor['file_offset'] = payload + tensor['relative_offset']
        assert tensor['file_offset'] % 16 == 0
        assert tensor['file_offset'] + tensor['bytes'] <= path.stat().st_size
    tensors.sort(key=lambda t: PROJECTIONS.index(t['projection']))
    assert len({t['name'] for t in tensors}) == 3
    for previous, current in zip(tensors, tensors[1:]):
        assert previous['file_offset'] + previous['bytes'] == current['file_offset']
    assert tensors[-1]['file_offset'] + tensors[-1]['bytes'] == path.stat().st_size
    return tensors


def route_cases(sample):
    cases = []
    for path in sorted(CAPTURES.glob('*.json')):
        if sample is not None and path.name != sample:
            continue
        data = json.loads(path.read_text())
        if 'expert_occupancy' not in data:
            continue
        counts = data['expert_occupancy']
        assert len(counts) == EXPERTS and all(isinstance(n, int) and n >= 0 for n in counts)
        assert data['derived'] is False and data['layer'] == 8
        assert data['rows'] in (1, 7, 42, 56)
        assert sum(counts) == data['rows'] * 10
        selected = [expert for expert, count in enumerate(counts) if count]
        cases.append(dict(source_metadata=path.name, source_metadata_sha256=sha(path),
                          source_tensor_file=data['tensor_file'], rows=data['rows'],
                          batch=data['batch'], query_len=data['query_len'],
                          selected_experts=selected, selected_count=len(selected),
                          assignments=sum(counts), max_expert_rows=max(counts),
                          assignments_per_selected_expert=sum(counts) / len(selected)))
    assert len(cases) == (1 if sample is not None else 26)
    return cases


def self_check():
    generator = np.random.default_rng(70193)
    checks = 0
    for vectors in (1, 31, 512, 577, 16385):
        words = generator.integers(0, 2**32, size=(3, vectors, 4), dtype=np.uint32)
        reference = np.bitwise_xor.reduce(words, axis=1)
        for chunk in CHUNK_KIB:
            size = chunk * 1024 // 16
            partial = [np.bitwise_xor.reduce(words[:, start:start+size], axis=1)
                       for start in range(0, vectors, size)]
            actual = np.bitwise_xor.reduce(np.stack(partial), axis=0)
            assert np.array_equal(reference, actual)
            checks += 1
    return dict(checks=checks, passed=True, scope='CPU XOR chunk partition and tail identities')


def api():
    library = ct.CDLL(str(WORK / 'selected_scan.so'))
    pointer = ct.c_void_p
    library.scan_create.argtypes = [ct.POINTER(pointer), ct.POINTER(ct.c_size_t), ct.c_int,
                                   ct.POINTER(pointer), ct.POINTER(ct.c_uint64)]
    library.scan_select.argtypes = [pointer, ct.POINTER(ct.c_uint32), ct.c_int]
    library.scan_measure.argtypes = [pointer, ct.c_int, ct.c_size_t, ct.c_int, ct.c_int, ct.POINTER(ct.c_float)]
    library.scan_checksums.argtypes = [pointer, ct.c_int, ct.c_size_t, ct.POINTER(ct.c_uint32), ct.c_size_t]
    library.scan_destroy.argtypes = [pointer]
    library.scan_error.argtypes = [ct.c_int]
    library.scan_error.restype = ct.c_char_p
    for name in ('scan_create', 'scan_select', 'scan_measure', 'scan_checksums', 'scan_destroy'):
        getattr(library, name).restype = ct.c_int
    return library


def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sample')
    parser.add_argument('--eviction-control', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--released', action='store_true')
    args = parser.parse_args()
    assert args.plan_only or args.released, 'GPU run requires root release'
    assert not args.output.exists(), 'preserve previous evidence'
    args.output.mkdir(parents=True)
    weights_path = CAPTURES / 'layer8.gguf'
    layout = gguf_layout(weights_path)
    cases = route_cases(args.sample)
    if args.validate_only and args.sample is None:
        names = ('c8_target_verify_b8_q7_04.json', 'c1_r00_target_decode_b1_q1_00.json',
                 'c8_target_verify_b8_q7_00.json')
        cases = [next(case for case in cases if case['source_metadata'] == name) for name in names]
    metadata = dict(complete=False, started_at=now(), command=list(os.sys.argv),
                    scope='Real layer-8 compressed selected-weight scan; no dequantization, MMA, activations, routing construction, or output model calculation. Precomputed expert-index lookup remains in the scan.',
                    limitation='Same-layer repeated warm-cache diagnostic. Q7 captures are fixed-depth diagnostic operands, not the current adaptive-depth serving distribution. Logical bytes per event time are not actual DRAM traffic, a physical lower bound, or an end-to-end throughput prediction.',
                    threads_per_block=256, chunk_kib=CHUNK_KIB, layout=layout, cases=cases,
                    rounds=0 if args.validate_only else ROUNDS,
                    repetitions=REPEATS, warmups=0 if args.validate_only else WARMUPS,
                    validation_only=args.validate_only,
                    cache_modes=['warm', 'eviction_attempt'] if args.eviction_control else ['warm'],
                    eviction_definition='Write a separate allocation of max(64 MiB,4*reported L2 size) before each individually timed scan. Flush excluded from CUDA events; eviction is attempted, not proven.',
                    cpu_self_check=self_check(), numpy_version=np.__version__,
                    sources={name:sha(WORK/name) for name in ['selected_scan.cu','selected_scan.so','run_scan.py','build.log']})
    save(args.output / 'metadata.json', metadata)
    if args.plan_only:
        metadata.update(complete=True, plan_only=True, finished_at=now())
        save(args.output / 'metadata.json', metadata)
        return
    def interrupted(number, _frame):
        raise InterruptedError(number)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    library = api()
    def check(error):
        if error:
            raise RuntimeError(library.scan_error(error).decode())
    context = ct.c_void_p()
    monitor = None
    monitor_files = []
    try:
        metadata['weights_file'] = str(weights_path)
        metadata['weights_file_sha256'] = sha(weights_path)
        validation = json.loads((CAPTURES.parent / 'capture.validation.json').read_text())
        expected = {entry['path']: entry['sha256'] for entry in validation['files']}
        assert metadata['weights_file_sha256'] == expected[weights_path.name]
        metadata['capture_validation_sha256'] = sha(CAPTURES.parent / 'capture.validation.json')
        for case in cases:
            assert case['source_metadata_sha256'] == expected[case['source_metadata']]
        mapped = np.memmap(weights_path, mode='r', dtype=np.uint8)
        views = [mapped[t['file_offset']:t['file_offset']+t['bytes']].view(np.uint32).reshape(EXPERTS, -1, 4) for t in layout]
        cpu_checksums = [np.bitwise_xor.reduce(view, axis=1) for view in views]
        save(args.output / 'cpu_checksums.json', {name:values.tolist() for name,values in zip(PROJECTIONS, cpu_checksums)})
        metadata['cpu_checksums_sha256'] = sha(args.output / 'cpu_checksums.json')
        pointers = (ct.c_void_p * 3)(*(view.ctypes.data for view in views))
        sizes = (ct.c_size_t * 3)(*(t['expert_bytes'] for t in layout))
        info = (ct.c_uint64 * 5)()
        check(library.scan_create(pointers, sizes, EXPERTS, ct.byref(context), info))
        metadata['device_memory'] = dict(zip(['free_before','total','l2_bytes','flush_bytes','free_after_upload'], map(int,info)))
        metadata['device_allocated_weight_bytes'] = sum(t['bytes'] for t in layout)
        command = ['nvidia-smi','--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu','--format=csv,nounits','--loop-ms=1000']
        metadata['monitor_command'] = command
        monitor_files = [(args.output/'gpu_monitor.csv').open('wb'), (args.output/'gpu_monitor.stderr.log').open('wb')]
        monitor = subprocess.Popen(command, stdout=monitor_files[0], stderr=monitor_files[1], start_new_session=True)
        metadata['monitor_pid'] = monitor.pid
        save(args.output/'metadata.json',metadata)
        def measure(projection, chunk, repeats, eviction):
            result = ct.c_float()
            check(library.scan_measure(context, projection, chunk, repeats, int(eviction), ct.byref(result)))
            assert np.isfinite(result.value) and result.value > 0
            return float(result.value)
        records = []
        for case_index, case in enumerate(cases):
            selected = np.asarray(case['selected_experts'], dtype=np.uint32)
            check(library.scan_select(context, selected.ctypes.data_as(ct.POINTER(ct.c_uint32)), len(selected)))
            validation_records = []
            timings = []
            for chunk_kib in CHUNK_KIB:
                chunk = chunk_kib * 1024
                for projection, tensor in enumerate(layout):
                    blocks = (tensor['expert_bytes'] + chunk - 1) // chunk
                    checksums = np.empty((len(selected), blocks, 4), dtype=np.uint32)
                    check(library.scan_checksums(context, projection, chunk, checksums.ctypes.data_as(ct.POINTER(ct.c_uint32)), checksums.size))
                    folded = np.bitwise_xor.reduce(checksums, axis=1)
                    assert np.array_equal(folded, cpu_checksums[projection][selected]), (case['source_metadata'], projection, chunk)
                    validation_records.append(dict(projection=PROJECTIONS[projection], chunk_kib=chunk_kib,
                                                   grid=[blocks,len(selected),1], block=[256,1,1],
                                                   logical_bytes=len(selected)*tensor['expert_bytes'],
                                                   checksum_bytes=checksums.nbytes, per_expert_checksum_verified=True,
                                                   checksum_sha256=hashlib.sha256(checksums.tobytes()).hexdigest()))
                if not args.validate_only:
                    for projection in (0,1,2,-1):
                        measure(projection, chunk, WARMUPS, False)
            for round_index in range(metadata['rounds']):
                for offset in range(len(CHUNK_KIB)):
                    chunk_kib = CHUNK_KIB[(round_index+case_index+offset) % len(CHUNK_KIB)]
                    for projection_offset in range(len(PROJECTIONS)):
                        label = PROJECTIONS[(round_index+projection_offset) % len(PROJECTIONS)]
                        projection = -1 if label == 'total' else PROJECTIONS.index(label)
                        expert_bytes = sum(t['expert_bytes'] for t in layout) if projection == -1 else layout[projection]['expert_bytes']
                        modes = metadata['cache_modes'] if round_index % 2 == 0 else list(reversed(metadata['cache_modes']))
                        for mode in modes:
                            elapsed = measure(projection, chunk_kib*1024, REPEATS, mode == 'eviction_attempt')
                            payload = len(selected)*expert_bytes
                            timings.append(dict(round=round_index,projection=label,chunk_kib=chunk_kib,cache_mode=mode,
                                                event_ms_per_scan=elapsed,logical_bytes_per_scan=payload,
                                                logical_payload_gb_per_s=payload/elapsed/1e6))
            record = dict(**case, validation=validation_records, timings=timings)
            records.append(record)
            save(args.output/'results.partial.json',dict(results=records))
            print(f"Completed {case_index+1}/{len(cases)}: {case['source_metadata']}",flush=True)
        save(args.output/'results.json',dict(results=records))
        assert monitor.poll() is None, 'GPU monitor stopped before completion'
        stop(monitor)
        metadata['monitor_returncode'] = monitor.returncode
        check(library.scan_destroy(context)); context = ct.c_void_p()
        metadata.update(complete=True, finished_at=now(), results_sha256=sha(args.output/'results.json'),
                        gpu_monitor_sha256=sha(args.output/'gpu_monitor.csv'))
        save(args.output/'metadata.json',metadata)
    except BaseException as error:
        metadata.update(complete=False,error=repr(error),finished_at=now())
        save(args.output/'metadata.json',metadata)
        raise
    finally:
        stop(monitor)
        for output in monitor_files:
            output.close()
        if context.value:
            library.scan_destroy(context)


if __name__ == '__main__':
    main()
