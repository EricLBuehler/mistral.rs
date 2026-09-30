#!/usr/bin/env python3
"""Run the prepared dense FFN benchmark after the GPU/build slot is released."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


REPO = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
SNAPSHOT = Path('/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a')
TEST = 'dense_qwen38_fp8_vs_q4k_microbenchmark'
SOURCE = REPO / 'mistralrs-quant/tests/dense_backend_bench.rs'
COMPILERS = {'cargo', 'rustc', 'clippy-driver', 'cargo-clippy', 'nvcc', 'cicc', 'ptxas', 'cc1plus', 'g++', 'clang++', 'rust-analyzer'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def isolation():
    seen = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            name = (path / 'comm').read_text().strip()
            if name not in COMPILERS and name != 'mistralrs':
                continue
            state = (path / 'stat').read_text().split(') ', 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
        seen.append({'pid': int(path.name), 'name': name, 'state': state})
        assert state in ('T', 't', 'Z'), f'Active compiler/model before isolated run: {seen[-1]}'
    return {'at': now(), 'processes': seen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path, default=WORK / 'dense_projection')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'provenance.json').exists(), 'Preserve earlier experiment'
    assert not (out / 'results.json').exists(), 'Preserve earlier results'
    binary = args.binary.resolve()
    command = [str(binary), '--exact', TEST, '--ignored', '--nocapture', '--test-threads=1']
    env = dict(os.environ)
    overrides = {'MISTRALRS_DENSE_BENCH_SAFETENSORS': str(SNAPSHOT), 'MISTRALRS_DENSE_BENCH_OUTPUT': str(out / 'results.json')}
    env.update(overrides)
    index = json.loads((SNAPSHOT / 'model.safetensors.index.json').read_text())['weight_map']
    names = [f'model.language_model.layers.0.mlp.{projection}.{suffix}'
             for projection in ('gate_proj', 'up_proj', 'down_proj')
             for suffix in ('weight', 'weight_scale_inv')]
    sources = [
        'mistralrs-quant/src/blockwise_fp8/mod.rs',
        'mistralrs-quant/src/blockwise_fp8/mma.rs',
        'mistralrs-quant/src/cutile/fp8_gemm.rs',
        'mistralrs-quant/kernels/blockwise_fp8/blockwise_fp8_mma.cu',
        'mistralrs-quant/src/gguf/mod.rs',
        'mistralrs-quant/src/gguf/fast_mmvq.rs',
        'mistralrs-quant/src/gguf/fast_mmq.rs',
        'mistralrs-quant/src/gguf/packed_affine.rs',
        'mistralrs-quant/kernels/mmvq_gguf/mmvq_gguf.cu',
        'mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh',
        'mistralrs-quant/kernels/mmq_gguf/mmq_instance_q4_k.cu',
        'mistralrs-core/src/layers.rs',
        'mistralrs-core/src/ops.rs',
        'mistralrs-quant/src/lib.rs',
        'Cargo.lock',
    ]
    metadata = {
        'complete': False, 'prepared_at': now(), 'command': command, 'cwd': str(REPO),
        'binary': str(binary), 'binary_sha256': sha(binary),
        'source': str(SOURCE), 'source_sha256': sha(SOURCE),
        'controller_sha256': sha(Path(__file__)),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=REPO)).hexdigest(),
        'environment_overrides': overrides,
        'performance_environment': {key: env.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'MISTRALRS_GGUF_AFFINE_BACKEND', 'CUDA_LAUNCH_BLOCKING', 'RAYON_NUM_THREADS', 'OMP_NUM_THREADS')},
        'snapshot': str(SNAPSHOT),
        'weight_map': {name: index[name] for name in names},
        'checkpoint_files': {name: sha(SNAPSHOT / name) for name in sorted({index[name] for name in names} | {'config.json', 'model.safetensors.index.json'})},
        'implementation_sha256': {name: sha(REPO / name) for name in sources},
        'limitations': ['Q4K derives from already-FP8 weights and is a timing control, not a quality comparison.',
                        'The model is not loaded; this is an isolated FFN pipeline using synthetic activations.',
                        'CUDA-event intervals include CPU launch gaps; graph replay is not measured.'],
    }
    def save():
        (out / 'provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copy2(SOURCE, out / 'dense_backend_bench.measured.rs')
    shutil.copy2(Path(__file__), out / 'run_dense_projection.py')
    shutil.copy2(WORK / 'paused_editor_processes.json', out / 'paused_editor_processes.json')
    metadata['isolation_before'] = isolation()
    metadata['started_at'] = now()
    save()
    print(f'{now()} DENSE_PROJECTION_BEGIN', flush=True)
    try:
        start = time.monotonic()
        with (out / 'test.log').open('w') as log:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        metadata.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - start,
                        finished_at=now(), log_sha256=sha(out / 'test.log'))
        save()
        result.check_returncode()
        metadata['isolation_after'] = isolation()
        raw = json.loads((out / 'results.json').read_text())
        summary = []
        base = {}
        for row in raw['results']:
            item = {'rows': row['rows']}
            for backend in ('fp8', 'q4k'):
                samples = [sample['stream_elapsed_ms_per_layer'] for sample in row[backend]['timings']]
                median = statistics.median(samples)
                if row['rows'] == 1:
                    base[backend] = median
                item[backend] = {'median_stream_ms': median, 'mean_stream_ms': statistics.mean(samples),
                                 'stdev_stream_ms': statistics.stdev(samples), 'samples_stream_ms': samples,
                                 'aggregate_row_throughput_scaling_vs_M1': row['rows'] * base[backend] / median,
                                 'reference_error': row[backend]['own_dequantized_weight_reference'],
                                 'dispatch': row[backend]['dispatch']}
            item['paired_q4k_over_fp8_ms_ratios'] = [q['stream_elapsed_ms_per_layer'] / f['stream_elapsed_ms_per_layer']
                                                    for f, q in zip(row['fp8']['timings'], row['q4k']['timings'])]
            summary.append(item)
        (out / 'summary.json').write_text(json.dumps({'scope': raw['pipeline'], 'limitations': raw['limitations'],
                                                    'results': summary}, indent=2, allow_nan=False) + '\n')
        metadata.update(complete=True, results_sha256=sha(out / 'results.json'), summary_sha256=sha(out / 'summary.json'))
        save()
        (out / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.name}\n' for path in sorted(out.iterdir()) if path.is_file() and path.name != 'SHA256SUMS'))
        print(f'{now()} DENSE_PROJECTION_END', flush=True)
    except BaseException as error:
        metadata['error'] = {'type': type(error).__name__, 'message': str(error), 'at': now()}
        save()
        raise


if __name__ == '__main__':
    main()
