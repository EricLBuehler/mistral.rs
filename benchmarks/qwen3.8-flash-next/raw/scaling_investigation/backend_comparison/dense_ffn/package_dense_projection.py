#!/usr/bin/env python3
"""Archive the two finished layer probes without altering their measured evidence."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

WORK = Path(__file__).resolve().parent
DEST = Path('/home/ericbuehler/mistral.rs/benchmarks/qwen3.8-flash-next/raw/scaling_investigation/backend_comparison/dense_ffn')

def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

def main():
    DEST.mkdir(parents=True, exist_ok=False)
    variants = [('default', 'dense_projection'), ('affine_on', 'dense_projection_affine_on')]
    runs = {}
    for name, directory in variants:
        source = WORK / directory
        provenance = json.loads((source / 'provenance.json').read_text())
        assert provenance['complete'] and provenance['returncode'] == 0
        for line in (source / 'SHA256SUMS').read_text().splitlines():
            expected, filename = line.split('  ', 1)
            assert sha(source / filename) == expected, filename
        shutil.copytree(source, DEST / name)
        runs[name] = json.loads((source / 'summary.json').read_text())
    shutil.copytree(WORK / 'dense_projection_build', DEST / 'build')
    shutil.copy2(__file__, DEST / Path(__file__).name)
    rows = []
    for default, affine in zip(runs['default']['results'], runs['affine_on']['results'], strict=True):
        assert default['rows'] == affine['rows']
        rows.append({
            'rows': default['rows'],
            'default_fp8': default['fp8'],
            'default_q4k': default['q4k'],
            'affine_on_fp8_control': affine['fp8'],
            'affine_on_q4k': affine['q4k'],
            'default_q4k_over_affine_q4k_median_ratio': default['q4k']['median_stream_ms'] / affine['q4k']['median_stream_ms'],
        })
    comparison = {
        'scope': runs['default']['scope'],
        'dimensions': {'hidden': 5120, 'intermediate': 17408},
        'layer': 0,
        'hardware': 'GB10, compute capability 12.1 (asserted by the probe)',
        'production_default': 'MISTRALRS_GGUF_AFFINE_BACKEND unset',
        'diagnostic_override': 'MISTRALRS_GGUF_AFFINE_BACKEND=on',
        'statistic': 'Median of seven alternating backend rounds, ten FFN iterations per round, after three warmups; raw samples preserved.',
        'scaling': 'Aggregate row-throughput ratio M * median_time(M=1) / median_time(M).',
        'source_dispatch': 'FP8 packed gate/up plus down; Q4K separate gate/up weights with production fused FFN/gate-up hooks plus down. All three Q4K FFN weights are Q4K, matching the dense loader plan.',
        'numerical_validation': 'Both backends checked against FP32 matmul using their own dequantized weights; finite outputs, relative RMS < 0.06, cosine > 0.995. Cross-backend differences recorded without equality assertions.',
        'build_command': 'cargo test --release -p mistralrs-quant --features cuda,cutile --test dense_backend_bench --no-run --message-format=json-render-diagnostics',
        'check_command': 'cargo check --release -p mistralrs-quant --features cuda,cutile --test dense_backend_bench',
        'format_command': 'rustfmt --edition 2021 --check mistralrs-quant/tests/dense_backend_bench.rs',
        'validation': {'build': 'passed', 'check': 'passed', 'format': 'passed', 'default_gpu_test': 'passed', 'affine_on_gpu_test': 'passed'},
        'limitations': runs['default']['limitations'] + [
            'Default and affine-on are separate fresh processes, not paired interleaved variants; FP8 samples in each process are retained as controls.',
            'Affine preparation is checked before warmup and timings. Conversion and repacking setup costs are excluded.',
            'Batch row count is a layer shape, not a serving concurrency or speculative acceptance measurement.',
            'The optional affine backend was not enabled for the matched full-model Q4K control. No full-model affine speedup is claimed.',
        ],
        'results': rows,
    }
    (DEST / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False) + '\n')
    lines = [
        '# Dense FFN backend probe', '',
        'Actual layer-0 Qwen3.8-27B-FP8 weights; hidden 5120, intermediate 17408. Each timing covers gate/up, SiLU product, and down projection through the production dispatch and fusion hooks. FP8 uses packed gate/up; immediate Q4K uses separate weights with fused computation.', '',
        'Q4K is derived from the same FP8 checkpoint after dequantization. This double-quantized experiment tests backend timing, not model quality. Both paths passed checks against their own dequantized-weight FP32 references.', '',
        '| Rows | Default FP8 ms | Default Q4K ms | Affine-on Q4K ms | Default / affine Q4K |',
        '| ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(f"| {row['rows']} | {row['default_fp8']['median_stream_ms']:.4f} | {row['default_q4k']['median_stream_ms']:.4f} | {row['affine_on_q4k']['median_stream_ms']:.4f} | {row['default_q4k_over_affine_q4k_median_ratio']:.3f}x |")
    lines += ['',
        'Default M1-to-M8 aggregate row-throughput scaling is 7.92x for FP8 and 6.02x for Q4K. The optional affine run prepares all three Q4K projections from M8 onward and reaches 8.45x scaling; M1 and M6 still use MMVQ. The default FFN probe is consistent with a partial backend contribution to the full-model scaling difference.', '',
        'Source-dispatch evidence: FP8 uses its eight-column MMA path through M32, then cuTile above M32. Default Q4K uses MMVQ through M8 and MMQ above M8. The optional `MISTRALRS_GGUF_AFFINE_BACKEND=on` run records successful affine preparation at M8 and above. These are checked dispatch branches, not measured kernel traces.', '',
        'Seven alternating rounds, ten iterations per round, three warmups; all raw stream/host intervals, errors and backend preparation flags are retained. Default and affine-on run in separate fresh processes. Compilation and model serving were absent during both probes; editor processes remained stopped. Release build, relevant release `cargo check`, and rustfmt passed.', '',
        'The inputs are synthetic BF16. Upstream fused activation quantization, graph replay, model-wide cache behavior, speculative acceptance, and serving overhead are outside this test. CUDA-event intervals include CPU launch gaps. Repacking setup costs are excluded. These layer timings do not predict a full-model affine speedup.', '',
        'Reproduction: build with the command in `comparison.json`, then run the archived controller with `--binary <test-executable> --output-dir <new-directory>`. Repeat in a fresh process with the explicit affine environment override. The controller uses the pinned local snapshot recorded in provenance; update its paths on another machine. `build/` contains the successful build/check logs; each run preserves source, runner, results, provenance and an inner manifest. External originals remain under `/home/ericbuehler/qwen4exp_work/kernel_comparison_20260930/`.', '',
    ]
    (DEST / 'README.md').write_text('\n'.join(lines))
    provenance = {'packaged_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'external_originals': {name: str(WORK / directory) for name, directory in variants},
                  'package_script_sha256': sha(Path(__file__)),
                  'originals_preserved': True}
    (DEST / 'archive_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    files = sorted(p for p in DEST.rglob('*') if p.is_file() and p != DEST / 'SHA256SUMS')
    (DEST / 'SHA256SUMS').write_text(''.join(f'{sha(path)}  {path.relative_to(DEST)}\n' for path in files))
    print(f'Packaged {len(files)} files at {DEST}')

if __name__ == '__main__':
    main()
