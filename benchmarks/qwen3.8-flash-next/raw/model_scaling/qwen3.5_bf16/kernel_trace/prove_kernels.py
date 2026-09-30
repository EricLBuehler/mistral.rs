#!/usr/bin/env python3
"""Inspect completed, aligned SQLite exports for actual cuTile MoE graph nodes."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'analysis')
    args = parser.parse_args()
    manifest_path = args.analysis / 'manifest.json'
    for name, expected_sha in json.loads(manifest_path.read_text()).items():
        if sha(args.analysis / name) != expected_sha:
            raise ValueError(f'Changed analysis artifact: {name}')
    provenance = json.loads((args.analysis / 'provenance.json').read_text())
    if not provenance.get('complete'):
        raise ValueError('Complete export/alignment before inspecting kernels')
    result = {
        'complete': False,
        'scope': 'Observed kernel presence in separate post-warmup diagnostic traces, not throughput or exhaustive coverage.',
        'analyzer_sha256': sha(Path(__file__)),
        'analysis_provenance_sha256': sha(args.analysis / 'provenance.json'),
        'phases': [],
        'source': {
            'path': '/home/ericbuehler/mistral.rs/mistralrs-quant/src/cutile/fused_moe.rs',
            'entry': 'fused_moe_kernel',
            'meaning': 'BF16 cuTile grouped GEMM used twice per gated-expert FFN (gate/up, then down); source and binary hashes are retained by the capture.',
        },
        'limitations': [
            'A positive graph-node match proves this kernel executed during captured decode-graph work; startup/JIT logs alone do not.',
            'Finite phases include prefill/startup/drain; graphs identify observed decode nodes, not a latency-isolated steady-state decode window.',
            'Nsight collection warnings remain attached. Absence of a fallback name does not prove every possible kernel event was collected.',
            'Recorded kernel time and launch counts are diagnostic; do not publish them as unprofiled serving performance.',
        ],
    }
    for phase in ('profile_c1', 'profile_c8'):
        database = args.analysis / f'{phase}.sqlite'
        expected = next(row for row in provenance['phases'] if row['name'] == phase)
        if sha(database) != expected['sqlite_sha256']:
            raise ValueError('SQLite changed after export')
        alignment = json.loads((args.analysis / f'{phase}.alignment.json').read_text())
        window = next(row for row in alignment['windows'] if row['name'] == 'client_request_envelope')
        with sqlite3.connect(f'file:{database.resolve()}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA cache_size=-2048')
            columns = {row['name'] for row in connection.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
            if not {'demangledName', 'graphId', 'start', 'end'} <= columns:
                raise ValueError('Kernel name/timing/graph schema missing')
            params = (window['end_ns'], window['start_ns'])
            query = '''SELECT s.value AS kernel, k.graphId, k.gridX, k.gridY, k.gridZ,
                       k.blockX, k.blockY, k.blockZ, COUNT(*) AS launches,
                       SUM(MIN(k.end, ?)-MAX(k.start, ?)) AS clipped_duration_ns
                       FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
                       WHERE k.start < ? AND k.end > ? AND INSTR(s.value, 'fused_moe_kernel') > 0
                       GROUP BY s.value,k.graphId,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ'''
            matches = [dict(row) for row in connection.execute(query, params + params)]
            graph_matches = sum(row['launches'] for row in matches if row['graphId'])
            all_matches = sum(row['launches'] for row in matches)
            fallback = [dict(row) for row in connection.execute('''
                SELECT s.value AS kernel,COUNT(*) AS launches FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.demangledName WHERE k.start < ? AND k.end > ?
                AND (LOWER(s.value) LIKE '%cutlass%moe%' OR LOWER(s.value) LIKE '%fused_moe_fp8%')
                GROUP BY s.value''', params)]
        result['phases'].append({
            'name': phase, 'kernel_launches': all_matches, 'graph_kernel_launches': graph_matches,
            'observed_decode_graph_execution': graph_matches > 0,
            'matches': matches, 'other_matching_moe_names': fallback,
            'exact_window': window, 'profiler_diagnostics': alignment['profiler_diagnostics'],
            'sqlite_sha256': expected['sqlite_sha256'],
        })
    result['complete'] = all(row['observed_decode_graph_execution'] for row in result['phases'])
    target = args.analysis / 'kernel_proof.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + '\n')
    proof_manifest = {
        'kernel_proof.json': sha(target),
        'analysis_manifest_sha256': sha(manifest_path),
        'proof_analyzer_sha256': sha(Path(__file__)),
    }
    (args.analysis / 'kernel_proof.manifest.json').write_text(json.dumps(proof_manifest, indent=2) + '\n')
    print(json.dumps({'complete': result['complete'], 'phases': [
        {key: row[key] for key in ('name', 'kernel_launches', 'graph_kernel_launches', 'observed_decode_graph_execution')}
        for row in result['phases']]}, indent=2))
    if not result['complete']:
        raise SystemExit('Expected cuTile decode graph nodes were not observed; inspect preserved evidence')


if __name__ == '__main__':
    main()
