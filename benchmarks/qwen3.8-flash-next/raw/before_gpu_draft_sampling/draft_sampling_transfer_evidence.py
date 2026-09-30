#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent
VOCAB = 248320
F32_BYTES = 4
QUERY = """SELECT COUNT(*) AS transfer_count,
       COALESCE(SUM(m.bytes),0) AS total_bytes,
       COALESCE(SUM(m.end-m.start),0) AS summed_transfer_duration_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
JOIN ENUM_CUDA_MEMCPY_OPER AS k ON k.id=m.copyKind
WHERE k.name='CUDA_MEMCPY_KIND_DTOH' AND m.bytes=?"""


def main():
    cases = []
    for tag in ['c1', 'c6', 'c8']:
        path = WORKDIR / f'profile_{tag}.sqlite'
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute('PRAGMA cache_size=-2048')
            counts = dict(connection.execute(QUERY, (VOCAB * F32_BYTES,)).fetchone())
            metadata = dict(connection.execute(
                'SELECT name,value FROM META_DATA_EXPORT WHERE name IN '
                "('EXPORT_PRODUCT_VERSION','EXPORT_SCHEMA_VERSION','EXPORT_PARAM_INPUT_PATH_ABS','EXPORT_TIME_UTC')"
            ).fetchall())
        cases.append(dict(capture=tag, source_sqlite=str(path),
                          source_sqlite_bytes=path.stat().st_size,
                          export_metadata=metadata, **counts))
    evidence = {
        'vocab_size': VOCAB,
        'bytes_per_logit': F32_BYTES,
        'matched_transfer_size_bytes': VOCAB * F32_BYTES,
        'query': QUERY,
        'query_parameters': [VOCAB * F32_BYTES],
        'cases': cases,
        'interpretation': 'Full F32 vocabulary D2H transfers are absent at C1 but numerous at C6/C8. Source tracing identifies per-row draft sampling as a consumer; transfer-size matching alone does not attribute every transfer to a source call.',
        'duration_caveat': 'Summed memcpy durations measure recorded transfer activity only. They exclude CPU argmax/softmax, allocation, host synchronization and launch/wait overhead, so they are not the full cost of draft sampling.',
        'profile_caveat': 'These are diagnostic profiles from Nsight2025.6.3 hardware tracing, not unprofiled throughput measurements. Kernel activity timestamps are missing in all three exports; memcpy records remain available.',
        'source_paths': {
            'draft_loop': 'mistralrs-core/src/speculative/builtin_mtp.rs:329',
            'row_sampling': 'mistralrs-core/src/speculative/proposer.rs:455 (pre-fastpath implementation)',
            'greedy_eligibility': 'mistralrs-core/src/sampler.rs:1365',
            'host_logits_copy': 'mistralrs-core/src/sampler.rs:2183',
            'host_argmax_and_softmax': 'mistralrs-core/src/sampler.rs:1163',
        },
        'reproduce': f'python3 {Path(__file__).resolve()}',
    }
    output = WORKDIR / 'draft_sampling_transfer_evidence.json'
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
