#!/usr/bin/env python3
"""Validate captured tensors and hash the diagnostic bundle without CUDA libraries."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_tensors(path):
    raw = path.read_bytes()
    header_size = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    payload = memoryview(raw)[8 + header_size:]
    tensors = {}
    for name, metadata in header.items():
        if name == '__metadata__':
            continue
        start, end = metadata['data_offsets']
        assert 0 <= start <= end <= len(payload)
        data = payload[start:end]
        dtype = metadata['dtype']
        if dtype == 'BF16':
            values = [struct.unpack('<f', struct.pack('<I', value << 16))[0]
                      for (value,) in struct.iter_unpack('<H', data)]
        elif dtype == 'F32':
            values = [value for (value,) in struct.iter_unpack('<f', data)]
        elif dtype == 'U32':
            values = [value for (value,) in struct.iter_unpack('<I', data)]
        else:
            raise AssertionError(f'unexpected capture dtype {dtype}')
        assert len(values) == math.prod(metadata['shape'])
        assert all(math.isfinite(value) for value in values)
        tensors[name] = (metadata, values)
    assert set(tensors) == {'xs', 'ids', 'weights', 'output'}
    return tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--requests', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve prior validation'
    records = []
    grouped = collections.defaultdict(list)
    files = {args.directory / 'layer8.gguf'}
    with (args.directory / 'layer8.gguf').open('rb') as handle:
        assert handle.read(4) == b'GGUF'
    for path in sorted(args.directory.glob('*.json')):
        metadata = json.loads(path.read_text())
        if 'tensor_file' not in metadata:
            continue
        tensor_path = args.directory / metadata['tensor_file']
        tensors = read_tensors(tensor_path)
        rows = metadata['rows']
        assert metadata['batch'] * metadata['query_len'] == rows
        assert metadata['layer'] == 8 and metadata['hidden'] == 2560
        assert metadata['topk'] == 10 and metadata['experts'] == 512
        assert metadata['derived'] is False
        assert metadata['output_shape'] == [rows, 2560] and metadata['output_dtype'] == 'BF16'
        for name in ['xs', 'output']:
            assert tensors[name][0]['shape'] == [rows, 2560]
            assert tensors[name][0]['dtype'] == 'BF16'
        for name, dtype in [('ids', 'U32'), ('weights', 'F32')]:
            assert tensors[name][0]['shape'] == [rows, 10]
            assert tensors[name][0]['dtype'] == dtype
        ids = tensors['ids'][1]
        occupancy = [0] * 512
        for start in range(0, len(ids), 10):
            assert len(set(ids[start:start+10])) == 10
            for expert in ids[start:start+10]:
                assert 0 <= expert < 512
                occupancy[expert] += 1
        assert metadata['expert_occupancy'] == occupancy
        assert all(weight >= 0 for weight in tensors['weights'][1])
        record = dict(metadata_file=path.name, case=metadata['case'], stage=metadata['stage'],
                      batch=metadata['batch'], query_len=metadata['query_len'], rows=rows,
                      active_experts=sum(count > 0 for count in occupancy),
                      max_expert_rows=max(occupancy), all_tensors_finite=True)
        records.append(record)
        grouped[(metadata['case'], metadata['stage'], rows)].append(record)
        files.update([path, tensor_path])
    assert records, 'no captures'
    assert sum(record['stage'] == 'target_verify' and record['rows'] == 7 for record in records) >= 4
    assert sum(record['stage'] == 'target_verify' and record['rows'] == 42 for record in records) >= 4
    assert sum(record['stage'] == 'target_verify' and record['batch'] == 8 and record['query_len'] == 7 for record in records) >= 4
    summary = dict(scope='bounded diagnostic captures, not a natural routing or MTP-depth distribution',
                   samples=records, by_case_stage_rows=[dict(case=case, stage=stage, rows=rows, samples=len(samples))
                       for (case, stage, rows), samples in sorted(grouped.items())])
    if args.requests:
        requests = json.loads(args.requests.read_text())
        assert requests['complete'] is True and len(requests['results']) == 24
        assert requests['logprobs_requested'] is False
        assert all(record['logprobs_requested'] is False and record['expected_output_count_met']
                   for record in requests['results'])
        summary['requests_complete'] = True
        summary['requests_sha256'] = digest(args.requests)
    summary['files'] = [dict(path=path.name, bytes=path.stat().st_size, sha256=digest(path))
                        for path in sorted(files)]
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['by_case_stage_rows'], indent=2))


if __name__ == '__main__':
    main()
