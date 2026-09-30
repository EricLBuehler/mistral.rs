#!/usr/bin/env python3
"""CPU-only validation regression: well-formed capture passes, BF16 NaN fails."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

WORK = Path(__file__).resolve().parent


def tensor_file(path, rows):
    data = {
        'xs': ('BF16', [rows, 2560], struct.pack('<H', 0x3f80) * (rows * 2560)),
        'output': ('BF16', [rows, 2560], struct.pack('<H', 0x4000) * (rows * 2560)),
        'ids': ('U32', [rows, 10], b''.join(struct.pack('<I', (row * 13 + rank) % 512) for row in range(rows) for rank in range(10))),
        'weights': ('F32', [rows, 10], struct.pack('<f', 0.1) * (rows * 10)),
    }
    header = {}
    payload = b''
    for name, (dtype, shape, value) in data.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(payload), len(payload) + len(value)])
        payload += value
    serialized = json.dumps(header).encode()
    serialized += b' ' * (-len(serialized) % 8)
    path.write_bytes(struct.pack('<Q', len(serialized)) + serialized + payload)


with tempfile.TemporaryDirectory(prefix='capture_validator_test_', dir=WORK) as directory:
    root = Path(directory)
    (root / 'layer8.gguf').write_bytes(b'GGUF')
    for batch in [1, 6, 8]:
        rows = batch * 7
        for sample in range(4):
            name = f'c{batch}_{sample}'
            path = root / f'{name}.safetensors'
            tensor_file(path, rows)
            occupancy = [0] * 512
            for row in range(rows):
                for rank in range(10):
                    occupancy[(row * 13 + rank) % 512] += 1
            metadata = dict(tensor_file=path.name, case=f'c{batch}', stage='target_verify', rows=rows,
                            batch=batch, query_len=7, layer=8, hidden=2560, topk=10, experts=512,
                            derived=False, output_shape=[rows,2560], output_dtype='BF16', expert_occupancy=occupancy)
            (root / f'{name}.json').write_text(json.dumps(metadata))
    command = [sys.executable, str(WORK / 'validate_capture.py'), str(root), '--output', str(root / 'valid.json')]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert result.returncode == 0, result.stdout
    assert len(json.loads((root / 'valid.json').read_text())['samples']) == 12
    path = root / 'c1_0.safetensors'
    raw = bytearray(path.read_bytes())
    offset = 8 + struct.unpack('<Q', raw[:8])[0]
    raw[offset:offset+2] = struct.pack('<H', 0x7fc0)
    path.write_bytes(raw)
    command[-1] = str(root / 'invalid.json')
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
    assert result.returncode != 0 and 'math.isfinite' in result.stdout, result.stdout
    assert not (root / 'invalid.json').exists()
print('PASS: native shape/occupancy fixtures accepted; non-finite BF16 input rejected before packaging')
