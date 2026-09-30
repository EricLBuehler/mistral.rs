#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Arm/disarm diagnostic capture without querying the server.')
parser.add_argument('action', choices=['arm', 'disarm'])
parser.add_argument('--directory', required=True, type=Path)
parser.add_argument('--case')
parser.add_argument('--rows', nargs='+', type=int)
parser.add_argument('--stages', nargs='+', default=['target_decode', 'target_verify'])
parser.add_argument('--max-per-shape', type=int, default=4)
parser.add_argument('--every-n', type=int, default=4)
parser.add_argument('--skip-per-shape', type=int, default=0)
args = parser.parse_args()
args.directory.mkdir(parents=True, exist_ok=True)
control = args.directory / 'control.json'
if args.action == 'disarm':
    control.unlink(missing_ok=True)
else:
    assert args.case and args.case.isascii() and all(c.isalnum() or c == '_' for c in args.case)
    assert args.rows and all(1 <= row <= 64 for row in args.rows)
    assert 1 <= args.max_per_shape <= 8 and args.every_n >= 1 and args.skip_per_shape >= 0
    assert all(stage in ['target_decode', 'target_verify', 'target_prefill'] for stage in args.stages)
    value = dict(case=args.case, row_counts=args.rows, stages=args.stages,
                 max_per_shape=args.max_per_shape, every_n=args.every_n,
                 skip_per_shape=args.skip_per_shape)
    temporary = control.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(control)
    print(json.dumps(value, sort_keys=True))
