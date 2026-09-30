#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import statistics

root = Path(__file__).resolve().parent
results = []
for scheme in ('gguf', 'isq'):
    report = json.loads((root / f'{scheme}.json').read_text())
    for case in report['results']:
        normal = [sample['stream_elapsed_ms_per_layer'] for sample in case['normal_bound']]
        tight = [sample['stream_elapsed_ms_per_layer'] for sample in case['tight_bound']]
        assert len(normal) == len(tight) == report['rounds']
        assert case['max_expert_rows'] <= case['tight_column_bound']
        row = {key: case[key] for key in ('scheme', 'routing', 'rows', 'active_experts', 'max_expert_rows', 'normal_column_bound', 'tight_column_bound', 'gate_dtype', 'down_dtype', 'numerical', 'packed_reference')}
        row.update(normal_median_ms=statistics.median(normal), tight_median_ms=statistics.median(tight),
                   normal_min_ms=min(normal), normal_max_ms=max(normal), tight_min_ms=min(tight), tight_max_ms=max(tight),
                   paired_speedup_ratios=[a/b for a,b in zip(normal,tight)])
        row['median_paired_speedup'] = statistics.median(row['paired_speedup_ratios'])
        results.append(row)
summary = {
    'scope': 'Actual layer8 Flash-Next weights, synthetic BF16 activations/routes, same eager unfused projection pipeline with normal bound=token rows versus known bound=8.',
    'validation': 'Before timing, GPU expert-bound counts must equal synthetic expected counts and maximum<=8. Shapes/finite values checked; tight versus normal RMS<1e-3 and cosine>.999999; normal versus packed production pipeline also checked.',
    'controls': 'At1/8 rows both bounds select identical8-column tile and identical grid coverage.',
    'limitations': [
        'The tight bound is valid only for these verified synthetic routes; applying it generally can omit rows.',
        'This measures ideal known-occupancy tile/grid behavior, not a safe fixed-width clamp preserving full coverage.',
        'Single warmed layer and deterministic synthetic routing do not establish full-model throughput or real routing occupancy.',
        'Projection pipeline quantizes gate/up activations separately on both sides; production packed pipeline shares that quantization.',
        'CUDA event elapsed includes host launch gaps; it is not summed kernel duration.',
    ],
    'results': results,
    'raw_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('gguf.json','isq.json')},
}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['Known-occupancy grouped MMQ diagnostic','',summary['scope'],'',
       'Scheme Routing Rows Active MaxRows Normal_ms Tight8_ms PairedSpeedup']
for row in results:
    lines.append(f"{row['scheme']} {row['routing']} {row['rows']} {row['active_experts']} {row['max_expert_rows']} {row['normal_median_ms']:.4f} {row['tight_median_ms']:.4f} {row['median_paired_speedup']:.3f}")
lines.extend(['',summary['validation'],'',summary['controls'],'',*summary['limitations']])
(root/'findings.txt').write_text('\n'.join(lines)+'\n')
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n' for path in sorted(root.iterdir()) if path.is_file() and path.name!='SHA256SUMS'))
print('\n'.join(lines))
