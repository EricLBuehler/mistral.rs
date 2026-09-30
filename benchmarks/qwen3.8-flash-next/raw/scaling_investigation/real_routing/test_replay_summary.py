#!/usr/bin/env python3
"""CPU-only paired-ratio regression for the replay summary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

WORK = Path(__file__).resolve().parent
rows = []
for index, (gemv, grouped) in enumerate([(2.0, 1.0), (8.0, 2.0)]):
    numerical = dict(relative_rms=0.01, cosine=0.9999)
    rows.append(dict(capture=dict(case=f'c1_r{index:02}', stage='target_verify', gate_dtype='Q4K', up_dtype='Q4K', down_dtype='Q4_1'),
        selection='native', derived=False, source_row_indices=list(range(7)), rows=7,
        expert_occupancy=[7]*10+[0]*502, active_experts=10, max_expert_rows=7, source_metadata=f'input{index}.json',
        gemv=[dict(stream_elapsed_ms_per_layer=gemv,host_elapsed_ms_per_layer=gemv+1)]*7,
        grouped=[dict(stream_elapsed_ms_per_layer=grouped,host_elapsed_ms_per_layer=grouped+1)]*7,
        numerical=numerical,captured_vs_gemv=numerical,captured_vs_grouped=numerical))
with tempfile.TemporaryDirectory(prefix='replay_summary_test_', dir=WORK) as directory:
    root=Path(directory)
    source=root/'input.json'
    output=root/'summary.json'
    source.write_text(json.dumps(dict(results=rows,topk=10,rounds=7,layer=8)))
    result=subprocess.run([sys.executable,str(WORK/'summarize_replay.py'),str(source),'--output',str(output)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    assert result.returncode==0,result.stdout
    summary=json.loads(output.read_text())['groups'][0]
    assert summary['samples']==2 and summary['gemv']['stream_elapsed_ms_per_layer']['median']==5
    assert summary['grouped']['stream_elapsed_ms_per_layer']['median']==1.5
    assert summary['gemv_over_grouped_stream_ratio']['median']==3
    assert summary['mean_assignments_per_selected_expert']['median']==7
    assert summary['selected_distinct_expert_weight_bytes']['median']==28672000
print('PASS: paired-ratio medians preserved separately from ratio-of-medians; logical occupancy/payload arithmetic checked')
