#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, datetime
W=Path(__file__).resolve().parent
RUNNER=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/tile_variants/run_variant.py')
parser=argparse.ArgumentParser();parser.add_argument('--released',action='store_true');args=parser.parse_args();assert args.released
validation=json.loads((W/'validation/metadata.json').read_text());assert validation['complete']
build=json.loads((W/'build/provenance.json').read_text());assert build['complete']
commands=[]
for name,binary,provenance,baseline in [('baseline',validation['binaries']['moe_dispatch_bench']['path'],W/'validation/metadata.json',W.parent/'masked_y_bounded/replay'),('compact',build['variant_binary'],W/'build/provenance.json',W/'baseline')]:
 cmd=['python3',str(RUNNER),'--binary',binary,'--variant','baseline','--provenance',str(provenance),'--test-source',str(W/'build/replay.source.rs'),'--output',str(W/name),'--baseline-dir',str(baseline),'--released']
 record=dict(name=name,command=cmd,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());commands.append(record);(W/'comparison_commands.json').write_text(json.dumps(commands,indent=2));print(name,flush=True)
 p=subprocess.run(cmd);record.update(returncode=p.returncode,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(W/'comparison_commands.json').write_text(json.dumps(commands,indent=2));assert p.returncode==0
cmd=['python3',str(RUNNER.with_name('summarize_variants.py')),'--baseline',str(W/'baseline'),'--variants',str(W/'compact'),'--output',str(W/'summary.json')]
print('summarize',flush=True);subprocess.run(cmd,check=True)
