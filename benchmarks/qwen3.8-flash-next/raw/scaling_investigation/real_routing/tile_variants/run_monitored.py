#!/usr/bin/env python3
import subprocess
from pathlib import Path
W=Path(__file__).resolve().parent
for variant,name in [('baseline','monitored_unset'),('16','monitored_16'),('32','monitored_32')]:
 command=['python3',str(W/'run_variant.py'),'--binary',str(W/'build/moe_dispatch_bench.variant'),'--variant',variant,'--provenance',str(W/'build/provenance.json'),'--test-source',str(W/'build/replay.source.rs'),'--output',str(W/name),'--baseline-dir',str(W/'monitored_baseline'),'--released']
 subprocess.run(command,check=True)
