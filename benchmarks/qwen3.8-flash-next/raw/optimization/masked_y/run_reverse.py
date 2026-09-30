from pathlib import Path
import subprocess
W=Path(__file__).resolve().parent;OLD=Path('/home/ericbuehler/qwen4exp_work/real_routing_20260930/tile_variants')
for binary,output in [(W/'masked_y/build/moe_dispatch_bench.variant',W/'masked_y/replay_reverse'),(OLD/'build/moe_dispatch_bench.baseline',W/'baseline_reverse')]:
 subprocess.run(['python3',str(OLD/'run_variant.py'),'--binary',str(binary),'--variant','baseline','--provenance',str(W/'masked_y/build/provenance.json'),'--test-source',str(W/'masked_y/build/replay.source.rs'),'--output',str(output),'--baseline-dir',str(W/'baseline'),'--released'],check=True)
