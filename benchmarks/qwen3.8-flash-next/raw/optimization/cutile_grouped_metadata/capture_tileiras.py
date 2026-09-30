#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent / 'jit'
ARGS = sys.argv[1:]
REAL = '/usr/local/cuda/bin/tileiras'
started = time.time()
result = subprocess.run([REAL, *ARGS])
if result.returncode == 0 and '-o' in ARGS:
    ROOT.mkdir(exist_ok=True)
    cubin = Path(ARGS[ARGS.index('-o') + 1])
    sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
    target = ROOT / (sha + '.cubin')
    shutil.copy2(cubin, target)
    with (ROOT / (sha + '.resources.txt')).open('w') as output:
        subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', str(target)], stdout=output, check=True)
    with (ROOT / 'compiles.jsonl').open('a') as output:
        output.write(json.dumps(dict(started=started, seconds=time.time()-started, args=ARGS, cubin=str(target), sha256=sha)) + '\n')
sys.exit(result.returncode)
