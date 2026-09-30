#!/usr/bin/env python3
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
EARLIER = ROOT.parent / 'known_occupancy'
REPO = Path('/home/ericbuehler/mistral.rs')
PAUSED = Path('/home/ericbuehler/qwen4exp_work/scaling_20260930/paused_editor_processes.json')
TEST = 'flash_next_known_occupancy_microbenchmark'
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = json.loads((EARLIER/'metadata.json').read_text())
assert metadata['complete']
metadata = {key:value for key,value in metadata.items() if key not in ['steps','complete','finished_at','queued_at']}
metadata.update(complete=False, started_at=now(), steps=[], earlier_exploratory_metadata=str(EARLIER/'metadata.json'),
                isolation_note='Editor compiler tree paused by root before these runs; no active compiler processes allowed at run boundaries. No build occurs in this controller.')
binary = Path(metadata['binary'])
assert sha(binary) == metadata['binary_sha256']
source = Path(metadata['source'])
assert sha(source) == metadata['source_sha256']
shutil.copyfile(source,ROOT/source.name)
shutil.copyfile(PAUSED,ROOT/PAUSED.name)


def save():
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


def assert_isolation(tag):
    observed=[]
    compilers={'cargo','rustc','clippy-driver','cargo-clippy','nvcc','cicc','ptxas','cc1plus','g++','clang++','rust-analyzer'}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            comm=(path/'comm').read_text().strip()
            if comm not in compilers:
                continue
            stat=(path/'stat').read_text().split(') ',1)[1].split()
            state=stat[0]
        except (FileNotFoundError,ProcessLookupError):
            continue
        observed.append({'pid':int(path.name),'comm':comm,'state':state})
        assert state in ('T','t','Z'), f'Active compiler at {tag}: {path.name} {comm} {state}'
    metadata.setdefault('compiler_state_checks',[]).append({'tag':tag,'at':now(),'observed':observed})
    save()


save()
try:
    assert_isolation('before_all')
    print(f'{now()} CLEAN_TIMINGS_BEGIN',flush=True)
    for scheme,variable,directory in [('gguf','MISTRALRS_MOE_BENCH_GGUF','/home/ericbuehler/qwen4exp_work/gguf/UD-Q4_K_XL'),
                                      ('isq','MISTRALRS_MOE_BENCH_SAFETENSORS','/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540')]:
        assert_isolation('before_'+scheme)
        env=dict(os.environ); extra={variable:directory,'MISTRALRS_MOE_BENCH_OUTPUT':str(ROOT/f'{scheme}.json')}; env.update(extra)
        command=[str(binary),'--ignored','--exact',TEST,'--nocapture']
        step={'tag':scheme,'command':command,'env_overrides':extra,'cwd':str(REPO),'started_at':now()}
        metadata['steps'].append(step); save()
        with (ROOT/f'{scheme}.log').open('w') as log:
            result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        step.update(returncode=result.returncode,finished_at=now(),log_sha256=sha(ROOT/f'{scheme}.log'));save()
        print(f'{now()} END {scheme} returncode={result.returncode}',flush=True)
        assert result.returncode == 0
        assert_isolation('after_'+scheme)
    metadata.update(complete=True,finished_at=now());save()
    print(f'{now()} CLEAN_TIMINGS_END GPU_AND_BUILD_FREE',flush=True)
except BaseException as error:
    metadata['error']={'type':type(error).__name__,'message':str(error)};save()
    raise
