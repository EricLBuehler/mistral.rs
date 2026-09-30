#!/usr/bin/env python3
"""Relink an identical-Rust masked baseline and validate an external compact CUDA archive."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, subprocess

WORK = Path(__file__).resolve().parent
REPO = Path('/home/ericbuehler/mistral.rs')
TESTS = ('grouped_mmq_tail_cuda_tests', 'grouped_mmq_packed_cuda_tests')
TIMEOUT_SECONDS = 240

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--released', action='store_true')
    args = parser.parse_args()
    assert args.released, 'Coordinator must release the GPU/build slot'
    build = json.loads((WORK/'build/provenance.json').read_text())
    assert build['complete'] and build['originals_unchanged']
    root = WORK/'validation'
    root.mkdir(exist_ok=False)
    meta = dict(complete=False, started_at=now(), commands=[], binaries={},
                build_provenance_sha256=sha(WORK/'build/provenance.json'),
                source_hashes={}, method='Baseline and compact link the same frozen replay source and Rust dependencies, replacing only Q4K/Q4_1 native objects. Independent CPU-oracle tails and dynamic-routing CUDA graph replay must pass before timings.')
    metadata = root/'metadata.json'
    children = []
    def save():
        metadata.write_text(json.dumps(meta, indent=2)+'\n')
    def run(command, name, env):
        record = dict(name=name, command=command, started_at=now())
        meta['commands'].append(record);save()
        with (root/(name+'.log')).open('wb') as log:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child);record['pid']=child.pid;save()
            code = child.wait(timeout=TIMEOUT_SECONDS)
        record.update(returncode=code, finished_at=now(), log_sha256=sha(root/(name+'.log')));save()
        assert code == 0, (name, code)
        return (root/(name+'.log')).read_text()
    def interrupted(number, _frame):
        raise InterruptedError(number)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    env = dict(os.environ, **build['rustc_environment'])
    env.pop('MISTRALRS_DIAGNOSTIC_MOE_MMQ_TILE', None)
    env.pop('MISTRALRS_GGUF_AFFINE_BACKEND', None)
    base_command = build['rustc_command']
    shadow = str(WORK/'build/shadow')
    replay_source = str(WORK/'build/replay.source.rs')
    try:
        meta['rust_dependency_hashes'] = {}
        for argument in base_command:
            if '=' in argument and (argument.endswith('.rlib') or argument.endswith('.so')):
                path = argument.split('=', 1)[1]
                if Path(path).is_file():meta['rust_dependency_hashes'][path]=sha(path)
        for name in ('moe_dispatch_bench', *TESTS):
            out = root/name;out.mkdir()
            command = list(base_command)
            command[command.index('--out-dir')+1]=str(out)
            if name == 'moe_dispatch_bench':
                command=[str(Path(build['native_archive']).parent) if value == shadow else value for value in command]
            else:
                source=REPO/'mistralrs-quant/tests'/(name+'.rs')
                frozen=out/(name+'.rs');frozen.write_bytes(source.read_bytes())
                meta['source_hashes'][str(source)]=sha(source)
                command=[name if value=='moe_dispatch_bench' else str(frozen) if value==replay_source else value for value in command]
            env['CARGO_CRATE_NAME']=name
            run(command, 'link_'+name, env)
            binary=out/(name+'-0dfba70e2949feab')
            meta['binaries'][name]=dict(path=str(binary), sha256=sha(binary));save()
        for name in TESTS:
            binary=meta['binaries'][name]['path']
            run([binary,'--test-threads=1'],name,env)
            result=run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck',
                        '--error-exitcode','99','--target-processes','all','--print-limit','20',
                        binary,'--test-threads=1'],name+'.memcheck',env)
            assert 'ERROR SUMMARY: 0 errors' in result
        for path, expected in build['original_sources'].items():assert sha(path)==expected,path
        assert sha(build['native_archive'])==build['native_archive_sha256']
        for path, expected in meta['rust_dependency_hashes'].items():assert sha(path)==expected,path
        for path, expected in meta['source_hashes'].items():assert sha(path)==expected,path
        meta.update(complete=True,finished_at=now());save()
        print('Compact correctness validation complete',flush=True)
    except BaseException as error:
        for child in children:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        meta.update(error=repr(error),failed_at=now());save();raise

if __name__ == '__main__':main()
