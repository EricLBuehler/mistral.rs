#!/usr/bin/env python3
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess

ROOT = Path('/home/ericbuehler/mistral.rs')
WORK = Path(__file__).resolve().parent
BASE = WORK.parent / 'compact_schedule/build/provenance.json'

def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--released', action='store_true')
    parser.add_argument('--attempt', default='build')
    parser.add_argument('--kernel-source', type=Path, default=WORK / 'grouped_gemv.cu')
    parser.add_argument('--harness-dir', type=Path, default=WORK)
    parser.add_argument('--object', type=Path)
    args = parser.parse_args()
    assert args.released, 'Coordinator must release build/GPU slot first'
    build = WORK / args.attempt
    build.mkdir(exist_ok=False)
    for name in ('grouped_gemv.cu', 'replay.source.rs', 'grouped_gemv_replay.rs'):
        shutil.copy2(args.kernel_source if name == 'grouped_gemv.cu' else args.harness_dir / name, build / name)
    baseline = json.loads(BASE.read_text())
    obj = build / 'grouped_gemv.o'
    nvcc = baseline['nvcc_commands'][0].copy()
    nvcc[nvcc.index('-c') + 1] = str(build / 'grouped_gemv.cu')
    nvcc[nvcc.index('-o') + 1] = str(obj)
    nvcc += ['--resource-usage']
    rustc = baseline['rustc_command'].copy()
    old_source = str(WORK.parent / 'compact_schedule/build/replay.source.rs')
    rustc[rustc.index(old_source)] = str(build / 'replay.source.rs')
    rustc[rustc.index('--out-dir') + 1] = str(build)
    original_native = Path(baseline['native_archive'])
    old_shadow = str(WORK.parent / 'compact_schedule/build/shadow')
    rustc = [str(original_native.parent) if item == old_shadow else item for item in rustc]
    rustc += ['-C', 'link-arg=' + str(obj)]
    output = build / 'moe_dispatch_bench-0dfba70e2949feab'
    sources = [build / name for name in ('grouped_gemv.cu', 'replay.source.rs', 'grouped_gemv_replay.rs')]
    originals = [original_native, ROOT / 'target/release/mistralrs', ROOT / 'mistralrs-quant/kernels/mmq_gguf/mmq_common.cuh', ROOT / 'mistralrs-quant/kernels/indexed_moe/indexed_moe.cu', ROOT / 'mistralrs-quant/kernels/mmq_gguf/mmq_gguf.cuh']
    metadata = dict(complete=False, started_at=now(), source_sha256={str(p): sha(p) for p in sources},
                    original_sha256={str(p): sha(p) for p in originals},
                    script_sha256=sha(__file__), commands=[], nvcc_command=nvcc,
                    rustc_command=rustc, rustc_environment=baseline['rustc_environment'],
                    scope='External Q4K gate/up plus Q4_1 down grouped GEMV, widths 2/4; original indexed GEMV arithmetic and regular Q8_1 activation format. Production archive and sources remain untouched.')
    path = build / 'provenance.json'
    def save():
        path.write_text(json.dumps(metadata, indent=2) + '\n')
    children = []
    def interrupted(sig, frame):
        raise InterruptedError(sig)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        save()
        if args.object:
            shutil.copy2(args.object, obj)
            metadata['reused_object'] = dict(path=str(args.object.resolve()), sha256=sha(args.object))
        commands = ([] if args.object else [('nvcc', nvcc, None)]) + [('rustc', rustc, dict(os.environ, **baseline['rustc_environment']))]
        for name, command, env in commands:
            record = dict(name=name, command=command, started_at=now())
            metadata['commands'].append(record)
            save()
            with (build / (name + '.log')).open('wb') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children.append(child)
                record['pid'] = child.pid
                save()
                record['returncode'] = child.wait()
            record['finished_at'] = now()
            record['log_sha256'] = sha(build / (name + '.log'))
            save()
            if record['returncode']:
                raise subprocess.CalledProcessError(record['returncode'], command)
        for p, digest in metadata['original_sha256'].items():
            assert sha(p) == digest, p
        metadata.update(complete=True, finished_at=now(), binary=str(output), binary_sha256=sha(output), object_sha256=sha(obj), originals_unchanged=True)
        save()
        print(output, flush=True)
    except BaseException as exc:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        metadata.update(error=repr(exc), failed_at=now())
        save()
        raise

if __name__ == '__main__':
    main()
