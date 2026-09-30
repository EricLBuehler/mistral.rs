import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path('/home/ericbuehler/mistral.rs')
OUT = pathlib.Path(__file__).parent
SNAPSHOT = pathlib.Path('/home/ericbuehler/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540')
BINARY = ROOT / 'target/release/mistralrs'
command = [str(BINARY), 'serve', '--no-ui', '-p', '1234', '--max-model-len', '16384', '--max-seqs', '8', '--prefix-cache-n', '0', '-m', str(SNAPSHOT), '--isq', 'q4k', '--mtp']
launch = command
env = dict(os.environ, HF_HUB_OFFLINE='1', RUST_LOG='info')
metadata = {
    'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': command,
    'launcher_command': launch,
    'env': {key: env[key] for key in ('HF_HUB_OFFLINE', 'RUST_LOG')},
    'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'source_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
    'binary_sha256': hashlib.file_digest(BINARY.open('rb'), 'sha256').hexdigest(),
}
def save():
    (OUT / 'isq_mtp_tuner.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
save()
stop = threading.Event()
start = time.monotonic()
def monitor():
    with (OUT / 'isq_mtp_tuner.memory.jsonl').open('w') as output:
        while not stop.is_set():
            memory = {line.split(':')[0]: int(line.split()[1]) for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'SwapFree:'))}
            memory['elapsed'] = time.monotonic() - start
            output.write(json.dumps(memory) + '\n')
            output.flush()
            stop.wait(2)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
with (OUT / 'isq_mtp_tuner.server.log').open('w') as log:
    server = subprocess.Popen(launch, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    metadata['launcher_pid'] = server.pid
    save()
    try:
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'Launcher exited {server.returncode} during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:1234/v1/models', timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() - start > 2400:
                raise TimeoutError('Startup exceeded 2400s')
            time.sleep(2)
        metadata['startup_seconds'] = time.monotonic() - start
        expected_cmdline = b'\0'.join(part.encode() for part in command) + b'\0'
        model_pids = []
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                if (proc / 'cmdline').read_bytes() == expected_cmdline:
                    model_pids.append(int(proc.name))
            except OSError:
                pass
        assert len(model_pids) == 1, model_pids
        metadata['model_pid'] = model_pids[0]
        save()
        (OUT / 'ready').write_text(str(metadata['startup_seconds']))
        print('Server ready:', metadata['startup_seconds'], flush=True)
        while not (OUT / 'stop').exists():
            time.sleep(2)
    finally:
        stop.set()
        thread.join()
        model_pid = metadata.get('model_pid')
        if model_pid is not None:
            try:
                os.kill(model_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
