#!/usr/bin/env python3
"""CPU-only lifecycle checks against an ephemeral local fake completion endpoint."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

WORK = Path(__file__).resolve().parent


def run_case(fail):
    with tempfile.TemporaryDirectory(prefix='capture_runner_test_', dir=WORK) as directory:
        root = Path(directory)
        capture = root / 'captures'
        output = root / 'requests.json'
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                control = json.loads((capture / 'control.json').read_text())
                observed.append(dict(case=control['case'], prompt=request['prompt']))
                assert 'logprobs' not in request, 'capture must preserve ordinary greedy verifier eligibility'
                count = request['max_tokens'] - 1 if fail else request['max_tokens']
                response = dict(usage=dict(prompt_tokens=10, completion_tokens=count),
                    choices=[dict(text='valid diagnostic output')])
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = subprocess.run([sys.executable, str(WORK / 'capture_requests.py'),
                '--directory', str(capture), '--output', str(output),
                '--base-url', f'http://127.0.0.1:{server.server_port}'],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        report = json.loads(output.read_text())
        assert not (capture / 'control.json').exists(), 'capture must disarm on success and failure'
        if fail:
            assert result.returncode != 0 and report['complete'] is False
            assert 'Expected 128 tokens, got' in result.stdout
            assert len(observed) == 1
        else:
            assert result.returncode == 0, result.stdout
            assert report['complete'] and len(report['results']) == 24
            assert len(report['controls']) == 10
            assert len(observed) == 24
            for prefix in ['c1', 'c6', 'c8']:
                prompts = [entry['prompt'] for entry in observed if entry['case'].startswith(prefix)]
                assert len(prompts) == 8 and len(set(prompts)) == 8
            assert {entry['case'] for entry in observed if entry['case'].startswith('c1')} == {
                f'c1_r{index:02}' for index in range(8)}


run_case(False)
run_case(True)
print('PASS: 24 diverse requests, ordinary greedy body without logprobs, per-prompt C1 controls, count mismatch rejection, cleanup on both paths')
