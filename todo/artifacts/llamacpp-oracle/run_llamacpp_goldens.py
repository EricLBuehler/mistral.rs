"""Run the 8 Kolibri-1 golden prompts through llama.cpp and save token-exact results.

The native /completion endpoint is used rather than /v1/completions because only the native
one returns `completion_probabilities`, which carries the generated token **ids** and the
per-position top-k logprobs. The OpenAI endpoint returns text only, and splitting that text
into "tokens" silently compares whitespace-separated words against token ids.

Prompts are sent as explicit token ids, so the run is token-exact against the goldens.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

GGUF = os.environ.get("KOLIBRI1_GGUF", "/tmp/kolibri1-llamacpp-auto.gguf")
BUILD = os.environ.get("LLAMACPP_BUILD", "/tmp/opencode/llamacpp-build/bin")
GOLDENS = os.environ.get(
    "KOLIBRI1_GOLDENS", "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens.json"
)
OUT = os.environ.get("OUT", "/tmp/opencode/kolibri1_goldens_llamacpp.json")
PORT = int(os.environ.get("PORT", "8124"))
THREADS = os.environ.get("THREADS", "32")
MAX_NEW = 32
N_PROBS = 20


def wait_for_server(url, timeout=3600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def main():
    goldens = json.load(open(GOLDENS))
    prompts = goldens["prompts"]
    if isinstance(prompts, dict):
        prompts = list(prompts.values())

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = BUILD + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        f"{BUILD}/llama-server",
        "-m", GGUF,
        "--port", str(PORT),
        "-c", "4096",
        "-t", THREADS,
        "--no-warmup",
        "-np", "1",
    ]
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{PORT}"
        if not wait_for_server(url):
            print("server did not come up", flush=True)
            return 1
        print("server up", flush=True)

        out = {
            "model": GGUF,
            "max_new_tokens": MAX_NEW,
            "decode": "greedy, temperature 0",
            "provenance": "llama.cpp commit 62318cd57 (Kolibri-1 support), CPU aarch64, bf16 GGUF",
            "prompts": [],
        }
        for i, p in enumerate(prompts):
            ids = p["input_ids"]
            body = json.dumps(
                {
                    "prompt": ids,
                    "n_predict": MAX_NEW,
                    "temperature": 0.0,
                    "top_k": 1,
                    "n_probs": N_PROBS,
                    "cache_prompt": False,
                }
            ).encode()
            req = urllib.request.Request(
                url + "/completion", data=body, headers={"Content-Type": "application/json"}
            )
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=3600) as r:
                res = json.loads(r.read())
            dt = time.time() - t0

            probs = res.get("completion_probabilities") or []
            gen_ids = [c["id"] for c in probs]
            entry = {
                "prompt": p["prompt"],
                "input_ids": ids,
                "generated_ids": gen_ids,
                "wall_s": round(dt, 2),
            }
            if probs:
                entry["final_topk"] = {
                    "top20_token_ids": [a["id"] for a in probs[-1].get("top_logprobs", [])],
                    "top20_logprobs": [a.get("logprob") for a in probs[-1].get("top_logprobs", [])],
                }
                entry["per_position_logprobs"] = [
                    {
                        "id": c["id"],
                        "logprob": c.get("logprob"),
                        "top": [
                            [a["id"], a.get("logprob")] for a in c.get("top_logprobs", [])
                        ],
                    }
                    for c in probs
                ]
            out["prompts"].append(entry)
            print(f"  [{i}] {p['prompt'][:40]!r} -> {gen_ids[:8]} ({dt:.1f}s)", flush=True)

        json.dump(out, open(OUT, "w"), indent=1)
        print("wrote", OUT)
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
