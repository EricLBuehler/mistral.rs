#!/usr/bin/env python3
"""Inspect a chat answer after the benchmark's long repetitive prompt."""

import argparse
import json
import urllib.request
from pathlib import Path

REPETITIONS = 900
MAX_TOKENS = 128
TIMEOUT_SECONDS = 120


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    args = parser.parse_args()
    prompt = (
        "The following is a repeated phrase.\n"
        + "The quiet river flows past the green forest. " * REPETITIONS
        + "\nWrite one sentence about the river."
    )
    body = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": MAX_TOKENS,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        result = json.load(response)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"request": body, "response": result}, indent=2, allow_nan=False)
        + "\n"
    )
    print(result["choices"][0]["message"])


if __name__ == "__main__":
    main()
