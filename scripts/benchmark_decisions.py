"""Benchmark native CLM and optionally compare with a saved run."""

import argparse
import copy
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

REQUEST = {
    "model": "default",
    "state": "Customer: my invoice was charged twice and nobody answers the phone!",
    "questions": {
        "urgency": {"type": "noul", "instructions": "Is this urgent?"},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Charges, invoices, refunds",
                "technical": "Bugs and outages",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}


def call(base_url, body):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/systemone",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    return result, (time.perf_counter() - start) * 1000


def difference(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max((difference(a[k], b[k]) for k in a), default=0)
    if isinstance(a, (int, float)):
        return abs(a - b)
    assert a == b, (a, b)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--idle-seconds", type=float, default=0)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if not math.isfinite(args.idle_seconds) or args.idle_seconds < 0:
        parser.error("--idle-seconds must be finite and nonnegative")
    cases = ["repeated_request", "new_state_fixed_actions", "new_state_32_actions"]
    reference = json.loads(args.compare.read_text()) if args.compare else None
    requests = []
    if reference:
        requests = [
            (r["case"], r["iteration"], r["request"]) for r in reference["samples"]
        ]
    else:
        nonce = time.time_ns()
        for case in cases:
            for iteration in range(args.iterations + 1):
                body = copy.deepcopy(REQUEST)
                if case != "repeated_request":
                    body["state"] += f" Ticket reference: {nonce}-{iteration}."
                if case == "new_state_32_actions":
                    body["questions"] = {
                        "topic": {
                            "type": "choice",
                            "instructions": "Which topic fits best?",
                            "criteria": {
                                str(i): f"Topic number {i}." for i in range(32)
                            },
                        }
                    }
                requests.append((case, iteration, body))
    samples = []
    for i, (case, iteration, body) in enumerate(requests):
        if args.idle_seconds:
            time.sleep(args.idle_seconds)
        answer, elapsed = call(args.base_url, body)
        if case != "repeated_request" and answer["usage"]["input_tokens"] == 0:
            raise RuntimeError(
                f"{case} was already cached; restart the server before replaying a baseline"
            )
        row = {
            "case": case,
            "iteration": iteration,
            "warmup": iteration == 0,
            "idle_seconds": args.idle_seconds,
            "ms": elapsed,
            "input_tokens": answer["usage"]["input_tokens"],
            "request": body,
            "answers": answer["answers"],
        }
        if reference:
            row["max_answer_difference"] = difference(
                reference["samples"][i]["answers"], answer["answers"]
            )
        samples.append(row)
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("request", "answers")}
            ),
            flush=True,
        )
    summary = {}
    for case in cases:
        measured = [r for r in samples if r["case"] == case and not r["warmup"]]
        summary[case] = {"median_ms": statistics.median(r["ms"] for r in measured)}
        if reference:
            summary[case]["baseline_median_ms"] = reference["summary"][case][
                "median_ms"
            ]
            summary[case]["max_answer_difference"] = max(
                r["max_answer_difference"] for r in measured
            )
    print(json.dumps(summary, indent=2))
    args.output.write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2)
    )


if __name__ == "__main__":
    main()
