"""Generate, inspect sampling traces, compare watermarking, and detect output."""

import argparse
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

SCHEMES = ("synthid", "kgw", "unigram", "exponential", "inverse_transform", "mpac", "textgrain")


def post(base_url, endpoint, body):
    request = Request(
        f"{base_url.rstrip('/')}/v1/{endpoint}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        return json.load(response)


def generated_text(response, endpoint):
    if endpoint == "completions":
        return response["choices"][0]["text"]
    if endpoint == "chat/completions":
        return response["choices"][0]["message"]["content"]
    if endpoint == "messages":
        return "".join(
            block["text"] for block in response["content"] if block["type"] == "text"
        )
    return "".join(
        block["text"]
        for item in response["output"]
        if item["type"] == "message"
        for block in item["content"]
        if block["type"] == "output_text"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheme", choices=SCHEMES)
    parser.add_argument(
        "--endpoint",
        choices=("chat/completions", "completions", "responses", "messages"),
        default="chat/completions",
    )
    parser.add_argument("--base-url", default="http://localhost:1234")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--trace", action="store_true", help="Include bounded sampling diagnostics")
    parser.add_argument("--compare", action="store_true", help="Trace both an unmarked and marked run")
    parser.add_argument("--trace-steps", type=int, default=32)
    parser.add_argument("--trace-candidates", type=int, default=32)
    parser.add_argument("--trace-layers", type=int, default=8)
    parser.add_argument(
        "--generation-tournament", action="store_true",
        help="Capture the actual generation bracket, when the selected policy executes one",
    )
    parser.add_argument("--trace-matches", type=int, default=4095)
    parser.add_argument("--textgrain-transport", action="store_true", help="Include native textGrain costs and coupling")
    parser.add_argument("--textgrain-iterations", type=int, help="Bound captured textGrain solver iterations")
    parser.add_argument(
        "--generation-policy", choices=("probability_updates", "tournament", "block_then_token"),
        help="Select SynthID or textGrain sampling independently of tracing",
    )
    parser.add_argument(
        "--depth", type=int,
        help="SynthID generation depth; explicit tournaments support at most 20",
    )
    parser.add_argument(
        "--teaching-tournament", type=int, choices=range(1, 5), metavar="ROUNDS",
        help="Include a separate SynthID teaching bracket with 1 to 4 rounds",
    )
    parser.add_argument("--teaching-seed", type=int, default=0)
    parser.add_argument(
        "--vocab-size", type=int, help="Model output vocabulary, including padding"
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Detect the returned text using model retokenization",
    )
    args = parser.parse_args()
    if args.generation_tournament:
        args.trace = True
    if args.depth is not None and args.scheme != "synthid":
        parser.error("depth requires synthid")
    if args.generation_policy is not None:
        allowed = {"synthid": ("probability_updates", "tournament"), "textgrain": ("probability_updates", "block_then_token")}
        if args.generation_policy not in allowed.get(args.scheme, ()):
            parser.error("generation policy is incompatible with the selected scheme")
    if args.textgrain_transport or args.textgrain_iterations is not None:
        args.trace = True
        if args.scheme != "textgrain":
            parser.error("native textGrain traces require textgrain")
    if args.teaching_tournament is not None:
        args.trace = True
        if args.scheme != "synthid":
            parser.error("teaching tournaments require synthid")
        if not 0 <= args.teaching_seed < 2**64:
            parser.error("teaching seed must be an unsigned 64-bit integer")
    if (args.trace or args.compare) and args.endpoint not in ("chat/completions", "completions"):
        parser.error("sampling traces require chat/completions or completions")
    config_path = (
        Path(__file__).resolve().parents[1] / "watermarking" / f"{args.scheme}.json"
    )
    config = json.loads(config_path.read_text())
    config["key"] = os.environ["MISTRALRS_WATERMARK_KEY"]
    if args.generation_policy is not None:
        config["generation_policy"] = args.generation_policy
    if args.depth is not None:
        config["depth"] = args.depth
    if args.vocab_size is not None and "vocab_size" in config:
        config["vocab_size"] = args.vocab_size
    text = "Write a long story about a lunar garden."
    body = {
        "model": args.model,
        "temperature": 0.8,
        "top_k": 40,
        "enable_thinking": False,
        "watermark": config,
        "seed": args.seed,
    }
    if args.endpoint == "completions":
        body.update(prompt=text, max_tokens=args.max_tokens)
    elif args.endpoint == "responses":
        body.update(input=text, max_output_tokens=args.max_tokens)
    else:
        body.update(messages=[{"role": "user", "content": text}], max_tokens=args.max_tokens)
    if args.trace or args.compare:
        body["sampling_trace"] = {
            "max_steps": args.trace_steps,
            "max_candidates": args.trace_candidates,
            "max_layers": args.trace_layers,
        }
        if args.teaching_tournament is not None:
            body["sampling_trace"]["teaching_tournament"] = {
                "rounds": args.teaching_tournament,
                "seed": args.teaching_seed,
            }
        if args.generation_tournament:
            body["sampling_trace"]["generation_tournament"] = {
                "max_matches": args.trace_matches
            }
        if args.textgrain_transport or args.textgrain_iterations is not None:
            body["sampling_trace"]["textgrain"] = {
                "max_iterations": args.textgrain_iterations or 0,
                "transport": args.textgrain_transport,
            }
        body["logprobs"] = 10 if args.endpoint == "completions" else True
        if args.endpoint == "chat/completions":
            body["top_logprobs"] = 10
    if args.compare:
        baseline = {name: value for name, value in body.items() if name != "watermark"}
        baseline["sampling_trace"] = {
            name: value for name, value in body["sampling_trace"].items()
            if name not in ("teaching_tournament", "textgrain")
        }
        print(json.dumps({"without_watermark": post(args.base_url, args.endpoint, baseline)}, indent=2))
    response = post(args.base_url, args.endpoint, body)
    print(json.dumps(response, indent=2))
    if args.detect:
        evidence = post(
            args.base_url,
            "watermark/detect",
            {
                "watermark": config,
                "input": {
                    "type": "text",
                    "model": args.model,
                    "text": generated_text(response, args.endpoint),
                },
            },
        )
        print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
