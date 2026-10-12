"""Run any token watermark, or inspect synthetic SemStamp embeddings.

Usage: MISTRALRS_WATERMARK_KEY=<64 hex chars> python examples/python/watermarking.py kgw
"""

import argparse
import json
import os
from pathlib import Path

from mistralrs import Architecture, ChatCompletionRequest, Runner, WatermarkConfig, Which

SCHEMES = (
    "synthid", "kgw", "unigram", "exponential", "inverse_transform", "mpac", "textgrain", "semstamp"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scheme", choices=SCHEMES, default="synthid", nargs="?")
    parser.add_argument(
        "--device",
        choices=("cpu", "metal", "cuda"),
        default="cpu",
        help="SemStamp example device",
    )
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parents[1] / "watermarking" / f"{args.scheme}.json"
    config = json.loads(config_path.read_text())
    config["key"] = os.environ["MISTRALRS_WATERMARK_KEY"]
    watermark = WatermarkConfig(**config)
    if args.scheme == "semstamp":
        embeddings = [[1.0, 0.3, 0.5], [-0.2, 0.9, 0.1], [0.4, -0.1, 1.0]]
        print("Synthetic embeddings only; use a fixed sentence encoder for real text.")
        print(watermark.accepts_embedding(embeddings[0], embeddings[1]))
        print(watermark.detect_embeddings(embeddings, prompt_len=1, device_name=args.device))
        return
    runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B", arch=Architecture.Qwen3))
    response = runner.send_chat_completion_request(
        ChatCompletionRequest(
            model="Qwen/Qwen3-4B",
            messages=[
                {"role": "user", "content": "Write a long story about a lunar garden."}
            ],
            enable_thinking=False,
            max_tokens=512,
            temperature=0.8,
            top_k=40,
            watermark=watermark,
        )
    )
    text = response.choices[0].message.content or ""
    print(text)
    # Retokenization is approximate; prefer original generated IDs when available.
    tokens = runner.tokenize_text(text, add_special_tokens=False, enable_thinking=False)
    print(watermark.detect(tokens))


if __name__ == "__main__":
    main()
