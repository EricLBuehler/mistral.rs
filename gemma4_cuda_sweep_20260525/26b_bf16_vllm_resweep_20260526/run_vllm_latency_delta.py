import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from vllm import LLM, SamplingParams


def timed_generate(llm: LLM, prompt_token_ids: list[int], max_tokens: int) -> float:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    llm.generate({"prompt_token_ids": prompt_token_ids}, params, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lengths", default="128,512,2048,4096,8192,16384")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--prompt-pattern", choices=["repeat", "sequential"], default="repeat")
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        language_model_only=True,
        disable_log_stats=True,
    )
    token_id = llm.get_tokenizer().encode(" the", add_special_tokens=False)[0]

    results = {}
    for length in lengths:
        prompt = (
            [token_id] * length
            if args.prompt_pattern == "repeat"
            else list(range(1000, 1000 + length))
        )
        timed_generate(llm, prompt, 1)
        timed_generate(llm, prompt, 257)
        o1 = [timed_generate(llm, prompt, 1) for _ in range(args.repeats)]
        o257 = [timed_generate(llm, prompt, 257) for _ in range(args.repeats)]
        o1_mean = sum(o1) / len(o1)
        o257_mean = sum(o257) / len(o257)
        results[str(length)] = {
            "o1_latencies_s": o1,
            "o257_latencies_s": o257,
            "o1_mean_s": o1_mean,
            "o257_mean_s": o257_mean,
            "prefill_tps_approx": length / o1_mean,
            "decode_tps_delta": 256 / (o257_mean - o1_mean),
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
