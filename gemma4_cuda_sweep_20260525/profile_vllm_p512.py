import argparse
import os
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from vllm import LLM, SamplingParams


def generate(llm: LLM, prompt_token_ids: list[int]) -> float:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
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
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        language_model_only=True,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )
    prompt = list(range(1000, 1000 + args.length))

    generate(llm, prompt)
    torch.cuda.nvtx.range_push("profile")
    elapsed = generate(llm, prompt)
    torch.cuda.nvtx.range_pop()
    print(f"elapsed_s={elapsed:.6f}")
    print(f"prefill_tps_approx={args.length / elapsed:.1f}")


if __name__ == "__main__":
    main()
