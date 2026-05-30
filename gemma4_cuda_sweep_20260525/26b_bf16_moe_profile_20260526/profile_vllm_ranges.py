import argparse
import ctypes
import os
from pathlib import Path

os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mistralrs_pycache")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/mistralrs_triton_cache")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import torch
from vllm import LLM, SamplingParams


def cudart():
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    raise RuntimeError("could not load libcudart")


def generate(llm: LLM, prompt_token_ids: list[int], max_tokens: int) -> None:
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    llm.generate({"prompt_token_ids": prompt_token_ids}, params, use_tqdm=False)
    torch.cuda.synchronize()


def capture(label: str, runtime, llm: LLM, prompt_token_ids: list[int], max_tokens: int, marker: Path) -> None:
    generate(llm, prompt_token_ids, max_tokens)
    marker.write_text(f"{label} start\n")
    runtime.cudaProfilerStart()
    generate(llm, prompt_token_ids, max_tokens)
    runtime.cudaProfilerStop()
    marker.write_text(f"{label} done\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-depth", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=17)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--mode", choices=["prompt", "decode", "both"], default="both")
    parser.add_argument("--marker", default="/tmp/mistralrs_vllm_profile_marker.txt")
    args = parser.parse_args()

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
    runtime = cudart()
    marker = Path(args.marker)

    if args.mode in ("prompt", "both"):
        capture("prompt", runtime, llm, [token_id] * args.prompt_len, 1, marker)
    if args.mode in ("decode", "both"):
        capture("decode", runtime, llm, [token_id] * args.decode_depth, args.decode_tokens, marker)


if __name__ == "__main__":
    main()
