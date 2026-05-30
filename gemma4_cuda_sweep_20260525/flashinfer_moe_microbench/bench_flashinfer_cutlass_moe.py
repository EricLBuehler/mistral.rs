import argparse
import os
import time

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/mistralrs_flashinfer")
os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/mistralrs_pycache")

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.tllm_enums import ActivationType


def make_inputs(num_tokens: int, device: str):
    num_experts = 128
    top_k = 8
    hidden = 2816
    intermediate = 704

    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)
    experts = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    scales = torch.rand((num_tokens, top_k), device=device, dtype=torch.float32)
    scales = scales / scales.sum(dim=-1, keepdim=True)
    fc1 = torch.randn((num_experts, intermediate * 2, hidden), device=device, dtype=torch.bfloat16)
    fc2 = torch.randn((num_experts, hidden, intermediate), device=device, dtype=torch.bfloat16)
    return x, experts, scales, fc1, fc2


def run_once(x, experts, scales, fc1, fc2, tune_max_num_tokens):
    out = cutlass_fused_moe(
        x,
        experts,
        scales,
        fc1,
        fc2,
        torch.bfloat16,
        [],
        tune_max_num_tokens=tune_max_num_tokens,
        enable_pdl=False,
        activation_type=ActivationType.Geglu,
    )
    if isinstance(out, list):
        out = out[0]
    return out


def bench(num_tokens: int, warmup: int, iters: int, tune_max_num_tokens: int):
    x, experts, scales, fc1, fc2 = make_inputs(num_tokens, "cuda")
    torch.cuda.synchronize()

    start = time.monotonic()
    for _ in range(warmup):
        out = run_once(x, experts, scales, fc1, fc2, tune_max_num_tokens)
    torch.cuda.synchronize()
    warmup_s = time.monotonic() - start

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        out = run_once(x, experts, scales, fc1, fc2, tune_max_num_tokens)
    end.record()
    torch.cuda.synchronize()

    total_ms = begin.elapsed_time(end)
    ms = total_ms / iters
    print(
        f"tokens={num_tokens} ms={ms:.3f} tok_per_s={num_tokens / (ms / 1000):.2f} "
        f"warmup_s={warmup_s:.2f} mean_abs={out.float().abs().mean().item():.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 16, 128, 512, 4096])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tune-max-num-tokens", type=int, default=4096)
    args = parser.parse_args()

    for tokens in args.tokens:
        iters = max(3, min(args.iters, 2000 // tokens if tokens > 0 else args.iters))
        bench(tokens, args.warmup, iters, args.tune_max_num_tokens)


if __name__ == "__main__":
    main()
