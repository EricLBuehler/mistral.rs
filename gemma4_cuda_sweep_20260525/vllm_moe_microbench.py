import os

import torch

from vllm.model_executor.layers.fused_moe.fused_moe import (
    invoke_fused_moe_triton_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.triton_utils import tl


NUM_TOKENS = 512
TOPK = 8
NUM_EXPERTS = 128
HIDDEN = 2816
INTER = 704
HOT_EXPERTS = 30
HOT_COUNT = 80


def make_topk_ids() -> torch.Tensor:
    ids = []
    for expert in range(HOT_EXPERTS):
        ids.extend([expert] * HOT_COUNT)
    cold_start = len(ids)
    for i in range(cold_start, NUM_TOKENS * TOPK):
        ids.append(HOT_EXPERTS + (i - cold_start) % (NUM_EXPERTS - HOT_EXPERTS))
    return torch.tensor(ids, dtype=torch.int32, device="cuda").view(NUM_TOKENS, TOPK)


def time_call(name: str, fn, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    elapsed_ms = start.elapsed_time(end)
    per_iter = elapsed_ms / iters
    print(f"{name}: iters={iters} elapsed_ms={elapsed_ms:.3f} per_iter_ms={per_iter:.6f}")
    return per_iter


def main() -> None:
    torch.set_default_device("cuda")
    iters = int(os.environ.get("MISTRALRS_MOE_MICRO_ITERS", "100"))
    warmup = int(os.environ.get("MISTRALRS_MOE_MICRO_WARMUP", "20"))
    block_m = int(os.environ.get("MISTRALRS_MOE_MICRO_BLOCK_M", "64"))
    block_n = int(os.environ.get("MISTRALRS_MOE_MICRO_BLOCK_N", "128"))
    block_k = int(os.environ.get("MISTRALRS_MOE_MICRO_BLOCK_K", "64"))
    num_warps = int(os.environ.get("MISTRALRS_MOE_MICRO_WARPS", "8"))
    group_m = int(os.environ.get("MISTRALRS_MOE_MICRO_GROUP_M", "1"))

    config = {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": group_m,
        "num_warps": num_warps,
    }

    xs = torch.zeros((NUM_TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")
    gate_up_w = torch.zeros(
        (NUM_EXPERTS, INTER * 2, HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    down_w = torch.zeros(
        (NUM_EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda"
    )
    cache1 = torch.empty((NUM_TOKENS, TOPK, INTER * 2), dtype=torch.bfloat16)
    cache2 = torch.zeros((NUM_TOKENS * TOPK, INTER), dtype=torch.bfloat16)
    cache3 = torch.empty((NUM_TOKENS, TOPK, HIDDEN), dtype=torch.bfloat16)
    topk_weights = torch.ones((NUM_TOKENS, TOPK), dtype=torch.float32, device="cuda")
    topk_ids = make_topk_ids()

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, NUM_EXPERTS
    )
    torch.cuda.synchronize()
    print(
        "sorted_len={} expert_blocks={} num_tokens_post_padded={}".format(
            sorted_token_ids.numel(),
            expert_ids.numel(),
            int(num_tokens_post_padded.cpu().item()),
        )
    )

    def gate() -> None:
        invoke_fused_moe_triton_kernel(
            xs,
            gate_up_w,
            cache1,
            None,
            None,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            TOPK,
            config,
            tl.bfloat16,
            False,
            False,
            False,
            False,
            False,
        )

    def down() -> None:
        invoke_fused_moe_triton_kernel(
            cache2,
            down_w,
            cache3,
            None,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            tl.bfloat16,
            False,
            False,
            False,
            False,
            False,
        )

    for _ in range(warmup):
        gate()
        down()
    torch.cuda.synchronize()

    gate_ms = time_call("gate", gate, iters)
    down_ms = time_call("down", down, iters)
    both_ms = time_call(
        "gate_down",
        lambda: (
            gate(),
            down(),
        ),
        iters,
    )
    print(f"sum_separate_ms={gate_ms + down_ms:.6f} combined_ms={both_ms:.6f}")


if __name__ == "__main__":
    main()
