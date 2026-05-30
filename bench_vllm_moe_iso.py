import time
import torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

E, TOPK, HID, MI = 128, 8, 2816, 704
dev = "cuda"
dt = torch.bfloat16

w1 = torch.randn(E, 2 * MI, HID, device=dev, dtype=dt) * 0.02
w2 = torch.randn(E, HID, MI, device=dev, dtype=dt) * 0.02

def run(M):
    hs = torch.randn(M, HID, device=dev, dtype=dt) * 0.1
    logits = torch.randn(M, E, device=dev, dtype=torch.float32)
    tw, tid = torch.topk(torch.softmax(logits, dim=-1), TOPK, dim=-1)
    tw = tw.to(torch.float32)
    tid = tid.to(torch.int32)
    for _ in range(5):
        fused_experts(hs, w1, w2, tw, tid, global_num_experts=E)
    torch.cuda.synchronize()
    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        fused_experts(hs, w1, w2, tw, tid, global_num_experts=E)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / iters
    flop = 2.0 * M * TOPK * (HID * 2 * MI + MI * HID)
    tf = flop / (ms * 1e-3) / 1e12
    print(f"  M={M:6d}: {ms:8.3f} ms/forward  | {tf:6.1f} TFLOPS (both GEMMs)")

print(f"vLLM fused_experts (E={E} topk={TOPK} hid={HID} moe_inter={MI}) bf16")
for M in [2048, 4096, 8192]:
    run(M)
