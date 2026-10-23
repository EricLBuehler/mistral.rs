import json, glob, torch
from safetensors import safe_open

FP8 = "/mnt/models/Aleph-Alpha/Kolibri-1"
BF16 = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16"


def index(p):
    return json.load(open(glob.glob(p + "/model.safetensors.index.json")[0]))["weight_map"]


def load(p, name):
    shard = index(p).get(name)
    if shard is None:
        return None
    with safe_open(f"{p}/{shard}", framework="pt") as f:
        return f.get_tensor(name)


NAME = "model.layers.0.mlp.experts.0.gate_proj.weight"
SCALE = NAME + "_scale_inv"

f_w = load(FP8, NAME).to(torch.float32)
f_s = load(FP8, SCALE).to(torch.float32)
b_w = load(BF16, NAME).to(torch.float32)

print("fp8  weight:", tuple(f_w.shape), " scale:", tuple(f_s.shape))
print("bf16 weight:", tuple(b_w.shape))

H, W = f_w.shape
br, bc = f_s.shape
print("=> block size: %d x %d" % (H // br, W // bc))

for conv, fn in [("fp8 * scale_inv", lambda x, s: x * s), ("fp8 / scale_inv", lambda x, s: x / s)]:
    deq = fn(f_w.reshape(br, H // br, bc, W // bc),
             f_s.reshape(br, 1, bc, 1)).reshape(H, W)
    d = (deq - b_w).abs()
    rel = (d.max() / b_w.abs().max()).item()
    exact = (deq == b_w).float().mean().item()
    print("  %-16s max_abs=%.6g  rel_max=%.3g  exact_bit_match=%.4f"
          % (conv, d.max().item(), rel, exact))

# dequantise with the winning convention and report the real difference
deq = (f_w.reshape(br, H // br, bc, W // bc) * f_s.reshape(br, 1, bc, 1)).reshape(H, W)
d = (deq - b_w).abs()
print("\nper-element abs diff: mean=%.6g  p99=%.6g  max=%.6g" % (
    d.mean().item(), d.flatten().kthvalue(int(0.99 * d.numel())).values.item(), d.max().item()))
print("bf16 value range: [%.4g, %.4g]" % (b_w.min().item(), b_w.max().item()))
print("fp8*scale range : [%.4g, %.4g]" % (deq.min().item(), deq.max().item()))
