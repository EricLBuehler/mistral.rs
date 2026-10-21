"""Is the bf16 GGUF bit-identical to the bf16 safetensors, across many tensors?

The peer agent's hypothesis for the llama.cpp-to-primary offset is the container: the GGUF is
`round_bf16(fp8 * scale_inv)`, which they say is a ~4.6e-5 relative difference from an
in-process fp8 dequant.

If that were the cause then the GGUF and the bf16 safetensors would differ. Two tensors were
checked earlier and matched exactly, but the whole claim rests on it, so this checks a sample
across layers, experts and all three projections.

The GGUF stores an expert stack as ggml [n_embd, n_ff, n_expert] with ne0 fastest, so expert
e's bytes start at e * n_ff * n_embd elements and within an expert the rows are
[n_ff, n_embd] with n_embd fastest - the same order as the safetensors' [n_ff, n_embd].

Usage: check_container_identity.py <gguf> <safetensors-dir>
"""

import json
import sys

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/home/lu_zero/Sources/llama.cpp/gguf-py")
from gguf import GGUFReader  # noqa: E402

GGUF = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kolibri1-llamacpp-auto.gguf"
ST = sys.argv[2] if len(sys.argv) > 2 else "/mnt/models/Aleph-Alpha/Kolibri-1-bf16"

index = json.load(open(f"{ST}/model.safetensors.index.json"))["weight_map"]

# (layer, expert, projection) samples spread across the stack.
SAMPLES = []
for layer in (0, 7, 20, 33, 49):
    for expert in (0, 3, 100, 383):
        for proj, gguf_name in (
            ("gate_proj", "ffn_gate_exps"),
            ("up_proj", "ffn_up_exps"),
            ("down_proj", "ffn_down_exps"),
        ):
            SAMPLES.append((layer, expert, proj, gguf_name))

r = GGUFReader(GGUF)
by_name = {t.name: t for t in r.tensors}

total = 0
identical = 0
mismatches = []
for layer, expert, proj, gguf_name in SAMPLES:
    st_name = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
    gguf_name_full = f"blk.{layer}.{gguf_name}.weight"
    t = by_name.get(gguf_name_full)
    if t is None:
        print(f"  MISSING {gguf_name_full}")
        continue
    dims = [int(d) for d in t.shape]
    # ggml shape [n_embd, n_ff, n_expert]: ne0 (n_embd) fastest.
    n_embd, n_ff, n_expert = dims[0], dims[1], dims[2]
    raw = t.data.view(np.uint16).reshape(n_expert, n_ff, n_embd)
    gguf_expert = raw[expert]

    with safe_open(f"{ST}/{index[st_name]}", framework="pt") as f:
        st = f.get_tensor(st_name)
    # bfloat16 has no numpy dtype, so read the raw words.
    st_expert = st.view(torch.int16).numpy().astype(np.uint16).reshape(n_ff, n_embd)

    total += st_expert.size
    if np.array_equal(gguf_expert, st_expert):
        identical += st_expert.size
    else:
        diff = np.abs(gguf_expert.astype(np.int32) - st_expert.astype(np.int32))
        mismatches.append((st_name, int((diff != 0).sum()), diff.max()))

print(f"compared {total} bf16 elements across {len(SAMPLES)} tensors")
print(f"  bit-identical: {identical}/{total} ({100.0*identical/total:.4f}%)")
if mismatches:
    print(f"  {len(mismatches)} tensors differ:")
    for name, n, mx in mismatches[:8]:
        print(f"    {name}: {n} elements differ, max ulp {mx}")
else:
    print("  every sampled tensor is bit-identical")
