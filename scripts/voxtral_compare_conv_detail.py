#!/usr/bin/env python3
"""Compare conv output between HF and Rust in detail."""

import numpy as np

# Load HF conv output [1, 748, 1280]
hf_conv = np.fromfile("hf_conv_out.bin", dtype=np.float32).reshape(1, 748, 1280)
# Load Rust conv output [1, 748, 1280]
rust_conv = np.fromfile("rust_conv_out.bin", dtype=np.float32).reshape(1, 748, 1280)

print(f"HF conv shape: {hf_conv.shape}, Rust conv shape: {rust_conv.shape}")
print(f"HF conv mean: {hf_conv.mean():.8f}, Rust conv mean: {rust_conv.mean():.8f}")

diff = np.abs(hf_conv - rust_conv)
print(f"Max abs diff: {diff.max():.8f}")
print(f"Mean abs diff: {diff.mean():.8f}")

a = hf_conv.flatten()
b = rust_conv.flatten()
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"Cosine similarity: {cos_sim:.10f}")

# Per-frame
for i in [0, 1, 2, 100, 400, 747]:
    a = hf_conv[0, i]
    b = rust_conv[0, i]
    cs = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    md = np.abs(a - b).max()
    print(f"  Frame {i}: cosine={cs:.8f}, max_diff={md:.8f}")
