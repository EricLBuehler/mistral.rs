#!/usr/bin/env python3
"""Compare per-layer encoder outputs between HF and Rust."""

import numpy as np

# HF layer outputs were saved earlier
# Rust layer outputs saved by debug code

for i in [0, 1, 2, 31]:
    try:
        hf = np.fromfile(f"hf_enc_layer_{i}.bin", dtype=np.float32).reshape(1, 748, 1280)
        rust = np.fromfile(f"rust_enc_layer_{i}.bin", dtype=np.float32).reshape(1, 748, 1280)

        diff = np.abs(hf - rust)
        a = hf.flatten()
        b = rust.flatten()
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        print(f"Layer {i:2d}: HF mean={hf.mean():.6f}, Rust mean={rust.mean():.6f}, "
              f"max_diff={diff.max():.6f}, mean_diff={diff.mean():.6f}, cosine={cos_sim:.8f}")

        # Per-frame comparison for layer 0
        if i == 0:
            for frame in [0, 1, 100, 400, 747]:
                fa = hf[0, frame]
                fb = rust[0, frame]
                fcs = np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb))
                fmd = np.abs(fa - fb).max()
                print(f"  Frame {frame}: cosine={fcs:.8f}, max_diff={fmd:.8f}")
    except Exception as e:
        print(f"Layer {i}: Error: {e}")
