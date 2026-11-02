mistralrs-paged-attn: Candle / MLX-derived Metal attention kernels

Copyright (c) 2023 Apple Inc.

The following portions are adapted from Candle's MLX-derived Metal scaled
dot-product attention kernels and are licensed under the Apache License,
Version 2.0. The complete license is provided in LICENSE-APACHE.

Covered portions:

* src/metal/kernels/paged_decode.metal:
  paged_decode_short, adapted from sdpa_vector.
* src/metal/kernels/paged_decode_hnd.metal:
  paged_decode_hnd_short, adapted from sdpa_vector;
  paged_decode_hnd_split, derived from that head-parallel attention body;
  paged_decode_hnd_split_reduce, directly adapted from sdpa_vector_2pass_2.

Upstream source:

Repository: https://github.com/huggingface/candle
Commit: 66a8cf184a5a519671454066b1b9efd446ec9f5c
File: candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal
Source URL:
https://github.com/huggingface/candle/blob/66a8cf184a5a519671454066b1b9efd446ec9f5c/candle-metal-kernels/src/metal_src/scaled_dot_product_attention.metal

The upstream file identifies MLX commit f70764a as its source.
MLX repository: https://github.com/ml-explore/mlx

The adaptations add paged KV-cache addressing, the HND layout, block-table
lookup, visible-window bounds, kernel bindings, dispatch grids and FP32
split partials. The split reducer retains the upstream reduction arithmetic
with adapted names, bindings and grid indexing.

Original mistral.rs code remains covered by the repository's MIT LICENSE.
This notice identifies the Apache-2.0 portions above and does not replace
other applicable third-party licenses or notices.
