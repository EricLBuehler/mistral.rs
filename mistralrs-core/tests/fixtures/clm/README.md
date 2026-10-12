# CLM reference fixture

These are randomly initialized test weights, not a useful decision model.
`generate.py` produces the tiny Qwen3 encoder, trained-head-shaped MLPs,
and reference answers using PyTorch, Transformers, and the upstream
[CLM code](https://github.com/Contrastive-LM/CLM/tree/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7).
CLM's reference implementation is Apache-2.0 licensed.

The fixture exercises the `model.` encoder weight prefix, causal attention,
last-token pooling, normalization, GELU, LayerNorm, learned scaling, and all
three System One question types with structured inputs and unequal lengths.
Expected answers come from the Python implementation; Rust tests do not
regenerate them.

`published-head-reference.json` contains only numerical reference outputs
from the published CLM-v0.1-8B checkpoint using deterministic synthetic
embeddings. Its optional Rust test requires the real checkpoint separately.
