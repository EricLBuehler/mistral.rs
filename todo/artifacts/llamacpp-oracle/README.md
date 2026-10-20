# The llama.cpp third oracle

Artifacts for the finding recorded in `../../mistralrs/07-kolibri1-logit-gap.md`: llama.cpp is
a third independent Kolibri-1 implementation, and it agrees with the transcription goldens
rather than with the primary oracle.

| file | what it is |
|---|---|
| `kolibri1_goldens_llamacpp.json` | the 8 golden prompts run through llama.cpp: token ids plus per-position top-20 logprobs, in the format the golden test already reads |
| `run_llamacpp_goldens.py` | runs the prompts through `llama-server` and writes that file |
| `compare_oracles.py` | transcription vs primary vs llama.cpp: first-token and full-sequence agreement |
| `diag_offset.py` | finds the one-position offset in the primary's own logprob rows |
| `diag_prefix.py` | compares the primary and llama.cpp along the common prefix only, and reports the margin at each disagreement |
| `cmp_fp8_bf16.py` | shows `Kolibri-1-bf16` is the dequantisation of the fp8 checkpoint, so the GGUF is a valid oracle |

## Reproducing

```
# 1. the GGUF, converted from the fp8 checkpoint, bf16 weights, 1-D tensors f32
python convert_hf_to_gguf.py /mnt/models/Aleph-Alpha/Kolibri-1 --outfile kolibri1-llamacpp-auto.gguf

# 2. llama.cpp at commit 62318cd57 (the Kolibri-1 merge), CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
cmake --build build -j 32

# 3. the prompts
LD_LIBRARY_PATH=build/bin python3 run_llamacpp_goldens.py
python3 compare_oracles.py
```

## Two traps, both of which cost time here

**The native `/completion` endpoint, not `/v1/completions`.** Only the native one returns
`completion_probabilities`, which carries the generated token **ids** and the per-position
top-k. The OpenAI endpoint returns text, and splitting that text yields whitespace-separated
words, not tokens - which compares nonsense against the goldens and looks like a model
disagreement.

**`LD_LIBRARY_PATH` must point at the build.** There is a stale llama.cpp installed at
`~/.local/lib64` and the loader prefers it, which makes every binary fail with
`undefined symbol: mtmd_get_cap_from_file` regardless of how it was built.

## Also handed to the vllm.cpp repo

Copies live there so the two oracles can be compared side by side, next to the primary rows:

```
tests/vllm/models/kolibri1_goldens_llamacpp.json
tests/vllm/models/kolibri1_llamacpp_topk_rows.json
scripts/compare-kolibri1-oracles.py
```

`compare-kolibri1-oracles.py` auto-detects the primary's row offset instead of assuming it,
so it works on the lagged file and on a repaired one unchanged.

The llama.cpp rows are **not** lagged: row `k` is the distribution for step `k` and its `token`
is the token that step generated, so `rows[k].token == rows[k].top[0]` holds for all 256 rows.
That invariant is the cheapest way to tell a fixed file from a lagged one.
