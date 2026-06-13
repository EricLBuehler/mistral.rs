---
title: Model family notes
description: Per-family behavior that differs from the defaults - thinking tags, MoE quantization, MLA, MatFormer slices, and chat template fixes.
---

Most models need nothing beyond `mistralrs run -m <id>` (see [Run any model](/mistral.rs/guides/models/run-any-model/)). This page collects the per-family exceptions. The full architecture inventory is in the [supported models reference](/mistral.rs/reference/supported-models/).

## Thinking models (Qwen3, SmolLM3)

Qwen3 and SmolLM3 are hybrid reasoning models; their chat templates enable thinking by default. Toggle it per request with prompt tags:

```text
How many rs are in blueberry? /no_think
Are you sure? /think
```

Without editing user text, the `enable_thinking` extension field does the same over HTTP and in the Python SDK (`true` forces on, `false` forces off, omit/`None` for the template default):

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "How many rs are in blueberry?"}],
    "enable_thinking": false
  }'
```

On the CLI, `mistralrs run --thinking false -m Qwen/Qwen3-4B` does the same for one-shot and interactive use.

Qwen3 also publishes FP8 pre-quantized checkpoints; pass the FP8 model ID directly when you want those weights instead of runtime ISQ.

## MoE models and MoQE

MoE families (DeepSeek V2/V3, GLM-4.7, GLM-4.7-Flash, Phi 3.5 MoE, Qwen3 MoE, Qwen3-VL MoE, Qwen3.5 MoE) support MoQE: quantizing only the routed experts, which dominate memory, while leaving the rest of the model alone. MoQE is an explicit ISQ layout, so it uses `--isq` rather than `--quant`:

```bash
mistralrs run --isq 4 --isq-organization moqe -m Qwen/Qwen3-30B-A3B
```

In the Python SDK, pass `organization=IsqOrganization.MoQE` inside `Which.Plain(...)` or `Which.MultimodalPlain(...)`. Expect small output differences between quantization levels: router decisions are sensitive to numerical noise.

## MLA models (DeepSeek V2/V3, GLM-4.7-Flash)

DeepSeek V2, DeepSeek V3 (including non-distill R1, which uses the V3 architecture), and GLM-4.7-Flash use Multi-head Latent Attention. The KV cache stores a low-dimensional latent instead of full K/V, so the cache footprint is substantially smaller than standard attention at the same context length.

On CUDA (Unix builds), a specialized MLA decode kernel is used when all of the following hold: single-token decode (no attention mask, sequence length 1), paged attention enabled, and FlashInfer paged metadata available. A parallel fast path covers prefill with prefix caching (paged attention on, CUDA device). Otherwise the generic attention path reconstructs the latent per step.

`MISTRALRS_NO_MLA=1` forces the generic path; use it when debugging suspected MLA kernel issues, and try `--paged-attn off` as a sanity check for unexpected paged-attention behavior. Background: the [DeepSeek V2 paper](https://arxiv.org/abs/2405.04434).

## GPT-OSS

GPT-OSS experts are stored pre-quantized in MXFP4, and its attention uses per-head sinks. Load it without a quantization flag first:

```bash
mistralrs run -m openai/gpt-oss-20b
```

ISQ applies only to the attention layers (and `lm_head`); the expert weights are already quantized.

## Qwen3 Next

Qwen3 Next mixes Gated Delta Network (linear attention) layers with full softmax attention, so its cost profile at long contexts differs from a pure softmax model. Qwen3-Coder-Next checkpoints use the same loader.

## Granite 4.0 (GraniteMoeHybrid)

IBM Granite 4.0 checkpoints (e.g. `ibm-granite/granite-4.0-micro`) mix Mamba-2 recurrent layers with attention layers. They load through auto-detection like any other text model.

## Gemma

Gemma repos are gated: accept the license on the Hugging Face model page, then authenticate with `mistralrs login`.

Gemma 4 (E2B/E4B) accepts image, audio, and video parts mixed in one message, and enforces its tool-call format through constrained decoding by default; see [tool calling](/mistral.rs/guides/agents/tool-calling-basics/).

## MatFormer

MatFormer-trained models encode multiple model sizes in one checkpoint; the desired slice is selected at load time with two values:

- `matformer_config_path`: path to the slice config file (CSV or JSON) shipped with the model card.
- `matformer_slice_name`: the named slice within that file.

Without these, the default (full) configuration loads. Gemma 3n (`google/gemma-3n-E4B-it`) is the MatFormer model in the supported list; the bundled `matformer_configs/gemma3n.csv` contains the full E4B configuration, the official E2B slice, and intermediate E1.96B-E3.79B slices:

```bash
mistralrs run -m google/gemma-3n-E4B-it \
  --matformer-config-path matformer_configs/gemma3n.csv \
  --matformer-slice-name "Config for E2.49B (block-level)"
```

The same flags are accepted by `mistralrs serve` and `mistralrs bench`, and the same fields exist as `matformer_config_path` / `matformer_slice_name` in TOML configs and the Python `Which` selectors; the Rust SDK model builders expose `with_matformer_config_path` / `with_matformer_slice_name`. Use the full configuration for quality and smaller slices for constrained devices.

## Mistral Small tool calling

Mistral Small 3 checkpoints can do tool calling, but some repos do not ship the right chat template. Use the bundled one:

```bash
mistralrs serve --quant 4 \
  --jinja-explicit chat_templates/mistral_small_tool_call.jinja \
  -m mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

## LLaVA chat templates

Mistral-backed LLaVA checkpoints work with the default template. Vicuna-backed checkpoints need the Vicuna template:

```bash
mistralrs run -m llava-hf/llava-v1.6-vicuna-7b-hf \
  -c chat_templates/vicuna.json --image photo.jpg -i "Describe this image"
```

## Qwen3-VL

Per-request video frame-sampling overrides are not exposed. In multi-turn conversations reusing prefix cache entries, pixel inputs are narrowed per turn by grid count, not image count.

## Llama 4

Llama 4 Scout supports up to 10M tokens of context. Using the full window requires paged attention with a large memory budget, generally with multi-GPU tensor parallelism.

## Multimodal device mapping

For most multimodal models the text backbone holds most of the parameters, so device mapping and topology apply mainly to the text portion; the vision, audio, or video encoder stays on its supported device path.

## Phi

Phi 3.5 Vision works best with a single image; multiple images are resized together. Phi 4 Multimodal accepts audio and image parts in the same message. Phi 3.5 MoE routes per-token to 16 experts and benefits from [MoQE](#moe-models-and-moqe).

## DiffusionGemma

Block-diffusion generation has its own page: [block-diffusion models](/mistral.rs/guides/models/use-block-diffusion/).

## Reporting issues

File an issue on [GitHub](https://github.com/EricLBuehler/mistral.rs/issues), with a reproducer when possible.
