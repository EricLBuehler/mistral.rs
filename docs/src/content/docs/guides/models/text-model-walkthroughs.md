---
title: Text model walkthroughs
description: Model-family notes for Qwen3, SmolLM3, DeepSeek, GLM, GPT-OSS, and other text-only backends.
sidebar:
  order: 6
---

Most text-only models use the same CLI, HTTP, Python, and Rust shape. This page collects the model-family details that matter when a model is not just another plain decoder.

## Common pattern

Start with auto-detection unless you have a reason to force an architecture:

```bash
mistralrs run --quant 4 -m Qwen/Qwen3-4B
mistralrs serve --quant 4 -p 1234 -m Qwen/Qwen3-4B
```

The server is OpenAI-compatible:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:1234/v1/")

completion = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Tell me a short story about Rust."}],
    max_tokens=256,
)
print(completion.choices[0].message.content)
```

The Python SDK selector is explicit when auto-detection is not enough:

```python
from mistralrs import Architecture, ChatCompletionRequest, Runner, Which

runner = Runner(
    which=Which.Plain(
        model_id="Qwen/Qwen3-4B",
        arch=Architecture.Qwen3,
    ),
    in_situ_quant="4",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=256,
    )
)
print(response.choices[0].message.content)
```

## Family quick starts

| Family | Example model | Python architecture | Notes |
|---|---|---|---|
| DeepSeek V2 | `deepseek-ai/DeepSeek-V2-Lite` | `Architecture.DeepseekV2` | MoE model with MLA. Supports MoQE for expert-only ISQ. |
| DeepSeek V3 / R1 | `deepseek-ai/DeepSeek-R1` | `Architecture.DeepseekV3` | Non-distill DeepSeek R1 uses the DeepSeek V3 architecture. Distill checkpoints load by model ID. |
| Gemma 2 | `google/gemma-2-9b-it` | `Architecture.Gemma2` | Plain text Gemma family. Requires Hugging Face license acceptance for gated repos. |
| GLM4 | `zai-org/GLM-4-32B-0414` | `Architecture.GLM4` | Text backbone for GLM4. |
| GLM-4.7 | `zai-org/GLM-4.7` | `Architecture.GLM4Moe` | MoE model with standard GQA attention and partial RoPE. |
| GLM-4.7-Flash | `zai-org/GLM-4.7-Flash` | `Architecture.GLM4MoeLite` | MoE model with MLA. Similar long-context KV-cache tradeoffs to DeepSeek. |
| GPT-OSS | `openai/gpt-oss-20b` | `Architecture.GptOss` | MXFP4-quantized experts. ISQ applies only to attention layers. Paged attention is not supported. |
| Phi 3.5 MoE | `microsoft/Phi-3.5-MoE-instruct` | `Architecture.Phi3_5MoE` | 16 expert MoE. MoQE is useful when quantizing only routed experts. |
| Qwen3 dense | `Qwen/Qwen3-4B` | `Architecture.Qwen3` | Hybrid reasoning model. Thinking is on by default. |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B` | `Architecture.Qwen3Moe` | Same thinking controls as dense Qwen3. MoE variants support MoQE. |
| Qwen3 Next | `Qwen/Qwen3-Next-80B-A3B-Instruct` | `Architecture.Qwen3Next` | Hybrid GDN plus full attention. Qwen3-Coder-Next checkpoints use the same loader when available. |
| SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | `Architecture.SmolLm3` | Small hybrid reasoning model. Thinking controls match Qwen3. |

## Thinking models

Qwen3 and SmolLM3 are hybrid reasoning models. The default chat template enables thinking. Disable or enable it per request with prompt tags:

```text
How many rs are in blueberry? /no_think
Are you sure? /think
```

The HTTP extension field does the same thing without editing user text:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "How many rs are in blueberry?"}],
    "enable_thinking": false
  }'
```

With the Python SDK, pass `enable_thinking=None` to use the model template default, `False` to force thinking off, or `True` to force it on:

```python
request = ChatCompletionRequest(
    model="default",
    messages=[{"role": "user", "content": "How many rs are in blueberry?"}],
    max_tokens=512,
    enable_thinking=False,
)
```

Qwen3 also has FP8 pre-quantized checkpoints. Use the FP8 model ID directly when you want the pre-quantized weights instead of runtime ISQ.

Qwen3 supports the same tool-calling and web-search surface as the agent guides. See [tool calling basics](/mistral.rs/guides/agents/tool-calling-basics/) and [web search](/mistral.rs/guides/agents/web-search/) when you need tool use rather than plain chat.

## MoE and MoQE

MoE families such as DeepSeek, GLM-4.7, Phi 3.5 MoE, and Qwen3 MoE can use MoQE to quantize expert weights separately from the rest of the model:

```bash
mistralrs run --isq 4 --isq-organization moqe -m deepseek-ai/DeepSeek-V2-Lite
mistralrs run --isq 4 --isq-organization moqe -m Qwen/Qwen3-30B-A3B
```

These examples use `--isq` because MoQE is an explicit runtime ISQ layout. MoQE is most useful when the routed experts dominate memory. Expect small output differences between quantization levels because router decisions are sensitive to numerical noise.

In the Python SDK, pass `organization=IsqOrganization.MoQE` inside `Which.Plain(...)` or `Which.MultimodalPlain(...)` for the same expert-first quantization behavior.

## MLA models

DeepSeek V2/V3 and GLM-4.7-Flash use Multi-head Latent Attention. MLA usually reduces KV-cache memory compared with standard attention. If you are debugging unexpected paged-attention behavior, first compare with paged attention disabled:

```bash
mistralrs run --paged-attn off -m deepseek-ai/DeepSeek-R1
```

## GPT-OSS

GPT-OSS is not a normal dense checkpoint. Its MoE experts are stored in MXFP4 and custom attention uses per-head sinks. Load it without an ISQ flag first:

```bash
mistralrs run -m openai/gpt-oss-20b
mistralrs serve -p 1234 -m openai/gpt-oss-20b
```

Paged attention is not supported for GPT-OSS. ISQ can be applied only to attention layers; the expert weights are already quantized.

## Qwen3 Next

Qwen3 Next mixes Gated Delta Network layers with full attention. It is good for long-context coding workloads, but its cost profile differs from a pure softmax-attention model.

```bash
mistralrs run --quant 4 -m Qwen/Qwen3-Next-80B-A3B-Instruct
mistralrs serve --quant 4 -p 1234 -m Qwen/Qwen3-Next-80B-A3B-Instruct
```

Qwen3-Coder-Next checkpoints use the same architecture when the checkpoint is available. GGUF checkpoints use the GGUF selector:

```bash
mistralrs run --format gguf -m Qwen/Qwen3-Coder-Next-GGUF -f <filename>
```

## UQFF examples

When a pre-quantized UQFF repo exists, load it directly:

```bash
mistralrs run -m EricB/SmolLM3-3B-UQFF --from-uqff smollm33b-q4k-0.uqff
```

See [UQFF format](/mistral.rs/reference/uqff-format/) for file layout and [use pre-quantized UQFF models](/mistral.rs/guides/perf/use-uqff/) for runtime usage.

## Example files

Long-form SDK examples live in the repository so they can compile with the Rust and Python APIs:

- Python: [`examples/python/`](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)
- HTTP/OpenAI clients: [`examples/server/`](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/server)
- Rust text models: [`mistralrs/examples/models/text_models/main.rs`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/models/text_models/main.rs)
