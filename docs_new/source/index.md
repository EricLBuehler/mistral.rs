# mistral.rs

:::{raw} html
<h1 style="text-align:center">
<strong>mistral.rs
</strong>
</h1>

<h3 style="text-align:center">
<strong>Blazingly fast LLM inference
</strong>
</h3>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/EricLBuehler/mistral.rs" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/EricLBuehler/mistral.rs/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/EricLBuehler/mistral.rs/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>
:::

mistral.rs is a fast, powerful, and easy to use AI inference platform:

**Broad model support**:
- Text models
- Vision models
- Diffusion models

**Easy**:
- Lightweight OpenAI API compatible HTTP server
- Python API
- Grammar support with JSON Schema, Regex, Lark, and Guidance via [llguidance library](https://github.com/microsoft/llguidance)
- ISQ (In situ quantization): run `.safetensors` models directly from 🤗 Hugging Face by quantizing in-place
    - Enhance precision with an imatrix!
- Automatic device mapping to easily load and run models across multiple GPUs and CPU.

**Fast**:
- Apple silicon support: ARM NEON, Accelerate, Metal
- Accelerated CPU inference with MKL, AVX support
- CUDA support with FlashAttention and cuDNN.
- Automatic tensor-parallelism support with NCCL

**Powerful**:
- LoRA support with weight merging
- First X-LoRA inference platform with first class support
- AnyMoE: Build a memory-efficient MoE model from anything, in seconds
- Various sampling and penalty methods
- Tool calling
- Prompt chunking

**Advanced features**:
- PagedAttention and continuous batching (CUDA and Metal support)
- FlashAttention V2/V3
- Prefix caching
- Topology: Configure ISQ and device mapping easily
- UQFF: Quantized file format for easy mixing of quants.
- Speculative Decoding: Mix supported models as the draft model or the target model
- Dynamic LoRA adapter activation with adapter preloading

**Quantization**:
- GGML: 2-bit, 3-bit, 4-bit, 5-bit, 6-bit and 8-bit, with imatrix support
- GPTQ: 2-bit, 3-bit, 4-bit and 8-bit, with [Marlin](https://github.com/IST-DASLab/marlin) kernel support in 4-bit and 8-bit.
- HQQ: 4-bit and 8 bit, with ISQ support
- FP8
- BNB: bitsandbytes int8, fp4, nf4 support

% Getting started

:::{toctree}
:caption: Getting Started
:maxdepth: 1

getting_started/installation
getting_started/quickstart

:::

% Python API documentation

:::{toctree}
:caption: Python API documentation
:maxdepth: 1

python_docs/api

:::

% Quantization

:::{toctree}
:caption: Quantization in mistral.rs
:maxdepth: 1

quantization/isq.md
quantization/quants.md
quantization/topology.md
quantization/uqff.md

:::

% Advanced features

:::{toctree}
:caption: Advanced features
:maxdepth: 1

advanced/index.md

:::
