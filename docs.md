# Documentation for Direct GPU Inference in mistral.rs

This document provides a guide to running model inference directly on a GPU using `mistral.rs`, with a focus on single-GPU, multi-GPU, and CPU offloading scenarios.

## Single-GPU Inference

Running inference on a single GPU is the most straightforward way to accelerate your models. `mistral.rs` simplifies this process by automatically detecting and using a GPU if one is available.

### Enabling GPU Support

To use a GPU, you need to build `mistral.rs` with the appropriate feature flag. For NVIDIA GPUs, use the `cuda` flag, and for Apple Silicon, use `metal`.

**Build with CUDA support:**
```bash
cargo build --release --features cuda
```

**Build with Metal support:**
```bash
cargo build --release --features metal
```

### Python Example

This example demonstrates how to load a model and run inference. `mistral.rs` will automatically place the model on the GPU if it's available.

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

# The model will be automatically loaded to the GPU if available
runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Explain the importance of GPU acceleration in deep learning."}
        ],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

### Rust Example

The process is similar in Rust. The `mistralrs` crate will handle device placement automatically.

```rust
use mistralrs::{MistralRs, Which, ChatCompletionRequest, RequestMessage, Role};

fn main() -> anyhow::Result<()> {
    // The model will be automatically loaded to the GPU if available
    let mistralrs = MistralRs::new(Which::Plain {
        model_id: "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
        arch: "mistral".to_string(),
    })?;

    let request = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: "Explain the importance of GPU acceleration in deep learning.".to_string(),
        }],
        max_tokens: Some(256),
        ..Default::default()
    };

    let response = mistralrs.send_chat_completion_request(request)?;
    println!("{}", response.choices[0].message.content);
    Ok(())
}
```

## Multi-GPU Inference

`mistral.rs` supports distributed inference across multiple GPUs to run larger models that wouldn't fit on a single GPU. It offers both automatic and manual configuration.

### Automatic Tensor Parallelism (NCCL for CUDA)

If you have multiple CUDA-enabled GPUs and build `mistral.rs` with the `cuda` feature, tensor parallelism is enabled automatically using NCCL. The model's layers are distributed across all available GPUs, providing seamless acceleration without manual configuration.

To disable this feature and use automatic device mapping (which might offload to CPU), you can set the following environment variable:
```bash
export MISTRALRS_NO_NCCL=1
```

### Heterogeneous Setups (Ring Backend)

For non-CUDA or mixed-device environments (e.g., multiple Metal GPUs), `mistral.rs` provides a Ring backend. This allows for distributed inference over TCP, enabling flexible, heterogeneous setups.

### Manual Device Mapping

For fine-grained control, you can manually specify how many layers of the model to place on each GPU. This method is deprecated in favor of automatic mapping but remains available for advanced use cases.

#### Python Example

In the Python API, you can pass a list of strings to the `Runner` to specify the device mapping. Each string is in the format `"ORDINAL:NUM_LAYERS"`.

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

# Manually map 16 layers to GPU 0 and 16 layers to GPU 1
runner = Runner(
    which=Which.Plain(
        model_id="gradientai/Llama-3-8B-Instruct-262k",
        arch=Architecture.Llama,
    ),
    device_map=["0:16", "1:16"]
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=32,
    )
)
print(res.choices[0].message.content)
```

#### Rust Example

In Rust, you can configure the device map when initializing `MistralRs`.

```rust
use mistralrs::{MistralRs, Which, ChatCompletionRequest, RequestMessage, Role, DeviceMap};

fn main() -> anyhow::Result<()> {
    // Manually map 16 layers to GPU 0 and 16 layers to GPU 1
    let mistralrs = MistralRs::new(Which::Plain {
        model_id: "gradientai/Llama-3-8B-Instruct-262k".to_string(),
        arch: "llama".to_string(),
    }, DeviceMap::from_str("0:16;1:16")?)?;

    let request = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: "Hello!".to_string(),
        }],
        max_tokens: Some(32),
        ..Default::default()
    };

    let response = mistralrs.send_chat_completion_request(request)?;
    println!("{}", response.choices[0].message.content);
    Ok(())
}
```

## CPU Offloading for Large Models

When a model is too large to fit entirely into GPU memory, `mistral.rs` provides an automatic device mapping feature that offloads the remaining layers to the CPU. This allows you to run larger models than your GPU VRAM would typically allow, at the cost of some performance.

### Automatic Device Mapping

This feature is enabled by default. `mistral.rs` will prioritize loading the model onto the GPU and then seamlessly place the rest of the model's layers onto the CPU. This process is transparent to the user and requires no special configuration.

You can influence the device mapping by providing hints about the expected workload, such as `max_seq_len` and `max_batch_size`. This helps `mistral.rs` make more informed decisions about how to distribute the model.

#### Python Example

In this example, we specify the maximum sequence length and batch size to guide the automatic device mapping.

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="meta-llama/Llama-3-70B-Instruct",
        arch=Architecture.Llama,
    ),
    max_seq_len=4096,
    max_batch_size=2,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "How does CPU offloading work in practice?"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

#### Rust Example

The same parameters can be configured in Rust to control the automatic device mapping.

```rust
use mistralrs::{MistralRs, Which, ChatCompletionRequest, RequestMessage, Role};

fn main() -> anyhow::Result<()> {
    let mistralrs = MistralRs::new(Which::Plain {
        model_id: "meta-llama/Llama-3-70B-Instruct".to_string(),
        arch: "llama".to_string(),
    })?
    .with_max_seq_len(4096)
    .with_max_batch_size(2);

    let request = ChatCompletionRequest {
        model: "default".to_string(),
        messages: vec![RequestMessage {
            role: Role::User,
            content: "How does CPU offloading work in practice?".to_string(),
        }],
        max_tokens: Some(256),
        ..Default::default()
    };

    let response = mistralrs.send_chat_completion_request(request)?;
    println!("{}", response.choices[0].message.content);
    Ok(())
}
```

## Performance Optimization

### FlashAttention

FlashAttention is a highly efficient attention mechanism that can significantly speed up inference. To use it, you need to build `mistral.rs` with the `flash-attn` feature flag.

```bash
cargo build --release --features "cuda flash-attn"
```

FlashAttention will be used automatically when available and supported by the model and hardware.

### Quantization

Quantization is a technique used to reduce the memory footprint and improve the performance of deep learning models. It works by reducing the precision of the model's weights from 32-bit floating-point to a lower precision, such as 8-bit or 4-bit integers. `mistral.rs` supports a variety of quantization methods, including GGUF, GPTQ, AWQ, and more.

#### GGUF (GPT-Generated Unified Format)

GGUF is a popular format for quantized models, especially in the Llama community. `mistral.rs` provides first-class support for loading and running GGUF models.

##### Python Example

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.GGUF(
        tok_model_id="HuggingFaceH4/zephyr-7b-beta",
        quantized_model_id="TheBloke/zephyr-7B-beta-GGUF",
        quantized_filename="zephyr-7b-beta.Q5_0.gguf",
    ),
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "What is GGUF?"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```

#### In-Situ Quantization (ISQ)

In-Situ Quantization is a powerful feature in `mistral.rs` that allows you to quantize a model on the fly as it's being loaded. This is particularly useful when you want to use a model that doesn't have a pre-quantized version available.

##### Python Example

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3-mini-4k-instruct",
        arch=Architecture.Phi3,
    ),
    in_situ_quant="4bit", # Specify the desired quantization level
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "What is In-Situ Quantization?"}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)
```
