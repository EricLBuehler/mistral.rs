# Speculative Decoding

Speculative decoding is an inference acceleration technique that uses a smaller "draft" model to propose tokens, which are then validated in parallel by the larger "target" model. This can significantly speed up generation when the draft model frequently predicts tokens the target model would also choose.

Mistral.rs implements speculative decoding based on the paper: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192).

## How It Works

1. The draft model generates `gamma` candidate tokens autoregressively
2. The target model evaluates all candidate tokens in a single forward pass
3. Using rejection sampling, tokens are accepted or rejected:
   - Accept if the target model's probability >= draft model's probability
   - Otherwise, accept with probability `p_target(x) / p_draft(x)`
   - If rejected, sample from the normalized difference distribution

This approach guarantees the same output distribution as running the target model alone, while often achieving significant speedups.

## Configuration

The key parameter is `gamma` - the number of draft tokens to generate per speculation step. Higher values can increase throughput when the draft model is accurate, but waste computation when predictions are frequently rejected.

**Recommended values:** Start with `gamma = 12-32` and tune based on your models and workload.

## Requirements

- **Same tokenizer:** Both target and draft models must share the same tokenizer vocabulary
- **Same model category:** Both must be the same type (e.g., both text models or both vision models)
- **KV cache enabled:** Both models must have KV caching enabled (default behavior)

## Limitations

> Note: PagedAttention is not currently supported with speculative decoding.

> Note: Prefix caching is not supported with speculative decoding.

> Note: Hybrid KV caches are not supported with speculative decoding.

## Using TOML Configuration

The recommended way to configure speculative decoding is via TOML. Create a config file (e.g., `speculative.toml`):

```toml
[model]
model_id = "meta-llama/Llama-3.1-8B-Instruct"

[speculative]
gamma = 12

[speculative.draft_model]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
```

Then run with:

```bash
mistralrs run --from-toml speculative.toml
```

The draft model can use any supported format (Plain, GGUF, etc.) and can have different quantization than the target model.

### TOML with GGUF Draft Model

```toml
[model]
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

[speculative]
gamma = 16

[speculative.draft_model]
model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
tok_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
```

### TOML with ISQ Quantization

```toml
[model]
model_id = "meta-llama/Llama-3.1-8B-Instruct"

[speculative]
gamma = 16

[speculative.draft_model]
model_id = "meta-llama/Llama-3.2-1B-Instruct"
isq = "Q8_0"
```

## Using the Python SDK

```python
from mistralrs import Runner, Which, ChatCompletionRequest, Architecture

runner = Runner(
    which=Which.Plain(
        model_id="mistralai/Mistral-7B-Instruct-v0.1",
        arch=Architecture.Mistral,
    ),
    which_draft=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    ),
    speculative_gamma=32,
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```

### Python SDK Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `which_draft` | `Which` | Draft model specification (Plain, GGUF, etc.) |
| `speculative_gamma` | `int` | Number of draft tokens per step (default: 32) |

## Using the Rust SDK

You can find this example at `mistralrs/examples/advanced/speculative/main.rs`.

```rust
use anyhow::Result;
use mistralrs::{
    IsqType, RequestBuilder, SpeculativeConfig, TextMessageRole, TextMessages,
    TextModelBuilder, TextSpeculativeBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let target = TextModelBuilder::new("meta-llama/Llama-3.1-8B-Instruct")
        .with_logging();
    let draft = TextModelBuilder::new("meta-llama/Llama-3.2-1B-Instruct")
        .with_logging()
        .with_isq(IsqType::Q8_0);
    let spec_cfg = SpeculativeConfig { gamma: 16 };

    let model = TextSpeculativeBuilder::new(target, draft, spec_cfg)?
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
```

## Choosing Draft and Target Models

For best performance:

1. **Use the same model family** - Draft models from the same family as the target (e.g., Llama 3.2-1B with Llama 3.1-8B) typically have higher acceptance rates
2. **Smaller is better for draft** - The draft model should be significantly smaller than the target for meaningful speedup
3. **Quantize the draft model** - Using ISQ or GGUF quantization on the draft model reduces memory and improves draft generation speed
4. **Tune gamma** - Monitor acceptance rates and adjust gamma accordingly

### Example Model Pairings

| Target Model | Draft Model | Notes |
|--------------|-------------|-------|
| Llama 3.1-8B | Llama 3.2-1B | Same family, good acceptance |
| Llama 3.1-70B | Llama 3.1-8B | Large speedup potential |
| Mistral-7B | Mistral-7B (Q4_K_M GGUF) | Same model, quantized draft |

## Performance Considerations

- **Acceptance rate:** Higher acceptance rates lead to better speedups. Monitor your logs for rejection statistics.
- **Draft model overhead:** If the draft model is too large relative to the target, the overhead may negate speedup benefits.
- **Batch size:** Speculative decoding is most beneficial for single-request scenarios. For high-throughput batch inference, standard decoding may be more efficient.
- **Memory usage:** Both models must fit in memory simultaneously. Consider quantizing one or both models.

## Combining with Other Features

Speculative decoding can be combined with:

- **ISQ quantization** - Quantize target, draft, or both models
- **X-LoRA adapters** - Use adapters on the target model
- **Device mapping** - Distribute models across multiple GPUs

See `examples/python/speculative_xlora.py` for an example combining speculative decoding with X-LoRA.
