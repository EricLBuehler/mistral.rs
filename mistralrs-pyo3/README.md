# mistral.rs Python SDK

`mistralrs` is the Python SDK for [mistral.rs](https://github.com/EricLBuehler/mistral.rs), a blazing-fast LLM inference engine.

## Quick Start

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.Plain(
        model_id="microsoft/Phi-3.5-mini-instruct",
    ),
    in_situ_quant="Q4K",
)

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[
            {"role": "user", "content": "Tell me a story about the Rust type system."}
        ],
        max_tokens=256,
        temperature=0.1,
    )
)
print(response.choices[0].message.content)
```

## Installation

```bash
pip install mistralrs-cuda   # NVIDIA GPUs
pip install mistralrs-metal  # Apple Silicon
pip install mistralrs        # CPU only
```

See the [Python Installation Guide](https://ericlbuehler.github.io/mistral.rs/PYTHON_INSTALLATION.html) for all options and building from source.

## Documentation

- [SDK Documentation](https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html) - Full SDK reference
- [Type Stubs](mistralrs.pyi) - Type hints for IDE support
- [Examples](https://github.com/EricLBuehler/mistral.rs/tree/master/examples/python)
- [Cookbook](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/cookbook.ipynb)

## Multi-model Support

For serving multiple models and dynamic model management (loading/unloading), see the [SDK documentation](https://ericlbuehler.github.io/mistral.rs/PYTHON_SDK.html#multi-model-support).
