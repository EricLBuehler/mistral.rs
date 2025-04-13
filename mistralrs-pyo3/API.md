# `mistralrs` API

These are API docs for the `mistralrs` package.

**Table of contents**
- Full API docs: [here](https://ericlbuehler.github.io/mistral.rs/pyo3/mistralrs.html)
- Docs for the `Which` enum: [here](#which)
- Example: [here](#example)

## `Which`

Each `*_model_id` may be a HF hub repo or a local path. For quantized GGUF models, a list is accepted if multiples files must be specified.

### Architecture for plain models
If you do not specify the architecture, an attempt will be made to use the model's config. If this fails, please raise an issue.

- `Mistral`
- `Gemma`
- `Mixtral`
- `Llama`
- `Phi2`
- `Phi3`
- `Qwen2`
- `Gemma2`
- `Starcoder2`
- `Phi3_5MoE`
- `DeepseekV2`
- `DeepseekV3`

### ISQ Organization
- `Default`
- `MoQE`: if applicable, only quantize MoE experts. https://arxiv.org/abs/2310.02410

### Architecture for vision models
- `Phi3V`
- `Idefics2`
- `LLaVaNext`
- `LLaVa`
- `VLlama`
- `Qwen2VL`
- `Idefics3`
- `MiniCpmO`
- `Phi4MM`
- `Qwen2_5VL`
- `Gemma3`
- `Mistral3`
- `Llama4`

### Architecture for diffusion models
- `Flux`
- `FluxOffloaded`

### ISQ Organization
- `Default`
- `MoQE`: if applicable, only quantize MoE experts. https://arxiv.org/abs/2310.02410

> Note: `from_uqff` specified a UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).

```py
class Which(Enum):
    @dataclass
    class Plain:
        model_id: str
        arch: Architecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        organization: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)
        calibration_file: str | None = None
        imatrix: str | None = None
        hf_cache_path: str | None = None

    @dataclass
    class XLora:
        xlora_model_id: str
        order: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        tgt_non_granular_index: int | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)
        hf_cache_path: str | None = None

    @dataclass
    class Lora:
        adapter_model_id: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)
        hf_cache_path: str | None = None

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class XLoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tgt_non_granular_index: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class LoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class GGML:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        gqa: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class XLoraGGML:
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        gqa: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class LoraGGML:
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = (None,)

    @dataclass
    class VisionPlain:
        model_id: str
        arch: VisionArchitecture
        tokenizer_json: str | None = None
        topology: str | None = None
        from_uqff: str | list[str] | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        max_edge: int | None = None
        auto_map_params: VisionAutoMapParams | None = (None,)
        calibration_file: str | None = None
        imatrix: str | None = None
        hf_cache_path: str | None = None

    @dataclass
    class DiffusionPlain:
        model_id: str
        arch: DiffusionArchitecture
        dtype: ModelDType = ModelDType.Auto
```


## Example
```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    which=Which.GGUF(
        tok_model_id="mistralai/Mistral-7B-Instruct-v0.1",
        quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    )
)

res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="mistral",
        messages=[{"role":"user", "content":"Tell me a story about the Rust type system."}],
        max_tokens=256,
        presence_penalty=1.0,
        top_p=0.1,
        temperature=0.1,
    )
)
print(res.choices[0].message.content)
print(res.usage)
```