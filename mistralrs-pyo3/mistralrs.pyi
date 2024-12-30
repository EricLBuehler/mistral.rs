from dataclasses import dataclass
from enum import Enum
from typing import Iterator

@dataclass
class ToolChoice(Enum):
    NoTools = "None"
    Auto = "Auto"

@dataclass
class ChatCompletionRequest:
    """
    A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.

    The messages type is as follows: (for normal chat completion, for chat completion with images, pretemplated prompt)
    """

    messages: (
        list[dict[str, str]] | list[dict[str, list[dict[str, str | dict[str, str]]]]]
    ) | str
    model: str
    logit_bias: dict[int, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    top_k: int | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    adapters: list[str] | None = None
    min_p: float | None = None
    min_p: float | None = None
    tool_schemas: list[str] | None = None
    tool_choice: ToolChoice | None = None

@dataclass
class CompletionRequest:
    """
    A CompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.
    """

    prompt: str
    model: str
    echo_prompt: bool = False
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    best_of: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    suffix: str | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    adapters: list[str] | None = None
    min_p: float | None = None
    tool_schemas: list[str] | None = None
    tool_choice: ToolChoice | None = None

@dataclass
class Architecture(Enum):
    Mistral = "mistral"
    Gemma = "gemma"
    Mixtral = "mixtral"
    Llama = "llama"
    Phi2 = "phi2"
    Qwen2 = "qwen2"
    Gemma2 = "gemma2"
    Starcoder2 = "starcoder2"
    Phi3_5MoE = "phi3.5moe"

@dataclass
class VisionArchitecture(Enum):
    Phi3V = "phi3v"
    Idefics2 = "idefics2"
    LLaVANext = "llava-next"
    LLaVA = "llava"
    VLlama = "vllama"
    Qwen2VL = "qwen2vl"
    Idefics3 = "idefics3"

@dataclass
class DiffusionArchitecture(Enum):
    Flux = "flux"
    FluxOffloaded = "flux-offloaded"

@dataclass
class IsqOrganization(Enum):
    Default = "default"
    MoQE = "moqe"

@dataclass
class ModelDType(Enum):
    Auto = "auto"
    BF16 = "bf16"
    F16 = "f16"
    F32 = "f32"

@dataclass
class ImageGenerationResponseFormat(Enum):
    Url = "url"
    B64Json = "b64json"

class Which(Enum):
    """
    Which model to select. See the docs for the `Which` enum in API.md for more details.
    Usage:
    ```python
    >>> Which.Plain(...)
    ```
    """
    @dataclass
    class Plain:
        model_id: str
        arch: Architecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        organization: str | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class XLora:
        xlora_model_id: str
        order: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        tgt_non_granular_index: int | None = None
        topology: str | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class Lora:
        adapters_model_id: str
        order: str
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto

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

    @dataclass
    class LoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class GGML:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        gqa: int | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto

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

    @dataclass
    class VisionPlain:
        model_id: str
        arch: VisionArchitecture
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        dtype: ModelDType = ModelDType.Auto
        max_edge: int | None = None

    @dataclass
    class DiffusionPlain:
        model_id: str
        arch: DiffusionArchitecture
        dtype: ModelDType = ModelDType.Auto

class Runner:
    def __init__(
        self,
        which: Which,
        max_seqs: int = 16,
        no_kv_cache: bool = False,
        prefix_cache_n: int = 16,
        token_source: str = "cache",
        speculative_gamma: int = 32,
        which_draft: Which | None = None,
        chat_template: str | None = None,
        num_device_layers: list[str] | None = None,
        in_situ_quant: str | None = None,
        anymoe_config: AnyMoeConfig | None = None,
        pa_gpu_mem: int | float | None = None,
        pa_blk_size: int | None = None,
        no_paged_attn: bool = False,
        paged_attn: bool = False,
        prompt_batchsize: int | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Load a model.

        - `which` specifies which model to load or the target model to load in the case of speculative decoding.
        - `max_seqs` specifies how many sequences may be running at any time.
        - `no_kv_cache` disables the KV cache.
        - `prefix_cache_n` sets the number of sequences to hold in the device prefix cache, others will be evicted to CPU.
        - `token_source` specifies where to load the HF token from.
            The token source follows the following format: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token.
        - `speculative_gamma` specifies the `gamma` parameter for specuative decoding, the ratio of draft tokens to generate before calling
            the target model. If `which_draft` is not specified, this is ignored.
        - `which_draft` specifies which draft model to load. Setting this parameter will cause a speculative decoding model to be loaded,
            with `which` as the target (higher quality) model and `which_draft` as the draft (lower quality) model.
        - `chat_template` specifies an optional JINJA chat template.
            The JINJA template should have `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
            It is used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
        - `num_device_layers` sets the number of layers to load and run on each device.
            Each element follows the format ORD:NUM where ORD is the device ordinal and NUM is
            the corresponding number of layers.
        - `in_situ_quant` sets the optional in-situ quantization for models that are not quantized (not GGUF or GGML).
        - `anymoe_config` specifies the AnyMoE config. If this is set, then the model will be loaded as an AnyMoE model.
        - `pa_gpu_mem`: GPU memory to allocate for KV cache with PagedAttention in MBs.
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
        - `pa_gpu_mem_usage`: Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
            If this is not set and the device is CUDA, it will default to `0.9`.
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
        - `pa_ctxt_len`: Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-gpu-mem-usage` (default = 0.9) > `pa-ctxt-len` > `pa-gpu-mem`.
        - `pa_blk_size` sets the block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA,
            it will default to 32. PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
        - `no_paged_attn` disables PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
        - `paged_attn` enables PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
        - `prompt_batchsize` Number of tokens to batch the prompt step into. This can help with OOM errors when in the prompt step, but reduces performance.
        - `seed`, used to ensure reproducible random number generation.
        """
        ...

    def send_chat_completion_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]:
        """
        Send a chat completion request to the mistral.rs engine, returning the response object or a generator
        over chunk objects.
        """

    def send_completion_request(self, request: CompletionRequest) -> CompletionResponse:
        """
        Send a chat completion request to the mistral.rs engine, returning the response object.
        """

    def generate_image(
        self,
        prompt: str,
        response_format: ImageGenerationResponseFormat,
        height: int = 720,
        width: int = 1280,
    ) -> ImageGenerationResponse:
        """
        Generate an image.
        """

    def send_re_isq(self, dtype: str) -> CompletionResponse:
        """
        Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML then nothing will happen.
        """

    def activate_adapters(self, adapter_names: list[str]) -> None:
        """
        Send a request to make the specified adapters the active adapters for the model.
        """

    def tokenize_text(self, text: str, add_speial_tokens: bool) -> list[int]:
        """
        Tokenize some text, returning raw tokens.
        """

    def detokenize_text(self, tokens: list[int], skip_speial_tokens: bool) -> str:
        """
        Detokenize some tokens, returning text.
        """

class AnyMoeExpertType(Enum):
    """
    Expert type for an AnyMoE model. May be:
    - `AnyMoeExpertType.FineTuned()`
    - `AnyMoeExpertType.LoraAdapter(rank: int, alpha: float, target_modules: list[str])`
    """
    @dataclass
    class FineTuned:
        pass

    @dataclass
    class LoraAdapter:
        rank: int
        alpha: float
        target_modules: list[str]

class AnyMoeConfig:
    def __init__(
        self,
        hidden_size: str,
        dataset_json: str,
        prefix: str,
        mlp: str,
        model_ids: list[str],
        expert_type: AnyMoeExpertType,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 4,
        gate_model_id: str | None = None,
        training: bool = False,
        loss_csv_path: str | None = None,
    ) -> None:
        """
        Create an AnyMoE config from the hidden size, dataset, and other metadata. The model IDs may be local paths.

        To find the prefix/mlp values:

            - Go to `https://huggingface.co/<MODEL ID>/tree/main?show_file_info=model.safetensors.index.json`
            - Look for the mlp layers: For example `model.layers.27.mlp.down_proj.weight` means that the prefix is `model.layers` and the mlp is `mlp`.

        To find the hidden size:

            - Can be found at `https://huggingface.co/<BASE MODEL ID>/blob/main/config.json`

        > Note: `gate_model_id` specifies the gating model ID. If `training == True`, then safetensors will be written here.
            Otherwise, the pretrained safetensors will be loaded and no training occurs.

        > Note: if `training == True`, `loss_csv_path` has no effect. Otherwise, an csv loss file will be saved here.
        """
        ...

@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    avg_tok_per_sec: float
    avg_prompt_tok_per_sec: float
    avg_compl_tok_per_sec: float
    total_time_sec: float
    total_prompt_time_sec: float
    total_completion_time_sec: float

@dataclass
class ToolCallType(Enum):
    Function = "function"

@dataclass
class CalledFunction:
    name: str
    arguments: str

@dataclass
class ToolCallResponse:
    id: str
    type: ToolCallType
    function: CalledFunction

@dataclass
class ResponseMessage:
    content: str
    role: str
    tool_calls: list[ToolCallResponse]

@dataclass
class TopLogprob:
    token: int
    logprob: float
    bytes: str

@dataclass
class ResponseLogprob:
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[TopLogprob]

@dataclass
class Logprobs:
    content: list[ResponseLogprob] | None

@dataclass
class Choice:
    finish_reason: str
    index: int
    message: ResponseMessage
    logprobs: Logprobs

@dataclass
class ChatCompletionResponse:
    id: str
    choices: list[Choice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage

@dataclass
class Delta:
    content: str
    role: str

@dataclass
class ChunkChoice:
    finish_reason: str | None
    index: int
    delta: Delta
    logprobs: ResponseLogprob | None

@dataclass
class ChatCompletionChunkResponse:
    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str

@dataclass
class CompletionChoice:
    finish_reason: str
    index: int
    text: str
    # NOTE(EricLBuehler): `logprobs` in undocumented

@dataclass
class CompletionResponse:
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage

@dataclass
class ImageChoice:
    url: str | None
    b64_json: str | None

@dataclass
class ImageGenerationResponse:
    choices: list[ImageChoice]
    created: int
