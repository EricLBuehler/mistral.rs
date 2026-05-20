from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping, Optional, Callable

class SearchContextSize(Enum):
    Low = "low"
    Medium = "medium"
    High = "high"

class AgentPermission(Enum):
    Auto = "auto"
    Ask = "ask"
    Deny = "deny"

class CodeExecutionPermission(Enum):
    Auto = "auto"
    Ask = "ask"
    Deny = "deny"

class NetworkMode(Enum):
    NoNetwork = "none"
    Loopback = "loopback"
    Full = "full"

class AgentToolSource(Enum):
    BuiltIn = "built_in"
    User = "user"
    Mcp = "mcp"
    External = "external"

class AgentToolKind(Enum):
    CodeExecution = "code_execution"
    WebSearch = "web_search"
    File = "file"
    Custom = "custom"
    External = "external"

class AgentToolApprovalDecisionKind(Enum):
    Approve = "approve"
    Deny = "deny"

@dataclass
class ApproximateUserLocation:
    city: str
    country: str
    region: str
    timezone: str

class WebSearchUserLocation:
    @staticmethod
    def approximate(approximate: ApproximateUserLocation) -> "WebSearchUserLocation": ...

@dataclass
class WebSearchOptions:
    search_context_size: Optional[SearchContextSize] = None
    user_location: Optional[WebSearchUserLocation] = None
    search_description: Optional[str] = None
    extract_description: Optional[str] = None

@dataclass
class ToolChoice(Enum):
    NoTools = "None"
    Auto = "Auto"

@dataclass
class AgentToolMetadata:
    """
    Stable metadata for the agent action being approved.
    """

    source: AgentToolSource
    kind: AgentToolKind
    label: str

@dataclass
class AgentToolApproval:
    """
    Approval request passed to `ChatCompletionRequest.agent_approval_callback`.
    """

    approval_id: str
    session_id: str
    round: int
    tool: AgentToolMetadata
    arguments_json: str
    code: str | None = None

    def arguments(self) -> Any: ...

@dataclass
class AgentToolApprovalDecision:
    """
    Approval callback return value with HTTP/Rust parity.
    """

    decision: AgentToolApprovalDecisionKind
    remember_for_session: bool = False
    message: str | None = None

    @staticmethod
    def approve(remember_for_session: bool = False) -> "AgentToolApprovalDecision": ...

    @staticmethod
    def deny(message: str | None = None) -> "AgentToolApprovalDecision": ...

@dataclass
class ChatCompletionRequest:
    """
    A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.

    The messages type is as follows: (for normal chat completion, for chat completion with images, pretemplated prompt)

    Agent permission fields:

    - `agent_permission`: `AgentPermission.Auto`, `.Ask`, or `.Deny`. Applies to server-executed
      agent actions such as code execution, web search, file tools, callbacks,
      and external tool dispatch.
    - `agent_approval_callback`: called when `agent_permission=AgentPermission.Ask` with an
      `AgentToolApproval`. Return `True`, `False`, or
      `AgentToolApprovalDecision`.

    See [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions)
    for the shared CLI, HTTP, Python, and Rust behavior.
    """

    messages: (
        list[dict[str, str]] | list[dict[str, list[dict[str, str | dict[str, str]]]]]
    ) | str
    model: str
    logprobs: bool = False
    n_choices: int = 1
    logit_bias: dict[int, float] | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    grammar: str | None = None
    grammar_type: str | None = None
    min_p: float | None = None
    tool_schemas: list[str] | None = None
    tool_choice: ToolChoice | None = None
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_sequence_breakers: list[str] | None = None
    web_search_options: WebSearchOptions | None = None
    enable_thinking: bool | None = None
    truncate_sequence: bool = False
    reasoning_effort: str | None = None
    max_tool_rounds: int | None = None
    tool_dispatch_url: str | None = None
    enable_code_execution: bool = False
    agent_permission: AgentPermission | None = None
    agent_approval_callback: Callable[
        [AgentToolApproval], bool | AgentToolApprovalDecision
    ] | None = None
    code_execution_permission: CodeExecutionPermission | None = None
    session_id: str | None = None
    files: list[RequestedFile] | None = None

@dataclass
class CompletionRequest:
    """
    A CompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.
    """

    prompt: str
    model: str
    best_of: int = 1
    echo_prompt: bool = False
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    suffix: str | None = None
    top_k: int | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    min_p: float | None = None
    tool_schemas: list[str] | None = None
    tool_choice: ToolChoice | None = None
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_sequence_breakers: list[str] | None = None
    truncate_sequence: bool = False

@dataclass
class EmbeddingRequest:
    """
    An EmbeddingRequest represents a request to compute embeddings for the provided input text.
    """

    input: str | list[str] | list[int] | list[list[int]]
    truncate_sequence: bool = False

@dataclass
class Architecture(Enum):
    Mistral = "mistral"
    Gemma = "gemma"
    Mixtral = "mixtral"
    Llama = "llama"
    Phi2 = "phi2"
    Phi3 = "phi3"
    Qwen2 = "qwen2"
    Gemma2 = "gemma2"
    Starcoder2 = "starcoder2"
    Phi3_5MoE = "phi3.5moe"
    DeepseekV2 = "deepseekv2"
    DeepseekV3 = "deepseekv3"
    Qwen3 = "qwen3"
    GLM4 = "glm4"
    GLM4Moe = "glm4moe"
    GLM4MoeLite = "glm4moelite"
    Qwen3Moe = "qwen3moe"
    SmolLm3 = "smollm3"
    GraniteMoeHybrid = "granitemoehybrid"
    GptOss = "gptoss"
    Qwen3Next = "qwen3next"

@dataclass
class EmbeddingArchitecture(Enum):
    EmbeddingGemma = "embeddinggemma"
    Qwen3Embedding = "qwen3embedding"

@dataclass
class MultimodalArchitecture(Enum):
    Phi3V = "phi3v"
    Idefics2 = "idefics2"
    LLaVANext = "llava-next"
    LLaVA = "llava"
    VLlama = "vllama"
    Qwen2VL = "qwen2vl"
    Idefics3 = "idefics3"
    MiniCpmO = "minicpmo"
    Phi4MM = "phi4mm"
    Qwen2_5VL = "qwen2_5vl"
    Gemma3 = "gemma3"
    Mistral3 = "mistral3"
    Llama4 = "llama4"
    Gemma3n = "Gemma3n"
    Qwen3VL = "Qwen3VL"
    Qwen3VLMoE = "Qwen3VLMoE"
    Qwen3_5 = "Qwen3_5"
    Qwen3_5Moe = "Qwen3_5Moe"
    Voxtral = "Voxtral"
    Gemma4 = "Gemma4"

@dataclass
class DiffusionArchitecture(Enum):
    Flux = "flux"
    FluxOffloaded = "flux-offloaded"

@dataclass
class SpeechLoaderType(Enum):
    Dia = "Dia"

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
    Url = "Url"
    B64Json = "B64Json"

@dataclass
class SpeechGenerationResponse:
    """
    This wraps PCM values, sampling rate and the number of channels.
    """

    pcm: list[float]
    rate: int
    channels: int

@dataclass
class TextAutoMapParams:
    """
    Auto-mapping parameters for a text model.
    These affect automatic device mapping but are not a hard limit.
    """

    max_seq_len: int = 4 * 1024
    max_batch_size: int = 1

@dataclass
class MultimodalAutoMapParams:
    """
    Auto-mapping parameters for a multimodal model.
    These affect automatic device mapping but are not a hard limit.
    """

    max_seq_len: int = 4 * 1024
    max_batch_size: int = 1
    max_num_images: int = 1
    max_image_length: int = 1024

class SandboxPolicy:
    """
    OS-level sandbox applied to the code-execution subprocess on Linux/macOS.

    Pass to `CodeExecutionConfig(sandbox_policy=...)` to enable the sandbox;
    omit (or pass `None`) to disable it. See the sandbox reference for the
    layered defenses: env scrub, namespaces, Landlock FS allowlist, rlimits,
    seccomp deny-list, and optional cgroup v2 on Linux.

    - `max_memory_mb`: per-session memory cap (default 2048).
    - `max_cpu_secs`: per-session CPU time cap (default 300).
    - `max_procs`: per-session process/thread cap (default 64).
    - `max_open_fds`: per-session open-fd cap (default 1024).
    - `max_file_sz_mb`: per-session max written-file size (default 256).
    - `network`: `NetworkMode.NoNetwork`, `.Loopback`, or `.Full`.
    - `extra_fs_read`: additional paths the sandboxed process may read.
    - `extra_fs_write`: additional paths the sandboxed process may read/write.
    - `extra_env`: additional environment variable names allowed through.
    - `strict`: fail initialization if requested filesystem or network
      isolation is unavailable.
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,
        max_cpu_secs: int = 300,
        max_procs: int = 64,
        max_open_fds: int = 1024,
        max_file_sz_mb: int = 256,
        network: NetworkMode = NetworkMode.Loopback,
        extra_fs_read: list[str] = [],
        extra_fs_write: list[str] = [],
        extra_env: list[str] = [],
        strict: bool = False,
    ) -> None: ...

class CodeExecutionConfig:
    """
    Configuration for the built-in Python code execution tool.

    Pass to `Runner(code_execution_config=...)` to enable the `execute_python`
    tool. Per-request, set `ChatCompletionRequest.enable_code_execution=True`.

    All fields are optional:

    - `python_path`: interpreter to run. Defaults to `python` on Windows,
      `python3` elsewhere.
    - `timeout_secs`: per-call timeout. Defaults to 30.
    - `working_directory`: shared working directory. Defaults to a per-session
      temp directory.
    - `sandbox_policy`: an OS-level sandbox to apply to the spawned interpreter
      on Linux/macOS. `None` (default) disables the sandbox; passing a
      `SandboxPolicy` enables it with the configured limits.
    - `permission`: `CodeExecutionPermission.Auto`, `.Ask`, or `.Deny`. For
      new code, prefer `ChatCompletionRequest.agent_permission`.
    - `approval_callback`: code-execution-specific callback. For new code,
      prefer `ChatCompletionRequest.agent_approval_callback`, which applies to
      all agent actions.
    """

    def __init__(
        self,
        python_path: str | None = None,
        timeout_secs: int | None = None,
        working_directory: str | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        permission: CodeExecutionPermission | None = None,
        approval_callback: Callable[[dict[str, object]], bool] | None = None,
    ) -> None: ...

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
        organization: IsqOrganization | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        imatrix: str | None = None
        calibration_file: str | None = None
        auto_map_params: TextAutoMapParams | None = None
        hf_cache_path: str | None = None
        matformer_config_path: str | None = None
        matformer_slice_name: str | None = None

    @dataclass
    class Embedding:
        model_id: str
        arch: EmbeddingArchitecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
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
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None
        hf_cache_path: str | None = None

    @dataclass
    class Lora:
        adapter_model_ids: list[str]
        arch: Architecture | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None
        hf_cache_path: str | None = None

    @dataclass
    class GGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

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
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class LoraGGUF:
        quantized_model_id: str
        quantized_filename: str | list[str]
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class GGML:
        quantized_model_id: str
        quantized_filename: str
        tok_model_id: str
        tokenizer_json: str | None = None
        gqa: int = 1
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class XLoraGGML:
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        tgt_non_granular_index: int | None = None
        gqa: int = 1
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class LoraGGML:
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tok_model_id: str | None = None
        tokenizer_json: str | None = None
        gqa: int = 1
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto
        auto_map_params: TextAutoMapParams | None = None

    @dataclass
    class MultimodalPlain:
        model_id: str
        arch: MultimodalArchitecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        max_edge: int | None = None
        calibration_file: str | None = None
        imatrix: str | None = None
        auto_map_params: MultimodalAutoMapParams | None = None
        hf_cache_path: str | None = None
        matformer_config_path: str | None = None
        matformer_slice_name: str | None = None
        organization: IsqOrganization | None = None

    @dataclass
    class DiffusionPlain:
        model_id: str
        arch: DiffusionArchitecture
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class Speech:
        model_id: str
        arch: SpeechLoaderType
        dac_model_id: str | None = None
        dtype: ModelDType = ModelDType.Auto

class PagedCacheType(Enum):
    Auto: int = 0
    F8E4M3: int = 1

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
        jinja_explicit: str | None = None,
        num_device_layers: list[str] | None = None,
        in_situ_quant: str | None = None,
        anymoe_config: AnyMoeConfig | None = None,
        pa_gpu_mem: int | None = None,
        pa_gpu_mem_usage: float | None = None,
        pa_ctxt_len: int | None = None,
        pa_blk_size: int | None = None,
        pa_cache_type: PagedCacheType | None = None,
        no_paged_attn: bool = False,
        paged_attn: bool = False,
        seed: int | None = None,
        enable_search: bool = False,
        search_embedding_model: str | None = None,
        search_callback: Callable[[str], list[dict[str, str]]] | None = None,
        tool_callbacks: Mapping[str, Callable[[str, dict], str]] | None = None,
        mcp_client_config: McpClientConfigPy | None = None,
        code_execution_config: CodeExecutionConfig | None = None,
    ) -> None:
        """
        Load a model.

        - `which` specifies which model to load or the target model to load in the case of speculative decoding.
        - `max_seqs` specifies how many sequences may be running at any time.
        - `no_kv_cache` disables the KV cache.
        - `prefix_cache_n` sets the number of sequences to hold in the device prefix cache, others will be evicted to CPU.
        - `token_source` specifies where to load the HF token from.
            The token source follows the following format: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token.
        - `speculative_gamma` specifies the `gamma` parameter for speculative decoding, the ratio of draft tokens to generate before calling
            the target model. If `which_draft` is not specified, this is ignored.
        - `which_draft` specifies which draft model to load. Setting this parameter will cause a speculative decoding model to be loaded,
            with `which` as the target (higher quality) model and `which_draft` as the draft (lower quality) model.
        - `chat_template` specifies an optional JINJA chat template as a JSON file.
            This chat template should have `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
            It is used if the automatic deserialization fails. If this ends with `.json` (i.e., it is a file) then that template is loaded.
        - `jinja_explicit` allows an explicit JINJA chat template file to be used. If specified, this overrides all other chat templates.
        - `num_device_layers` sets the number of layers to load and run on each device.
            Each element follows the format ORD:NUM where ORD is the device ordinal and NUM is
            the corresponding number of layers. Note: this is deprecated in favor of automatic device mapping.
        - `in_situ_quant` sets the optional in-situ quantization for a model.
        - `anymoe_config` specifies the AnyMoE config. If this is set, then the model will be loaded as an AnyMoE model.
        - `pa_gpu_mem`: GPU memory to allocate for KV cache with PagedAttention in MBs.
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
        - `pa_gpu_mem_usage`: Percentage of GPU memory to utilize after allocation of KV cache with PagedAttention, from 0 to 1.
            If this is not set and the device is CUDA, it will default to `0.9`.
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
        - `pa_ctxt_len`: Total context length to allocate the KV cache for (total number of tokens which the KV cache can hold).
            PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
            The priority is as follows: `pa-ctxt-len` > `pa-gpu-mem-usage` > `pa-gpu-mem`.
            This is the default setting, and it defaults to the `max-seq-len` specified in after the model type.
        - `pa_blk_size` sets the block size (number of tokens per block) for PagedAttention. If this is not set and the device is CUDA,
            it will default to 32. PagedAttention is supported on CUDA and Metal. It is automatically activated on CUDA but not on Metal.
        - `pa_cache_type` sets the PagedAttention KV cache type (auto or f8e4m3). Defaults to `auto`.
        - `no_paged_attn` disables PagedAttention on CUDA. Because PagedAttention is already disabled on Metal, this is only applicable on CUDA.
        - `paged_attn` enables PagedAttention on Metal. Because PagedAttention is already enabled on CUDA, this is only applicable on Metal.
        - `seed`, used to ensure reproducible random number generation.
        - `enable_search`: Enable searching compatible with the OpenAI `web_search_options` setting. This loads the selected search embedding reranker (EmbeddingGemma by default).
        - `search_embedding_model`: select which built-in search embedding model to load (currently `"embedding_gemma"`).
        - `search_callback`: Custom Python callable to perform web searches. Should accept a query string and return a list of dicts with keys "title", "description", "url", and "content".
        - `tool_callbacks`: Mapping from tool name to Python callable invoked for generic tool calls. Each callable receives the tool name and a dict of arguments and should return the tool output as a string.
        - `code_execution_config`: enables the built-in Python code execution tool. Pass a `CodeExecutionConfig` to configure the interpreter, per-call timeout, and working directory. Per-request, set `ChatCompletionRequest.enable_code_execution=True`.
        """
        ...

    def send_chat_completion_request(
        self, request: ChatCompletionRequest, model_id: str | None = None
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]:
        """
        Send a chat completion request to the mistral.rs engine, returning the response object or a generator
        over chunk objects.

        Args:
            request: The chat completion request.
            model_id: Optional model ID to send the request to. If None, uses the default model.
        """

    def send_completion_request(
        self, request: CompletionRequest, model_id: str | None = None
    ) -> CompletionResponse:
        """
        Send a completion request to the mistral.rs engine, returning the response object.

        Args:
            request: The completion request.
            model_id: Optional model ID to send the request to. If None, uses the default model.
        """

    def send_embedding_request(
        self, request: EmbeddingRequest, model_id: str | None = None
    ) -> list[list[float]]:
        """
        Generate embeddings for the supplied inputs and return one embedding vector per input.

        Args:
            request: The embedding request.
            model_id: Optional model ID to send the request to. If None, uses the default model.
        """

    def generate_image(
        self,
        prompt: str,
        response_format: ImageGenerationResponseFormat,
        height: int = 720,
        width: int = 1280,
        model_id: str | None = None,
        save_file: str | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate an image.

        Args:
            prompt: The image generation prompt.
            response_format: The response format (Url or B64Json).
            height: Image height in pixels.
            width: Image width in pixels.
            model_id: Optional model ID to send the request to. If None, uses the default model.
            save_file: Optional path where the PNG is written when response_format is Url. Defaults to an auto-generated filename.
        """

    def generate_audio(
        self, prompt: str, model_id: str | None = None
    ) -> SpeechGenerationResponse:
        """
        Generate audio given a (model specific) prompt. PCM and sampling rate as well as the number of channels is returned.

        Args:
            prompt: The audio generation prompt.
            model_id: Optional model ID to send the request to. If None, uses the default model.
        """

    def send_re_isq(self, dtype: str, model_id: str | None = None) -> None:
        """
        Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML then nothing will happen.

        Args:
            dtype: The ISQ dtype (e.g., "Q4K", "Q8_0").
            model_id: Optional model ID to re-ISQ. If None, uses the default model.
        """

    def tokenize_text(
        self,
        text: str,
        add_special_tokens: bool,
        enable_thinking: bool | None,
        model_id: str | None = None,
    ) -> list[int]:
        """
        Tokenize some text, returning raw tokens.

        Args:
            text: The text to tokenize.
            add_special_tokens: Whether to add special tokens.
            enable_thinking: Enables thinking for models that support this configuration.
            model_id: Optional model ID to use for tokenization. If None, uses the default model.
        """

    def detokenize_text(
        self, tokens: list[int], skip_special_tokens: bool, model_id: str | None = None
    ) -> str:
        """
        Detokenize some tokens, returning text.

        Args:
            tokens: The tokens to detokenize.
            skip_special_tokens: Whether to skip special tokens.
            model_id: Optional model ID to use for detokenization. If None, uses the default model.
        """

    def max_sequence_length(self, model_id: str | None = None) -> int | None:
        """
        Return the maximum supported sequence length for the current or specified model, or None when
        the concept does not apply (such as diffusion or speech models).

        Args:
            model_id: Optional model ID to query. If None, uses the default model.
        """

    # Multi-model management methods

    def list_models(self) -> list[str]:
        """
        List all available model IDs (aliases if configured).

        Returns:
            A list of model ID strings.
        """

    def get_default_model_id(self) -> str | None:
        """
        Get the current default model ID.

        Returns:
            The default model ID, or None if no default is set.
        """

    def set_default_model_id(self, model_id: str) -> None:
        """
        Set the default model ID. The model must already be loaded.

        Args:
            model_id: The model ID to set as default.

        Raises:
            ValueError: If the model ID is not found.
        """

    def is_model_loaded(self, model_id: str) -> bool:
        """
        Check if a model is currently loaded in memory.

        Args:
            model_id: The model ID to check.

        Returns:
            True if the model is loaded, False otherwise.
        """

    def unload_model(self, model_id: str) -> None:
        """
        Unload a model from memory while preserving its configuration for later reload.
        The model can be reloaded manually with reload_model() or automatically when
        a request is sent to it.

        Args:
            model_id: The model ID to unload.
        """

    def reload_model(self, model_id: str) -> None:
        """
        Manually reload a previously unloaded model.

        Args:
            model_id: The model ID to reload.
        """

    def list_models_with_status(self) -> list[tuple[str, str]]:
        """
        List all models with their current status.

        Returns:
            A list of (model_id, status) tuples where status is one of:
            - "loaded": Model is loaded and ready
            - "unloaded": Model is unloaded but can be reloaded
            - "reloading": Model is currently being reloaded
        """

    def list_unloaded_models(self) -> list[str]:
        """
        List model IDs that are currently unloaded (but can be reloaded).
        """

    def get_model_status(self, model_id: str) -> str | None:
        """
        Get the status of a model: "loaded", "unloaded", "reloading", or None if not found.
        """

    def remove_model(self, model_id: str) -> None:
        """
        Remove a model by ID in multi-model mode.
        """

    # Per-model routing

    def send_chat_completion_request_to_model(
        self, request: ChatCompletionRequest, model_id: str
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]:
        """
        Send a chat completion request to a specific model, returning the response
        object or a generator over chunk objects.
        """

    def send_completion_request_to_model(
        self, request: CompletionRequest, model_id: str
    ) -> CompletionResponse:
        """
        Send a completion request to a specific model.
        """

    # Agentic session management

    def export_session(
        self, session_id: str, model_id: str | None = None
    ) -> str | None:
        """
        Export an agentic session by ID as a JSON string.

        Returns None if the session does not exist.
        """

    def import_session(
        self,
        session_id: str,
        session_json: str,
        model_id: str | None = None,
    ) -> None:
        """
        Import an agentic session from a JSON string.

        Replaces any existing session with the same ID.
        """

    def delete_session(
        self, session_id: str, model_id: str | None = None
    ) -> bool:
        """
        Delete an agentic session. Returns whether the session existed.
        """

    def list_session_ids(self, model_id: str | None = None) -> list[str]:
        """
        List all stored agentic session IDs.
        """

    def find_file(self, file_id: str) -> File | None:
        """
        Look up a produced file by id. Returns the full body even if the
        file was wire-truncated in the response payload.
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
        hidden_size: int,
        dataset_json: str,
        prefix: str,
        mlp: str,
        model_ids: list[str],
        expert_type: AnyMoeExpertType,
        layers: list[int] = [],
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 4,
        gate_model_id: str | None = None,
        training: bool = True,
        loss_csv_path: str | None = None,
    ) -> None:
        """
        Create an AnyMoE config from the hidden size, dataset, and other metadata. The model IDs may be local paths.

        To find the prefix/mlp values:

        - Go to `https://huggingface.co/<MODEL ID>/tree/main?show_file_info=model.safetensors.index.json`
        - Look for the mlp layers: for example `model.layers.27.mlp.down_proj.weight` means the prefix is `model.layers` and the mlp is `mlp`.

        To find the hidden size:

        - Look it up in `https://huggingface.co/<BASE MODEL ID>/blob/main/config.json`.

        Note: `gate_model_id` specifies the gating model ID. If `training == True`, safetensors are written here; otherwise the pretrained safetensors are loaded and no training occurs.

        Note: if `training == True`, `loss_csv_path` has no effect. Otherwise, a CSV loss file is saved at that path.
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
    index: int
    id: str
    tp: ToolCallType
    function: CalledFunction

@dataclass
class ResponseMessage:
    content: str | None
    role: str
    tool_calls: list[ToolCallResponse] | None
    reasoning_content: str | None = None

@dataclass
class TopLogprob:
    token: int
    logprob: float
    bytes: str | None

@dataclass
class ResponseLogprob:
    token: str
    logprob: float
    bytes: list[int] | None
    top_logprobs: list[TopLogprob]

@dataclass
class Logprobs:
    content: list[ResponseLogprob] | None

@dataclass
class Choice:
    finish_reason: str
    index: int
    message: ResponseMessage
    logprobs: Logprobs | None = None

@dataclass
class AgenticToolCallRecord:
    round: int
    name: str
    arguments: str
    result_content: str
    result_images_base64: list[str]
    file_ids: list[str]

@dataclass
class ChatCompletionResponse:
    id: str
    choices: list[Choice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage
    agentic_tool_calls: list[AgenticToolCallRecord] | None = None
    files: list[File] | None = None
    session_id: str | None = None

@dataclass
class Delta:
    content: str | None
    role: str
    tool_calls: list[ToolCallResponse] | None = None
    reasoning_content: str | None = None

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
    usage: Usage | None = None
    session_id: str | None = None

@dataclass
class CompletionChoice:
    finish_reason: str
    index: int
    text: str
    logprobs: Logprobs | None = None

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
    data: list[ImageChoice]
    created: int

# Files

class RequestedFile:
    """A required output file declared on a request. The runtime tells the
    model about declared files; if produced by a tool, they surface in
    `ChatCompletionResponse.files`. If missing, an error placeholder is
    surfaced instead."""

    name: str
    format: str | None
    description: str | None
    def __init__(
        self,
        name: str,
        format: str | None = None,
        description: str | None = None,
    ) -> None: ...

@dataclass
class FileSource:
    """Where a file was produced."""

    tool: str
    round: int
    turn: int

class File:
    """First-class output from an agentic run.

    Files exist independently of the transcript. The body is inline for
    small files (`text` for text content, `data_base64` for binary). Large
    files have a server-side url and `text`/`data_base64` will be `None` -
    use `is_truncated()` to detect."""

    id: str
    name: str
    format: str | None
    mime_type: str | None
    bytes: int
    source: FileSource
    text: str | None
    data_base64: str | None
    preview: str | None
    def is_text(self) -> bool: ...
    def is_binary(self) -> bool: ...
    def is_image(self) -> bool: ...
    def is_video(self) -> bool: ...
    def is_error(self) -> bool: ...
    def is_truncated(self) -> bool: ...
    def save(self, path: str) -> None: ...

# MCP (Model Context Protocol) Client Types

class McpServerSourcePy:
    """MCP server transport source. Construct via the variant factories below. All arguments are positional and required; pass `None` explicitly for unused fields."""

    @staticmethod
    def Http(
        url: str,
        timeout_secs: int | None,
        headers: dict[str, str] | None,
    ) -> "McpServerSourcePy": ...
    @staticmethod
    def Process(
        command: str,
        args: list[str],
        work_dir: str | None,
        env: dict[str, str] | None,
    ) -> "McpServerSourcePy": ...
    @staticmethod
    def WebSocket(
        url: str,
        timeout_secs: int | None,
        headers: dict[str, str] | None,
    ) -> "McpServerSourcePy": ...

@dataclass
class McpServerConfigPy:
    """Configuration for an individual MCP server"""

    id: str
    name: str
    source: McpServerSourcePy
    enabled: bool = True
    tool_prefix: Optional[str] = None
    resources: Optional[list[str]] = None
    bearer_token: Optional[str] = None

@dataclass
class McpClientConfigPy:
    """Configuration for MCP client integration"""

    servers: list[McpServerConfigPy]
    auto_register_tools: bool = True
    tool_timeout_secs: Optional[int] = None
    max_concurrent_calls: Optional[int] = None
