from dataclasses import dataclass, field
from enum import Enum
from os import PathLike
from typing import Any, Callable, Iterator, Literal, Mapping, Optional

class CalibrationStatus:
    collecting: bool
    layers: int
    layers_tracking: int
    total_rows: int
    min_rows: int
    max_rows: int

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
    """No network access. On Linux, opening a socket fails."""
    Loopback = "loopback"
    """Only `localhost` is reachable."""
    Full = "full"
    """No network restrictions."""

class AgentToolSource(Enum):
    BuiltIn = "built_in"
    User = "user"
    Mcp = "mcp"
    External = "external"

class AgentToolKind(Enum):
    CodeExecution = "code_execution"
    WebSearch = "web_search"
    Shell = "shell"
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
    def approximate(
        approximate: ApproximateUserLocation,
    ) -> "WebSearchUserLocation": ...

@dataclass
class WebSearchOptions:
    search_context_size: Optional[SearchContextSize] = None
    user_location: Optional[WebSearchUserLocation] = None
    search_description: Optional[str] = None
    extract_description: Optional[str] = None

@dataclass
class ToolChoice(Enum):
    NoTools = "None"
    """The model can't call tools."""
    Auto = "Auto"
    """The model decides whether to call a tool."""

@dataclass
class AgentToolMetadata:
    """
    Describes the tool behind an agent action that is waiting for approval.
    """

    source: AgentToolSource
    """Where the tool comes from: built in, registered by your app, an MCP server, or an external dispatch URL."""
    kind: AgentToolKind
    """What the tool does, such as code execution, shell, or web search."""
    label: str
    """Display name, such as `"Python code"`, `"Shell command"`, or the tool's own name."""

@dataclass
class AgentToolApproval:
    """
    An agent action waiting for approval. Your `ChatCompletionRequest.agent_approval_callback` receives one.
    """

    approval_id: str
    """ID of this approval request."""
    session_id: str
    """Agent session the action belongs to."""
    round: int
    """Tool-call round within the request."""
    tool: AgentToolMetadata
    """The tool the model wants to run."""
    arguments_json: str
    """The tool call's arguments, as a JSON string."""
    code: str | None = None
    """The `code` argument when it is a string, as it is for Python code execution. Otherwise `None`."""

    def arguments(self) -> Any:
        """Parse `arguments_json` and return the result as Python objects."""

@dataclass
class AgentToolApprovalDecision:
    """
    A richer answer for an approval callback than `True` or `False`. Use it to send the model a reason
    for a denial, or to approve every later action in the session.
    """

    decision: AgentToolApprovalDecisionKind
    """Approve or deny."""
    remember_for_session: bool = False
    """On approve, skip approval for later actions in the same `session_id`."""
    message: str | None = None
    """On deny, text returned to the model as the tool result."""

    @staticmethod
    def approve(remember_for_session: bool = False) -> "AgentToolApprovalDecision":
        """Approve the action."""
    @staticmethod
    def deny(message: str | None = None) -> "AgentToolApprovalDecision":
        """Deny the action, optionally with a message for the model."""

@dataclass(frozen=True)
class LoraAdapterGeneration:
    """
    Pins a request to one exact version of a LoRA adapter, instead of whatever its alias points to now.

    Each set of adapter files you load is a generation, identified by a 64-character hash of its config
    and weights. Loading different files under an alias gives it a new generation. Loading the same
    files again gives the same one. See
    [pin an exact adapter version](/guides/customize/lora-adapters/#pin-an-exact-adapter-version).

    Get one from `LoraAdapterInfo.exact()`, or pass a generation ID. A malformed ID raises `ValueError`.
    """

    generation: str
    """The 64-character generation ID."""

class LoraAdapterError(ValueError):
    """
    Raised when a dynamic LoRA operation fails. See the
    [error codes](/guides/customize/lora-adapters/#troubleshooting).
    """

    code: str
    """Why it failed, such as `lora_generation_mismatch` or `lora_adapter_not_found`."""

@dataclass
class ChatCompletionRequest:
    """
    A chat request. Send it with `Runner.send_chat_completion_request`.

    `messages` takes one of three forms:

    - A list of `{"role": ..., "content": "..."}` dicts.
    - A list of dicts whose `content` is a list of parts, for images, audio, or video. Part types are
      `text`, `image_url`, `input_audio`, `audio_url`, and `video_url`. Only `user` messages can hold media.
    - A plain string, sent as a single user message.

    Sampling fields left as `None` use the model's `generation_config.json` values when it has them.

    Reasoning controls:

    - `reasoning_effort` accepts `off`, `low`, `medium`, `high`, and `xhigh`. `none` means `off` and
      `max` means `xhigh`. Case and surrounding spaces are ignored.
    - If you set neither control, thinking is on for text output and off for speech output.
    - Contradictory values, such as `enable_thinking=False` with `reasoning_effort="high"`, raise `ValueError`.

    Audio:

    - Input: add a part `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`.
      `format` is `wav` or `mp3`. For other formats, use an `audio_url` part.
    - Speech output: set `modalities=["audio"]` or `["text", "audio"]`, and `audio={"format": ...}`.
      Formats are `wav`, `pcm16`, `mp3`, `flac`, `opus`, and `aac`. All but `wav` and `pcm16` need FFmpeg.
      `voice` can be omitted or set to `"default"`.
    - Streaming speech requires `pcm16` and arrives as `delta.audio.data` chunks.
    - The HTTP API's `audio.id` history replay isn't available here.
    - Text sampling fields don't change LFM3's separate speech sampler.

    For the agent fields, see [agent permissions](/guides/agents/permissions-and-approvals/).
    """

    messages: (
        list[dict[str, str]] | list[dict[str, list[dict[str, str | dict[str, str]]]]]
    ) | str
    """The conversation, in one of the forms above."""
    model: str
    """Not used to choose a model. Pass `model_id` to the `Runner` method instead."""
    logprobs: bool = False
    """Return the log probability of each output token."""
    n_choices: int = 1
    """Number of completions to generate."""
    logit_bias: dict[int, float] | None = None
    """Map of token ID to a bias added to that token's logit before sampling."""
    top_logprobs: int | None = None
    """With `logprobs`, how many of the most likely tokens to return at each step. Defaults to 1."""
    max_tokens: int | None = None
    """Maximum number of tokens to generate."""
    presence_penalty: float | None = None
    """Flat penalty on any token already generated. 0 turns it off."""
    frequency_penalty: float | None = None
    """Penalty that grows each time a token is generated again. 0 turns it off."""
    repetition_penalty: float | None = None
    """Penalty on tokens already in the prompt or output. 1.0 turns it off."""
    stop_seqs: list[str] | None = None
    """Stop generating when any of these strings appears."""
    temperature: float | None = None
    """Sampling temperature. 0 picks the most likely token every time."""
    top_p: float | None = None
    """Sample only from the most likely tokens whose probabilities add up to `top_p`."""
    top_k: int | None = None
    """Sample only from the `top_k` most likely tokens."""
    stream: bool = False
    """Return an iterator of `ChatCompletionChunkResponse` instead of one response."""
    grammar: str | None = None
    """Constraint that the output must match. Set `grammar_type` with it."""
    grammar_type: str | None = None
    """How to read `grammar`: `"regex"`, `"lark"`, `"json_schema"`, or `"llguidance"`."""
    min_p: float | None = None
    """Drop tokens less likely than `min_p` times the most likely token."""
    tool_schemas: list[str] | None = None
    """Tools the model may call, each a JSON string in the OpenAI format (`{"type": "function", "function": {...}}`)."""
    tool_choice: ToolChoice | None = None
    """`ToolChoice.Auto` lets the model decide whether to call a tool. `ToolChoice.NoTools` turns tool calls off."""
    dry_multiplier: float | None = None
    """Turns on DRY ("don't repeat yourself") sampling, which penalizes continuing a sequence that already appeared. Higher is stronger."""
    dry_base: float | None = None
    """How fast the DRY penalty grows with the length of the repeat. Defaults to 1.75."""
    dry_allowed_length: int | None = None
    """Repeats up to this many tokens long aren't penalized. Defaults to 2."""
    dry_sequence_breakers: list[str] | None = None
    """Strings that end a repeat for DRY. Defaults to newline, `:`, `"`, and `*`."""
    web_search_options: WebSearchOptions | None = None
    """Turns on web search for this request."""
    enable_thinking: bool | None = None
    """Turn the model's thinking on or off. See the reasoning controls above."""
    truncate_sequence: bool = False
    """If the prompt is too long for the model, drop tokens from its start instead of failing."""
    reasoning_effort: (
        Literal["off", "none", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    """How much a reasoning model thinks. See the reasoning controls above."""
    max_tool_rounds: int | None = None
    """Most tool-call rounds the server runs by itself. Setting it turns on the agent loop. Defaults to 256."""
    tool_dispatch_url: str | None = None
    """URL that receives, as a POST, any tool call that no built-in tool or `tool_callbacks` entry handles."""
    enable_code_execution: bool = False
    """Let the model run Python. The `Runner` needs a `code_execution_config`."""
    enable_shell: bool = False
    """Let the model run shell commands. The `Runner` needs a `shell_config`. Non-empty `shell_skills` also turns it on."""
    shell_skills: list[ShellSkillMount] | None = None
    """Skill directories to mount for the shell tool."""
    agent_permission: AgentPermission | None = None
    """Whether agent actions run (`Auto`), wait for approval (`Ask`), or are refused (`Deny`)."""
    agent_approval_callback: (
        Callable[[AgentToolApproval], bool | AgentToolApprovalDecision] | None
    ) = None
    """Called when `agent_permission` is `Ask`. Return `True`, `False`, or an `AgentToolApprovalDecision`. Raising or returning anything else denies the action."""
    code_execution_permission: CodeExecutionPermission | None = None
    """Older, code-only form of `agent_permission`. Used when `agent_permission` is unset."""
    session_id: str | None = None
    """Agent session to continue. If unset, the server matches an earlier session by its messages or starts a new one."""
    files: list[RequestedFile] | None = None
    """Output files the model must produce."""
    input_files: list[InputFile] | None = None
    """Files to give the model."""
    ignore_eos: bool = False
    """Keep generating past the end-of-sequence token, until `max_tokens` or a stop string."""
    adapter: str | LoraAdapterGeneration | None = field(default=None, kw_only=True)
    """LoRA adapter to use: an alias, or a `LoraAdapterGeneration` for one exact version."""
    modalities: list[Literal["text", "audio"]] | None = field(default=None, kw_only=True)
    """Output types for a speech model: `["text"]`, `["audio"]`, or `["text", "audio"]`."""
    audio: dict[str, str] | None = field(default=None, kw_only=True)
    """Speech output settings: `format`, and optionally `voice`. Requires `"audio"` in `modalities`."""

@dataclass
class CompletionRequest:
    """
    A raw text completion request, with no chat template. Send it with `Runner.send_completion_request`.

    Fields shared with `ChatCompletionRequest` work the same way.
    """

    prompt: str
    """Text to continue."""
    model: str
    """Not used to choose a model. Pass `model_id` to the `Runner` method instead."""
    best_of: int = 1
    """How many of the `n_choices` completions to return, keeping those with the highest log probability."""
    echo_prompt: bool = False
    """Put the prompt at the start of each returned completion."""
    presence_penalty: float | None = None
    """Flat penalty on any token already generated. 0 turns it off."""
    frequency_penalty: float | None = None
    """Penalty that grows each time a token is generated again. 0 turns it off."""
    repetition_penalty: float | None = None
    """Penalty on tokens already in the prompt or output. 1.0 turns it off."""
    logit_bias: dict[int, float] | None = None
    """Map of token ID to a bias added to that token's logit before sampling."""
    max_tokens: int | None = None
    """Maximum number of tokens to generate."""
    n_choices: int = 1
    """Number of completions to generate."""
    stop_seqs: list[str] | None = None
    """Stop generating when any of these strings appears."""
    temperature: float | None = None
    """Sampling temperature. 0 picks the most likely token every time."""
    top_p: float | None = None
    """Sample only from the most likely tokens whose probabilities add up to `top_p`."""
    suffix: str | None = None
    """Text appended to each returned completion."""
    top_k: int | None = None
    """Sample only from the `top_k` most likely tokens."""
    grammar: str | None = None
    """Constraint that the output must match. Set `grammar_type` with it."""
    grammar_type: str | None = None
    """How to read `grammar`: `"regex"`, `"lark"`, `"json_schema"`, or `"llguidance"`."""
    min_p: float | None = None
    """Drop tokens less likely than `min_p` times the most likely token."""
    tool_schemas: list[str] | None = None
    """Tools the model may call, each a JSON string in the OpenAI format."""
    tool_choice: ToolChoice | None = None
    """`ToolChoice.Auto` lets the model decide whether to call a tool. `ToolChoice.NoTools` turns tool calls off."""
    dry_multiplier: float | None = None
    """Turns on DRY ("don't repeat yourself") sampling. Higher is stronger."""
    dry_base: float | None = None
    """How fast the DRY penalty grows with the length of the repeat. Defaults to 1.75."""
    dry_allowed_length: int | None = None
    """Repeats up to this many tokens long aren't penalized. Defaults to 2."""
    dry_sequence_breakers: list[str] | None = None
    """Strings that end a repeat for DRY. Defaults to newline, `:`, `"`, and `*`."""
    truncate_sequence: bool = False
    """If the prompt is too long for the model, drop tokens from its start instead of failing."""
    ignore_eos: bool = False
    """Keep generating past the end-of-sequence token, until `max_tokens` or a stop string."""
    adapter: str | LoraAdapterGeneration | None = field(default=None, kw_only=True)
    """LoRA adapter to use: an alias, or a `LoraAdapterGeneration` for one exact version."""

class DecisionQuestion:
    """A typed decision question; spoken instructions accept audio_url/input_audio content parts."""
    def __init__(
        self,
        kind: Literal["noul", "choice", "score", "text", "number", "date", "box", "boxes"],
        instructions: str | dict[str, Any],
        *,
        criteria: dict[str, Any] | list[str] | None = None,
        multiple: bool = False,
        range: tuple[float, float] | None = None,
        step: float | None = None,
        unit: str | None = None,
        max: int | None = None,
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class DecisionRequest:
    """State is any JSON value. Media accepts image_url, audio_url, input_audio, and video_url parts."""
    def __init__(
        self,
        state: Any,
        questions: dict[str, DecisionQuestion | dict[str, Any]],
        *,
        images: list[str] | None = None,
        media: list[dict[str, Any]] | None = None,
        set_size: int | None = None,
    ) -> None: ...

class DecisionAnswer:
    """Fields depend on type; absent optional fields return None. to_dict preserves the wire object."""
    type: Literal["noul", "choice", "score", "text", "number", "date", "box", "boxes"]
    noul: float
    choice: str | None
    choices: list[str] | None
    confidence: float | None
    probabilities: dict[str, float]
    score: float
    legend: dict[str, str]
    text: str | None
    span: list[int] | None
    found: float | None
    path: str | None
    number: float | None
    unit: str | None
    interval: list[float] | None
    date: str | None
    box: dict[str, float] | None
    boxes: list[dict[str, Any]]
    count: int
    at_least: bool
    presence: list[float]
    def to_dict(self) -> dict[str, Any]: ...

class DecisionUsage:
    input_tokens: int
    output_tokens: int

class DecisionResponse:
    answers: dict[str, DecisionAnswer]
    usage: DecisionUsage
    def to_dict(self) -> dict[str, Any]: ...

@dataclass
class EmbeddingRequest:
    """
    Text or token IDs to embed. Send it with `Runner.send_embedding_request`, which returns one vector per input.
    """

    input: str | list[str] | list[int] | list[list[int]]
    """A string, a list of strings, a list of token IDs, or a list of token-ID lists."""
    truncate_sequence: bool = False
    """If an input is too long for the model, drop tokens from its end instead of failing."""

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
    Flint = "flint"
    SmolLm3 = "smollm3"
    GraniteMoeHybrid = "granitemoehybrid"
    GptOss = "gptoss"
    HunYuanDenseV1 = "hunyuanv1dense"
    HunYuanMoEV1 = "hunyuanv1moe"
    Qwen3Next = "qwen3next"
    Lfm2 = "lfm2"
    Lfm3 = "lfm3"
    Lfm2Moe = "lfm2_moe"

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
    Lfm2Vl = "lfm2vl"
    Lfm3Vl = "lfm3vl"
    Lfm3Audio = "lfm3audio"
    Lfm3Omni = "lfm3omni"
    Lfm3D1 = "lfm3d1"
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
    Qwen4Exp = "Qwen4Exp"
    Voxtral = "Voxtral"
    Gemma4 = "Gemma4"
    MuseGlimmer = "MuseGlimmer"
    DiffusionGemma = "DiffusionGemma"

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
    """Quantize every layer that in-situ quantization can quantize."""
    MoQE = "moqe"
    """Quantize only the MoE expert weights and leave the other layers as they are."""

@dataclass
class ModelDType(Enum):
    Auto = "auto"
    """Pick a type the device supports."""
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

class SandboxPolicy:
    """
    Limits on the processes that the Python and shell tools start, on Linux and macOS.

    Pass one as `CodeExecutionConfig(sandbox_policy=...)` or `ShellConfig(sandbox_policy=...)`. Without
    one, those processes run with no sandbox. The sandbox controls:

    - which files they can read and write
    - which network they can reach
    - which environment variables they see
    - how much memory, CPU time, and how many processes they can use

    macOS applies the file and network rules but not the resource limits. The
    [sandbox reference](/reference/sandbox/) explains how each rule is enforced.

    The defaults match the CLI's `restricted` profile.
    """

    def __init__(
        self,
        max_memory_mb: int = 2048,
        max_cpu_secs: int = 600,
        max_procs: int = 64,
        max_open_fds: int = 1024,
        max_file_sz_mb: int = 256,
        network: NetworkMode = NetworkMode.Loopback,
        extra_fs_read: list[str] = [],
        extra_fs_write: list[str] = [],
        extra_env: list[str] = [],
        strict: bool = False,
    ) -> None:
        """
        Args:
            max_memory_mb: Memory (address space) limit, in MB.
            max_cpu_secs: CPU time limit, in seconds. On Linux, a limit shorter than the tool's timeout is raised to match it.
            max_procs: Limit on processes and threads.
            max_open_fds: Limit on open file descriptors.
            max_file_sz_mb: Largest file a process can write, in MB.
            network: Network access. See `NetworkMode`.
            extra_fs_read: More paths the process can read.
            extra_fs_write: More paths the process can read and write.
            extra_env: More environment variable names to pass through. Their values come from the current environment.
            strict: Fail at startup if a requested file or network restriction isn't available, instead of skipping it.
        """

class CodeExecutionConfig:
    """
    Turns on the built-in Python tool, `execute_python`.

    Pass it as `Runner(code_execution_config=...)`, then set
    `ChatCompletionRequest.enable_code_execution=True` on each request that may run code.
    """

    def __init__(
        self,
        python_path: str | None = None,
        timeout_secs: int | None = None,
        working_directory: str | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        permission: CodeExecutionPermission | None = None,
        approval_callback: Callable[[dict[str, object]], bool] | None = None,
    ) -> None:
        """
        Args:
            python_path: Interpreter to run. Defaults to `python` on Windows and `python3` elsewhere.
            timeout_secs: Time limit per call, in seconds. Defaults to 60.
            working_directory: One directory that every session runs in. Defaults to a new temporary directory per session.
            sandbox_policy: Sandbox for the interpreter. `None` runs it without one.
            permission: Older form of `ChatCompletionRequest.agent_permission`, which you should prefer. Defaults to `Auto`.
            approval_callback: Older, code-only approval callback. Prefer `ChatCompletionRequest.agent_approval_callback`.
                It receives a dict with `approval_id`, `session_id`, `code`, `outputs`, and `working_directory`,
                and returns `True` to run the code. An exception counts as a denial.
        """

class ShellConfig:
    """
    Turns on the built-in shell tool.

    Pass it as `Runner(shell_config=...)`, then set `ChatCompletionRequest.enable_shell=True` or pass
    `shell_skills` on each request that may run commands.
    """

    def __init__(
        self,
        shell_path: str | None = None,
        timeout_secs: int | None = None,
        working_directory: str | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        permission: AgentPermission | None = None,
    ) -> None:
        """
        Args:
            shell_path: Shell to run. Defaults to `cmd` on Windows and `/bin/sh` elsewhere.
            timeout_secs: Time limit per call, in seconds. Defaults to 600.
            working_directory: Parent directory for sessions. Each session runs in its own `<working_directory>/<session_id>`.
                Defaults to a new temporary directory per session.
            sandbox_policy: Sandbox for the shell. `None` runs it without one.
            permission: Default permission for shell commands. Defaults to `Auto`. For per-request control, use `ChatCompletionRequest.agent_permission`.
        """

class ShellSkillMount:
    """
    A local skill directory to mount for the shell tool, in the OpenAI Skill layout (a folder with a `SKILL.md`).

    Pass instances in `ChatCompletionRequest.shell_skills`. With the HTTP server, upload skills through
    `/v1/skills` instead. Each skill appears at `skills/<name>` in the shell's working directory.
    """

    def __init__(
        self,
        name: str,
        description: str,
        source_path: str,
    ) -> None:
        """
        Args:
            name: Skill name, shown to the model.
            description: What the skill does, shown to the model.
            source_path: Local directory that holds the skill.
        """

@dataclass
class LoraAdapter:
    """A LoRA adapter to load at startup, for `Which.Lora` or `Which.GGUF`."""

    alias: str
    """Name that requests use to select the adapter. Case-sensitive, up to 256 bytes."""
    source: str
    """Hugging Face repo or local directory holding the adapter."""
    revision: str | None = None
    """Branch, tag, or commit to download for a Hub `source`. Defaults to `main`."""

class Which(Enum):
    """
    Chooses the kind of model to load and where its weights come from. Pass one variant to `Runner(which=...)`.

    ```python
    runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B"))
    ```
    """
    @dataclass
    class Plain:
        """A text model from safetensors weights."""

        model_id: str
        """Hugging Face model ID or local directory."""
        arch: Architecture | None = None
        """Architecture to load it as. Detected from `config.json` if unset."""
        tokenizer_json: str | None = None
        """Path to a `tokenizer.json` to use instead of the model's own."""
        topology: str | None = None
        """Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer."""
        organization: IsqOrganization | None = None
        """Which layers in-situ quantization touches. See `IsqOrganization`."""
        write_uqff: str | None = None
        """Save the quantized weights to this `.uqff` path while loading. Use with `Runner(in_situ_quant=...)`."""
        from_uqff: str | list[str] | None = None
        """Pre-quantized [UQFF](/guides/quantization/uqff/) file, or list of files, to load from the model's repo."""
        dtype: ModelDType = ModelDType.Auto
        """Data type to load the model in."""
        imatrix: str | None = None
        """Importance matrix (`.imatrix` or `.cimatrix` file) that guides in-situ quantization."""
        calibration_file: str | None = None
        """Text file to run through the model at load time to build an importance matrix."""
        hf_cache_path: str | None = None
        """Hugging Face cache directory to use instead of the default."""
        matformer_config_path: str | None = None
        """[MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n."""
        matformer_slice_name: str | None = None
        """Slice to load from that table. Without it, the full model loads."""

    @dataclass
    class Embedding:
        """An embedding model from safetensors weights. Fields match `Which.Plain`."""

        model_id: str
        arch: EmbeddingArchitecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        hf_cache_path: str | None = None
        imatrix: str | None = None
        calibration_file: str | None = None

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
        hf_cache_path: str | None = None

    @dataclass
    class Lora:
        """
        A text model that can switch LoRA adapters per request. See [LoRA adapters](/guides/customize/lora-adapters/).

        Other fields match `Which.Plain`.
        """

        model_id: str
        adapters: list[LoraAdapter] | None = None
        """Adapters to load at startup. Leave unset to start with none and load them later."""
        arch: Architecture | None = None
        tokenizer_json: str | None = None
        topology: str | None = None
        write_uqff: str | None = None
        from_uqff: str | list[str] | None = None
        dtype: ModelDType = ModelDType.Auto
        hf_cache_path: str | None = None
        max_adapters: int = 16
        """Limit on loaded aliases, and separately on adapter generations in memory, counting old ones still finishing requests."""
        max_rank: int = 256
        """Highest adapter rank accepted."""
        max_bytes: int = 8589934592
        """Memory limit for all loaded adapters, in bytes (8 GiB by default)."""

    @dataclass
    class GGUF:
        """
        A model from a GGUF file. See [run GGUF models](/guides/models/run-gguf/).

        - Dynamic LoRA turns on when you pass `adapters`, even `[]`, or change `max_adapters`, `max_rank`, or `max_bytes`.
        - With `mmproj_filename`, adapters apply to the language model only.
        - To re-quantize compatible GGUF weights while loading, pass `in_situ_quant` to `Runner`.
        """

        quantized_model_id: str
        """Hugging Face repo or local directory holding the GGUF file."""
        quantized_filename: str | list[str]
        """GGUF file name, or a list of names for a split model."""
        tok_model_id: str | None = None
        """The original model's repo or local directory, for config, tokenizer, and processor files the GGUF doesn't include."""
        topology: str | None = None
        """Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer."""
        dtype: ModelDType = ModelDType.Auto
        """Data type to load the model in."""
        tokenizer_json: str | None = field(default=None, kw_only=True)
        """Path to a `tokenizer.json` to use instead of the model's own."""
        mmproj_filename: str | list[str] | None = field(default=None, kw_only=True)
        """Projector GGUF file, or files, holding a multimodal model's vision or audio encoder."""
        organization: IsqOrganization | None = field(default=None, kw_only=True)
        """Which layers to quantize when `in_situ_quant` re-quantizes the weights."""
        write_uqff: str | None = field(default=None, kw_only=True)
        """Save the re-quantized weights to this `.uqff` path while loading."""
        imatrix: str | None = field(default=None, kw_only=True)
        """Importance matrix (`.imatrix` or `.cimatrix` file) that guides re-quantization."""
        calibration_file: str | None = field(default=None, kw_only=True)
        """Text file to run through the model at load time to build an importance matrix."""
        max_edge: int | None = field(default=None, kw_only=True)
        """Shrink images so their longest edge is at most this many pixels, keeping the aspect ratio."""
        adapters: list[LoraAdapter] | None = field(default=None, kw_only=True)
        """LoRA adapters to load at startup. Pass `[]` to start with none and load them later."""
        max_adapters: int = field(default=16, kw_only=True)
        """Limit on loaded aliases, and separately on adapter generations in memory."""
        max_rank: int = field(default=256, kw_only=True)
        """Highest adapter rank accepted."""
        max_bytes: int = field(default=8589934592, kw_only=True)
        """Memory limit for all loaded adapters, in bytes (8 GiB by default)."""
        hf_cache_path: str | None = field(default=None, kw_only=True)
        """Hugging Face cache directory to use instead of the default."""
        matformer_config_path: str | None = field(default=None, kw_only=True)
        """[MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n."""
        matformer_slice_name: str | None = field(default=None, kw_only=True)
        """Slice to load from that table. Without it, the full model loads."""
        encoder_cache_memory_bytes: int | None = field(default=None, kw_only=True)
        """Memory limit, in bytes, for cached image, audio, and video encoder outputs. Unset keeps the model's default entry-count limit."""

    @dataclass
    class XLoraGGUF:
        """Select X-LoRA for a Phi3 GGUF configuration."""

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
        """Select legacy static LoRA for a Phi3 GGUF configuration.

        For dynamic adapters on a supported GGUF, pass `adapters` to `Which.GGUF`.
        """

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
        tok_model_id: str
        tokenizer_json: str | None = None
        gqa: int = 1
        topology: str | None = None
        dtype: ModelDType = ModelDType.Auto

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

    @dataclass
    class MultimodalPlain:
        """A model that takes images, audio, or video, from safetensors weights."""

        model_id: str
        """Hugging Face model ID or local directory."""
        arch: MultimodalArchitecture | None = None
        """Architecture to load it as. Detected from `config.json` if unset."""
        tokenizer_json: str | None = None
        """Path to a `tokenizer.json` to use instead of the model's own."""
        topology: str | None = None
        """Path to a [topology](/guides/perf/topology/) file that sets the device and quantization for each layer."""
        write_uqff: str | None = None
        """Save the quantized weights to this `.uqff` path while loading. Use with `Runner(in_situ_quant=...)`."""
        from_uqff: str | list[str] | None = None
        """Pre-quantized [UQFF](/guides/quantization/uqff/) file, or list of files, to load from the model's repo."""
        dtype: ModelDType = ModelDType.Auto
        """Data type to load the model in."""
        max_edge: int | None = None
        """Shrink images so their longest edge is at most this many pixels, keeping the aspect ratio."""
        calibration_file: str | None = None
        """Text file to run through the model at load time to build an importance matrix."""
        imatrix: str | None = None
        """Importance matrix (`.imatrix` or `.cimatrix` file) that guides in-situ quantization."""
        hf_cache_path: str | None = None
        """Hugging Face cache directory to use instead of the default."""
        matformer_config_path: str | None = None
        """[MatFormer](/guides/models/model-family-notes/#matformer) slice table (CSV), for Gemma 3n."""
        matformer_slice_name: str | None = None
        """Slice to load from that table. Without it, the full model loads."""
        organization: IsqOrganization | None = None
        """Which layers in-situ quantization touches. See `IsqOrganization`."""
        encoder_cache_memory_bytes: int | None = None
        """Memory limit, in bytes, for cached image, audio, and video encoder outputs. Unset keeps the model's default entry-count limit."""

    @dataclass
    class DiffusionPlain:
        """An image generation model."""

        model_id: str
        arch: DiffusionArchitecture
        dtype: ModelDType = ModelDType.Auto

    @dataclass
    class Speech:
        """A dedicated speech generation model, such as Dia."""

        model_id: str
        arch: SpeechLoaderType
        dac_model_id: str | None = None
        dtype: ModelDType = ModelDType.Auto

class PagedCacheType(Enum):
    Auto: int = 0
    """Store the KV cache in the model's activation type."""
    F8E4M3: int = 1
    """Store the KV cache in FP8 (E4M3), half the memory of BF16. Needs F16, BF16, or F32 activations."""

@dataclass
class LoraAdapterInfo:
    """
    A loaded adapter alias and the generation it points to now. A generation is one exact set of
    adapter files, identified by a hash of its config and weights. See `LoraAdapterGeneration`.
    """

    alias: str
    """Name that requests use to select the adapter."""
    source: str
    """Repo or directory the adapter was loaded from."""
    revision: str | None
    """Hub revision requested at load, if any."""
    generation: str
    """64-character ID of the generation the alias points to now."""
    rank: int
    """Highest LoRA rank in the adapter's config."""
    bytes: int
    """Memory used by this generation's adapter tensors, in bytes."""

    def exact(self) -> LoraAdapterGeneration:
        """Return a selector for this exact generation, to pass as a request's `adapter`."""

@dataclass
class LoraResidentGenerationInfo:
    """
    One adapter generation held in memory. When an alias is replaced or unloaded, its old generation
    is retired: no alias points to it, but it stays in memory until the requests using it finish.
    """

    generation: str
    """64-character generation ID."""
    aliases: list[str]
    """Aliases that point to this generation now."""
    rank: int
    """Highest LoRA rank in the adapter's config."""
    bytes: int
    """Memory used by this generation's adapter tensors, in bytes."""
    retired: bool
    """True if no alias points to it and it stays only for requests still using it."""
    active_leases: int
    """How many in-flight requests are using it."""

@dataclass
class LoraRuntimeStatus:
    """Adapter memory use and limits for one model."""

    adapters: list[LoraAdapterInfo]
    """Loaded aliases."""
    generations: list[LoraResidentGenerationInfo]
    """Every generation in memory, including retired ones."""
    resident_generations: int
    """Number of generations in memory."""
    retired_generations: int
    """How many of those are retired and kept only for running requests."""
    resident_bytes: int
    """Memory used by all adapter tensors, in bytes."""
    max_adapters: int
    """`max_adapters` limit set on `Which.Lora` or `Which.GGUF`."""
    max_rank: int
    """`max_rank` limit set on `Which.Lora` or `Which.GGUF`."""
    max_bytes: int
    """`max_bytes` limit set on `Which.Lora` or `Which.GGUF`."""

class Runner:
    def __init__(
        self,
        which: Which,
        max_seqs: int = 32,
        no_kv_cache: bool = False,
        prefix_cache_n: int | None = None,
        token_source: str = "cache",
        mtp_model: str | None = None,
        mtp_n_predict: int | None = None,
        chat_template: str | None = None,
        jinja_explicit: str | None = None,
        num_device_layers: list[str] | None = None,
        in_situ_quant: str | None = None,
        anymoe_config: AnyMoeConfig | None = None,
        memory_goal: str | float | None = None,
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
        shell_config: ShellConfig | None = None,
    ) -> None:
        """
        Load a model.

        - `which`: the model to load. See [Which](/reference/python/which/).
        - `max_seqs`: how many requests can run at once.
        - `no_kv_cache`: turn off the KV cache, and paged attention with it.
        - `prefix_cache_n`: how many finished requests' caches to keep, so a later request that starts
          with the same prompt can skip recomputing it.
            - `None` (default): keep up to 16. Models with recurrent layers may keep fewer to fit memory.
            - A number: keep exactly that many.
            - `0`: turn prefix caching off.
        - `token_source`: where to read the Hugging Face token. One of `"cache"`, `"none"`,
          `"literal:<token>"`, `"env:<VAR>"`, or `"path:<file>"`.
        - `mtp_model`: a separate drafter model, by ID or path, for
          [speculative decoding](/guides/perf/speculative-decoding/).
        - `mtp_n_predict`: how many tokens the drafter proposes per step. If unset, mistral.rs picks the
          depth as it runs. See [draft depth](/guides/perf/speculative-decoding/#draft-depth).
        - `chat_template`: fallback chat template, used only when the model's files don't include one. A JSON
          file with a `chat_template` key, plus optional `bos_token`, `eos_token`, and `unk_token`.
        - `jinja_explicit`: a `.jinja` chat template file. It overrides every other template source.
        - `num_device_layers`: place layers by hand instead of mapping them automatically. Each entry is
          `"ORD:NUM"`, putting `NUM` layers on device `ORD`. A single number puts that many on device 0.
        - `in_situ_quant`: quantize the weights while loading. Takes a format such as `"q4k"`, or a bit width
          such as `"4"` that picks a format for the device. See
          [quantize a model](/guides/quantization/quantize-a-model/).
        - `anymoe_config`: load the model as an AnyMoE model. See `AnyMoeConfig`.
        - `memory_goal`: how much memory to plan for. A fraction or a size requires paged attention.
          See [paged attention](/guides/perf/paged-attention/#memory-budget).
            - `None` (default): size the KV cache from the memory left after loading.
            - A fraction such as `0.9`: keep weights, working memory, and cache under that share of GPU memory.
            - A size such as `"8GiB"` or `"65536tokens"`: give the KV cache exactly that much.
        - `pa_blk_size`: tokens per paged attention block: 8, 16, or 32. Defaults to 32.
        - `pa_cache_type`: KV cache storage type. See `PagedCacheType`.
        - `no_paged_attn`: turn paged attention off. It's on by default on CUDA and Metal.
        - `paged_attn`: require paged attention, failing if the device doesn't support it. Can't be combined
          with `no_paged_attn`.
        - `seed`: random seed, for reproducible sampling.
        - `enable_search`: load a reranker for web search results. The default is EmbeddingGemma.
        - `search_embedding_model`: which reranker to load. The only option is `"embedding_gemma"`.
        - `search_callback`: a function to run web searches yourself. It takes the query string and returns
          a list of dicts with `"title"`, `"description"`, `"url"`, and `"content"`.
        - `tool_callbacks`: map of tool name to a Python function that runs that tool. The function gets the
          tool name and a dict of arguments, and returns the result as a string.
        - `mcp_client_config`: connect to MCP servers and give the model their tools. See [MCP](/reference/python/mcp/).
        - `code_execution_config`: turn on the built-in Python tool. See `CodeExecutionConfig`.
        - `shell_config`: turn on the built-in shell tool. See `ShellConfig`.
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

    def send_decision_request(
        self, request: DecisionRequest, model_id: str | None = None
    ) -> DecisionResponse:
        """Evaluate named decision questions with the selected model, without text generation."""

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
        Generate speech with a dedicated speech model such as Dia or a chat speech
        model such as LFM3-Audio or LFM3-Omni. Returns PCM samples, sample rate, and
        channel count. LFM3 uses audio-only synthesis with its default voice.

        Use send_chat_completion_request with modalities and audio for custom chat
        prompts, mixed text/speech output, encoded formats, or PCM16 streaming.
        HTTP instructions, speed, and audio.id history are not options on this helper.

        Args:
            prompt: The audio generation prompt.
            model_id: Optional model ID to send the request to. If None, uses the default model.
        """

    def send_re_isq(self, dtype: str, model_id: str | None = None) -> None:
        """
        Re-ISQ a model that was loaded with `in_situ_quant`.

        Args:
            dtype: The ISQ dtype (e.g., "Q4K", "Q8_0").
            model_id: Optional model ID to re-ISQ. If None, uses the default model.
        """

    def begin_calibration(self, model_id: str | None = None) -> CalibrationStatus:
        """
        Begin online calibration: collect activation statistics from live traffic on every
        ISQ-tracked layer. The model must have been loaded with ISQ.

        Args:
            model_id: Optional model ID. If None, uses the default model.
        """

    def calibration_status(self, model_id: str | None = None) -> CalibrationStatus:
        """
        Report per-layer calibration collection progress.

        Args:
            model_id: Optional model ID. If None, uses the default model.
        """

    def apply_calibration(
        self, save_cimatrix: str | None = None, model_id: str | None = None
    ) -> CalibrationStatus:
        """
        Requantize from the source weights with the collected statistics and hot-swap the
        layers into the live model. Returns the pre-apply status.

        Args:
            save_cimatrix: Optional `.cimatrix` path to save the collected importance matrix.
            model_id: Optional model ID. If None, uses the default model.
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
            enable_thinking: Compatibility argument; raw text tokenization does not render a chat template.
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

    def load_lora_adapter(
        self,
        alias: str,
        adapter_dir: str | PathLike[str],
        model_id: str | None = None,
        load_inplace: bool = False,
        expected_generation: str | None = None,
    ) -> LoraAdapterInfo:
        """
        Load a LoRA adapter from a local directory under the name `alias`.

        By default this fails if `alias` is already loaded. With `load_inplace=True`, it replaces the
        alias instead. Requests already running keep the adapter version they started with.

        To avoid overwriting someone else's change, also pass `expected_generation`, the generation you
        last saw for the alias. The replace then happens only if the alias still points at it. If it was
        changed first, the call fails with code `lora_generation_mismatch` and nothing changes.

        Args:
            alias: Name that requests use to select the adapter.
            adapter_dir: Directory with `adapter_config.json` and `adapter_model.safetensors`.
            model_id: Model to load it into. If None, uses the default model.
            load_inplace: Replace `alias` if it's already loaded.
            expected_generation: Replace only if the alias still points at this generation. Requires `load_inplace=True`.

        Returns:
            The alias and its new generation.

        Raises:
            LoraAdapterError: The load failed. `code` says why.
        """

    def unload_lora_adapter(
        self,
        alias: str,
        model_id: str | None = None,
        expected_generation: str | None = None,
    ) -> LoraAdapterInfo:
        """
        Remove `alias`. Requests already running keep using its adapter until they finish.

        Args:
            alias: Alias to remove.
            model_id: Model to remove it from. If None, uses the default model.
            expected_generation: Remove only if the alias still points at this generation. If it was
                changed first, the call fails with code `lora_generation_mismatch`.

        Returns:
            The removed alias.

        Raises:
            LoraAdapterError: The unload failed. `code` says why.
        """

    def list_lora_adapters(
        self,
        model_id: str | None = None,
    ) -> list[LoraAdapterInfo]:
        """
        List the loaded adapter aliases for a model.

        Raises:
            LoraAdapterError: The call failed, for example because the model has no LoRA runtime. `code` says why.
        """

    def lora_adapter_status(
        self,
        model_id: str | None = None,
    ) -> LoraRuntimeStatus:
        """
        Report loaded aliases, adapter generations in memory, and the adapter limits for a model.

        Raises:
            LoraAdapterError: The call failed, for example because the model has no LoRA runtime. `code` says why.
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

    def delete_session(self, session_id: str, model_id: str | None = None) -> bool:
        """
        Delete an agentic session. Returns whether the session existed.
        """

    def list_session_ids(self, model_id: str | None = None) -> list[str]:
        """
        List all stored agentic session IDs.
        """

    def find_file(self, file_id: str) -> File | None:
        """
        Look up a file by ID and return it with its full body, even if the response left the body out.
        Returns None if no file has that ID. Files are kept for 30 minutes.
        """

class AnyMoeExpertType(Enum):
    """
    What each AnyMoE expert in `AnyMoeConfig.model_ids` is: a full fine-tune or a LoRA adapter.
    """
    @dataclass
    class FineTuned:
        """Each expert is a full fine-tune of the base model. Its MLP weights become the expert."""

    @dataclass
    class LoraAdapter:
        """Each expert is a LoRA adapter, applied to the base model's MLP."""

        rank: int
        """LoRA rank of the adapters."""
        alpha: float
        """LoRA alpha of the adapters."""
        target_modules: list[str]
        """MLP modules the adapters change."""

class AnyMoeConfig:
    """
    Builds a mixture of experts from a base model and fine-tunes of it, with a small gate that learns
    which expert should handle each token. Pass it as `Runner(anymoe_config=...)`. See
    [AnyMoE](/guides/customize/lora-adapters/#anymoe).
    """

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
        To find `prefix` and `mlp`, open the model's `model.safetensors.index.json` and find an MLP weight.
        For example, `model.layers.27.mlp.down_proj.weight` means `prefix` is `model.layers` and `mlp` is `mlp`.
        `hidden_size` is in the base model's `config.json`.

        Args:
            hidden_size: The base model's hidden size.
            dataset_json: Training data: a JSON file with a `rows` list of `{"prompt": ..., "expert": N}` objects.
                Expert `0` is the base model, `1` the first entry in `model_ids`, and so on.
            prefix: Weight name prefix before the layer number, such as `model.layers`.
            mlp: Name of the MLP module within each layer, such as `mlp`.
            model_ids: One Hugging Face ID or local path per expert.
            expert_type: Whether the experts are full fine-tunes or LoRA adapters.
            layers: Layers to turn into mixtures. Empty means every layer.
            lr: Learning rate for training the gate.
            epochs: Training epochs.
            batch_size: Training batch size.
            gate_model_id: With `training=True`, a directory to save the trained gate to, as `gate.safetensors`.
                With `training=False`, a Hub ID or local directory to load a trained gate from.
            training: Train the gate at load time. Set it to False to load a saved gate from `gate_model_id`.
            loss_csv_path: With `training=True`, save each step's training loss to this `.csv` file. Ignored when not training.
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
class ChatAudioResponse:
    id: str | None = None
    data: str | None = None
    expires_at: int | None = None
    transcript: str | None = None

@dataclass
class ResponseMessage:
    content: str | None
    role: str
    tool_calls: list[ToolCallResponse] | None
    reasoning_content: str | None = None
    audio: ChatAudioResponse | None = None

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
    adapter_generation: str | None = None
    agentic_tool_calls: list[AgenticToolCallRecord] | None = None
    files: list[File] | None = None
    session_id: str | None = None

@dataclass
class Delta:
    content: str | None
    role: str
    tool_calls: list[ToolCallResponse] | None = None
    reasoning_content: str | None = None
    audio: ChatAudioResponse | None = None

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
    adapter_generation: str | None = None
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
    adapter_generation: str | None = None

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
    """
    An output file the model must produce. Pass a list of them as `ChatCompletionRequest.files`.

    The model is told to write each file with a tool, so the request also needs a tool that can, such
    as code execution or the shell. Otherwise the request fails. Written files appear in
    `ChatCompletionResponse.files`. A file the model never wrote appears there as an error placeholder,
    for which `File.is_error()` returns True.
    """

    name: str
    """File name the model must write."""
    format: str | None
    """File format, such as `csv`. Inferred from the name's extension if unset."""
    description: str | None
    """What the file should contain, shown to the model."""
    def __init__(
        self,
        name: str,
        format: str | None = None,
        description: str | None = None,
    ) -> None: ...

class InputFile:
    """
    A file you give the model. Pass a list of them as `ChatCompletionRequest.input_files`.

    - The model sees each file's name, ID, MIME type, and size.
    - For a text file, it also sees the first 4,096 characters, and up to 32,768 across all files. It can
      read the rest with the built-in `mistralrs_read_file` tool.
    - When the Python or shell tool runs, every file, text or binary, is copied into its working directory.

    See [file inputs](/guides/agents/file-inputs/).
    """

    id: str
    """ID assigned when the object is created."""
    name: str
    """File name shown to the model."""
    mime_type: str | None
    """MIME type. Inferred from the name's extension if unset."""
    bytes: int
    """Size in bytes."""

    def __init__(
        self,
        name: str,
        data: bytes,
        mime_type: str | None = None,
    ) -> None: ...
    @staticmethod
    def from_text(
        name: str,
        text: str,
        mime_type: str = "text/plain",
    ) -> "InputFile": ...
    @staticmethod
    def from_path(
        path: str,
        mime_type: str | None = None,
        name: str | None = None,
    ) -> "InputFile": ...

@dataclass
class FileSource:
    """Where a file was produced."""

    tool: str
    """Name of the tool that wrote it."""
    round: int
    """Tool-call round within the turn, counting from 0."""
    turn: int
    """Turn within the session, counting from 0. Always 0 without a session."""

class File:
    """
    A file that the model wrote during an agent run, such as a CSV from the Python tool. Read them from
    `ChatCompletionResponse.files`.

    A file's body is in `text` for text files and in `data_base64` for binary ones. Bodies over 8 MiB
    are left out of the response: `text` and `data_base64` are `None` and `is_truncated()` returns True.
    Fetch the full file with `Runner.find_file(file.id)`.
    """

    id: str
    """File ID, for `Runner.find_file`."""
    name: str
    """File name."""
    format: str | None
    """File format, such as `csv`, from the name's extension."""
    mime_type: str | None
    """MIME type."""
    bytes: int
    """Size in bytes."""
    source: FileSource
    """Which tool wrote the file, and when."""
    text: str | None
    """Body of a text file. `None` for binary files and truncated bodies."""
    data_base64: str | None
    """Base64 body of a binary file. `None` for text files and truncated bodies."""
    preview: str | None
    """Short preview of a text file, or the full text when there is no preview."""
    def is_text(self) -> bool:
        """True for a text file."""
    def is_binary(self) -> bool:
        """True for a binary file."""
    def is_image(self) -> bool:
        """True if the MIME type starts with `image/`."""
    def is_video(self) -> bool:
        """True if the MIME type starts with `video/`."""
    def is_error(self) -> bool:
        """True for the placeholder of a requested file the model never wrote."""
    def is_truncated(self) -> bool:
        """True if the body was left out of the response because it was too large."""
    def save(self, path: str) -> None:
        """Write the body to `path`. Raises `OSError` for a truncated body or an error placeholder."""

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
