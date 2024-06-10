from dataclasses import dataclass
from enum import Enum
from typing import Iterator

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

@dataclass
class Architecture(Enum):
    Mistral = "mistral"
    Gemma = "gemma"
    Mixtral = "mixtral"
    Llama = "llama"
    Phi2 = "phi2"

@dataclass
class VisionArchitecture(Enum):
    Phi3V = "phi3v"

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
        arch: Architecture
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class XLora:
        arch: Architecture
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class Lora:
        arch: Architecture
        adapters_model_id: str
        order: str
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class GGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        repeat_last_n: int = 64
    @dataclass
    class XLoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        repeat_last_n: int = 64
    @dataclass
    class LoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        repeat_last_n: int = 64
    @dataclass
    class GGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class XLoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class LoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
    @dataclass
    class VisionPlain:
        model_id: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64
        arch: VisionArchitecture

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
        num_device_layers: int | None = None,
        in_situ_quant: str | None = None,
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
        - `num_device_layers` sets the number of layers to load and run on the device.
        - `in_situ_quant` sets the optional in-situ quantization for models that are not quantized (not GGUF or GGML).
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

    def send_re_isq(self, dtype: str) -> CompletionResponse:
        """
        Send a request to re-ISQ the model. If the model was loaded as GGUF or GGML then nothing will happen.
        """

    def activate_adapters(self, adapter_names: list[str]) -> None:
        """
        Send a request to make the specified adapters the active adapters for the model.
        """

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
class ResponseMessage:
    content: str
    role: str

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
