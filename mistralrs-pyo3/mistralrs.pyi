from dataclasses import dataclass
from enum import Enum
from typing import Iterator

@dataclass
class ChatCompletionRequest:
    """
    A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.
    """

    messages: list[Message] | str
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

@dataclass
class _Base:
    tokenizer_json: str | None
    repeat_last_n: int | None

@dataclass
class _Normal(_Base):
    model_id: str

@dataclass
class _Quantized(_Base):
    tok_model_id: str
    quantized_model_id: str
    quantized_filename: str

@dataclass
class _XLoraQuantized(_Quantized):
    xlora_model_id: str
    order: str
    tgt_non_granular_index: int | None

@dataclass
class _XLoraNormal(_Normal):
    xlora_model_id: str
    order: str
    tgt_non_granular_index: int | None

class Which(Enum):
    """
    Which model to select.
    """
    @dataclass
    class Mistral(_Normal): ...
    @dataclass
    class XLoraMistral(_XLoraNormal): ...
    @dataclass
    class Gemma(_Normal): ...
    @dataclass
    class XLoraGemma(_XLoraNormal): ...
    @dataclass
    class Llama(_Normal): ...
    @dataclass
    class XLoraLlama(_XLoraNormal): ...
    @dataclass
    class Mixtral(_Normal): ...
    @dataclass
    class XLoraMixtral(_XLoraNormal): ...
    @dataclass
    class Phi2(_Normal): ...
    @dataclass
    class XLoraPhi2(_XLoraNormal): ...
    @dataclass
    class LoraMistral(_XLoraNormal): ...
    @dataclass
    class LoraMixtral(_XLoraNormal): ...
    @dataclass
    class LoraLlama(_XLoraNormal): ...
    @dataclass
    class GGUF(_Quantized): ...
    @dataclass
    class XLoraGGUF(_XLoraQuantized): ...
    @dataclass
    class LoraGGUF(_XLoraQuantized): ...
    @dataclass
    class GGML(_Quantized): ...
    @dataclass
    class XLoraGGML(_XLoraQuantized): ...
    @dataclass
    class LoraGGML(_XLoraQuantized): ...

class Runner:
    def __init__(
        self,
        which: Which,
        max_seqs: int = 16,
        no_kv_cache=False,
        prefix_cache_n: int = 16,
        token_source="cache",
        chat_template=None,
    ) -> None:
        """
        Load a model.

        - `which` specified which model to load.
        - `max_seqs` specifies how many sequences may be running at any time.
        - `no_kv_cache` disables the KV cache.
        - `prefix_cache_n` sets the number of sequences to hold in the device prefix cache, others will be evicted to CPU.
        - `token_source` specifies where to load the HF token from.
            The token source follows the following format: "literal:<value>", "env:<value>", "path:<value>", "cache" to use a cached token or "none" to use no token.
        - `chat_template` specifies an optional JINJA chat template.
            The JINJA template should have `messages`, `add_generation_prompt`, `bos_token`, `eos_token`, and `unk_token` as inputs.
            It is used if the automatic deserialization fails. If this ends with `.json` (ie., it is a file) then that template is loaded.
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

@dataclass
class Role(Enum):
    """
    The role for each `Message` of a chat completion request.
    """

    User = 1
    Assistant = 2

@dataclass
class Message:
    """
    A message for a chat completion request.
    """

    role: Role
    content: str

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
