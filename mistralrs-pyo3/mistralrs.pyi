from dataclasses import dataclass
from enum import Enum

@dataclass
class ChatCompletionRequest:
    """
    A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
    about input data, sampling, and how to return the response.
    """

    messages: list[dict[str, str]] | str
    model: str
    logit_bias: dict[int, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    stop_token_ids: list[int] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    top_k: int | None = None
    grammar: str | None = None
    grammar_type: str | None = None

class Runner:
    """
    The Runner is a class with no constructor. It is only created via one of the loader classes.
    """
    def send_chat_completion_request(self, request: ChatCompletionRequest) -> str:
        """
        Send a chat completion request to the mistral.rs engine, returning the response as a string.
        This can be parsed as JSON.
        """

class ModelKind(Enum):
    """
    The model kind is passed to a loader and specifies the type of model to load.
    """

    Normal = 1
    XLoraNormal = 2
    XLoraGGUF = 3
    XLoraGGML = 4
    QuantizedGGUF = 5
    QuantizedGGML = 6

class DType(Enum):
    """
    The data type for a model.
    """

    U8 = 1
    U32 = 2
    I64 = 3
    BF16 = 4
    F16 = 5
    F32 = 6
    F64 = 7

class LoaderMixin:
    def load(
        self,
        token_source: str = "cache",
        max_seqs: int = 16,
        truncate_sequence: bool = False,
        logfile: str | None = None,
        revision: str | None = None,
        token_source_value: str | None = None,
        dtype: DType | None = None,
    ) -> Runner:
        """
         Load a model.

        - `token_source="cache"`
        Specify token source and token source value as the following pairing:
            - "cache" -> None
            - "literal" -> str
            - "envvar" -> str
            - "path" -> str
            - "none" -> None

        - `max_seqs`: Maximum running sequences at any time.

        - `truncate_sequence`:
        If a sequence is larger than the maximum model length, truncate the number
        of tokens such that the sequence will fit at most the maximum length.
        If `max_tokens` is not specified in the request, space for 10 tokens will be reserved instead.

        - `logfile`: Log all responses and requests to this file.

        - `revision`: HF revision.

        - `token_source_value`: Value of token source value for `token_source`

        - `dtype`: Datatype to load the model into, only applicable for non-quantized models.
        """

class NormalLoader(LoaderMixin):
    """
    A loader to load "normal" models, those without X-LoRA or quantization.
    """
    def __init__(
        self,
        loader_class,
        model_id: str,
        no_kv_cache: bool = False,
        use_flash_attn: bool = False,
        repeat_last_n: int = 64,
        gqa: int | None = None,
        chat_template: str | None = None,
        tokenizer_json: str | None = None,
    ):
        """
        - `loader_class`: Loader class.
        - `model_id`: Base model ID, or tokenizer ID if quantized model type.
        - `no_kv_cache=False`: Disable kv cache.
        - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
        - `repeat_last_n=64`: Repeat last n context window.
        - `gqa=None`: GQA, irrelevant if non quantized model type.
        - `chat_template=None`: Chat template literal or file.
        - `tokenizer_json=None`: Tokenizer json file.
        """

class XLoraLoader(LoaderMixin):
    """
    A loader to load X-LoRA models.
    """
    def __init__(
        self,
        loader_class,
        model_id: str,
        no_kv_cache: bool = False,
        use_flash_attn: bool = False,
        repeat_last_n: int = 64,
        gqa: int | None = None,
        order_file: str | None = None,
        xlora_model_id: str | None = None,
        chat_template: str | None = None,
        tokenizer_json: str | None = None,
    ):
        """
        - `loader_class`: Loader class.
        - `model_id`: Base model ID, or tokenizer ID if quantized model type.
        - `no_kv_cache=False`: Disable kv cache.
        - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
        - `repeat_last_n=64`: Repeat last n context window.
        - `gqa=None`: GQA, irrelevant if non quantized model type.
        - `order_file=None`: Ordering JSON file.
        - `xlora_model_id=None`: X-LoRA model.
        - `chat_template=None`: Chat template literal or file.
        - `tokenizer_json=None`: Tokenizer json file.
        """

class QuantizedLoader(LoaderMixin):
    """
    A loader to load quantized models.
    """
    def __init__(
        self,
        loader_class,
        model_id: str,
        is_gguf: bool,
        no_kv_cache: bool = False,
        use_flash_attn: bool = False,
        repeat_last_n: int = 64,
        gqa: int | None = None,
        quantized_model_id: str | None = None,
        quantized_filename: str | None = None,
        chat_template: str | None = None,
        tokenizer_json: str | None = None,
    ):
        """
        - `loader_class`: Loader class.
        - `model_id`: Base model ID, or tokenizer ID if quantized model type.
        - `is_gguf`: Is the quantized model GGUF.
        - `no_kv_cache=False`: Disable kv cache.
        - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
        - `repeat_last_n=64`: Repeat last n context window.
        - `gqa=None`: GQA, irrelevant if non quantized model type.
        - `quantized_model_id=None`: Ordering JSON file.
        - `quantized_filename=None`: X-LoRA model.
        - `chat_template=None`: Chat template literal or file.
        - `tokenizer_json=None`: Tokenizer json file.
        """

class XLoraQuantizedLoader(LoaderMixin):
    """
    A loader to load X-LoRA models with quantization.
    """
    def __init__(
        self,
        loader_class,
        model_id: str,
        is_gguf: bool,
        no_kv_cache: bool = False,
        use_flash_attn: bool = False,
        repeat_last_n: int = 64,
        gqa: int | None = None,
        order_file: str | None = None,
        quantized_model_id: str | None = None,
        quantized_filename: str | None = None,
        xlora_model_id: str | None = None,
        chat_template: str | None = None,
        tokenizer_json: str | None = None,
    ):
        """
        - `loader_class`: Loader class.
        - `model_id`: Base model ID, or tokenizer ID if quantized model type.
        - `is_gguf`: Is the quantized model GGUF.
        - `no_kv_cache=False`: Disable kv cache.
        - `use_flash_attn=None`: Use flash attn, only used if feature is enabled.
        - `repeat_last_n=64`: Repeat last n context window.
        - `gqa=None`: GQA, irrelevant if non quantized model type.
        - `order_file=None`: Ordering JSON file.
        - `quantized_model_id=None`: Ordering JSON file.
        - `quantized_filename=None`: X-LoRA model.
        - `xlora_model_id=None`: X-LoRA model.
        - `chat_template=None`: Chat template literal or file.
        - `tokenizer_json=None`: Tokenizer json file.
        """
