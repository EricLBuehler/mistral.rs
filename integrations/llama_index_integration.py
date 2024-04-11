from typing import Any, Callable, Dict, Optional, Sequence

import json
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode
from llama_index.core.utils import get_cache_dir
from tqdm import tqdm

from mistralrs import (
    MistralLoader,
    MixtralLoader,
    GemmaLoader,
    LlamaLoader,
    ChatCompletionRequest,
    NormalLoader,
    Runner,
    XLoraLoader,
    QuantizedLoader,
    XLoraQuantizedLoader,
)

DEFAULT_MISTRAL_RS_GGML_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve"
    "/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
)
DEFAULT_MISTRAL_RS_GGUF_MODEL = (
    "https://huggingface.co/TheBloke/Ll ama-2-13B-chat-GGUF/resolve"
    "/main/llama-2-13b-chat.Q4_0.gguf"
)
DEFAULT_TOPK = 32
DEFAULT_TOPP = 0.1
DEFAULT_TOP_LOGPROBS = 10
DEFAULT_REPEAT_LAST_N = 64
DEFAULT_MAX_SEQS = 10


class MistralRS(CustomLLM):
    r"""MistralRS LLM.

    Examples:
        Install mistralrs following instructions:
        https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md#installation

        Then `pip install llama-index-llms-mistral-rs`

        ```python
        from llama_index.llms.mistral_rs import MistralRS

        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        model_url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"

        llm = MistralRS(
            model_url=model_url,
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},  # if compiled to use GPU
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
        )

        response = llm.complete("Hello, how are you?")
        print(str(response))
        ```
    """

    model_url: Optional[str] = Field(
        description="The URL llama-cpp model to download and use."
    )
    model_path: Optional[str] = Field(
        description="The path to the llama-cpp model to use."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    _runner: Runner = PrivateAttr("Mistral.rs model runner.")

    def __init__(
        self,
        arch: str,
        model_id: Optional[str] = None,
        quantized_model_id: Optional[str] = None,
        quantized_filename: Optional[str] = None,
        xlora_order_file: Optional[str] = None,
        xlora_model_id: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        top_k: int = DEFAULT_TOPK,
        top_p: int = DEFAULT_TOPP,
        top_logprobs: Optional[int] = DEFAULT_TOP_LOGPROBS,
        callback_manager: Optional[CallbackManager] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = {},
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:
        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "top_k": top_k,
                "top_p": top_p,
                "top_logprobs": top_logprobs,
            }
        )
        splits = list(map(lambda x: x.lower(), arch.split("-")))
        if splits[0:2] == ["x", "lora"]:
            is_xlora = True
        else:
            is_xlora = False
        if splits[-1] == "gguf":
            is_gguf = True
            is_ggml = False
        elif splits[-1] == "ggml":
            is_ggml = True
            is_gguf = False
        else:
            is_gguf = False
            is_ggml = False

        if len(splits) == 1:
            model = splits[0]
        elif len(splits) == 2 and is_xlora:
            model = splits[1]
        elif len(splits) == 2 and not is_xlora and (is_ggml or is_gguf):
            model = splits[0]
        elif len(splits) == 2 and is_xlora and (is_ggml or is_gguf):
            model = splits[1]

        match model:
            case "mistral":
                model_loader = MistralLoader
            case "mixtral":
                model_loader = MixtralLoader
            case "llama":
                model_loader = LlamaLoader
            case "gemma":
                model_loader = GemmaLoader
            case _:
                raise ValueError(
                    f"Unexpected model {model}, value values are one of mistral, mixtral, llama, gemma"
                )

        match (is_gguf, is_ggml, is_xlora):
            case (False, False, False):
                loader = NormalLoader(
                    model_loader,
                    model_id,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get(
                        "repeat_last_n", DEFAULT_REPEAT_LAST_N
                    ),
                    gqa=model_kwargs.get("gqa", None),
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case (True, False, False):
                loader = QuantizedLoader(
                    model_loader,
                    model_id,
                    is_gguf=True,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get(
                        "repeat_last_n", DEFAULT_REPEAT_LAST_N
                    ),
                    gqa=model_kwargs.get("gqa", None),
                    quantized_model_id=quantized_model_id,
                    quantized_filename=quantized_filename,
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case (False, True, False):
                loader = QuantizedLoader(
                    model_loader,
                    model_id,
                    is_gguf=False,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get(
                        "repeat_last_n", DEFAULT_REPEAT_LAST_N
                    ),
                    gqa=model_kwargs.get("gqa", None),
                    quantized_model_id=quantized_model_id,
                    quantized_filename=quantized_filename,
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case (False, False, True):
                loader = XLoraLoader(
                    model_loader,
                    model_id,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get(
                        "repeat_last_n", DEFAULT_REPEAT_LAST_N
                    ),
                    gqa=model_kwargs.get("gqa", None),
                    order_file=xlora_order_file,
                    xlora_model_id=xlora_model_id,
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case (True, False, True):
                loader = XLoraQuantizedLoader(
                    model_loader,
                    model_id,
                    is_gguf=True,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get("repeat_last_n", None),
                    gqa=model_kwargs.get("gqa", None),
                    order_file=xlora_order_file,
                    quantized_model_id=quantized_model_id,
                    quantized_filename=quantized_filename,
                    xlora_model_id=xlora_model_id,
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case (False, True, True):
                loader = XLoraQuantizedLoader(
                    model_loader,
                    model_id,
                    is_gguf=False,
                    no_kv_cache=model_kwargs.get("no_kv_cache", False),
                    use_flash_attn=True,  # will be disabled by &
                    repeat_last_n=model_kwargs.get("repeat_last_n", None),
                    gqa=model_kwargs.get("gqa", None),
                    order_file=xlora_order_file,
                    quantized_model_id=quantized_model_id,
                    quantized_filename=quantized_filename,
                    xlora_model_id=xlora_model_id,
                    chat_template=model_kwargs.get("chat_template", None),
                    tokenizer_json=model_kwargs.get("tokenizer_json", None),
                )
            case _:
                raise ValueError(
                    f"Invalid model architecture {arch}. Expected <x-lora>-arch-<gguf | ggml>."
                )

        super().__init__(
            model_path=model_id,
            model_url=model_id,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs={},
            verbose=True,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

        self._runner = loader.load(
            token_source=model_kwargs.get("token_source", {"source": "cache"})[
                "source"
            ],  # default source is "cache"
            max_seqs=model_kwargs.get("max_seqs", DEFAULT_MAX_SEQS),
            logfile=None,
            revision=model_kwargs.get("revision", None),
            token_source_value=model_kwargs.get("token_source", {"value": None})[
                "value"
            ],
            dtype=None,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MistralRS_llm"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        messages_raw = []
        for message in messages:
            messages_raw.append({"role": str(message.role), "content": message.content})
        request = ChatCompletionRequest(
            messages=messages_raw,
            model="",
            max_tokens=self.generate_kwargs["max_tokens"],
            logit_bias=None,
            logprobs=False,
            top_logprobs=None,
            top_k=self.generate_kwargs["top_k"],
            top_p=self.generate_kwargs["top_p"],
            presence_penalty=self.generate_kwargs.get("presence_penalty", None),
            repetition_penalty=self.generate_kwargs.get("repetition_penalty", None),
            temperature=self.generate_kwargs.get("temperature", None),
        )
        completion_response = self._runner.send_chat_completion_request(request)
        json_resp = json.loads(completion_response)
        return completion_response_to_chat_response(
            CompletionResponse(
                text=json_resp["choices"][0]["message"]["content"],
                raw=json_resp,
            )
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            max_tokens=self.generate_kwargs["max_tokens"],
            logit_bias=None,
            logprobs=False,
            top_logprobs=None,
            top_k=self.generate_kwargs["top_k"],
            top_p=self.generate_kwargs["top_p"],
            presence_penalty=self.generate_kwargs.get("presence_penalty", None),
            repetition_penalty=self.generate_kwargs.get("repetition_penalty", None),
            temperature=self.generate_kwargs.get("temperature", None),
        )
        completion_response = self._runner.send_chat_completion_request(request)
        json_resp = json.loads(completion_response)
        return CompletionResponse(
            text=json_resp["choices"][0]["message"]["content"],
            raw=json_resp,
        )

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError(".stream_complete is not implemented yet.")
