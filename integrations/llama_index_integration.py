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
)
from llama_index.core.types import BaseOutputParser, PydanticProgramMode

from mistralrs import (
    ChatCompletionRequest,
    Runner,
    Which,
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
DEFAULT_MAX_SEQS = 16
DEFAULT_PREFIX_CACHE_N = 16


class MistralRS(CustomLLM):
    r"""MistralRS LLM.

    Examples:
        Install `mistralrs` following instructions:
        https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/README.md#installation

        Then `pip install llama-index-llms-mistral-rs`

        ```python
        from llama_index.llms.mistral_rs import MistralRS
        from mistralrs import Which

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

        llm = MistralRS(
            which = Which.XLora(
                model_id=None,  # Automatically determine from ordering file
                tokenizer_json=None,
                repeat_last_n=64,
                xlora_model_id="lamm-mit/x-lora"
                order="xlora-paper-ordering.json", # Make sure you copy the ordering file from `mistral.rs/orderings`
                tgt_non_granular_index=None,
                arch=Architecture.Mistral,
            ),
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
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
        which: Which,
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

        super().__init__(
            model_path="local",
            model_url="local",
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

        self._runner = Runner(
            which=which,
            token_source=model_kwargs.get("token_source", "cache"),
            max_seqs=model_kwargs.get("max_seqs", DEFAULT_MAX_SEQS),
            prefix_cache_n=model_kwargs.get("prefix_cache_n", DEFAULT_PREFIX_CACHE_N),
            no_kv_cache=model_kwargs.get("no_kv_cache", False),
            chat_template=model_kwargs.get("chat_template", None),
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
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        self.generate_kwargs.update({"stream": True})

        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            logit_bias=None,
            logprobs=False,
            **self.generate_kwargs,
        )

        streamer = self._runner.send_chat_completion_request(request)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in streamer:
                delta = response.choices[0].text
                text += delta
                yield CompletionResponse(delta=delta, text=text)

        return gen()

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            logit_bias=None,
            logprobs=False,
            **self.generate_kwargs,
        )
        completion_response = self._runner.send_chat_completion_request(request)
        return CompletionResponse(text=completion_response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        self.generate_kwargs.update({"stream": True})
        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        request = ChatCompletionRequest(
            messages=prompt,
            model="",
            logit_bias=None,
            logprobs=False,
            **self.generate_kwargs,
        )

        streamer = self._runner.send_chat_completion_request(request)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in streamer:
                delta = response.choices[0].text
                text += delta
                yield CompletionResponse(delta=delta, text=text)

        return gen()
