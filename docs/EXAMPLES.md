# Examples

A comprehensive index of all examples in the repository, organized by SDK and category.

## Python SDK

Examples using the `mistralrs` Python package directly.

### Getting Started

| Example | Description |
|---|---|
| [plain.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/plain.py) | Basic text generation with a plain (unquantized) model |
| [streaming.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/streaming.py) | Streaming chat completions token-by-token with a GGUF model |
| [gguf.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gguf.py) | Load and run a GGUF-quantized model |
| [embedding_gemma.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/embedding_gemma.py) | Generate text embeddings with EmbeddingGemma |
| [qwen3_embedding.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_embedding.py) | Generate text embeddings with Qwen3-Embedding |
| [token_source.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/token_source.py) | Configure a custom HF token source (literal, env, path, cache) |

### Models

#### Text Models

| Example | Description |
|---|---|
| [deepseekr1.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/deepseekr1.py) | Run DeepSeek-R1 reasoning model |
| [deepseekv2.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/deepseekv2.py) | Run DeepSeek-V2-Lite |
| [qwen3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3.py) | Run Qwen3 with thinking mode toggling via `/think` and `/no_think` |
| [qwen3_5.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_5.py) | Run Qwen3.5 dense or MoE variant with vision input |
| [qwen3_next.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_next.py) | Run Qwen3-Coder-Next |
| [smollm3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/smollm3.py) | Run SmolLM3 text model |
| [granite.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/granite.py) | Run IBM Granite 4.0 MoE Hybrid model |
| [gpt_oss.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gpt_oss.py) | Run GPT-OSS MoE model with MXFP4 quantized experts |
| [glm4_moe_lite.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/glm4_moe_lite.py) | Run GLM-4.7-Flash MoE model |

#### Vision Models

| Example | Description |
|---|---|
| [gemma3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma3.py) | Run Gemma 3 multimodal model with image input |
| [gemma3n.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma3n.py) | Run Gemma 3n multimodal model with image input |
| [gemma4.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/gemma4.py) | Run Gemma 4 multimodal model with image input |
| [qwen2vl.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen2vl.py) | Run Qwen2-VL vision-language model |
| [qwen3_vl.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/qwen3_vl.py) | Run Qwen3-VL vision-language model with thinking |
| [llama4.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llama4.py) | Run Llama 4 Scout multimodal model with ISQ |
| [llama_vision.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llama_vision.py) | Run Llama 3.2 Vision model |
| [mistral3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mistral3.py) | Run Mistral Small 3.1 vision model with ISQ |
| [phi3v.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v.py) | Run Phi-3.5 Vision with a remote image URL |
| [phi3v_base64.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_base64.py) | Run Phi-3.5 Vision with a base64-encoded image |
| [phi3v_local_img.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v_local_img.py) | Run Phi-3.5 Vision with a local image file path |
| [phi4mm.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi4mm.py) | Run Phi-4 Multimodal with image input |
| [phi4mm_audio.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi4mm_audio.py) | Run Phi-4 Multimodal with combined audio and image input |
| [idefics2.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/idefics2.py) | Run Idefics 2 vision model |
| [idefics3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/idefics3.py) | Run Idefics 3 vision model |
| [llava_next.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llava_next.py) | Run LLaVA-NeXT vision model |
| [smolvlm.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/smolvlm.py) | Run SmolVLM vision model |
| [minicpmo_2_6.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/minicpmo_2_6.py) | Run MiniCPM-o 2.6 multimodal model |

#### Image Generation and Speech

| Example | Description |
|---|---|
| [flux.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/flux.py) | Generate images with Flux diffusion model |
| [dia.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/dia.py) | Text-to-speech synthesis with Dia and WAV output |

### Tools and Agents

| Example | Description |
|---|---|
| [tool_call.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_call.py) | Client-side tool calling with manual tool execution loop |
| [agentic_tools.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/agentic_tools.py) | Agentic tool callbacks with automatic multi-round execution |
| [custom_tool_call.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_tool_call.py) | Custom tool callback for local filesystem search |
| [web_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/web_search.py) | Web-search-augmented generation with WebSearchOptions |
| [custom_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/custom_search.py) | Custom search callback replacing the default web search |
| [mcp_client.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mcp_client.py) | MCP client connecting to an external tool server |

### Constrained Generation

| Example | Description |
|---|---|
| [json_schema.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/json_schema.py) | JSON schema-constrained generation |
| [pydantic_schema.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/pydantic_schema.py) | Constrained generation from a Pydantic model schema |
| [regex.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/regex.py) | Regex-constrained generation |
| [lark.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lark.py) | Lark grammar-constrained generation (JSON via grammar) |
| [lark_llg.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lark_llg.py) | Lark grammar with inline `%json` schema directive |
| [llguidance.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/llguidance.py) | llguidance grammar with named JSON schema references |

### Quantization

| Example | Description |
|---|---|
| [isq.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/isq.py) | In-situ quantization (ISQ) at load time |
| [imatrix.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/imatrix.py) | ISQ with importance-matrix calibration data |
| [topology.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/topology.py) | Per-layer quantization control via a topology YAML file |
| [mixture_of_quant_experts.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/mixture_of_quant_experts.py) | MoQE: quantize only MoE expert layers differently |

### Adapters (LoRA / X-LoRA)

| Example | Description |
|---|---|
| [lora_zephyr.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lora_zephyr.py) | Load a LoRA adapter on a GGUF model and select adapters per request |
| [lora_activation.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/lora_activation.py) | Programmatically activate/deactivate LoRA adapters |
| [xlora_zephyr.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/xlora_zephyr.py) | Run X-LoRA with GGUF on Zephyr |
| [xlora_gemma.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/xlora_gemma.py) | Run X-LoRA on Gemma |

### AnyMoE

| Example | Description |
|---|---|
| [anymoe.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/anymoe.py) | Train AnyMoE gating layer from fine-tuned expert models |
| [anymoe_inference.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/anymoe_inference.py) | Run AnyMoE inference with a pretrained gating model |
| [anymoe_lora.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/anymoe_lora.py) | AnyMoE with LoRA adapters as experts |

### Advanced

| Example | Description |
|---|---|
| [paged_attention.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/paged_attention.py) | Enable PagedAttention for efficient KV-cache memory |
| [speculative.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/speculative.py) | Speculative decoding with a smaller GGUF draft model |
| [speculative_xlora.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/speculative_xlora.py) | Speculative decoding combined with X-LoRA |
| [text_auto_device_map.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/text_auto_device_map.py) | Automatic multi-GPU device mapping for text models |
| [multimodal_auto_device_map.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/multimodal_auto_device_map.py) | Automatic multi-GPU device mapping for multimodal models |
| [multi_model_example.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/multi_model_example.py) | Load, manage, and dispatch requests across multiple models |
| [test_multi_model.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/test_multi_model.py) | Multi-model management test harness |

---

## HTTP Server

Examples using the OpenAI-compatible HTTP API (start the server with `mistralrs serve`).

### Getting Started

| Example | Description |
|---|---|
| [chat.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/chat.py) | Interactive multi-turn chat with the OpenAI client |
| [completion.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/completion.py) | Text completion (non-chat) endpoint |
| [streaming.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/streaming.py) | Streaming chat completions token-by-token |
| [streaming_completion.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/streaming_completion.py) | Streaming text completion endpoint |
| [embedding.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/embedding.py) | Generate text embeddings via the embeddings endpoint |

### Models

#### Vision Models

| Example | Description |
|---|---|
| [gemma3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gemma3.py) | Gemma 3 multimodal chat with image input |
| [gemma3n.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gemma3n.py) | Gemma 3n multimodal chat with image input |
| [gemma4.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gemma4.py) | Gemma 4 multimodal chat with image input |
| [gemma4_video.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gemma4_video.py) | Gemma 4 with video input |
| [qwen2vl.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen2vl.py) | Qwen2-VL vision-language chat |
| [qwen3_vl.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3_vl.py) | Qwen3-VL vision-language chat |
| [llama4.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llama4.py) | Llama 4 Scout multimodal chat |
| [llama_vision.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llama_vision.py) | Llama 3.2 Vision chat |
| [mistral3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/mistral3.py) | Mistral Small 3.1 vision chat |
| [phi3v.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v.py) | Phi-3.5 Vision with remote image URL |
| [phi3v_base64.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_base64.py) | Phi-3.5 Vision with base64-encoded image |
| [phi3v_local_img.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3v_local_img.py) | Phi-3.5 Vision with local image file |
| [phi4mm.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi4mm.py) | Phi-4 Multimodal with image input |
| [phi4mm_audio.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi4mm_audio.py) | Phi-4 Multimodal with audio and image input |
| [idefics2.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/idefics2.py) | Idefics 2 vision chat |
| [idefics3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/idefics3.py) | Idefics 3 vision chat |
| [llava.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llava.py) | LLaVA 1.5 vision chat |
| [llava_next.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llava_next.py) | LLaVA-NeXT vision chat |
| [minicpmo_2_6.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/minicpmo_2_6.py) | MiniCPM-o 2.6 multimodal chat |
| [smollm3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/smollm3.py) | SmolLM3 chat |

#### Text Models

| Example | Description |
|---|---|
| [qwen3.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3.py) | Qwen3 chat |
| [qwen3_5.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3_5.py) | Qwen3.5 chat |
| [qwen3_next.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/qwen3_next.py) | Qwen3-Coder-Next chat |
| [gpt_oss.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/gpt_oss.py) | GPT-OSS MoE chat |

#### Image Generation and Speech

| Example | Description |
|---|---|
| [flux.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/flux.py) | Generate images via the images endpoint |
| [dia.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/dia.py) | Text-to-speech via the audio speech endpoint |

### OpenAI Responses API

| Example | Description |
|---|---|
| [responses.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/responses.py) | Multi-turn conversation using the Responses API with reasoning |
| [responses_vision.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/responses_vision.py) | Responses API with vision (image) input |
| [responses_audio.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/responses_audio.py) | Responses API with audio input and transcription |

### Tools and Agents

| Example | Description |
|---|---|
| [tool_calling.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_calling.py) | Client-side tool calling with the OpenAI client |
| [streaming_tool_calling.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/streaming_tool_calling.py) | Streaming tool calls with multi-round execution loop |
| [tool_dispatch.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/tool_dispatch.py) | Server-side tool dispatch via an external HTTP endpoint |
| [agentic_tool_rounds.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/agentic_tool_rounds.py) | Server-side agentic loop with max_tool_rounds |
| [web_search.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/web_search.py) | Web-search-augmented generation via web_search_options |
| [mcp_chat.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/mcp_chat.py) | Chat with MCP tool server integration |

### Constrained Generation

| Example | Description |
|---|---|
| [json_schema.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/json_schema.py) | JSON schema-constrained output via response_format |
| [openai_response_format.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/openai_response_format.py) | Structured output with Pydantic via `beta.chat.completions.parse` |
| [regex.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/regex.py) | Regex-constrained generation and token healing |
| [lark.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/lark.py) | Lark grammar-constrained generation (C code grammar) |
| [llguidance.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/llguidance.py) | llguidance grammar with named JSON schema references |

### Advanced

| Example | Description |
|---|---|
| [adapter_chat.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/adapter_chat.py) | Chat with per-request LoRA adapter selection |
| [multi_model_chat.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/multi_model_chat.py) | Multi-model server: list models, route requests, compare outputs |
| [stream_completion_bench.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/stream_completion_bench.py) | Benchmark streaming vs non-streaming latency |

---

## Rust SDK

Examples using the `mistralrs` Rust crate (in `mistralrs/examples/`).

### Getting Started

| Example | Description |
|---|---|
| [text_generation](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/text_generation) | Basic text generation with ISQ quantization and chat |
| [streaming](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/streaming) | Streaming text generation with token-by-token output |
| [gguf](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/gguf) | Load and run a GGUF-quantized model from Hugging Face |
| [gguf_locally](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/gguf_locally) | Load and run a GGUF model from a local file path |
| [multimodal](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/multimodal) | Simple multimodal "hello world" with image input |
| [embedding](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/getting_started/embedding) | Generate text embeddings using an embedding model |

### Models

| Example | Description |
|---|---|
| [text_models](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/text_models) | Unified text model example with all supported model IDs |
| [multimodal_models](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/multimodal_models) | Unified multimodal model example with all supported model IDs |
| [multimodal](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/multimodal) | Multimodal streaming with combined image and audio inputs |
| [multimodal_multiturn](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/multimodal_multiturn) | Multi-turn conversation with a multimodal model |
| [audio](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/audio) | Audio input processing with a multimodal model |
| [asr](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/asr) | Automatic speech recognition (ASR) with Voxtral |
| [speech](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/speech) | Text-to-speech synthesis using a speech model |
| [diffusion](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/models/diffusion) | Image generation using a diffusion model |

### Quantization

| Example | Description |
|---|---|
| [isq](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/isq) | In-situ quantization with explicit and automatic type selection |
| [imatrix](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/imatrix) | ISQ with importance-matrix calibration data |
| [topology](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/topology) | Per-layer quantization control using a Topology |
| [uqff](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/uqff) | Load a pre-quantized UQFF text model |
| [uqff_multimodal](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/uqff_multimodal) | Load a pre-quantized UQFF multimodal model |
| [mixture_of_quant_experts](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/quantization/mixture_of_quant_experts) | MoQE: quantize only MoE expert layers at a different precision |

### Tools and Agents

| Example | Description |
|---|---|
| [agent](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/agent) | Agentic loop with `#[tool]` macro and `AgentBuilder` (non-streaming) |
| [agent_streaming](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/agent_streaming) | Agentic loop with streaming output and real-time events |
| [tools](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/tools) | Tool calling (function calling) with manual tool definitions |
| [tool_callback](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/tool_callback) | Tool callbacks for automatic server-side tool execution |
| [web_search](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/web_search) | Web-search-augmented generation |
| [search_callback](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/search_callback) | Custom search callback to override default web search |
| [mcp_client](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/mcp_client) | MCP client connecting to an external tool server |

### Constrained Generation

| Example | Description |
|---|---|
| [json_schema](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/json_schema) | JSON schema-constrained generation for typed output |
| [grammar](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/grammar) | Constrained generation using a GBNF grammar |
| [llguidance](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/llguidance) | Constrained generation using an llguidance grammar |

### Adapters and AnyMoE

| Example | Description |
|---|---|
| [lora](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/lora) | Load and run a model with a LoRA adapter |
| [xlora](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/xlora) | X-LoRA adapter mixing |
| [anymoe](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/anymoe) | AnyMoE: create a MoE model from fine-tuned adapters |
| [anymoe_lora](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/anymoe_lora) | AnyMoE with LoRA adapters for expert specialization |

### Advanced

| Example | Description |
|---|---|
| [paged_attn](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/paged_attn) | Enable PagedAttention for efficient KV-cache memory management |
| [speculative](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/speculative) | Speculative decoding with a smaller draft model |
| [auto_device_map](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/auto_device_map) | Automatic device mapping across multiple GPUs |
| [multi_model](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/multi_model) | Load and dispatch requests across multiple models |
| [batching](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/batching) | Concurrent request batching with parallel requests |
| [embeddings](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/embeddings) | Compute and compare embeddings with cosine similarity |
| [batching_embeddings](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/batching_embeddings) | Batch multiple embedding requests for parallel encoding |
| [logits_processor](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/logits_processor) | Custom logits processor to modify token probabilities |
| [perplexity](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/perplexity) | Compute perplexity of a text file |
| [error_handling](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/error_handling) | Error handling patterns with error variant matching |
| [file_logging](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/advanced/file_logging) | Log model output to a file using the tracing framework |

### Cookbook

| Example | Description |
|---|---|
| [agent](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/cookbook/agent) | Code review agent using `#[tool]` macro and `AgentBuilder` |
| [multiturn](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/cookbook/multiturn) | Interactive multi-turn chatbot with streaming output |
| [rag](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/cookbook/rag) | Simple RAG with embedding retrieval and text generation |
| [structured](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/cookbook/structured) | Structured data extraction with `generate_structured<T>()` |

---

## Jupyter Notebooks

| Notebook | Description |
|---|---|
| [cookbook.ipynb](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/cookbook.ipynb) | Python API cookbook: loading GGUF, Plain, and X-LoRA models |
| [tool_calling.ipynb](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/tool_calling.ipynb) | Interactive tool calling walkthrough with Python SDK |
| [phi3_duckduckgo_mistral.rs.ipynb](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/phi3_duckduckgo_mistral.rs.ipynb) | Phi-3 tool calling with DuckDuckGo search via LangChain |
