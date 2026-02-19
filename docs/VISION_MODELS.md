# Vision model support in mistral.rs

Mistral.rs supports various modalities of models, including vision models. Vision models take images and text as input and have the capability to reason over both.

Please see docs for the following model types:

- Phi 3 Vision: [PHI3V.md](PHI3V.md)
- Idefics2: [IDEFICS2.md](IDEFICS2.md)
- LLaVA and LLaVANext: [LLAVA.md](LLaVA.md)
- Llama 3.2 Vision: [VLLAMA.md](VLLAMA.md)
- Qwen2-VL: [QWEN2VL.md](QWEN2VL.md)
- Idefics 3 and Smol VLM: [IDEFICS3.md](IDEFICS3.md)
- Phi 4 Multimodal: [PHI4MM.md](PHI4MM.md)
- Gemma 3: [GEMMA3.md](GEMMA3.md)
- Gemma 3n: [GEMMA3N.md](GEMMA3N.md)
- Mistral 3: [MISTRAL3.md](MISTRAL3.md)
- Llama 4: [LLAMA4.md](LLAMA4.md)
- Qwen 3-VL: [QWEN3VL.md](QWEN3VL.md)
- MiniCPM-O 2.6: [MINICPMO_2_6.md](MINICPMO_2_6.md)

> Note for the Python and HTTP APIs:
> We follow the OpenAI specification for structuring the image messages and allow both base64 encoded images as well as a URL/path to the image. There are many examples of this, see [this Python example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/phi3v.py).