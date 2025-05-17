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

> Note for the Python and HTTP APIs:
> We follow the OpenAI specification for structuring the image messages and allow both base64 encoded images as well as a URL/path to the image. There are many examples of this, see [this Python example](../examples/python/phi3v.py).