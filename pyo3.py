from mistralrs import ModelKind, MistalLoader, Request

kind = ModelKind.QuantizedGGUF
loader = MistralLoader(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    kind=kind,
    no_kv_cache=False,
    repeat_last_n=64,
    quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
)
runner=loader.load()
res = runner.add_request(Request(messages = {"role": "user", "content": "Tell me a story about the Rust type system."}, max_tokens = 256, frequency_penalty = 1.0, top_p = 0.1, temperature = 0.1))
print(res)