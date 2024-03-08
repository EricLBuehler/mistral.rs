import mistralrs

kind = mistralrs.ModelKind.QuantizedGGUF
loader = mistralrs.MistralLoader(
    model_id="mistralai/Mistral-7B-Instruct-v0.1",
    kind=kind,
    no_kv_cache=False,
    repeat_last_n=64,
    quantized_model_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    quantized_filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
)
runner=loader.load()