# mistralrs-cli TOML Config

`mistralrs-cli` can run entirely from a single TOML file. This config supports multiple models with no aliases — the `model_id` you provide is the only identifier.

## Usage

```bash
mistralrs from-config --file path/to/config.toml
```

## Serve example (multi-model)

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 8080

[runtime]
max_seqs = 32
enable_search = true
search_embedding_model = "embedding-gemma"

[paged_attn]
mode = "auto"

[[models]]
kind = "auto"
model_id = "meta-llama/Llama-3.2-3B-Instruct"
dtype = "auto"

[models.quantization]
in_situ_quant = "4"

[[models]]
kind = "vision"
model_id = "Qwen/Qwen2-VL-2B-Instruct"

[models.vision]
max_num_images = 4

[[models]]
kind = "embedding"
model_id = "google/embeddinggemma-300m"
```

## Run example (interactive)

```toml
command = "run"
enable_thinking = true

[runtime]
max_seqs = 16

[[models]]
kind = "text"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
```

## Model kinds

- `auto`: Auto loader (recommended for most text/vision models).
- `text`: Explicit text model setup (format, adapters, quantization).
- `vision`: Explicit vision model setup.
- `diffusion`: Image generation models (e.g., FLUX).
- `speech`: Speech models (e.g., Dia).
- `embedding`: Embedding models.

## Per-model options

Each `[[models]]` entry supports the same logical groupings as the CLI:

- Top-level: `model_id`, `tokenizer`, `arch`, `dtype`
- `[models.format]`: `format`, `quantized_file`, `tok_model_id`, `gqa`
- `[models.adapter]`: `lora`, `xlora`, `xlora_order`, `tgt_non_granular_index`
- `[models.quantization]`: `in_situ_quant`, `from_uqff`, `isq_organization`, `imatrix`, `calibration_file`
- `[models.device]`: `cpu` (must match across models), `device_layers`, `topology`, `hf_cache`, `max_seq_len`, `max_batch_size`
- `[models.vision]`: `max_edge`, `max_num_images`, `max_image_length`

Per-model overrides:

- `chat_template`
- `jinja_explicit`

## Notes

- `default_model_id` (serve only) must match one of the `model_id` values in `[[models]]`.
- `cpu` is global for the run; if specified, it must be consistent across all models.
