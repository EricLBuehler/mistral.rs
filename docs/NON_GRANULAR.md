# X-LoRA non-granular scalings

A key limitation of the X-LoRA architecture is the need for 2 forward passes of the model per generation step. To trade off model performance for speed, mistral.rs allows the user to reduce the granularity of the scalings by caching them in a technique we call Non Granular Scalings.

## How it works
For the first $k$ generation steps, the scalings are calculated normally for each token. However, for the rest of the tokens, it is cached and re-used. In this way, we are able to avoid the second forward pass and the performance is increased significantly. To maintain correctness, enabling non-granular scalings will restrict the engine to processing one sequence at a time.

## How to use it
### Command line
This can be enabled by passing `--tgt-non-granular-index` followed by $k$:
```bash
mistralrs serve -p 1234 --xlora lamm-mit/x-lora --xlora-order orderings/xlora-paper-ordering.json --tgt-non-granular-index 5
```

### Python
Set the `tgt_non_granular_index` attribute to a non-`None` value in the `Which` selection:
```py
from mistralrs import Runner, Which

runner = Runner(
    which=Which.XLoraGGUF(
        tok_model_id=None,  # Automatically determine from ordering file
        quantized_model_id="TheBloke/zephyr-7B-beta-GGUF",
        quantized_filename="zephyr-7b-beta.Q4_0.gguf",
        xlora_model_id="lamm-mit/x-lora",
        order="orderings/xlora-paper-ordering.json",
        tgt_non_granular_index=5,
    )
)

...
```
