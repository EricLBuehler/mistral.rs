# Examples of LoRA and X-LoRA models

- X-LoRA with no quantization

To start an X-LoRA server with the exactly as presented in [the paper](https://arxiv.org/abs/2402.07148):

```bash
mistralrs serve -p 1234 --xlora lamm-mit/x-lora --xlora-order orderings/xlora-paper-ordering.json
```
- LoRA with a model from GGUF

To start a LoRA server with adapters from the X-LoRA paper (you should modify the ordering file to use only one adapter, as the adapter static scalings are all 1 and so the signal will become distorted):

```bash
mistralrs serve -p 1234 --format gguf -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q8_0.gguf --lora lamm-mit/x-lora
```

Normally with a LoRA model you would use a custom ordering file. However, for this example we use the ordering from the X-LoRA paper because we are using the adapters from the X-LoRA paper.
