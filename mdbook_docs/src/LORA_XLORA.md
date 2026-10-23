# Examples of LoRA and X-LoRA models

- X-LoRA with no quantization

To start an X-LoRA server with the exactly as presented in [the paper](https://arxiv.org/abs/2402.07148):

```bash
./mistralrs-server --port 1234 x-lora-plain -o orderings/xlora-paper-ordering.json -x lamm-mit/x-lora
```
- LoRA with a model from GGUF

To start an LoRA server with adapters from the X-LoRA paper (you should modify the ordering file to use only one adapter, as the adapter static scalings are all 1 and so the signal will become distorted):

```bash
./mistralrs-server --port 1234 lora-gguf -o orderings/xlora-paper-ordering.json -m TheBloke/zephyr-7B-beta-GGUF -f zephyr-7b-beta.Q8_0.gguf -a lamm-mit/x-lora
```

Normally with a LoRA model you would use a custom ordering file. However, for this example we use the ordering from the X-LoRA paper because we are using the adapters from the X-LoRA paper.