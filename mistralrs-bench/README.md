# `mistralrs-bench`

> **Deprecated:** The standalone `mistralrs-bench` package is deprecated. Use `mistralrs bench` instead for the same functionality.
>
> **Migration:**
> ```bash
> # Old
> cargo run --release --features cuda --package mistralrs-bench -- plain -m model-id
>
> # New
> mistralrs bench -m model-id
> ```

This is our official benchmarking application, which allows you to collect structured information about the speed of `mistral.rs`.

> [!NOTE]
> You should replace `--features ...` with one of the features specified [here](../README.md#supported-accelerators), or remove it for pure CPU inference.

To run: `cargo run --release --features ... --package mistralrs-bench`

```bash
Fast and easy LLM serving.

Usage: mistralrs-bench [OPTIONS] <COMMAND>

Commands:
  plain        Select a plain model
  x-lora       Select an X-LoRA architecture
  lora         Select a LoRA architecture
  gguf         Select a GGUF model
  x-lora-gguf  Select a GGUF model with X-LoRA
  lora-gguf    Select a GGUF model with LoRA
  ggml         Select a GGML model
  x-lora-ggml  Select a GGML model with X-LoRA
  lora-ggml    Select a GGML model with LoRA
  help         Print this message or the help of the given subcommand(s)

Options:
  -p, --n-prompt <N_PROMPT>
          Number of prompt tokens to run [default: 512]
  -g, --n-gen <N_GEN>
          Number of generations tokens to run [default: 128]
  -c, --concurrency <CONCURRENCY>
          Number of concurrent requests to run. Default is 1
  -r, --repetitions <REPETITIONS>
          Number of times to repeat each test [default: 5]
  -n, --num-device-layers <NUM_DEVICE_LAYERS>
          Number of device layers to load and run on the device. All others will be on the CPU
  -h, --help
          Print help
  -V, --version
          Print version
```