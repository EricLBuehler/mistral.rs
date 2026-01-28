# Enhancing ISQ with an imatrix

Mistral.rs supports enhancing the performance of models quantized with ISQ by collecting an imatix from [calibration data](https://github.com/EricLBuehler/mistral.rs/tree/master/calibration_data/). The following quantizations are supported with an imatrix:

- `Q2K`
- `Q3K`
- `Q4K`
- `Q5K`
- `Q6K`

> **What is an imatrix?** An imatrix (importance matrix) is generated from data collected during the execution of the model on calibration data. This data is used to enhance the performance of the model by enabling a weighted RMSE minimization when quantizing the tensor. For more information, see the [original PR](https://github.com/ggerganov/llama.cpp/pull/4861).

Using an imatrix causes the quantization process to take longer as the data must be collected, but there is no inference-time performance decrease.

> Note: mistral.rs will automatically generate a **.cimatrix** file which can be used within mistral.rs as a replacement for a .imatrix file. The primary advantage is the in-situ generation within mistral.rs. The format is incompatible with llama.cpp.

To use this, simply specify the calibration data file in the various APIs as detailed below.

## With the CLI
```bash
mistralrs run --isq 4 -m meta-llama/Llama-3.2-3B-Instruct --calibration-file calibration_data/calibration_datav3_small.txt
```

## With the Rust SDK
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/tree/master/mistralrs/examples/imatrix/).

```rust
let model = TextModelBuilder::new("meta-llama/Llama-3.2-3B-Instruct")
    .with_isq(IsqType::Q4K)
    .with_calibration_file("calibration_data/calibration_datav3_small.txt".into())
    .with_logging()
    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
    .build()
    .await?;
```

## With the Python SDK
You can find this example [here](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/imatrix.py).

```python
runner = Runner(
    which=Which.Plain(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        calibration_file="calibration_data/calibration_datav3_small.txt"
    ),
    in_situ_quant="4",
)
```