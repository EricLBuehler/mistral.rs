# Enhancing ISQ with an imatrix

Mistral.rs supports enhancing the performance of models quantized with ISQ by collecting an imatix from [calibration data](../calibration_data/). The following quantizations are supported with an imatrix:

- `Q2K`
- `Q3K`
- `Q4K`
- `Q5K`
- `Q6K`

Using an imatrix causes the quantization process to take longer as the data must be collected, but there is no inference-time performance decrease.

To use this, simply specify the calibration data file in the various APIs:

## With the CLI
```
./mistralrs-server -i --isq Q4K plain -m meta-llama/Llama-3.2-3B-Instruct --calibration-file calibration_data/calibration_datav3_small.txt
```

## With the Rust API
You can find this example [here](../mistralrs/examples/imatrix/).

```rust
let model = TextModelBuilder::new("meta-llama/Llama-3.2-3B-Instruct")
    .with_isq(IsqType::Q4K)
    .with_calibration_file("calibration_data/calibration_datav3_small.txt".into())
    .with_logging()
    .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
    .build()
    .await?;
```

## With the Python API
You can find this example [here](../examples/python/imatrix.py).

```python
runner = Runner(
    which=Which.Plain(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        calibration_file="calibration_data/calibration_datav3_small.txt"
    ),
    in_situ_quant="Q4K",
)
```