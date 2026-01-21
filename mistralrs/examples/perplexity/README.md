# Perplexity

What is peplexity? See: https://huggingface.co/docs/transformers/en/perplexity

To calculate perplexity over some text contained in a file, run:
```
cargo run --release --features ... --example perplexity -- --model-id ... --file ... --isq ...
```

For example,
```
wget https://huggingface.co/datasets/EricB/wikitext2/resolve/main/wiki.test_mini.raw
cargo run --release --features ... --example perplexity -- --model-id meta-llama/Llama-3.1-8B-Instruct --file wiki.test_mini.raw --isq 4
```

> Note: A table of perplexity benchmarks will be coming soon!

## Benchmark

All benchmarks on an M3 Max (Metal) with model: `meta-llama/Llama-3.1-8B-Instruct`, calibration data: `calibration_datav3_small.txt` (if applicable), and test data: `wiki.test_small.raw`.

|Quantization|Mistral.rs Imatrix    |Mistral.rs PPL     |Mistral.rs ΔPPL|
|--          |--                    |--                 |--             |
|None|None                          |8.538557±2.3113003 |0±0|
|Q6K|calibration_datav3_small.txt   |8.559697±2.3130114 |0.02114±0.0253533|
|Q5K|calibration_datav3_small.txt   |8.599381±2.3180215 |0.0608253±0.0438977|
|Q4K|calibration_datav3_small.txt   |8.7962885±2.3615937|0.257732±0.0893235|
|Q3K|None                           |10.583125±2.794593 |2.04457±0.586526|