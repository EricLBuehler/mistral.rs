# Perplexity

What is peplexity? See: https://huggingface.co/docs/transformers/en/perplexity

To calculate perplexity over some text contained in a file, run:
```
cargo run --release --features ... --example perplexity -- --model-id ... --file ... --isq ...
```

For example,
```
wget https://huggingface.co/datasets/EricB/wikitext2/resolve/main/wiki.test_mini.raw
cargo run --release --features ... --example perplexity -- --model-id meta-llama/Llama-3.1-8B-Instruct --file wiki.test_mini.raw --isq q4k
```

> Note: A table of perplexity benchmarks will be coming soon!