# Watermark scheme examples

Each JSON file selects one algorithm. Token examples target Qwen/Qwen3-4B's
151936 output slots; adjust `vocab_size` for another model. Zero keys are fixtures,
not production keys. All runners replace the key with `MISTRALRS_WATERMARK_KEY`.

| Scheme | Configuration | Execution |
| --- | --- | --- |
| SynthID | `synthid.json` | Text generation and token detection |
| KGW | `kgw.json` | Text generation and token detection |
| Unigram | `unigram.json` | Text generation and token detection |
| Exponential race | `exponential.json` | Keyed selection and cost detection |
| Inverse transform | `inverse_transform.json` | Keyed selection and cost detection |
| MPAC | `mpac.json` | Payload encoding and decoding |
| SemStamp | `semstamp.json` | Synthetic embedding acceptance/detection |

Run from the repository root:

```bash
cargo run -p mistralrs --example watermarking -- examples/watermarking/kgw.json
python examples/python/watermarking.py kgw
python examples/server/watermarking.py kgw --endpoint chat/completions
```

Use each scheme's name in place of `kgw`. HTTP token generation does not support
SemStamp: it needs sentence embeddings and a host-owned sentence retry workflow.
Rust/Python examples demonstrate its embedding API instead. For GPU Rust builds,
add `--features metal` or `--features cuda`; SemStamp also reads
`MISTRALRS_WATERMARK_DEVICE`. The Python SemStamp example accepts `--device`.

The device integration boundaries and proposed library extensions are documented
in `docs/src/content/docs/guides/customize/watermarking-gpu.md`.
