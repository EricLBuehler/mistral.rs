# Device mapping

There are 2 ways to do device mapping:
1) Specify the number of layers to put on the GPU - this uses the GPU with ordinal 0.
2) Specify the ordinals and number of layers - this allows for cross-GPU device mapping.

The format for the ordinals and number of layers is `ORD:NUM;...` where ORD is the unique ordinal and NUM is the number of layers for that GPU. This may be repeated as many times as necessary.

> Note: We refer to GPU layers as "device layers" throughout mistral.rs.

## Example of specifying ordinals
```
cargo run --release --features cuda -- -n "0:16;1:16" -i plain -m gradientai/Llama-3-8B-Instruct-262k -a llama
```

> Note: In the Python API, the "0:16;1:16" string is passed as the list `["0:16", "1:16"]`.

## Example of specifying the number of GPU layers
```
cargo run --release --features cuda -- -n 16 -i plain -m gradientai/Llama-3-8B-Instruct-262k -a llama
```