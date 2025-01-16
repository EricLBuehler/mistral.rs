# Device mapping

In mistral.rs, device mapping is **automatically managed** to be as performant and easy as possible. Automatic device mapping is enabled
by default in the CLI/server and Python API and does not make any changes when the model fits entirely on the GPU.

Automatic device mapping works by prioritizing loading models into GPU memory, and any remaining parts are loaded into CPU memory.
Models architectures such as vision models which greatly benefit from GPU acceleration also automatically prioritize keeping those
components on the GPU.

To control the mapping across devices, you can set the maximum sequence length and maximum batch size which the model should expect. For vision models,
you can also specify the maximum image size and number of images. These parameters do not translate to hard limits during runtime, they only control
the mapping.

---

If you want to manually device map the model (not recommended), please continue reading.

> [!NOTE]
> Manual device mapping is deprecated in favor of automatic device mapping due to the possibility for user error in manual.

## Manual device mapping

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