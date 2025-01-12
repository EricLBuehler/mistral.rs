# Device mapping

In mistral.rs, device mapping is **automatically managed** to be as performant and easy as possible. Automatic device mapping is enabled
by default in the CLI/server and Python API and does not make any changes when the model fits entirely on the GPU.

Automatic device mapping works by prioritizing loading models into GPU memory, and any remaining parts are loaded into CPU memory.
Models architectures such as vision models which greatly benefit from GPU acceleration also automatically prioritize keeping those
components on the GPU.

### When the model does not fit on 1 GPU
If the model does not fit on 1 GPU, it will be loaded onto any other GPUs detected and any residual parts onto the CPU.

Some memory is reserved for runtime (KV cache allocation, activations, etc), which can be controlled with the `mb_resrv_per_gpu` parameter (CLI/Python) or the `MbReservePerGpu` enum (Rust).

### When the model fits on 1 GPU
If the whole model fits on one GPU then no addition memory is reserved as described above. Memory can still be reserved (triggering possible CPU offloading)
with the `mb_resrv_per_gpu` parameters (CLI/Python) or the `MbReservePerGpu` enum (Rust). 

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