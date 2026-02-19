# Device mapping

In mistral.rs, device mapping is **automatically managed** to be as performant and easy as possible. Automatic device mapping is enabled
by default in the CLI/server and Python SDK and does not make any changes when the model fits entirely on the GPU.

> [!NOTE]
> If your system has more than one CUDA device, mistral.rs will automatically use [tensor parallelism](DISTRIBUTED/DISTRIBUTED.md). If the model does not
> completely fit on the available GPUs, or you wish to use automatic device mapping, you can disable tensor parallelism by setting `MISTRALRS_NO_NCCL=1`.

Automatic device mapping works by prioritizing loading models into GPU memory, and any remaining parts are loaded into CPU memory.
Models architectures such as vision models which greatly benefit from GPU acceleration also automatically prioritize keeping those
components on the GPU.

To control the mapping across devices, you can set the following maximum parameters which the model should expect in a prompt.

- maximum sequence length (default: 4096)
- maximum batch size (default: 1)
- (vision models) maximum image length (length refers to the edge length) (default: 1024)
- (vision models) maximum number of images (default: 1)

These parameters do not translate to hard limits during runtime, they only control the mapping.

### Unified memory systems

On integrated GPU systems (e.g. Apple Silicon, NVIDIA Grace Blackwell, Jetson) where GPU and CPU share the same physical RAM, the auto device mapper caps the GPU memory budget to a fraction of system RAM (75% by default for CUDA iGPUs, configurable via `MISTRALRS_IGPU_MEMORY_FRACTION`; Metal uses the `iogpu.wired_limit_mb` sysctl). CPU offload capacity is limited to the remaining fraction to prevent over-subscription of shared memory. Use `mistralrs doctor` to check whether your device is detected as unified memory.

> [!NOTE]
> The maximum sequence length is also used to ensure that a KV cache will fit for with and without PagedAttention.

## Examples
- Python
    - Text models [text_auto_device_map.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/text_auto_device_map.py)
    - Vision models [vision_auto_device_map.py](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/vision_auto_device_map.py)
- Rust
    - Text models [text_auto_device_map/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/auto_device_map/main.rs)
    - Vision models [vision_auto_device_map/main.rs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/auto_device_map/main.rs)
- Server
    - Text models:
    ```bash
    mistralrs run --isq 4 -m meta-llama/Llama-3.3-70B-Instruct --max-seq-len 4096 --max-batch-size 2
    ```
    - Vision models:
    ```bash
    mistralrs run --isq 4 -m meta-llama/Llama-3.2-11B-Vision-Instruct --max-seq-len 4096 --max-batch-size 2 --max-num-images 2 --max-image-length 1024
    ```

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
mistralrs run -n "0:16;1:16" -m gradientai/Llama-3-8B-Instruct-262k
```

> Note: In the Python SDK, the "0:16;1:16" string is passed as the list `["0:16", "1:16"]`.

## Example of specifying the number of GPU layers
```
mistralrs run -n 16 -m gradientai/Llama-3-8B-Instruct-262k
```