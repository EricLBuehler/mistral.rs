# Device mapping

In mistral.rs, device mapping is **automatically managed** to be as performant and easy as possible. Automatic device mapping is enabled
by default in the CLI/server and Python API and does not make any changes when the model fits entirely on the GPU.

> [!NOTE]
> If your system has more than one CUDA device, mistral.rs will automatically use [tensor parallelism](DISTRIBUTED.md). If the model does not
> completely fit on the available GPUs, or you with to use automatic device mapping, you can disable tensor parallelism by setting `MISTRALRS_NO_NCCL=1`.

Automatic device mapping works by prioritizing loading models into GPU memory, and any remaining parts are loaded into CPU memory.
Models architectures such as vision models which greatly benefit from GPU acceleration also automatically prioritize keeping those
components on the GPU.

To control the mapping across devices, you can set the following maximum parameters which the model should expect in a prompt.

- maximum sequence length (default: 4096)
- maximum batch size (default: 1)
- (vision models) maximum image length (length refers to the edge length) (default: 1024)
- (vision models) maximum number of images (default: 1)

These parameters do not translate to hard limits during runtime, they only control the mapping.

> [!NOTE]
> The maximum sequence length is also used to ensure that a KV cache will fit for with and without PagedAttention.

## Examples
- Python
    - Text models [text_auto_device_map.py](../examples/python/text_auto_device_map.py)
    - Vision models [vision_auto_device_map.py](../examples/python/vision_auto_device_map.py)
- Rust
    - Text models [text_auto_device_map/main.rs](../mistralrs/examples/text_auto_device_map/main.rs)
    - Vision models [vision_auto_device_map/main.rs](../mistralrs/examples/vision_auto_device_map/main.rs)
- Server
    - Text models: 
    ```
    ./mistralrs-server -i --isq q4k plain -m meta-llama/Llama-3.3-70B-Instruct --max-seq-len 4096 --max-batch-size 2
    ```
    - Vision models:
    ```
    ./mistralrs-server -i --isq q4k vision-plain -m meta-llama/Llama-3.2-11B-Vision-Instruct --max-seq-len 4096 --max-batch-size 2 --max-num-images 2 --max-image-length 1024
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
cargo run --release --features cuda -- -n "0:16;1:16" -i plain -m gradientai/Llama-3-8B-Instruct-262k -a llama
```

> Note: In the Python API, the "0:16;1:16" string is passed as the list `["0:16", "1:16"]`.

## Example of specifying the number of GPU layers
```
cargo run --release --features cuda -- -n 16 -i plain -m gradientai/Llama-3-8B-Instruct-262k -a llama
```