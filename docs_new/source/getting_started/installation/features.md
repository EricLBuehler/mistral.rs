# Feature flags in mistral.rs

mistral.rs controls compilation of hardware accelerator-specific and optimized components with feature flags. These are used for the CLI and the from-scratch Python APIs.

Feature flags can be chained by using commas as delimiters: `--features feature1,feature2`.

**Use the following table to determine which features to use for your GPU:**

|GPU accelerator|Base feature flag to use|
|--|--|
|Nvidia GPUs|`cuda`|
|Apple Silicon|`metal`|
|CPU (no GPU used)|None|

**Use the following table to activate optimized components:**

You can find more details about advanced optimized components [here](../../advanced/index.md).

|GPU accelerator|Optimized component|When to use|Additional feature flag to use|
|--|--|--|--|
|CUDA|FlashAttention V2|All models|`flash-attn`|
|CUDA|FlashAttention V3|All models|`flash-attn-v3`|
|CUDA|cuDNN|Vision models|`flash-attn-v3`|
|CUDA|NCCL|Multiple GPUs|`nccl`|
|CPU (Apple)|Accelerate|All models|`accelerate`|
|CPU (Intel)|MKL|All models|`mkl`|
