# Adding a Model
## 0) Fork the repository
Start by forking the mistral.rs repository. This will enable you to modify and test the codebase.

## 1) Bring the model code to mistralrs-core/src/models
Clone the non quantized source code from either Candle or your implementation. When copying code, please be sure to review and adhere to the copywright and licensing terms.

## 2) Change some types
Mistral.rs implements fused kernels for `RmsNorm`, `QRmsNorm` (quantized) and `RotaryEmbedding`. In your model,
replace Candle implementations (types) with these. Mistral.rs has its own Candle fork which is kept up to date, but has some modifications. The `RmsNorm`, `QRmsNorm` and `RotaryEmbedding` types can be found in `candle_nn`.

## 3) Update the `forward` methods

The forward methods should have the following 2 parameters added to their signatures:

```diff
pub fn forward(
        &mut self,
        input_ids: &Tensor,
+       seqlen_offsets: &[usize],
+       start_offsets_kernel: Tensor,
    ) -> Result<Tensor> {
```

For forward methods that are not on the model struct, the KV cache must also be passed:

```diff
pub fn forward(
        &mut self,
        input_ids: &Tensor,
+       seqlen_offsets: &[usize],
+       start_offsets_kernel: Tensor,
+       kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
```

## 4) Add a `Cache`

The model struct should contain a `Cache`, found at `mistralrs_core::models`. A `Cache` mangages the KV cache. When initializing it,
be sure to set the `is_xlora` state to false as this is a non X-LoRA model: `Cache::new(cfg.num_hidden_layers, false)`.

In the model struct's forward method, the normal model cache should be accessed with the `.lock()` method: `self.cache.lock()`. Then, caches for each layer should be passed by accessing the locked cache with `cache[layer]`. In the attention block, the KV cache is updated. See the following code for reference:

```rust
let (k, v) = match &*kv_cache {
    None => (k, v),
    Some((prev_k, prev_v)) => {
        let k = candle_nn::ops::kvconcat(prev_k, &k, 2)?;
        let v = candle_nn::ops::kvconcat(prev_v, &v, 2)?;
        (k, v)
    }
};
*kv_cache = Some((k.clone(), v.clone()));
```

## 5) Update the RoPE application
Next, replace code that applies RoPE to Q or K vectors with the following:

```rust
self.rotary_emb.forward(
    seqlen_offsets,
    &start_offsets_kernel,
    &mut q,
    &mut k,
    b_sz,
)?;

if q.rank() == 3 {
    q = q
        .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    k = k
        .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
}
```

## 6) Implement a `Pipeline` and `Loader` in mistralrs-core/src/pipeline
The `Loader` is in charge of downloading and loading the model. The `download_model` method is pretty general and can be copied from an existing implementation

The `_setup_model` method instantiates the `Pipeline`. It handles loading the different model kinds. The `Pipeline` is responsible for running and sampling the model. For example, please see the [`mistral pipeline`](mistralrs-core/src/pipeline/mistral.rs).

A quantized model should be handled by the `quantized_llama.rs` implementation. For example the [`llama pipeline`](mistralrs-core/src/pipeline/llama.rs) loads GGUF and GGML model with `quantized_llama`.rs


## 7) Adding an X-LoRA counterpart
If you prefer, we can add an X-LoRA counterpart of your model when you submit a PR. However, if you wish to add it yourself, you can follow the below guide:

To add an X-LoRA counterpart, start by copying your model to the `xlora_models` directory. 

Then, implement `ScalingsMaker` for model. This requires that you change the original `forward` method to `inner_forward`. A new forward method should be written, which because it is the same across most models can be copied:

```rust
pub fn forward(
    &mut self,
    input_ids: &Tensor,
    input_ids_full: &Tensor,
    seqlen_offsets: &[usize],
    seqlen_offsets_full: &[usize],
    start_offsets_kernel: Tensor,
    start_offsets_kernel_full: Tensor,
    no_kv_cache: bool,
    non_granular_state: &Option<NonGranularState>,
) -> Result<Tensor> {
    let (_b_size, seq_len_full) = input_ids_full.dims2()?;
    let (_, seq_len) = input_ids.dims2()?;

    let scalings = self.get_scalings(
        input_ids,
        input_ids_full,
        seqlen_offsets,
        seqlen_offsets_full,
        &start_offsets_kernel,
        &start_offsets_kernel_full,
        no_kv_cache,
        non_granular_state,
    )?;

    if no_kv_cache {
        self.inner_forward(
            input_ids_full,
            seqlen_offsets_full,
            start_offsets_kernel_full,
            scalings,
            true,
            no_kv_cache,
            None,
        )?
        .apply(&self.lm_head)?
        .narrow(1, seq_len_full - 1, 1)
    } else {
        // is_full_pass=true is ok because no_kv_cache=false
        self.inner_forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            scalings,
            true,
            no_kv_cache,
            None,
        )?
        .apply(&self.lm_head)?
        .narrow(1, seq_len - 1, 1)
    }
}
```

Next, the `Linear` layers should be replaced by `Arc<dyn LinearLayerLike>` and the appropriate method must be passed between initialization functions to correctly handle the layer count and configs:
```diff
fn new(
    rotary_emb: Arc<RotaryEmbedding>,
    cfg: &Config,
    vb: VarBuilder,
+   lora_config: &[((String, String), LoraConfig)],
+   count: &mut usize,
+   ord: &Ordering,
) -> Result<Self> {
```

Additionally, the pipeline should store a non granular state: 

```diff
pub struct MistralPipeline {
    model: Model,
    tokenizer: Tokenizer,
    config: MistralSpecificConfig,
    no_kv_cache: bool,
    chat_template: ChatTemplate,
+   non_granular_state: Option<NonGranularState>,
    model_id: String,
}
```
