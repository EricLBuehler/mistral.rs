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

In the model struct's forward method, the normal model cache should be accessed with the `.lock()` method: `self.cache.lock()`. Then, caches for each layer should be passed by accessing the locked cache with `cache.get_mut(layer).unwrap()`. In the attention block, the KV cache is updated. See the following code for reference:

```rust
let (key_states, value_states) = match &*kv_cache {
    None => (key_states, value_states),
    Some((prev_k, prev_v)) => {
        let key_states = candle_nn::ops::kvconcat(prev_k, &key_states, 2)?;
        let value_states = candle_nn::ops::kvconcat(prev_v, &value_states, 2)?;
        (key_states, value_states)
    }
};
*kv_cache = Some((key_states.clone(), value_states.clone()));
```

## 5) Update the RoPE application
Next, replace code that applies RoPE to Q or K vectors with the following:

```rust
self.rotary_emb.forward(
    seqlen_offsets,
    &start_offsets_kernel,
    &mut query_states,
    &mut key_states,
    b_sz,
)?;

if query_states.rank() == 3 {
    query_states = query_states
        .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
    key_states = key_states
        .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
        .transpose(1, 2)?
        .contiguous()?;
}
```

## 6) Implement a `Pipeline` and `Loader` in mistralrs-core/src/pipeline
The `Loader` is in charge of downloading and loading the model. The `download_model` method is pretty general and can be copied from an existing implementation

The `_setup_model` method instantiates the `Pipeline`. It handles loading the different model kinds. The `Pipeline` is responsible for running and sampling the model. For example, please see the [`mistral pipeline`](mistralrs-core/src/pipeline/mistral.rs).