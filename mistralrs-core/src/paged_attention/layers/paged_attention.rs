use candle_core::{DType, Device, Result, Tensor};

use mistralrs_paged_attn::{paged_attention, reshape_and_cache};

use crate::{
    attention::SdpaParams,
    layers::Sdpa,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self { alibi_slopes })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        let slot_mapping = input_metadata
            .slot_mappings
            .get(&query.device().location())
            .unwrap();
        let dims = slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            &slot_mapping.flatten(0, dims.len())?
        } else {
            slot_mapping
        };

        let block_tables = input_metadata
            .block_tables
            .as_ref()
            .unwrap()
            .get(&query.device().location())
            .unwrap();
        let context_lens = input_metadata
            .context_lens
            .as_ref()
            .unwrap()
            .get(&query.device().location())
            .unwrap();

        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(query.device())?)
        } else {
            None
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (_, key_value_heads, _, _) = key.shape().dims4()?;

        #[allow(clippy::cast_possible_truncation)]
        let att = match attention_mask {
            None => None,
            Some(mask) => Some(Sdpa.run_attention(
                query,
                key,
                value,
                Some(mask),
                flash_params,
                sdpa_params,
            )?),
        };

        // paged-attn expects [batch_size, num_tokens, num_heads, head_size]
        let (query, key, value) = if seq_len > 1 {
            let q = query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?;
            let k = key
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            let v = value
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        } else {
            // avoid unnecessary transpose for decoding
            let q = query.reshape(((), attention_heads, head_size))?;
            let k = key.reshape(((), key_value_heads, head_size))?;
            let v = value.reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        };

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                slot_mapping,
            )?;
        }

        if let Some(att) = att {
            // Return result in prefill or first prefix chunk
            return Ok(att);
        }

        let query = query.reshape((batch_size, seq_len, attention_heads, head_size))?;

        let mut key_cache = key_cache.as_ref().unwrap().clone();
        let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = key_cache.dims5()?;
        // (num_blocks, num_heads_kc, head_size_kc, block_size, x) -> (num_blocks, block_size, num_heads_kc, head_size_kc * x)
        key_cache = key_cache.permute((0, 3, 1, 2, 4))?.reshape((
            num_blocks,
            block_size,
            num_heads_kc,
            head_size_kc * x,
        ))?;

        // (num_blocks, num_heads_vc, head_size_vc, block_size) -> (num_blocks, block_size, num_heads_vc, head_size_vc)
        let value_cache = value_cache.as_ref().unwrap().permute((0, 3, 1, 2))?;

        return candle_flash_mla::flash_attn_mla(
            &query,
            &key_cache,
            &value_cache,
            block_tables.to_dtype(DType::I32)?,
            context_lens.to_dtype(DType::I32)?,
            sdpa_params.softmax_scale,
        );

        // //  Args:
        // //  output: shape = [num_generation_tokens, num_heads, head_size]
        // //
        // //  query: shape = [num_generation_tokens, num_heads, head_size]
        // //
        // //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        // //      block_size, x]
        // //
        // //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
        // //      block_size]
        // //
        // //  input_metadata: metadata for paged attention.
        // //
        // //  alibi_slopes: shape = [num_heads]
        // #[allow(clippy::cast_possible_truncation)]
        // paged_attention(
        //     &query,
        //     key_cache.as_ref().unwrap(),
        //     value_cache.as_ref().unwrap(),
        //     block_tables,
        //     context_lens,
        //     alibi_slopes.as_ref(),
        //     input_metadata.max_context_len.unwrap(),
        //     sdpa_params.softmax_scale,
        //     sdpa_params.softcap.unwrap_or(1.0f32),
        // )
    }
}
