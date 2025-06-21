use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor};

use mistralrs_paged_attn::{paged_attention, reshape_and_cache};

use crate::{
    attention::SdpaParams,
    get_mut_arcmutex,
    layers::Sdpa,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

#[derive(Clone)]
enum KvScaleCalculator {
    InProgress {
        k_scale: Tensor,
        v_scale: Tensor,
        n: usize,
    },
    Done {
        k_scale: Tensor,
        v_scale: Tensor,
    },
}

impl KvScaleCalculator {
    fn new(device: &Device) -> Result<Self> {
        Ok(Self::InProgress {
            k_scale: Tensor::new(1f32, device)?,
            v_scale: Tensor::new(1f32, device)?,
            n: 0,
        })
    }

    fn collect(&mut self, k_scale_new: &Tensor, v_scale_new: &Tensor) -> Result<usize> {
        match self {
            Self::InProgress {
                k_scale,
                v_scale,
                n,
            } => {
                *k_scale = k_scale.clone().maximum(k_scale_new)?;
                *v_scale = v_scale.clone().maximum(v_scale_new)?;
                *n += 1;
                return Ok(*n);
            }
            Self::Done { .. } => {
                candle_core::bail!("KvScaleCalculator::collect requires InProgress scales");
            }
        }
    }

    fn finish(&mut self) -> Result<()> {
        match self {
            Self::InProgress {
                k_scale,
                v_scale,
                n: _,
            } => {
                *self = Self::Done {
                    k_scale: k_scale.clone(),
                    v_scale: v_scale.clone(),
                }
            }
            Self::Done { .. } => {
                candle_core::bail!("KvScaleCalculator::finalize requires InProgress scales");
            }
        }

        Ok(())
    }

    fn compute_scale(x: &Tensor) -> Result<Tensor> {
        let mut absmax = x.abs()?.to_dtype(DType::F32)?;
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        (absmax / 240.)?.to_dtype(DType::F32)
    }
}

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    k_v_scale: Option<Arc<Mutex<KvScaleCalculator>>>,
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            alibi_slopes,
            k_v_scale: Some(Arc::new(Mutex::new(KvScaleCalculator::new(device)?))),
        })
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

        if let Some(collector) = &self.k_v_scale {
            let collector = &mut *get_mut_arcmutex!(collector);
            if let KvScaleCalculator::InProgress {
                k_scale,
                v_scale,
                n: _,
            } = collector.clone()
            {
                let k_scale = KvScaleCalculator::compute_scale(&key)?;
                let v_scale = KvScaleCalculator::compute_scale(&value)?;
                let n = collector.collect(&k_scale, &v_scale)?;

                if n == 100 {
                    collector.finish()?;
                    assert!(matches!(collector, KvScaleCalculator::Done { .. }));
                }
            }
        }

        let k_v_scale = if let Some(collector) = &self.k_v_scale {
            match &*get_mut_arcmutex!(collector) {
                // Use in progress during collection
                KvScaleCalculator::Done { k_scale, v_scale } => {
                    Some((k_scale.clone(), v_scale.clone()))
                }
                KvScaleCalculator::InProgress {
                    k_scale,
                    v_scale,
                    n,
                } => Some((k_scale.clone(), v_scale.clone())),
            }
        } else {
            None
        };
        assert!(k_v_scale.is_some());

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                k_v_scale.as_ref(),
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                slot_mapping,
            )?;
        }

        if let Some(att) = att {
            // Return result in prefill or first prefix chunk
            return Ok(att);
        }

        //  Args:
        //  output: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  query: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        //      block_size, x]
        //
        //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
        //      block_size]
        //
        //  input_metadata: metadata for paged attention.
        //
        //  alibi_slopes: shape = [num_heads]
        #[allow(clippy::cast_possible_truncation)]
        let res = paged_attention(
            &query,
            k_v_scale.as_ref(),
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            block_tables,
            context_lens,
            alibi_slopes.as_ref(),
            input_metadata.max_context_len.unwrap(),
            sdpa_params.softmax_scale,
            sdpa_params.softcap.unwrap_or(1.0f32),
        )?;

        Ok(res)
    }
}
