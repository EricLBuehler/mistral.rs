use std::sync::Arc;

use super::config::Config;
use crate::models::mistral::Attention;
use candle_core::{Result, Tensor};
use mistralrs_quant::{
    ColumnParallelLayer, MatMul, QuantMethod, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    attention::SdpaParams,
    kv_cache::KvCache,
    layers::{RotaryEmbedding, Sdpa},
    paged_attention::PagedAttention,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

struct LinearAttention {
    qkv_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    o_gate: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,

    slopte_rate: Arc<dyn QuantMethod>,
    q_decay: Arc<dyn QuantMethod>,
    k_decay: Arc<dyn QuantMethod>,
    diagonal_decay: Arc<dyn QuantMethod>,
}

impl LinearAttention {
    fn slope_rate(
        num_heads: usize,
        layer: usize,
        num_hidden_layers: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Tensor> {
        let base = 1.0 / (2.0_f32.powf(8.0 / num_heads as f32));

        // Create exponent tensor: [1, 2, 3, ..., num_attention_heads]
        let exponents = Tensor::arange(1.0, num_heads as f32, vb.device())?;

        // Calculate factor: 1 - layer_idx / (num_hidden_layers - 1 + 1e-5) + 1e-5
        let denominator = (num_hidden_layers - 1) as f32 + 1e-5;
        let factor = 1.0 - (layer as f32 / denominator) + 1e-5;

        // Compute base^exponent
        let base_tensor = Tensor::new(&[base], vb.device())?;
        let rate = base_tensor.pow(&exponents)?;

        // Multiply by factor
        let factor = Tensor::new(&[factor], vb.device())?;
        let rate = rate.broadcast_mul(&factor)?;

        // Reshape to [num_attention_heads, 1, 1]
        rate.reshape((num_heads, 1, 1))
    }

    fn decay_factors(
        slope_rate: &Tensor,
        block_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let block_size_range = Tensor::arange(1.0, block_size as f32, vb.device())?;

        // query_decay = exp(-slope_rate * block_size_range)
        let query_decay = (slope_rate.neg()? * &block_size_range)?.exp()?;

        // key_decay = exp(-slope_rate * (block_size - block_size_range))
        let key_decay_input = ((block_size as f64) - &block_size_range)?;
        let key_decay = (slope_rate.neg() * &key_decay_input)?.exp()?;

        // diagonal_decay = exp(slope_rate * diagonal_decay_mask)
        let diagonal_decay_range = block_size_range.clone();
        let diagonal_decay = diagonal_decay_range.broadcast_sub(&diagonal_decay_range)?;
        let diagonal_decay = (slope_rate * &diagonal_decay)?;
        let diagonal_decay = diagonal_decay.where_cond(
            &diagonal_decay.ge(0.0)?,
            &Tensor::new(&[f32::NEG_INFINITY], vb.device())?,
        )?;
        let diagonal_decay = diagonal_decay.exp()?;

        Ok((query_decay, key_decay, diagonal_decay))
    }

    pub(crate) fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer: usize,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("q_proj"),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            vb.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        candle_core::bail!("ohno");
        // Ok(Self {
        //     q_proj,
        //     k_proj,
        //     v_proj,
        //     o_proj,
        //     num_heads: num_heads / comm.world_size(),
        //     num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
        //     head_dim,
        //     rotary_emb,
        //     paged_attn,
        //     sdpa_params: SdpaParams {
        //         n_kv_groups: mistralrs_quant::compute_n_kv_groups(
        //             cfg.num_key_value_heads,
        //             cfg.num_attention_heads,
        //             comm,
        //         ),
        //         softcap: None,
        //         softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        //         sliding_window: cfg.sliding_window,
        //     },
        // })
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
    }
}
pub(crate) enum FullOrLinearAttention {
    Full(Attention),
    Linear(LinearAttention),
}

impl FullOrLinearAttention {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        match self {
            FullOrLinearAttention::Full(f) => f.forward(
                xs,
                attention_mask,
                seqlen_offsets,
                kv_cache,
                metadata,
                flash_params,
            ),
            FullOrLinearAttention::Linear(f) => f.forward(
                xs,
                attention_mask,
                seqlen_offsets,
                kv_cache,
                metadata,
                flash_params,
            ),
        }
    }
}
