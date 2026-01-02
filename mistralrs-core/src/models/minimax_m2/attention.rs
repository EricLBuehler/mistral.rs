use std::sync::Arc;

use super::config::Config;
use crate::{layers::Activation, models::mixtral::Attention as MixtralAttention};
use candle_core::{Result, Tensor};
use candle_nn::Module;
use mistralrs_quant::{ColumnParallelLayer, QuantMethod, RowParallelLayer, ShardedVarBuilder};

use crate::{
    kv_cache::KvCache,
    layers::RotaryEmbedding,
    paged_attention::PagedAttention,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

pub(crate) struct LinearAttention {
    qkv_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    o_gate: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    activation: Activation,

    slopte_rate: Arc<Tensor>,
    q_decay: Arc<Tensor>,
    k_decay: Arc<Tensor>,
    diagonal_decay: Arc<Tensor>,
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
        let qkv_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim * 3,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("qkv_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_proj"),
        )?;
        let o_gate = RowParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("o_gate"),
        )?;
        let slope_rate = Self::slope_rate(num_heads, layer, hidden_sz, vb.pp("slope_rate"))?;
        let (q_decay, k_decay, diagonal_decay) =
            Self::decay_factors(&slope_rate, cfg.block_size, vb.pp("decay_factors"))?;

        Ok(Self {
            qkv_proj,
            o_proj,
            o_gate,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            slopte_rate: Arc::new(slope_rate),
            q_decay: Arc::new(q_decay),
            k_decay: Arc::new(k_decay),
            diagonal_decay: Arc::new(diagonal_decay),
            activation: cfg.hidden_act.clone(),
        })
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
        let (batch_sz, seq_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.qkv_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let qkv_states = self.qkv_proj.forward(&xs)?;
        let qkv_states = self.activation.forward(&&qkv_states)?;
        let qkv_states =
            qkv_states.reshape((batch_sz, seq_len, self.num_heads, 3 * self.head_dim))?;

        // Split into Q, K, V
        let qkv_split = qkv_states.chunk(self.head_dim, 3)?;
        let mut query_states = qkv_split[0].transpose(1, 2)?; // [batch, num_heads, seq_len, head_dim]
        let mut key_states = qkv_split[1].transpose(1, 2)?;
        let mut value_states = qkv_split[2].transpose(1, 2)?;

        unimplemented!()
    }
}

pub(crate) enum FullOrLinearAttention {
    Full(MixtralAttention),
    Linear(LinearAttention),
}
impl FullOrLinearAttention {
    pub fn is_linear_layer(layer_idx: usize) -> bool {
        layer_idx % 2 == 1
    }
    pub fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_idx: usize,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mixtral_config = cfg.clone().into();
        let is_linear = Self::is_linear_layer(layer_idx);
        let self_attn = if is_linear {
            FullOrLinearAttention::Linear(LinearAttention::new(
                rotary_emb,
                cfg,
                vb.pp(format!("linear_layer_{}", layer_idx)),
                layer_idx,
                comm,
            )?)
        } else {
            FullOrLinearAttention::Full(MixtralAttention::new(
                rotary_emb,
                &mixtral_config,
                vb.pp(format!("full_layer_{}", layer_idx)),
                paged_attn,
                comm,
            )?)
        };
        Ok(self_attn)
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
