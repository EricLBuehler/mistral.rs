use std::sync::Arc;

use candle_nn::Module;
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    device_map::DeviceMapper,
    kv_cache::KvCache,
    layers::{RmsNorm, RotaryEmbedding},
    models::{
        minimax_m2::{attention::FullOrLinearAttention, Config},
        mixtral::SparseMoeBlock,
    },
    paged_attention::PagedAttention,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};
use candle_core::{Result, Tensor};

struct DecoderLayer {
    self_attn: FullOrLinearAttention,
    moe: super::super::mixtral::SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    //fixme layer cache for lin attenation
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mixtral_config = cfg.clone().into();
        let self_attn = FullOrLinearAttention::new(
            rotary_emb,
            cfg,
            vb.pp("attention"),
            layer_idx,
            paged_attn,
            comm,
        )?;
        let block_sparse_moe = SparseMoeBlock::new(
            &mixtral_config,
            mapper.set_device(layer_idx, vb.pp("block_sparse_moe"), loading_isq),
            comm,
        )?;
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            moe: block_sparse_moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        // skipping alpha/beta factors, none of minimax inference configs has them set.
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.post_attention_layernorm)?
            .apply(&self.moe)?
            .to_dtype(residual.dtype())?;
        residual + xs
    }
}
