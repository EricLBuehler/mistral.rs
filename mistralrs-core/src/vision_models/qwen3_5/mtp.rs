//! Multi-token prediction head shipped inside Qwen3.5 / Qwen3.8 checkpoints (`mtp.*`).
//!
//! Mirrors vLLM's `Qwen3_5MultiTokenPredictor`: one full-attention decoder block fed with
//! `fc(concat(norm(embed(next_token)), norm(target_hidden)))`, sharing the target's embeddings and
//! `lm_head`. Unlike the Gemma 4 assistant it keeps its own paged KV cache, addressed through the
//! sequence's block table as the layer right after the main stack.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    attention::AttentionMask,
    device_map::DeviceMapper,
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding},
    paged_attention::{AttentionImplementation, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use super::{config::TextConfig, text::DecoderLayer};

pub const MTP_FC_WEIGHT: &str = "mtp.fc.weight";

pub struct Qwen3_5MtpHead {
    pre_fc_norm_embedding: GemmaRmsNorm,
    pre_fc_norm_hidden: GemmaRmsNorm,
    fc: Arc<dyn QuantMethod>,
    layer: DecoderLayer,
    norm: GemmaRmsNorm,
    kv_layer_idx: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3_5MtpHead {
    /// `vb` is the checkpoint root; the head lives on the non-mapped device with the final norm and
    /// `lm_head` it feeds.
    pub fn load(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        normal_loading_metadata: &NormalLoadingMetadata,
        attention_mechanism: &AttentionImplementation,
    ) -> Result<Self> {
        if !crate::layers::contains_tensor_or_uqff(&vb, MTP_FC_WEIGHT) {
            candle_core::bail!(
                "`--mtp` requested but the checkpoint has no built-in MTP head (`{MTP_FC_WEIGHT}`)."
            );
        }
        if cfg.mtp_num_hidden_layers != 1 {
            candle_core::bail!(
                "Qwen3.5 MTP supports exactly one MTP layer, config has {}",
                cfg.mtp_num_hidden_layers
            );
        }
        if cfg.mtp_use_dedicated_embeddings {
            candle_core::bail!("Qwen3.5 MTP with dedicated embeddings is not supported");
        }
        let loading_isq = normal_loading_metadata.loading_isq;
        let device = normal_loading_metadata.real_device.clone();
        let vb_mtp = vb.pp("mtp");
        let vb_quant = mapper.set_nm_device(vb_mtp.clone(), loading_isq);
        let vb_plain = mapper.set_nm_device(vb_mtp, false);
        let comm = mapper.get_comm_for(cfg.mtp_layer_idx())?;

        let pre_fc_norm_embedding = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_embedding"),
        )?;
        let pre_fc_norm_hidden = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_plain.pp("pre_fc_norm_hidden"),
        )?;
        let fc = ReplicatedLayer::new(
            2 * cfg.hidden_size,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            vb_quant.pp("fc"),
        )?;
        let rotary_emb = Arc::new(Qwen3VLRotaryEmbedding::new(
            cfg.rope_theta() as f32,
            cfg.rot_dim(),
            &device,
            cfg.mrope_section().to_vec(),
        )?);
        let paged_attn = match attention_mechanism {
            AttentionImplementation::Eager => None,
            AttentionImplementation::PagedAttention => {
                Some(PagedAttention::new(cfg.head_dim, &device, None)?)
            }
        };
        let vb_layer = vb_plain.pp("layers").pp(0);
        let vb_layer_quant = vb_quant.pp("layers").pp(0);
        let layer = DecoderLayer::load_full_attention(
            vb_layer_quant,
            vb_layer,
            cfg,
            rotary_emb,
            paged_attn,
            &comm,
        )?;
        let norm = GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_plain.pp("norm"))?;

        Ok(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            norm,
            kv_layer_idx: cfg.mtp_layer_idx(),
            device,
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Absolute paged-KV layer index the head writes to and reads from.
    pub fn kv_layer_idx(&self) -> usize {
        self.kv_layer_idx
    }

    /// One drafter forward over `rows` independent query rows laid out as `[1, rows, hidden]`.
    /// `positions` is `[3, 1, rows]` (MRoPE) and `metadata` describes each row's own paged context.
    /// Returns the normed hidden state, which is both the next chained input and the `lm_head` input.
    pub fn forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        kv_cache: (Tensor, Tensor),
        metadata: &PagedAttentionInputMetadata,
    ) -> Result<Tensor> {
        let embeds = self.pre_fc_norm_embedding.forward(input_embeds)?;
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden)?;
        let xs = self
            .fc
            .forward(&Tensor::cat(&[embeds, hidden], D::Minus1)?)?;
        let rotary_emb = self
            .layer
            .rotary_emb()
            .expect("MTP layer is a full-attention block");
        let cos_sin = rotary_emb.compute_cos_sin(positions, xs.dtype())?;
        let flash_params = FlashParams::empty(false);
        let xs = self.layer.forward_attention(
            &xs,
            &AttentionMask::None,
            &cos_sin,
            None,
            Some((kv_cache, metadata)),
            &flash_params,
        )?;
        self.norm.forward(&xs)
    }

    pub fn residual_tensors(&self, uvb: &UnVarBuilder) {
        let uvb_mtp = uvb.pp("mtp");
        uvb_mtp
            .pp("pre_fc_norm_embedding")
            .add(&self.pre_fc_norm_embedding);
        uvb_mtp
            .pp("pre_fc_norm_hidden")
            .add(&self.pre_fc_norm_hidden);
        uvb_mtp.pp("norm").add(&self.norm);
        let uvb_l = uvb_mtp.pp("layers").pp(0);
        uvb_l.pp("input_layernorm").add(&self.layer.input_layernorm);
        uvb_l
            .pp("post_attention_layernorm")
            .add(&self.layer.post_attention_layernorm);
        if let super::text::LayerImpl::FullAttention(attn) = &self.layer.layer_impl {
            uvb_l.pp("self_attn").pp("q_norm").add(&attn.q_norm);
            uvb_l.pp("self_attn").pp("k_norm").add(&attn.k_norm);
        }
    }
}
