//! Multi-token prediction head shipped inside Qwen4-Exp checkpoints (`mtp.*`).
//!
//! Mirrors vLLM's `Qwen4ExpMultiTokenPredictor`: `fc_embedding(norm(embed(next_token)))` is added to every
//! stream of `fc_hidden(norm(target_hidden))`, then one QSA + MoE block with hyper-connections runs and the
//! head's own final mixer collapses the streams for `lm_head`. The pre-mixer 4-stream residual feeds the next
//! chained draft.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::{
    config::{QsaConfig, TextConfig},
    hyper::GatedResidual,
    qsa::{QsaAttention, QsaMode, QsaStep},
    text::SparseMoeBlock,
};
use crate::{
    device_map::DeviceMapper,
    layers::{GemmaRmsNorm, Qwen3VLRotaryEmbedding},
    paged_attention::AttentionImplementation,
    speculative::builtin_mtp::{MtpAttentionInputs, MtpDraftOutput},
    utils::unvarbuilder::UnVarBuilder,
};

pub const MTP_FC_WEIGHT: &str = "mtp.fc_embedding.weight";

pub(super) struct Qwen4ExpMtpHead {
    pre_fc_norm_embedding: GemmaRmsNorm,
    pre_fc_norm_hidden: GemmaRmsNorm,
    fc_embedding: Arc<dyn QuantMethod>,
    fc_hidden: Arc<dyn QuantMethod>,
    attn: QsaAttention,
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    moe: SparseMoeBlock,
    mixer: GatedResidual,
    rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    qsa: QsaConfig,
    max_seq_len: usize,
    kv_layer_idx: usize,
    hc: usize,
    hidden: usize,
    device: Device,
    dtype: DType,
}

impl Qwen4ExpMtpHead {
    /// `vb` is the checkpoint root; the head lives on the non-mapped device with `lm_head`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        vb: &ShardedVarBuilder,
        cfg: &TextConfig,
        qsa: QsaConfig,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
        device: &Device,
        attention_mechanism: &AttentionImplementation,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    ) -> Result<Self> {
        if !crate::layers::contains_tensor_or_uqff(vb, MTP_FC_WEIGHT) {
            candle_core::bail!(
                "`--mtp` requested but the checkpoint has no built-in MTP head (`{MTP_FC_WEIGHT}`)."
            );
        }
        if cfg.mtp_num_hidden_layers != 1 {
            candle_core::bail!(
                "Qwen4-Exp MTP supports exactly one MTP layer, config has {}",
                cfg.mtp_num_hidden_layers
            );
        }
        if cfg.mtp_use_dedicated_embeddings {
            candle_core::bail!("Qwen4-Exp MTP with dedicated embeddings is not supported");
        }
        let device = device.clone();
        let vb_mtp = vb.pp("mtp");
        let vb_quant = mapper.set_nm_device(vb_mtp.clone(), loading_isq);
        let vb_plain = mapper.set_nm_device(vb_mtp, false);
        let kv_layer_idx = cfg.num_hidden_layers;
        let comm = mapper.get_comm_for(kv_layer_idx)?;
        let hc_hidden = cfg.hc_hidden_size();

        let linear = |vb: ShardedVarBuilder| {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                vb,
            )
        };
        let vb_layer_quant = vb_quant.pp("layers").pp(0);
        let vb_layer = vb_plain.pp("layers").pp(0);
        Ok(Self {
            pre_fc_norm_embedding: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb_plain.pp("pre_fc_norm_embedding"),
            )?,
            pre_fc_norm_hidden: GemmaRmsNorm::new(
                hc_hidden,
                cfg.rms_norm_eps,
                vb_plain.pp("pre_fc_norm_hidden"),
            )?,
            fc_embedding: linear(vb_quant.pp("fc_embedding"))?,
            fc_hidden: linear(vb_quant.pp("fc_hidden"))?,
            attn: QsaAttention::load(
                vb_layer_quant.pp("self_attn"),
                vb_layer.pp("self_attn"),
                cfg,
                qsa,
                rotary_emb.clone(),
                matches!(attention_mechanism, AttentionImplementation::PagedAttention),
                &comm,
            )?,
            attn_hc: GatedResidual::load(cfg, vb_layer.pp("attn_hyper_connection"), true)?,
            mlp_hc: GatedResidual::load(cfg, vb_layer.pp("mlp_hyper_connection"), true)?,
            moe: SparseMoeBlock::new(
                cfg,
                vb_layer_quant.pp("mlp"),
                device.clone(),
                loading_isq,
                &comm,
            )?,
            mixer: GatedResidual::load(cfg, vb_plain.pp("hyper_connection_mixer"), false)?,
            rotary_emb,
            qsa,
            max_seq_len: cfg.max_position_embeddings,
            kv_layer_idx,
            hc: cfg.hc_count,
            hidden: cfg.hidden_size,
            device,
            dtype: vb.dtype(),
        })
    }

    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    pub(super) fn dtype(&self) -> DType {
        self.dtype
    }

    /// Absolute paged-KV layer index the head writes to and reads from.
    pub(super) fn kv_layer_idx(&self) -> usize {
        self.kv_layer_idx
    }

    /// One drafter forward over `[b, rows, hidden]` embeddings and `[b, rows, hc * hidden]` target (or chained)
    /// residuals with `[3, b, rows]` MRoPE positions.
    pub(super) fn forward(
        &self,
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
        attention: MtpAttentionInputs<'_>,
    ) -> Result<MtpDraftOutput> {
        let (batch, rows, _) = input_embeds.dims3()?;
        let embeds = self
            .fc_embedding
            .forward(&self.pre_fc_norm_embedding.forward(input_embeds)?)?;
        let hidden = self.pre_fc_norm_hidden.forward(target_hidden)?.reshape((
            batch,
            rows * self.hc,
            self.hidden,
        ))?;
        let hidden =
            self.fc_hidden
                .forward(&hidden)?
                .reshape((batch, rows, self.hc, self.hidden))?;
        let res = hidden.broadcast_add(&embeds.unsqueeze(2)?)?.reshape((
            batch,
            rows,
            self.hc * self.hidden,
        ))?;

        let cos_sin = self.rotary_emb.compute_cos_sin(positions, res.dtype())?;
        let n_tokens = batch * rows;
        let (mode, spans, kv_lens) = match attention.chunk_ranges {
            Some(ranges) => {
                let spans = ranges
                    .iter()
                    .enumerate()
                    .map(|(b, (start, end))| (b * rows, end - start))
                    .collect::<Vec<_>>();
                let kv_lens = ranges.iter().map(|(_, end)| *end).collect::<Vec<_>>();
                let max_kv = kv_lens.iter().copied().max().unwrap_or(0);
                (QsaMode::for_max_kv_len(&self.qsa, max_kv), spans, kv_lens)
            }
            None => (
                QsaMode::Sparse {
                    max_blocks: self.max_seq_len / self.qsa.compress_ratio,
                },
                Vec::new(),
                Vec::new(),
            ),
        };
        #[cfg(feature = "cuda")]
        let layout = if !res.device().is_cuda() {
            None
        } else if attention.chunk_ranges.is_some() {
            Some(crate::cuda::qwen4_exp::TokenLayout::from_host(
                &spans,
                &kv_lens,
                n_tokens,
                res.device(),
            )?)
        } else {
            Some(crate::cuda::qwen4_exp::TokenLayout::rectangular(
                n_tokens, 1,
            ))
        };
        #[cfg(not(feature = "cuda"))]
        let _ = (spans, kv_lens, n_tokens);
        let step = QsaStep {
            mode,
            positions: &[],
            #[cfg(feature = "cuda")]
            layout: layout.as_ref(),
        };

        let xn = self.attn_hc.norm(&res)?;
        let h = self.attn_hc.mix(&xn)?;
        let inject = self.attn_hc.inject(&xn)?;
        let out = self.attn.forward(
            &h,
            attention.attention_mask,
            &cos_sin,
            None,
            Some((attention.kv_cache, attention.metadata)),
            attention.flash_params,
            &step,
        )?;
        let (res, xn) = self
            .attn_hc
            .combine(&res, &out, &inject, Some(&self.mlp_hc))?;
        let xn = match xn {
            Some(xn) => xn,
            None => self.mlp_hc.norm(&res)?,
        };
        let h = self.mlp_hc.mix(&xn)?;
        let inject = self.mlp_hc.inject(&xn)?;
        let out = self.moe.forward(&h, "mtp_draft")?;
        let (res, xn) = self
            .mlp_hc
            .combine(&res, &out, &inject, Some(&self.mixer))?;
        let xn = match xn {
            Some(xn) => xn,
            None => self.mixer.norm(&res)?,
        };
        Ok(MtpDraftOutput {
            logits_hidden: self.mixer.mix(&xn)?,
            chain_hidden: Some(res),
        })
    }

    pub(super) fn residual_tensors(&self, uvb: &UnVarBuilder) {
        let uvb_mtp = uvb.pp("mtp");
        uvb_mtp
            .pp("pre_fc_norm_embedding")
            .add(&self.pre_fc_norm_embedding);
        uvb_mtp
            .pp("pre_fc_norm_hidden")
            .add(&self.pre_fc_norm_hidden);
        let add_hc = |uvb: &UnVarBuilder, hc: &GatedResidual| {
            uvb.pp("hc_norm")
                .add_tensor("weight", (&hc.norm_weight - 1.0).expect("hc norm weight"));
        };
        add_hc(&uvb_mtp.pp("hyper_connection_mixer"), &self.mixer);
        let uvb_l = uvb_mtp.pp("layers").pp(0);
        add_hc(&uvb_l.pp("attn_hyper_connection"), &self.attn_hc);
        add_hc(&uvb_l.pp("mlp_hyper_connection"), &self.mlp_hc);
        let sa = uvb_l.pp("self_attn");
        sa.pp("q_norm").add(&self.attn.q_norm);
        sa.pp("k_norm").add(&self.attn.k_norm);
        sa.pp("indexer")
            .pp("q_layernorm")
            .add(&self.attn.index_q_norm);
        sa.pp("indexer")
            .pp("k_layernorm")
            .add(&self.attn.index_k_norm);
        self.moe.add_residual_tensors(&uvb_l.pp("mlp"));
    }
}
