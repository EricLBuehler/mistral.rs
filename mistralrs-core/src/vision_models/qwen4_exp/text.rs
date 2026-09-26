#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use mistralrs_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};

use super::{
    config::{LayerType, QsaConfig, TextConfig},
    hyper::GatedResidual,
    ple::{PleBatch, PleLayer, PleState},
    qsa::{aux_dim, QsaAttention, QsaMode, QsaStep},
};
use crate::{
    attention::AttentionMask,
    device_map::{DeviceMappedMask, DeviceMapper},
    gdn::{
        GatedDeltaNet, GdnConfig, GdnGateActivation, GdnInputProjectionKind, GdnLayerCache,
        GdnVHeadLayout,
    },
    kv_cache::{
        HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType, RecurrentLayerConfig,
        RecurrentStateSpec,
    },
    layers::{self, Qwen3VLRotaryEmbedding},
    moe::{MoEExperts, MoEExpertsConfig},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{EitherCache, IsqModel, ModelForwardContext, NormalLoadingMetadata},
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
    vision_models::qwen3_5::packed_gdn::{forward_packed_gdn, packed_gdn_layout},
};

impl GdnConfig for TextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }
    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }
    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }
    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }
    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }
    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }
    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }
    fn v_head_layout(&self) -> GdnVHeadLayout {
        self.gdn_v_head_layout
    }
    fn output_gate_activation(&self) -> GdnGateActivation {
        self.output_gate_activation()
            .expect("output gate activation is validated at load")
    }
}

struct SparseMoeBlock {
    gate: Linear,
    experts: MoEExperts,
    shared_expert: layers::Mlp,
    shared_expert_gate: Linear,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        layer_device: Device,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            cfg.num_experts,
            vb.pp("gate").set_device(layer_device.clone()),
        )?;
        let experts = MoEExperts::new(
            &MoEExpertsConfig {
                num_experts: cfg.num_experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                hidden_size: cfg.hidden_size,
                moe_intermediate_size: cfg.moe_intermediate_size,
                expert_proj_names: crate::moe::ExpertProjNames::DEFAULT,
            },
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            &cfg.quantization_config,
            cfg.hidden_act,
        )?;
        let shared_expert = layers::Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.shared_expert_intermediate_size,
            &cfg.quantization_config,
            cfg.hidden_act,
            comm,
        )?;
        let shared_expert_gate = Linear::new(
            vb.pp("shared_expert_gate")
                .get((1, cfg.hidden_size), "weight")?
                .to_device(&layer_device)?,
            None,
        );
        Ok(Self {
            gate,
            experts,
            shared_expert,
            shared_expert_gate,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = self.gate.forward(&xs_flat)?;
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        let y = self
            .experts
            .forward(xs, topk.values, &topk.indices)?
            .reshape((b_size, seq_len, hidden_dim))?;
        let shared_gate = candle_nn::ops::sigmoid(&self.shared_expert_gate.forward(&xs_flat)?)?
            .reshape((b_size, seq_len, 1))?;
        let shared_out = self
            .shared_expert
            .forward(xs)?
            .broadcast_mul(&shared_gate)?;
        y + shared_out
    }
}

enum Mixer {
    Attention(QsaAttention),
    Linear(GatedDeltaNet),
}

struct DecoderLayer {
    mixer: Mixer,
    ple: Option<PleLayer>,
    attn_hc: GatedResidual,
    mlp_hc: GatedResidual,
    moe: SparseMoeBlock,
}

/// Recurrent-pool layer index that holds the PLE state (one past the decoder layers).
fn ple_state_layer(cfg: &TextConfig) -> usize {
    cfg.num_hidden_layers
}

pub struct Qwen4ExpTextModel {
    embed_tokens: Arc<dyn QuantMethod>,
    layers: Vec<DecoderLayer>,
    final_mixer: GatedResidual,
    pub(super) layer_types: Vec<LayerType>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
    rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    qsa: QsaConfig,
    hc: usize,
    pub(super) cache: EitherCache,
    pub(super) cfg: ModelConfigMetadata,
    pub(super) device: Device,
    pub(super) dtype: DType,
    pub(super) max_seq_len: usize,
    pub(super) aux_dim: usize,
    pub(super) dense_kv_cap: usize,
}

fn text_model_vb(vb: &ShardedVarBuilder) -> ShardedVarBuilder {
    if layers::contains_tensor_or_uqff(vb, "language_model.model.embed_tokens.weight") {
        vb.pp("language_model").pp("model")
    } else {
        vb.pp("model").pp("language_model")
    }
}

impl Qwen4ExpTextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        tie: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        cfg.validate()?;
        let qsa = cfg.qsa()?.ok_or_else(|| {
            candle_core::Error::msg("Qwen4-Exp checkpoints without a QSA indexer are unsupported")
        })?;
        let ple = cfg.ple()?;
        let mapper = normal_loading_metadata.mapper;
        let vb_m = text_model_vb(&vb);
        let real_device = normal_loading_metadata.real_device.clone();

        let embed_tokens = layers::embedding_with_legacy_tied_uqff(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), normal_loading_metadata.loading_isq),
            tie.then(|| {
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq)
            }),
            &cfg.quantization_config,
        )?;

        let layer_types = cfg.layer_types();
        let yarn = cfg.yarn_rope_config()?;
        let make_rope = |device: &Device| -> Result<Qwen3VLRotaryEmbedding> {
            match yarn.as_ref() {
                Some(yarn) => {
                    Qwen3VLRotaryEmbedding::new_yarn(yarn, device, cfg.mrope_section().to_vec())
                }
                None => Qwen3VLRotaryEmbedding::new(
                    cfg.rope_theta() as f32,
                    cfg.rot_dim(),
                    device,
                    cfg.mrope_section().to_vec(),
                ),
            }
        };
        let rotary_emb = Arc::new(make_rope(&real_device)?);
        let mut ropes = HashMap::new();
        for (layer_idx, layer_type) in layer_types.iter().enumerate() {
            if *layer_type != LayerType::FullAttention {
                continue;
            }
            let device = mapper.device_for(layer_idx, false).unwrap_or(&real_device);
            if let std::collections::hash_map::Entry::Vacant(entry) = ropes.entry(device.location())
            {
                entry.insert(Arc::new(make_rope(device)?));
            }
        }

        let vb_l = vb_m.pp("layers");
        let use_paged = matches!(attention_mechanism, AttentionImplementation::PagedAttention);
        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|layer_idx| {
            let comm = mapper.get_comm_for(layer_idx)?;
            let layer_device = mapper
                .device_for(layer_idx, false)
                .cloned()
                .unwrap_or(real_device.clone());
            let vb_layer = vb_l.pp(layer_idx);
            let mixer = match layer_types[layer_idx] {
                LayerType::FullAttention => Mixer::Attention(QsaAttention::load(
                    vb_layer.clone(),
                    cfg,
                    qsa,
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    ropes
                        .get(&layer_device.location())
                        .expect("rope per attention device")
                        .clone(),
                    use_paged,
                    &comm,
                )?),
                LayerType::LinearAttention => Mixer::Linear(GatedDeltaNet::load(
                    vb_layer.clone(),
                    cfg as &dyn GdnConfig,
                    &*mapper,
                    layer_idx,
                    normal_loading_metadata.loading_isq,
                    &comm,
                    GdnInputProjectionKind::Split,
                )?),
            };
            let vb_plain = mapper.set_device(layer_idx, vb_layer.clone(), false);
            let ple = match &ple {
                Some(ple_cfg) if ple_cfg.layer_idx == layer_idx => {
                    Some(PleLayer::load(cfg, ple_cfg, vb_plain.pp("ple"))?)
                }
                _ => None,
            };
            let moe = SparseMoeBlock::new(
                cfg,
                mapper.set_device(
                    layer_idx,
                    vb_layer.pp("mlp"),
                    normal_loading_metadata.loading_isq,
                ),
                layer_device,
                normal_loading_metadata.loading_isq,
                &comm,
            )?;
            Ok(DecoderLayer {
                mixer,
                ple,
                attn_hc: GatedResidual::load(cfg, vb_plain.pp("attn_hyper_connection"), true)?,
                mlp_hc: GatedResidual::load(cfg, vb_plain.pp("mlp_hyper_connection"), true)?,
                moe,
            })
        })?;

        let mut layers = layers;
        for (layer_idx, layer) in layers.iter_mut().enumerate() {
            if let Some(ple) = layer.ple.as_mut() {
                let device = mapper.device_for(layer_idx, false).unwrap_or(&real_device);
                ple.make_table_resident(device)?;
            }
        }
        let final_mixer = GatedResidual::load(
            cfg,
            mapper.set_nm_device(vb_m.pp("hyper_connection_mixer"), false),
            false,
        )?;
        let lm_head = if !tie {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
            )?
        } else {
            embed_tokens.clone()
        };

        let mut hybrid_layer_types = layer_types
            .iter()
            .map(|lt| match lt {
                LayerType::FullAttention => HybridLayerType::Attention,
                LayerType::LinearAttention => HybridLayerType::Recurrent,
            })
            .collect::<Vec<_>>();
        let mut layer_devices = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&real_device)
                    .clone()
            })
            .collect::<Vec<_>>();
        let mut overrides = HashMap::new();
        if let Some(ple_cfg) = &ple {
            hybrid_layer_types.push(HybridLayerType::Recurrent);
            layer_devices.push(layer_devices[ple_cfg.layer_idx].clone());
            overrides.insert(
                ple_state_layer(cfg),
                RecurrentLayerConfig {
                    conv_dim: cfg.hc_hidden_size(),
                    conv_width: ple_cfg.conv_state_len(),
                    state: RecurrentStateSpec::Opaque {
                        dims: vec![ple_cfg.context_len()],
                    },
                    recurrent_dtype: Some(DType::F32),
                },
            );
        }
        let hybrid_cache = HybridCache::new_with_recurrent_overrides(
            HybridCacheConfig {
                layer_types: hybrid_layer_types,
                max_seq_len: cfg.max_position_embeddings,
                recurrent: RecurrentLayerConfig {
                    conv_dim: cfg.linear_conv_dim(),
                    conv_width: cfg.linear_conv_kernel_dim,
                    state: RecurrentStateSpec::Gdn {
                        heads: cfg.linear_num_value_heads,
                        key_dim: cfg.linear_key_head_dim,
                        value_dim: cfg.linear_value_head_dim,
                    },
                    recurrent_dtype: Some(cfg.mamba_ssm_dtype.dtype()),
                },
            },
            vb_m.dtype(),
            &layer_devices,
            &overrides,
        )
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create hybrid cache: {e}")))?;

        Ok(Self {
            embed_tokens,
            layers,
            final_mixer,
            layer_types,
            rotary_emb,
            qsa,
            aux_dim: aux_dim(&qsa, cfg.rot_dim() / 2),
            dense_kv_cap: qsa.max_selected_tokens(),
            hc: cfg.hc_count,
            lm_head,
            cache: EitherCache::Hybrid(Arc::new(Mutex::new(hybrid_cache))),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: real_device,
            dtype: vb.dtype(),
            mapper,
        })
    }

    /// The decode step is graph-capturable unless PLE rows must be gathered on the host.
    #[cfg(feature = "cuda")]
    pub fn supports_decode_graphs(&self) -> bool {
        self.layers
            .iter()
            .filter_map(|layer| layer.ple.as_ref())
            .all(PleLayer::gathers_on_device)
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding_forward(input_ids, self.dtype)
    }

    /// `(start, len)` of every logical sequence on the flattened token axis of `xs: [b, s, h]`.
    fn sequence_spans(xs: &Tensor, ctx: &ModelForwardContext<'_>) -> Result<Vec<(usize, usize)>> {
        let (batch, seq_len, _) = xs.dims3()?;
        let query_lens = ctx
            .paged_input_metadata()
            .and_then(|metadata| metadata.query_lens.clone());
        if ctx.flash_params().packed {
            let query_lens = query_lens.ok_or_else(|| {
                candle_core::Error::msg("packed Qwen4-Exp prefill requires query lengths")
            })?;
            let mut start = 0;
            return Ok(query_lens
                .into_iter()
                .map(|len| {
                    let span = (start, len);
                    start += len;
                    span
                })
                .collect());
        }
        Ok((0..batch)
            .map(|b| {
                let len = query_lens
                    .as_ref()
                    .and_then(|lens| lens.get(b).copied())
                    .unwrap_or(seq_len)
                    .min(seq_len);
                (b * seq_len, len)
            })
            .collect())
    }

    /// Kv length after this step of every logical sequence.
    fn kv_lens(spans: &[(usize, usize)], ctx: &ModelForwardContext<'_>) -> Vec<usize> {
        if let Some(lens) = ctx
            .paged_input_metadata()
            .and_then(|metadata| metadata.full_paged_context_lens_cpu.clone())
            .filter(|lens| lens.len() == spans.len())
        {
            return lens;
        }
        let offsets = ctx.seqlen_offsets();
        spans
            .iter()
            .enumerate()
            .map(|(i, (_, len))| offsets.get(i).copied().unwrap_or(0) + len)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        xs: Tensor,
        input_ids: &Tensor,
        attention_mask: &AttentionMask,
        position_ids: &Tensor,
        ctx: &ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut hybrid_cache = self.cache.hybrid();
        let recurrent_metadata = ctx
            .recurrent_metadata()
            .cloned()
            .ok_or_else(|| candle_core::Error::msg("Qwen4-Exp requires recurrent metadata"))?;
        let packed_layout = packed_gdn_layout(&xs, ctx)?;
        let cos_sin = self.rotary_emb.compute_cos_sin(position_ids, xs.dtype())?;
        let attention_mask = DeviceMappedMask::new(attention_mask.clone(), &*self.mapper)?;

        let is_decode = ctx
            .paged_input_metadata()
            .is_some_and(|metadata| metadata.is_decode_step());
        let (batch, seq_len, _) = xs.dims3()?;
        let spans = if is_decode {
            (0..batch).map(|b| (b, 1)).collect::<Vec<_>>()
        } else {
            Self::sequence_spans(&xs, ctx)?
        };
        let kv_lens = Self::kv_lens(&spans, ctx);
        let mode = if is_decode {
            QsaMode::Sparse {
                max_blocks: self.max_seq_len / self.qsa.compress_ratio,
            }
        } else {
            QsaMode::for_max_kv_len(&self.qsa, kv_lens.iter().copied().max().unwrap_or(0))
        };
        let mut positions = vec![None; batch * seq_len];
        if !ctx.is_paged() {
            for (&(start, len), &kv_len) in spans.iter().zip(&kv_lens) {
                for local in 0..len {
                    positions[start + local] = Some(kv_len - len + local);
                }
            }
        }
        #[cfg(feature = "cuda")]
        let layout = if xs.device().is_cuda() {
            Some(if is_decode {
                crate::cuda::qwen4_exp::TokenLayout::decode(batch)
            } else {
                crate::cuda::qwen4_exp::TokenLayout::from_host(
                    &spans,
                    &kv_lens,
                    batch * seq_len,
                    xs.device(),
                )?
            })
        } else {
            None
        };
        let step = QsaStep {
            mode,
            positions: &positions,
            #[cfg(feature = "cuda")]
            layout: layout.as_ref(),
        };

        let mut res = xs.repeat((1, 1, self.hc))?;
        let mut pending_norm: Option<Tensor> = None;
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter().enumerate() {
            res = self.mapper.map(res, i)?;
            if let Some(ple) = &layer.ple {
                let indices = hybrid_cache
                    .state_indices_for_layer(n_layers)?
                    .ok_or_else(|| candle_core::Error::msg("PLE state indices are missing"))?;
                let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(n_layers) else {
                    candle_core::bail!("Qwen4-Exp PLE state pool is missing");
                };
                let state = PleState {
                    conv: &pool.conv_state,
                    history: &pool.recurrent_state,
                    slots: &indices,
                };
                res = ple.forward(
                    &res,
                    &input_ids.to_device(res.device())?,
                    &state,
                    &PleBatch {
                        seqs: &spans,
                        #[cfg(feature = "cuda")]
                        layout: layout.as_ref(),
                    },
                )?;
                pending_norm = None;
            }
            let xn = match pending_norm.take() {
                Some(xn) if xn.device().same_device(res.device()) => xn,
                _ => layer.attn_hc.norm(&res)?,
            };
            let h = layer.attn_hc.mix(&xn)?;
            let inject = layer.attn_hc.inject(&xn)?;
            let out = match &layer.mixer {
                Mixer::Attention(attn) => {
                    let Some(HybridLayerCache::Attention(kv_cache)) = hybrid_cache.get_mut(i)
                    else {
                        candle_core::bail!("Hybrid cache layer {i} is not an attention cache");
                    };
                    attn.forward(
                        &h,
                        &attention_mask.get(h.device()),
                        &cos_sin,
                        kv_cache,
                        ctx.paged_layer(i),
                        ctx.flash_params(),
                        &step,
                    )?
                }
                Mixer::Linear(gdn) => {
                    let indices = hybrid_cache.state_indices_for_layer(i)?.ok_or_else(|| {
                        candle_core::Error::msg(format!(
                            "Hybrid cache layer {i} has no state indices"
                        ))
                    })?;
                    let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(i) else {
                        candle_core::bail!("Hybrid cache layer {i} is not recurrent");
                    };
                    let mut gdn_cache = if packed_layout.is_some() {
                        GdnLayerCache::gathered(
                            pool.gather_conv_state(&indices)?,
                            pool.gather_recurrent_state(&indices)?,
                            pool.state_layout(),
                        )
                    } else {
                        GdnLayerCache::checkout(pool, &indices)?
                    };
                    let out = match &packed_layout {
                        Some(packed) => forward_packed_gdn(
                            gdn,
                            &h,
                            &mut gdn_cache,
                            recurrent_metadata.batch_kind(),
                            packed,
                        )?,
                        None => gdn.forward(&h, &mut gdn_cache, recurrent_metadata.batch_kind())?,
                    };
                    gdn_cache.commit(pool, &indices, recurrent_metadata.state_indices_host())?;
                    out
                }
            };
            let (next_res, xn) = layer
                .attn_hc
                .combine(&res, &out, &inject, Some(&layer.mlp_hc))?;
            res = next_res;
            let xn = match xn {
                Some(xn) => xn,
                None => layer.mlp_hc.norm(&res)?,
            };
            let h = layer.mlp_hc.mix(&xn)?;
            let inject = layer.mlp_hc.inject(&xn)?;
            let out = layer.moe.forward(&h)?;
            let next = match self.layers.get(i + 1) {
                Some(next) if next.ple.is_none() => Some(&next.attn_hc),
                Some(_) => None,
                None => Some(&self.final_mixer),
            };
            let (next_res, xn) = layer.mlp_hc.combine(&res, &out, &inject, next)?;
            res = next_res;
            pending_norm = xn;
        }
        let res = res.to_device(&self.device)?;
        let xn = match pending_norm {
            Some(xn) if xn.device().same_device(&self.device) => xn,
            _ => self.final_mixer.norm(&res)?,
        };
        let xn = ctx.logits(&xn)?;
        let xs = self.final_mixer.mix(&xn)?;
        ctx.lm_head(&*self.lm_head, &xs)
    }
}

impl IsqModel for Qwen4ExpTextModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        let uvb_lm = uvb.pp("model").pp("language_model");
        uvb_lm.pp("embed_tokens").add(&self.embed_tokens);
        let add_hc = |uvb: &UnVarBuilder, hc: &GatedResidual| {
            uvb.pp("hc_norm")
                .add_tensor("weight", (&hc.norm_weight - 1.0).expect("hc norm weight"));
        };
        add_hc(&uvb_lm.pp("hyper_connection_mixer"), &self.final_mixer);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_lm.pp("layers").pp(layer_idx);
            add_hc(&uvb_l.pp("attn_hyper_connection"), &layer.attn_hc);
            add_hc(&uvb_l.pp("mlp_hyper_connection"), &layer.mlp_hc);
            match &layer.mixer {
                Mixer::Attention(attn) => {
                    let sa = uvb_l.pp("self_attn");
                    sa.pp("q_norm").add(&attn.q_norm);
                    sa.pp("k_norm").add(&attn.k_norm);
                    sa.pp("indexer").pp("q_layernorm").add(&attn.index_q_norm);
                    sa.pp("indexer").pp("k_layernorm").add(&attn.index_k_norm);
                }
                Mixer::Linear(gdn) => {
                    let la = uvb_l.pp("linear_attn");
                    la.add_tensor("conv1d.weight", gdn.conv1d_weight.clone());
                    la.add_tensor("dt_bias", gdn.dt_bias.clone());
                    la.add_tensor("A_log", gdn.a_log.clone());
                    la.pp("norm").add_tensor("weight", gdn.norm.weight.clone());
                }
            }
            uvb_l
                .pp("mlp")
                .pp("gate")
                .add_tensor("weight", layer.moe.gate.weight().clone());
            uvb_l
                .pp("mlp")
                .pp("shared_expert_gate")
                .add_tensor("weight", layer.moe.shared_expert_gate.weight().clone());
        }
        uvb.to_safetensors()
    }
}
