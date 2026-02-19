#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{
    linear_no_bias, ColumnParallelLayer, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, Activation, CausalMasker, Llama3RotaryEmbedding, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    moe::{MoEExperts, MoEExpertsConfig},
    ops::{TopKLastDimOp, TopKOutput},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

use super::config::TextConfig;

struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<Llama3RotaryEmbedding>,
    max_seq_len: usize,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    norm: Option<RmsNorm>,
    use_rope: bool,
    floor_scale: Option<f32>,
    attn_scale: Option<f32>,
    attn_temperature_tuning: Option<f32>,
}

impl CausalSelfAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        layer_idx: usize,
        loading_isq: bool,
        mapper: &dyn DeviceMapper,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = ColumnParallelLayer::new(
            size_in,
            size_q,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            size_in,
            size_kv,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            size_q,
            size_in,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let use_rope = !(layer_idx + 1).is_multiple_of(4);
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let norm = if cfg.use_qk_norm && use_rope {
            let vb = mapper.set_device(layer_idx, vb, false);
            Some(RmsNorm::from_w(
                Tensor::ones(head_dim, vb.dtype(), vb.device())?,
                1e-6,
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads / comm.world_size(),
            num_key_value_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            norm,
            use_rope,
            floor_scale: cfg.floor_scale,
            attn_scale: cfg.attn_scale,
            attn_temperature_tuning: cfg.attn_temperature_tuning,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let mut q = self.q_proj.forward_autocast(x)?;
        let mut k = self.k_proj.forward_autocast(x)?;
        let mut v = self.v_proj.forward_autocast(x)?;

        q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        if self.use_rope {
            (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;
        }

        if let Some(qk_norm) = &self.norm {
            q = qk_norm.forward(&q)?;
            k = qk_norm.forward(&k)?;
        }

        if self.attn_temperature_tuning.is_some() && !self.use_rope {
            let floor_scale = self.floor_scale.unwrap() as f64;
            let attn_scale = self.attn_scale.unwrap() as f64;
            let floor = ((position_ids.to_dtype(DType::F32)? + 1.)? / floor_scale)?.floor()?;
            let attn_scales = (((floor + 1.0)?.log()? * attn_scale)? + 1.0)?;

            q = q
                .to_dtype(DType::F32)?
                .broadcast_mul(&attn_scales.unsqueeze(D::Minus1)?)?
                .to_dtype(q.dtype())?;
        }

        let mut y = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask.clone().as_ref(),
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask.clone().as_ref(),
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q.contiguous()?,
                    &k.contiguous()?,
                    &v.contiguous()?,
                    attention_mask.clone().as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        y = if attention_mask.is_some() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };
        self.o_proj.forward_autocast(&y)
    }
}

struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn new(
        vb: ShardedVarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        quantization_config: &Option<QuantizedConfig>,
        hidden_act: Activation,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up: ColumnParallelLayer::new(
                hidden_size,
                intermediate_size,
                quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down: RowParallelLayer::new(
                intermediate_size,
                hidden_size,
                quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward_autocast(xs)?;
        let rhs = self.up.forward_autocast(xs)?;

        self.down
            .forward_autocast(&crate::ops::mul_and_act(&lhs, &rhs, self.act)?)
    }
}

/// MoE layer for Llama4 using the unified MoEExperts
struct TextMoe {
    experts: MoEExperts,
    shared_expert: Mlp,
    router: Arc<dyn QuantMethod>,
    topk: usize,
}

impl TextMoe {
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        quantization_config: &Option<QuantizedConfig>,
        comm: &Arc<mistralrs_quant::Comm>,
        loading_isq: bool,
        layer_device: Device,
    ) -> Result<Self> {
        let moe_cfg = MoEExpertsConfig {
            num_experts: cfg.num_local_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: cfg.hidden_size,
            moe_intermediate_size: cfg.intermediate_size,
        };

        let experts = MoEExperts::new(
            &moe_cfg,
            vb.clone(),
            layer_device.clone(),
            comm,
            loading_isq,
            quantization_config,
            cfg.hidden_act,
        )?;

        let router = linear_no_bias(
            cfg.hidden_size,
            cfg.num_local_experts,
            quantization_config,
            vb.pp("router").set_device(layer_device),
        )?;

        let shared_expert = Mlp::new(
            vb.pp("shared_expert"),
            cfg.hidden_size,
            cfg.intermediate_size,
            quantization_config,
            cfg.hidden_act,
            comm,
        )?;

        Ok(Self {
            experts,
            shared_expert,
            router,
            topk: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = self.router.forward_autocast(&xs_flat)?;

        let TopKOutput {
            values: router_top_value,
            indices: router_indices,
        } = router_logits.topk(self.topk)?;

        let router_scores = candle_nn::ops::sigmoid(&router_top_value.to_dtype(DType::F32)?)?
            .to_dtype(router_top_value.dtype())?;

        // Forward through routed experts (is_prefill determined internally)
        let routed_out = self
            .experts
            .forward(xs, router_scores, &router_indices)?
            .reshape((bs, seq_len, hidden_dim))?;

        // Forward through shared expert and add
        let out = self.shared_expert.forward(xs)?;

        out + routed_out
    }
}

enum MoeOrMlp {
    Mlp(Mlp),
    Moe(TextMoe),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(l) => l.forward(xs),
            Self::Moe(l) => l.forward(xs),
        }
    }
}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    ff: MoeOrMlp,
    use_chunked_attention: bool,
}

impl Block {
    #[allow(clippy::too_many_arguments)]
    fn new(
        vb: ShardedVarBuilder,
        cfg: &TextConfig,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<Llama3RotaryEmbedding>,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
        real_device: Device,
    ) -> Result<Self> {
        let use_chunked_attention = !(layer_idx + 1).is_multiple_of(4);
        let attn = CausalSelfAttention::new(
            vb.pp("self_attn"),
            cfg,
            layer_idx,
            loading_isq,
            mapper,
            rope,
            paged_attn,
            comm,
        )?;
        let is_moe_layer = cfg.moe_layers().contains(&layer_idx);
        let layer_device = mapper
            .device_for(layer_idx, false)
            .cloned()
            .unwrap_or(real_device);
        let ff = if is_moe_layer {
            let moe = TextMoe::new(
                mapper.set_device(layer_idx, vb.pp("feed_forward"), loading_isq),
                cfg,
                &cfg.quantization_config,
                comm,
                loading_isq,
                layer_device,
            )?;
            MoeOrMlp::Moe(moe)
        } else {
            let mlp = Mlp::new(
                mapper.set_device(layer_idx, vb.pp("feed_forward"), loading_isq),
                cfg.hidden_size,
                cfg.intermediate_size_mlp,
                &cfg.quantization_config,
                cfg.hidden_act,
                comm,
            )?;
            MoeOrMlp::Mlp(mlp)
        };
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            ff,
            use_chunked_attention,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        position_ids: &Tensor,
        attention_mask: &Option<Tensor>,
        chunked_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mask = if self.use_chunked_attention {
            chunked_mask
        } else {
            attention_mask
        };
        let x = (self.attn.forward(
            &x,
            position_ids,
            mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )? + residual)?;
        let residual = &x;
        let x = (self.ff.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }
}

pub struct TextModel {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    kv_cache: crate::pipeline::EitherCache,
    device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
    attention_chunk_size: usize,
}

impl TextModel {
    pub fn new(
        cfg: &TextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &TextConfig,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(wte.embeddings(), normal_loading_metadata.loading_isq)?,
                None,
            ))?
        };
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut ropes = HashMap::new();
        for i in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Llama3RotaryEmbedding::new_llama4(
                    vb_m.dtype(),
                    cfg,
                    device,
                    is_gptx,
                )?),
            );
        }
        let blocks = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading text repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .par_iter_if_isq(|i| {
            let device = mapper
                .device_for(i, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => {
                    Some(PagedAttention::new(head_dim, device, None)?)
                }
            };
            let comm = mapper.get_comm_for(i)?;
            Block::new(
                vb_m.pp(format!("layers.{i}")),
                cfg,
                &*mapper,
                i,
                normal_loading_metadata.loading_isq,
                rotary_emb,
                paged_attn,
                &comm,
                normal_loading_metadata.real_device.clone(),
            )
        })?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            kv_cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: None,
                k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            mapper,
            attention_chunk_size: cfg.attention_chunk_size,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.wte.forward(input_ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut x = input_embeds;
        let cache = &mut self.kv_cache.normal().0;
        let cache_for_mask = metadata
            .as_ref()
            .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
            .unwrap_or(cache as &dyn PastKvLenCache);

        let position_ids = Tensor::new(
            seqlen_offsets.iter().map(|o| *o as i32).collect::<Vec<_>>(),
            input_ids.device(),
        )?;

        let mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            cache_for_mask,
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
        let chunked_mask = CausalMasker.make_chunked_mask_matrix(
            input_ids,
            self.attention_chunk_size,
            cache_for_mask,
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
        // PagedAttention prompt chunking
        let mask = mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        // PagedAttention prompt chunking
        let chunked_mask = chunked_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        let mask = DeviceMappedMask::new(mask, &*self.mapper)?;
        let chunked_mask = DeviceMappedMask::new(chunked_mask, &*self.mapper)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            let mask_for_layer = mask.as_ref().map(|m| m.get(x.device()).clone());
            let chunked_mask_for_layer = chunked_mask.as_ref().map(|m| m.get(x.device()).clone());
            x = block.forward(
                &x,
                &position_ids.to_device(x.device())?,
                &mask_for_layer,
                &chunked_mask_for_layer,
                seqlen_offsets,
                &mut cache[block_idx],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[block_idx].clone(), *metadata)),
                flash_params,
            )?;
        }
        let x = x.to_device(&self.device)?;
        let x = self.ln_f.forward(&x)?;
        let x = extract_logits(&x, context_lens)?;
        self.lm_head.forward_autocast(&x)
    }

    pub fn residual_tensors_m(&self, uvb_m: UnVarBuilder) -> Vec<(String, Tensor)> {
        uvb_m.pp("embed_tokens").add(&self.wte);
        uvb_m.pp("norm").add(&self.ln_f);

        for (layer_idx, layer) in self.blocks.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.rms_1);
            uvb_l.pp("post_attention_layernorm").add(&layer.rms_2);
        }

        uvb_m.to_safetensors()
    }
}

impl IsqModel for TextModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            match &mut layer.ff {
                MoeOrMlp::Mlp(x) => {
                    tensors.push((&mut x.gate, Some(i)));
                    tensors.push((&mut x.up, Some(i)));
                    tensors.push((&mut x.down, Some(i)));
                }
                MoeOrMlp::Moe(x) => {
                    tensors.push((&mut x.router, Some(i)));
                    for layer in x.experts.get_isq_layers() {
                        tensors.push((layer, Some(i)));
                    }
                    tensors.push((&mut x.shared_expert.gate, Some(i)));
                    tensors.push((&mut x.shared_expert.up, Some(i)));
                    tensors.push((&mut x.shared_expert.down, Some(i)));
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.residual_tensors_m(uvb.pp("model"))
    }
}

impl NormalModel for TextModel {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _seqlen_offsets: &[usize],
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
    ) -> Result<Tensor> {
        unreachable!()
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &crate::pipeline::EitherCache {
        &self.kv_cache
    }
    fn cache_mut(&mut self) -> &mut crate::pipeline::EitherCache {
        &mut self.kv_cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.blocks[0].attn.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for TextModel {}
