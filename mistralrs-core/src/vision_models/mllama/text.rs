#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Embedding, Module};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use crate::{
    attention::SdpaParams,
    device_map::{DeviceMappedMask, DeviceMapper},
    layers::{embedding, CausalMasker, Llama3RotaryEmbedding, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        extract_logits, EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
};

use super::config::MLlamaTextConfig;

struct MLlamaTextMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl MLlamaTextMlp {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                cfg.intermediate_size,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("down_proj"),
            )?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut res = self.down_proj.forward(
            &self
                .act
                .forward(&self.gate_proj.forward(&xs)?)?
                .broadcast_mul(&self.up_proj.forward(&xs)?)?,
        )?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MLlamaTextSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    sdpa_params: SdpaParams,
    rope: Arc<Llama3RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl MLlamaTextSelfAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            rope,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let mut hidden_states = hidden_states.clone();
        let original_dtype = hidden_states.dtype();
        if let Some(t) = self.q_proj.quantized_act_type() {
            hidden_states = hidden_states.to_dtype(t)?;
        }
        let mut q = self.q_proj.forward(&hidden_states)?;
        let mut k = self.k_proj.forward(&hidden_states)?;
        let mut v = self.v_proj.forward(&hidden_states)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let (q, k, mut v) = if q_len != 1 {
            let q = q
                .reshape((bs, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((bs, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((bs, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((bs, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, mut k) = self.rope.forward(&q, &k, seqlen_offsets)?;

        (k, v) = kv_cache.append(&k, &v)?;

        let mut attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attention_mask,
                None,
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_len, ()))?
            .to_dtype(q.dtype())?;

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.o_proj.forward(&attn_output)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MLlamaSelfAttentionDecoderLayer {
    attn: MLlamaTextSelfAttention,
    mlp: MLlamaTextMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MLlamaSelfAttentionDecoderLayer {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
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
        let attn = MLlamaTextSelfAttention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            rope,
            comm,
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states =
            self.attn
                .forward(&hidden_states, attention_mask, seqlen_offsets, kv_cache)?;
        hidden_states = (residual + hidden_states)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;

        residual + hidden_states
    }
}

struct MLlamaTextCrossAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sdpa_params: SdpaParams,
}

impl MLlamaTextCrossAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("q_proj"),
            )?,
            k_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("k_proj"),
            )?,
            v_proj: ColumnParallelLayer::new(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("v_proj"),
            )?,
            o_proj: RowParallelLayer::new(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                comm,
                vb.pp("o_proj"),
            )?,
            q_norm: RmsNorm::new(
                cfg.head_dim(),
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("q_norm"), false),
            )?,
            k_norm: RmsNorm::new(
                cfg.head_dim(),
                cfg.rms_norm_eps,
                mapper.set_device(layer_idx, vb.pp("k_norm"), false),
            )?,
            num_heads: cfg.num_attention_heads / comm.world_size(),
            num_kv_heads: (cfg.num_key_value_heads / comm.world_size()).max(1),
            head_dim: cfg.head_dim(),
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (cfg.head_dim() as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (bs, q_len, _) = hidden_states.dims3()?;

        let mut hidden_states = hidden_states.clone();
        let original_dtype = hidden_states.dtype();
        if let Some(t) = self.q_proj.quantized_act_type() {
            hidden_states = hidden_states.to_dtype(t)?;
        }
        let mut q = self.q_proj.forward(&hidden_states)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
        }
        q = q
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        q = self.q_norm.forward(&q)?;

        let (k, v) = if let Some(cross_attn_states) = cross_attn_states {
            let mut cross_attn_states = cross_attn_states.clone();
            let original_dtype = cross_attn_states.dtype();
            if let Some(t) = self.k_proj.quantized_act_type() {
                cross_attn_states = cross_attn_states.to_dtype(t)?;
            }
            let mut k = self.k_proj.forward(&cross_attn_states)?;
            k = k
                .reshape((bs, (), self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            if self.q_proj.quantized_act_type().is_some() {
                k = k.to_dtype(original_dtype)?;
            }
            k = self.k_norm.forward(&k)?;

            let mut v = self.v_proj.forward(&cross_attn_states)?;
            if self.q_proj.quantized_act_type().is_some() {
                v = v.to_dtype(original_dtype)?;
            }
            v = v
                .reshape((bs, (), self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;

            (k, v)
        } else {
            candle_core::bail!("Cross attn cannot find k,v cache or cross attn hidden states!")
        };

        let mut attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attention_mask
                    .map(|m| m.repeat((1, self.num_heads, 1, 1)).unwrap())
                    .as_ref(),
                None,
                &self.sdpa_params,
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_len, ()))?
            .to_dtype(q.dtype())?;

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        let mut res = self.o_proj.forward(&attn_output)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct MLlamaCrossAttentionDecoderLayer {
    attn: MLlamaTextCrossAttention,
    attn_gate: Tensor,
    mlp: MLlamaTextMlp,
    mlp_gate: Tensor,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl MLlamaCrossAttentionDecoderLayer {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
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
        let attn = MLlamaTextCrossAttention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("cross_attn"), loading_isq),
            mapper,
            layer_idx,
            comm,
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            attn_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_attn_gate")?,
            mlp_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_mlp_gate")?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        full_text_row_masked_out_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states = self
            .attn
            .forward(&hidden_states, cross_attn_states, attention_mask)?;
        hidden_states = (residual + hidden_states.broadcast_mul(&self.attn_gate.tanh()?)?)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;
        if let Some(full_text_row_masked_out_mask) = full_text_row_masked_out_mask {
            hidden_states = full_text_row_masked_out_mask
                .to_dtype(hidden_states.dtype())?
                .i((.., 0))?
                .broadcast_mul(&hidden_states)?;
        }

        residual + hidden_states.broadcast_mul(&self.mlp_gate.tanh()?)?
    }
}

enum MLlamaDecoderLayer {
    CrossAttn(MLlamaCrossAttentionDecoderLayer),
    SelfAttn(MLlamaSelfAttentionDecoderLayer),
}

pub(super) struct MLlamaTextModel {
    embed_tokens: Embedding,
    lm_head: Arc<dyn QuantMethod>,
    norm: RmsNorm,
    layers: Vec<MLlamaDecoderLayer>,
    pub(crate) cfg: ModelConfigMetadata,
    pub(crate) cache: EitherCache,
    pub(crate) device: Device,
    pub(crate) max_position_embeddings: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl MLlamaTextModel {
    pub(super) fn new(
        cfg: &MLlamaTextConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb)
            );
        }
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            candle_core::bail!("Expected eager attention implementation");
        }
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size + 8,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("model.embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb.pp("lm_head"), false),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(embed_tokens.embeddings(), false)?,
                None,
            ))?
        };

        let vb = vb.pp("model");

        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("norm"), false),
        )?;

        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(Llama3RotaryEmbedding::new_mllama3(
                    vb.dtype(),
                    cfg,
                    device,
                    is_gptx,
                )?),
            );
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let comm = mapper.get_comm_for(i)?;
            if cfg.cross_attention_layers.contains(&i) {
                layers.push(MLlamaDecoderLayer::CrossAttn(
                    MLlamaCrossAttentionDecoderLayer::new(
                        cfg,
                        vb.pp(format!("layers.{i}")),
                        &*mapper,
                        i,
                        false,
                        &comm,
                    )?,
                ))
            } else {
                let device = mapper
                    .device_for(i, false)
                    .unwrap_or(&normal_loading_metadata.real_device);
                layers.push(MLlamaDecoderLayer::SelfAttn(
                    MLlamaSelfAttentionDecoderLayer::new(
                        cfg,
                        vb.pp(format!("layers.{i}")),
                        ropes
                            .get(&device.location())
                            .expect("No RoPE for device location!")
                            .clone(),
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        &comm,
                    )?,
                ))
            }
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            device: normal_loading_metadata.real_device,
            max_position_embeddings: cfg.max_position_embeddings,
            mapper,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        input_ids: &Tensor,
        cross_attn_states: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
        full_text_row_masked_out_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let cache = &mut self.cache.normal().0;
        let self_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            cache as &dyn PastKvLenCache,
            hidden_states.dtype(),
            self.cfg.num_attn_heads,
        )?;

        let self_mask = DeviceMappedMask::new(self_mask, &*self.mapper)?;
        let cross_attention_mask =
            DeviceMappedMask::new(cross_attention_mask.cloned(), &*self.mapper)?;
        let full_text_row_masked_out_mask =
            DeviceMappedMask::new(full_text_row_masked_out_mask.cloned(), &*self.mapper)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = self.mapper.map(hidden_states, i)?;
            match layer {
                MLlamaDecoderLayer::SelfAttn(attn) => {
                    hidden_states = attn.forward(
                        &hidden_states,
                        self_mask.as_ref().map(|m| m.get(hidden_states.device())),
                        seqlen_offsets,
                        &mut cache[i],
                    )?;
                }
                MLlamaDecoderLayer::CrossAttn(attn) => {
                    // For text-only path we should skip cross attention layers.
                    // Let's check if the layer is cross attention layer and if we have cross attention states
                    // or cached cross attention states.
                    if cross_attn_states.is_none() {
                        continue;
                    }
                    hidden_states = attn.forward(
                        &hidden_states,
                        cross_attn_states
                            .as_ref()
                            .map(|x| x.to_device(hidden_states.device()).unwrap())
                            .as_ref(),
                        cross_attention_mask
                            .as_ref()
                            .map(|m| m.get(hidden_states.device())),
                        full_text_row_masked_out_mask
                            .as_ref()
                            .map(|m| m.get(hidden_states.device())),
                    )?;
                }
            }
        }

        hidden_states = hidden_states.to_device(&self.device)?;
        hidden_states = self.norm.forward(&hidden_states)?;

        hidden_states = self
            .lm_head
            .forward(&extract_logits(&hidden_states, context_lens)?)?;

        Ok(hidden_states)
    }
}

impl IsqModel for MLlamaTextModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            match layer {
                MLlamaDecoderLayer::CrossAttn(_cross) => {
                    // tensors.push((&mut cross.attn.q_proj, Some(i)));
                    // tensors.push((&mut cross.attn.k_proj, Some(i)));
                    // tensors.push((&mut cross.attn.v_proj, Some(i)));
                    // tensors.push((&mut cross.attn.o_proj, Some(i)));
                    // tensors.push((&mut cross.mlp.gate_proj, Some(i)));
                    // tensors.push((&mut cross.mlp.up_proj, Some(i)));
                    // tensors.push((&mut cross.mlp.down_proj, Some(i)));
                }
                MLlamaDecoderLayer::SelfAttn(self_attn) => {
                    tensors.push((&mut self_attn.attn.q_proj, Some(i)));
                    tensors.push((&mut self_attn.attn.k_proj, Some(i)));
                    tensors.push((&mut self_attn.attn.v_proj, Some(i)));
                    tensors.push((&mut self_attn.attn.o_proj, Some(i)));
                    tensors.push((&mut self_attn.mlp.gate_proj, Some(i)));
                    tensors.push((&mut self_attn.mlp.up_proj, Some(i)));
                    tensors.push((&mut self_attn.mlp.down_proj, Some(i)));
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("model.embed_tokens").add(&self.embed_tokens);
        uvb.pp("lm_head").add(&self.lm_head);

        let uvb = uvb.pp("model");

        uvb.pp("norm").add(&self.norm);

        for (i, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb.pp("layers").pp(i);
            match layer {
                MLlamaDecoderLayer::CrossAttn(crossattn) => {
                    // Cross attention layers are not quantized
                    uvb_l
                        .pp("post_attention_layernorm")
                        .add(&crossattn.post_attention_layernorm);
                    uvb_l.pp("input_layernorm").add(&crossattn.input_layernorm);
                    uvb_l.add_tensor("cross_attn_attn_gate", crossattn.attn_gate.clone());
                    uvb_l.add_tensor("cross_attn_mlp_gate", crossattn.mlp_gate.clone());

                    let uvb_attn = uvb_l.pp("cross_attn");
                    uvb_attn.pp("q_proj").add(&crossattn.attn.q_proj);
                    uvb_attn.pp("k_proj").add(&crossattn.attn.k_proj);
                    uvb_attn.pp("v_proj").add(&crossattn.attn.v_proj);
                    uvb_attn.pp("o_proj").add(&crossattn.attn.o_proj);
                    uvb_attn.pp("q_norm").add(&crossattn.attn.q_norm);
                    uvb_attn.pp("k_norm").add(&crossattn.attn.k_norm);

                    let uvb_mlp = uvb_l.pp("mlp");
                    uvb_mlp.pp("gate_proj").add(&crossattn.mlp.gate_proj);
                    uvb_mlp.pp("up_proj").add(&crossattn.mlp.up_proj);
                    uvb_mlp.pp("down_proj").add(&crossattn.mlp.down_proj);
                }
                MLlamaDecoderLayer::SelfAttn(selfattn) => {
                    uvb_l
                        .pp("post_attention_layernorm")
                        .add(&selfattn.post_attention_layernorm);
                    uvb_l.pp("input_layernorm").add(&selfattn.input_layernorm);
                }
            }
        }

        uvb.to_safetensors()
    }
}
