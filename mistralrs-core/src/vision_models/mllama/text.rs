#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::{embedding, Activation, Embedding, Module, VarBuilder};
use mistralrs_quant::{linear_no_bias, QuantMethod, QuantMethodConfig, UnquantLinear};

use crate::{
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{repeat_kv, CausalMasker, Llama3RotaryEmbedding, MatMul, RmsNorm, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{extract_logits, Cache, IsqModel, NormalLoadingMetadata},
};

use super::config::MLlamaTextConfig;

struct MLlamaTextMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl MLlamaTextMlp {
    fn new(cfg: &MLlamaTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                &cfg.quantization_config,
                vb.pp("up_proj"),
            )?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                &cfg.quantization_config,
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
        vb: VarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
                vb.pp("o_proj"),
            )?,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
                use_flash_attn: false,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
            },
            rope,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
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

        q = q.reshape((bs * q_len, self.num_heads, self.head_dim))?;
        k = k.reshape((bs * q_len, self.num_kv_heads, self.head_dim))?;
        v = v
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        self.rope
            .forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, bs)?;

        if q.rank() == 3 {
            q = q
                .reshape((bs, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }

        (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

        let mut attn_output = Sdpa
            .run_attention(&q, &k, &v, attention_mask, None, &self.sdpa_params)?
            .transpose(1, 2)?
            .reshape((bs, q_len, ()))?;

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
        vb: VarBuilder,
        rope: Arc<Llama3RotaryEmbedding>,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
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
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states = self.attn.forward(
            &hidden_states,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
        )?;
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
}

impl MLlamaTextCrossAttention {
    fn new(
        cfg: &MLlamaTextConfig,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim(),
                &cfg.quantization_config,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim(),
                cfg.hidden_size,
                &cfg.quantization_config,
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
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        kv_cache: &mut Option<(Tensor, Tensor)>,
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

            k = repeat_kv(k.clone(), self.num_heads / self.num_kv_heads)?.contiguous()?;
            v = repeat_kv(v.clone(), self.num_heads / self.num_kv_heads)?.contiguous()?;

            (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;
            (k, v)
        } else if let Some((k_cache, v_cache)) = kv_cache {
            (k_cache.clone(), v_cache.clone())
        } else {
            candle_core::bail!("Cross attn cannot find k,v cache or cross attn hidden states!")
        };

        let mut attn_output = {
            let att = MatMul.matmul_affine_div(
                &q.contiguous()?,
                &k.t()?.contiguous()?,
                (self.head_dim as f64).sqrt(),
            )?;

            let att = match attention_mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            MatMul
                .matmul(&att, &v.contiguous()?)?
                .transpose(1, 2)?
                .reshape((bs, q_len, ()))?
        };

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
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let mlp = MLlamaTextMlp::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
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
        )?;

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            // NOTE: Preapply the tanh
            attn_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_attn_gate")?
                .tanh()?,
            // NOTE: Preapply the tanh
            mlp_gate: mapper
                .set_device(layer_idx, vb.clone(), false)
                .get((1,), "cross_attn_mlp_gate")?
                .tanh()?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cross_attn_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        full_text_row_masked_out_mask: Option<&Tensor>,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = hidden_states;

        let mut hidden_states = self.input_layernorm.forward(hidden_states)?;

        hidden_states =
            self.attn
                .forward(&hidden_states, cross_attn_states, attention_mask, kv_cache)?;
        hidden_states = (residual + hidden_states.broadcast_mul(&self.attn_gate)?)?;

        let residual = &hidden_states;
        let mut hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        hidden_states = self.mlp.forward(&hidden_states)?;
        if let Some(full_text_row_masked_out_mask) = full_text_row_masked_out_mask {
            hidden_states = full_text_row_masked_out_mask
                .to_dtype(hidden_states.dtype())?
                .i((.., 0))?
                .broadcast_mul(&hidden_states)?;
        }

        residual + hidden_states.broadcast_mul(&self.mlp_gate)?
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
    pub(crate) self_attn_cache: Cache,
    pub(crate) device: Device,
    pub(crate) max_position_embeddings: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl MLlamaTextModel {
    pub(super) fn new(
        cfg: &MLlamaTextConfig,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if !matches!(attention_mechanism, AttentionImplementation::Eager) {
            candle_core::bail!("Expected eager attention implementation");
        }
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size + 8,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("model.embed_tokens"), false),
        )?;

        let lm_head = if !cfg.tie_word_embeddings {
            mistralrs_quant::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                &None,
                mapper.set_nm_device(vb.pp("lm_head"), false),
            )?
        } else {
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(
                    mapper.cast_nm_device(embed_tokens.embeddings(), false)?,
                    None,
                ),
            ))?)
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
            if cfg.cross_attention_layers.contains(&i) {
                layers.push(MLlamaDecoderLayer::CrossAttn(
                    MLlamaCrossAttentionDecoderLayer::new(
                        cfg,
                        vb.pp(format!("layers.{i}")),
                        &*mapper,
                        i,
                        false,
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
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                head_dim: None,
            },
            self_attn_cache: Cache::new(cfg.num_hidden_layers, false),
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
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        let mut self_cache = self.self_attn_cache.lock();
        let self_mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            hidden_states.dtype(),
            self.cfg.num_attn_heads,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = self.mapper.map(hidden_states, i)?;
            match layer {
                MLlamaDecoderLayer::SelfAttn(attn) => {
                    hidden_states = attn.forward(
                        &hidden_states,
                        self_mask.as_ref(),
                        seqlen_offsets,
                        start_offsets_kernel.clone(),
                        &mut self_cache[i],
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
                        cross_attn_states,
                        cross_attention_mask,
                        full_text_row_masked_out_mask,
                        &mut self_cache[i],
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
                    // ISQ for cross attn shows poor performance!
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
}
