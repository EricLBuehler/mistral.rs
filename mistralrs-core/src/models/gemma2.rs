#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, RotaryEmbedding, VarBuilder};
use mistralrs_quant::{QuantMethod, QuantMethodConfig, QuantizedConfig, UnquantLinear};

use crate::{
    amoe::{
        AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainableLayer, MlpLayer,
        MoeMlp,
    },
    attention::SdpaParams,
    device_map::DeviceMapper,
    get_delta_from_lora_ab,
    layers::{Activation, CausalMasker, MatMul, RmsNorm, Sdpa},
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalLoadingMetadata, NormalModel,
    },
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Config {
    pub attention_bias: bool,
    pub head_dim: usize,
    // The code gemma configs include both hidden_act and hidden_activation.
    pub hidden_act: Option<Activation>,
    pub hidden_activation: Option<Activation>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub sliding_window: usize,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub query_pre_attn_scalar: usize,
    pub max_position_embeddings: usize,
    pub quantization_config: Option<QuantizedConfig>,
    pub use_flash_attn: bool,
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn hidden_act(&self) -> Result<Activation> {
        match (self.hidden_act, self.hidden_activation) {
            (None, Some(act)) | (Some(act), None) => Ok(act),
            (Some(act), Some(_)) => {
                // If both are set just use hidden_act
                Ok(act)
            }
            (None, None) => candle_core::bail!("none of hidden_act and hidden_activation are set"),
        }
    }
}

#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
    params: Vec<usize>,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = mistralrs_quant::linear_b(
            hidden_sz,
            intermediate_sz,
            false,
            &cfg.quantization_config,
            vb.pp("gate_proj"),
        )?;
        let up_proj = mistralrs_quant::linear_b(
            hidden_sz,
            intermediate_sz,
            false,
            &cfg.quantization_config,
            vb.pp("up_proj"),
        )?;
        let down_proj = mistralrs_quant::linear_b(
            intermediate_sz,
            hidden_sz,
            false,
            &cfg.quantization_config,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act()?,
            params: vec![hidden_sz, intermediate_sz],
        })
    }
}

impl AnyMoeTrainableLayer for MLP {}

impl MlpLayer for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = MatMul
            .qmethod_matmul(&xs, &*self.gate_proj)?
            .apply(&self.act_fn)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        let mut res = MatMul.qmethod_matmul(&(lhs * rhs)?, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_layers(&mut self) -> Vec<&mut Arc<dyn QuantMethod>> {
        vec![&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    // gate, up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let gate_proj = if let Some(ref delta) = deltas[0] {
            self.gate_proj.add_delta_w(delta)?
        } else {
            self.gate_proj.clone()
        };
        let up_proj = if let Some(ref delta) = deltas[1] {
            self.up_proj.add_delta_w(delta)?
        } else {
            self.up_proj.clone()
        };
        let down_proj = if let Some(ref delta) = deltas[2] {
            self.down_proj.add_delta_w(delta)?
        } else {
            self.down_proj.clone()
        };

        Ok(Box::new(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: self.act_fn,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        self.gate_proj.dtype_and_device()
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    attn_logit_softcapping: Option<f64>,
    use_sliding_window: bool,
    sliding_window: Option<usize>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        layer_idx: usize,
        vb: VarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = mistralrs_quant::linear_b(
            hidden_sz,
            num_heads * head_dim,
            bias,
            &cfg.quantization_config,
            vb.pp("q_proj"),
        )?;
        let k_proj = mistralrs_quant::linear_b(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            &cfg.quantization_config,
            vb.pp("k_proj"),
        )?;
        let v_proj = mistralrs_quant::linear_b(
            hidden_sz,
            num_kv_heads * head_dim,
            bias,
            &cfg.quantization_config,
            vb.pp("v_proj"),
        )?;
        let o_proj = mistralrs_quant::linear_b(
            num_heads * head_dim,
            hidden_sz,
            bias,
            &cfg.quantization_config,
            vb.pp("o_proj"),
        )?;
        let sliding_window = if layer_idx % 2 == 0 {
            // ^ Order is SWA, global, SWA
            Some(cfg.sliding_window)
        } else {
            None
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            use_sliding_window: layer_idx % 2 == 0, // Order is SWA, global, SWA
            sliding_window,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                use_flash_attn: cfg.use_flash_attn,
                softcap: cfg.attn_logit_softcapping.map(|x| x as f32),
                softmax_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
                sliding_window,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let mut q = q.reshape((b_sz * q_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((b_sz * q_len, self.num_kv_heads, self.head_dim))?;
        let v = if q_len != 1 {
            v.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
        } else {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?
        };

        self.rotary_emb
            .forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 && q_len != 1 {
            q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        } else if q.rank() == 3 {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            q = q
                .reshape((b_sz, self.num_heads, q_len, self.head_dim))?
                .contiguous()?;
            k = k
                .reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?
                .contiguous()?;
        }

        let mask = if self.use_sliding_window {
            sliding_attention_mask
        } else {
            attention_mask
        };

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    self.attn_logit_softcapping,
                )?
            }
            None => {
                // self.sliding_window is None if !self.use_sliding_window
                let (k, v, mask) =
                    kv_cache.append_sliding_window(&k, &v, mask, self.sliding_window)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    mask.as_ref(),
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Box<dyn MlpLayer>,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            layer_idx,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
        )?;
        let mlp = MLP::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
        let input_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm"), false),
        )?;
        Ok(Self {
            self_attn,
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        sliding_attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(
                &xs,
                attention_mask,
                sliding_attention_mask,
                seqlen_offsets,
                start_offsets_kernel,
                kv_cache,
                metadata,
                flash_params,
            )?
            .apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    hidden_size: usize,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    sliding_window: usize,
    final_logit_softcapping: Option<f64>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization in {} bits.",
                quant_cfg.quant_method.to_string(),
                quant_cfg.bits
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let vb_m = vb.pp("model");
        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    cfg.head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let head_dim = cfg.head_dim;
            let sliding_window = if layer_idx % 2 == 0 {
                // ^ Order is SWA, global, SWA
                Some(cfg.sliding_window)
            } else {
                None
            };
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                    cfg.num_attention_heads,
                    head_dim,
                    (1.0 / (cfg.query_pre_attn_scalar as f64).sqrt()) as f32,
                    Some(cfg.num_key_value_heads),
                    sliding_window,
                    device,
                    None,
                )?),
            };
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = mapper.cast_nm_device(
            embed_tokens.embeddings(),
            normal_loading_metadata.loading_isq,
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
                Linear::new(lm_head, None),
            ))?),
            device: normal_loading_metadata.real_device,
            hidden_size: cfg.hidden_size,
            cache: EitherCache::Normal(NormalCache::new(
                cfg.num_hidden_layers,
                cfg.max_position_embeddings,
            )),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            sliding_window: cfg.sliding_window,
            final_logit_softcapping: cfg.final_logit_softcapping,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                head_dim: Some(cfg.head_dim),
            },
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        mut metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(input_ids)?;
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_causal_mask_matrix(
            input_ids,
            &*cache,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        let sliding_attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            &*cache,
            Some(self.sliding_window),
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                sliding_attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
                metadata
                    .as_mut()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), &mut **metadata)),
                flash_params,
            )?;
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }

        let mut xs = MatMul.qmethod_matmul(&xs, &*self.lm_head)?;

        if let Some(final_logit_softcapping) = self.final_logit_softcapping {
            xs = (xs / final_logit_softcapping)?;
            xs = xs.tanh()?;
            xs = (xs * final_logit_softcapping)?;
        }

        extract_logits(&xs, context_lens)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_layers()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm.undo_gemma().unwrap());

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l
                .pp("input_layernorm")
                .add(&layer.input_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("pre_feedforward_layernorm")
                .add(&layer.pre_feedforward_layernorm.undo_gemma().unwrap());
            uvb_l
                .pp("post_feedforward_layernorm")
                .add(&layer.post_feedforward_layernorm.undo_gemma().unwrap());
        }

        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _start_offsets_kernel: Tensor,
        _start_offsets_kernel_full: Tensor,
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        let mut mlps = Vec::new();
        for layer in &self.layers {
            mlps.push(&*layer.mlp);
        }
        mlps
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        let mut mlps = Vec::new();
        for layer in &mut self.layers {
            mlps.push(&mut layer.mlp);
        }
        mlps
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<VarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        mut layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<VarBuilder>,
    ) -> Result<()> {
        let mut experts: Vec<Vec<Box<dyn MlpLayer>>> = Vec::new();
        if layers.is_empty() {
            layers = (0..self.layers.len()).collect::<Vec<_>>();
        }
        for _ in 0..layers.len() {
            experts.push(Vec::new());
        }
        for vb in additional_vbs {
            let vb = vb.pp(&prefix);
            for (layer, row) in experts.iter_mut().enumerate() {
                if !layers.contains(&layer) {
                    continue;
                }

                let intermediate_size = self.layers[layer].mlp.get_params()[1];
                let hidden_size = self.layers[layer].mlp.get_params()[0];
                match expert_type {
                    AnyMoeExpertType::FineTuned => {
                        let (dtype, device) = self.layers[layer].mlp.dtype_device();
                        row.push(Box::new(MLP::new(
                            &Config {
                                intermediate_size: self.layers[layer].mlp.get_params()[1],
                                hidden_size: self.layers[layer].mlp.get_params()[0],
                                ..Default::default()
                            },
                            vb.pp(layer).pp(&mlp).set_dtype(dtype).set_device(device),
                        )?));
                    }
                    AnyMoeExpertType::LoraAdapter {
                        rank,
                        alpha,
                        ref target_modules,
                    } => {
                        let vb_mlp = vb.pp(layer).pp(&mlp);

                        let gate_proj_delta = if target_modules.contains(&"gate_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "gate_proj"
                            ))
                        } else {
                            None
                        };
                        let up_proj_delta = if target_modules.contains(&"up_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (hidden_size, intermediate_size),
                                "up_proj"
                            ))
                        } else {
                            None
                        };
                        let down_proj_delta = if target_modules.contains(&"down_proj".to_string()) {
                            Some(get_delta_from_lora_ab!(
                                vb_mlp,
                                rank,
                                alpha,
                                (intermediate_size, hidden_size),
                                "down_proj"
                            ))
                        } else {
                            None
                        };

                        row.push(self.layers[layer].mlp.new_added_delta(vec![
                            gate_proj_delta,
                            up_proj_delta,
                            down_proj_delta,
                        ])?);
                    }
                }
            }
        }
        for (layer, expert) in layers.into_iter().zip(experts) {
            let mut experts_all = vec![self.layers[layer].mlp.clone()];
            experts_all.extend(expert);
            let (dtype, device) = self.layers[layer].mlp.dtype_device();
            self.layers[layer].mlp = Box::new(MoeMlp::new(
                experts_all,
                config.clone(),
                dtype,
                &device,
                layer,
                gate_vb.as_ref(),
            )?);
        }
        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
