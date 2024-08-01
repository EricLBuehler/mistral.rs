#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{quantized::QMatMul, DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Activation, RotaryEmbedding, VarBuilder};
use std::sync::Arc;

use crate::{
    amoe::{
        AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainableLayer, MlpLayer,
        MoeMlp,
    },
    device_map::DeviceMapper,
    get_delta_from_lora_ab,
    layers::{repeat_kv, CausalMasker, MatMul, QLinear, RmsNorm, ScaledDotProductAttention},
    layers_masker::PastKvLenCache,
    merge_delta,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits, text_models_inputs_processor::PagedAttentionInputMetadata, Cache, IsqModel,
        NormalLoadingMetadata, NormalModel,
    },
    utils::progress::NiceProgressBar,
};

#[derive(Debug, Clone, PartialEq, serde::Deserialize, Default)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
    pub use_flash_attn: bool,
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    params: Vec<usize>,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: QMatMul::Tensor(gate_proj.weight().clone()),
            up_proj: QMatMul::Tensor(up_proj.weight().clone()),
            down_proj: QMatMul::Tensor(down_proj.weight().clone()),
            act_fn: cfg.hidden_act,
            params: vec![hidden_sz, intermediate_sz],
        })
    }
}

impl AnyMoeTrainableLayer for MLP {}

impl MlpLayer for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.gate_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let lhs = MatMul.qmatmul(&xs, &self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = MatMul.qmatmul(&xs, &self.up_proj)?;
        let mut res = MatMul.qmatmul(&(lhs * rhs)?, &self.down_proj)?;
        if matches!(self.gate_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
    fn get_isq_tensors(&mut self) -> Vec<&mut QMatMul> {
        vec![&mut self.gate_proj, &mut self.up_proj, &mut self.down_proj]
    }
    fn get_isq_biases(&mut self) -> Vec<Option<&mut Tensor>> {
        vec![None, None, None]
    }
    fn clone(&self) -> Box<dyn MlpLayer> {
        Box::new(Clone::clone(self))
    }
    fn get_params(&self) -> &[usize] {
        &self.params
    }
    // gate, up, down
    fn new_added_delta(&self, deltas: Vec<Option<Tensor>>) -> Result<Box<dyn MlpLayer>> {
        let new_gate = if let Some(ref delta) = deltas[0] {
            merge_delta!(self.gate_proj, delta)
        } else {
            self.gate_proj.clone()
        };
        let new_up = if let Some(ref delta) = deltas[1] {
            merge_delta!(self.up_proj, delta)
        } else {
            self.up_proj.clone()
        };
        let new_down = if let Some(ref delta) = deltas[2] {
            merge_delta!(self.down_proj, delta)
        } else {
            self.down_proj.clone()
        };

        Ok(Box::new(Self {
            gate_proj: new_gate,
            up_proj: new_up,
            down_proj: new_down,
            act_fn: self.act_fn,
            params: self.params.clone(),
        }))
    }

    fn dtype_device(&self) -> (DType, Device) {
        match &self.gate_proj {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }
}

struct Attention {
    q_proj: QLinear,
    k_proj: QLinear,
    v_proj: QLinear,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
    paged_attn: Option<PagedAttention>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: VarBuilder,
        paged_attn: Option<PagedAttention>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj: QLinear::from_linear(q_proj),
            k_proj: QLinear::from_linear(k_proj),
            v_proj: QLinear::from_linear(v_proj),
            o_proj: QMatMul::Tensor(o_proj.weight().clone()),
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            use_flash_attn: cfg.use_flash_attn,
            paged_attn,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if self.q_proj.is_quant() {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut q = self.q_proj.forward(&xs)?;
        let mut k = self.k_proj.forward(&xs)?;
        let mut v = self.v_proj.forward(&xs)?;
        if self.q_proj.is_quant() {
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
                )?
            }
            None => {
                let (k, v) = Cache::update_kv_cache(kv_cache, k, v, false)?;

                let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
                let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

                ScaledDotProductAttention.run_attention(
                    &q,
                    &k,
                    &v,
                    self.num_heads,
                    self.head_dim,
                    attention_mask,
                    self.use_flash_attn,
                    b_sz,
                    q_len,
                )?
            }
        };

        if self.q_proj.is_quant() {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmatmul(&attn_output, &self.o_proj)?;
        if self.q_proj.is_quant() {
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
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            paged_attn,
        )?;
        let mlp = MLP::new(cfg, mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq))?;
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
            mlp: Box::new(mlp),
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        metadata: Option<((Tensor, Tensor), &mut PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
            metadata,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    sliding_window: usize,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
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
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        let vb = vb.set_dtype(mapper.get_min_dtype()?);
        let vb_m = vb.pp("model");

        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = Arc::new(RotaryEmbedding::new(
                cfg.rope_theta as f32,
                head_dim,
                cfg.max_position_embeddings,
                device,
                is_gptx,
                vb.dtype(),
            )?);
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(PagedAttention::new(
                    cfg.num_attention_heads,
                    head_dim,
                    (1.0 / (head_dim as f64).sqrt()) as f32,
                    Some(cfg.num_key_value_heads),
                    Some(cfg.sliding_window),
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
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            cfg: ModelConfigMetadata {
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: Some(cfg.sliding_window),
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
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mut cache = self.cache.lock();
        let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(&*cache as &dyn PastKvLenCache),
            Some(self.sliding_window),
            xs.dtype(),
            self.layers[0].self_attn.num_heads,
        )?;
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
                metadata
                    .as_mut()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), &mut **metadata)),
            )?
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if matches!(self.lm_head, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        extract_logits(&MatMul.qmatmul(&xs, &self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_matmuls(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((layer.self_attn.q_proj.inner(), Some(i)));
            tensors.push((layer.self_attn.k_proj.inner(), Some(i)));
            tensors.push((layer.self_attn.v_proj.inner(), Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_tensors()
                    .into_iter()
                    .map(|m| (m, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
    }
    fn get_biases(&mut self) -> (Vec<(Option<&mut Tensor>, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((layer.self_attn.q_proj.bias_mut(), Some(i)));
            tensors.push((layer.self_attn.k_proj.bias_mut(), Some(i)));
            tensors.push((layer.self_attn.v_proj.bias_mut(), Some(i)));
            tensors.extend(
                layer
                    .mlp
                    .get_isq_biases()
                    .into_iter()
                    .map(|b| (b, Some(i)))
                    .collect::<Vec<_>>(),
            );
        }
        (tensors, &*self.mapper)
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
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
            metadata,
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
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &Cache {
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
