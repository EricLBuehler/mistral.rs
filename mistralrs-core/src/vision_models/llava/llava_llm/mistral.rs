#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_core::{quantized::QMatMul, DType, Device, Module, Result, Tensor};
use candle_nn::{linear_no_bias, Activation, VarBuilder};

use crate::{
    device_map::DeviceMapper,
    layers::{repeat_kv, CausalMasker, MatMul, RmsNorm, ScaledDotProductAttention},
    pipeline::{extract_logits, Cache, IsqModel, NormalLoadingMetadata},
    utils::progress::NiceProgressBar,
};

use super::{LLaVALLM, OrdinaryRoPE};
use crate::models::mistral::Config;

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
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
        })
    }
}

impl Module for MLP {
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
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj: QMatMul::Tensor(q_proj.weight().clone()),
            k_proj: QMatMul::Tensor(k_proj.weight().clone()),
            v_proj: QMatMul::Tensor(v_proj.weight().clone()),
            o_proj: QMatMul::Tensor(o_proj.weight().clone()),
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            use_flash_attn: cfg.use_flash_attn,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        rope_parameter: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        let mut q = MatMul.qmatmul(&xs, &self.q_proj)?;
        let mut k = MatMul.qmatmul(&xs, &self.k_proj)?;
        let mut v = MatMul.qmatmul(&xs, &self.v_proj)?;
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        let mut q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        q = OrdinaryRoPE::forward(&q, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        k = OrdinaryRoPE::forward(&k, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (k, v, attn_mask) = Cache::update_kv_cache_sliding_window(
            kv_cache,
            k,
            v,
            attention_mask,
            self.sliding_window,
            false,
        )?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let mut attn_output = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.num_heads,
            self.head_dim,
            attn_mask.as_ref(),
            self.use_flash_attn,
            b_sz,
            q_len,
        )?;

        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            attn_output = attn_output.to_dtype(DType::F32)?;
        }
        let mut res = MatMul.qmatmul(
            &attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.hidden_size))?,
            &self.o_proj,
        )?;
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Config,
        vb: VarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
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
            mlp,
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
        rope_parameter: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let residual = xs;
        let mut xs = self.input_layernorm.forward(xs)?;
        xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
            rope_parameter,
        )?;
        xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        xs + residual
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    sliding_window: Option<usize>,
    pub device: Device,
    pub cache: Cache,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    rope_parameters: (Tensor, Tensor),
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(cfg, vb_m, vb_lm_head, is_gptx, normal_loading_metadata)
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: VarBuilder,
        vb_lm_head: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        //let vb_m = vb_m.set_dtype(mapper.get_min_dtype()?);
        //let vb_lm_head = vb_lm_head.set_dtype(mapper.get_min_dtype()?);

        let embed_tokens = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let rope_parameters = OrdinaryRoPE::create_parameters(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta as f32,
            vb_m.dtype(),
            &normal_loading_metadata.real_device,
        )?;
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let layer = DecoderLayer::new(
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
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
            mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: Cache::new(cfg.num_hidden_layers, false),
            mapper,
            rope_parameters,
        })
    }

    pub fn get_input_embeddings(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.embed_tokens.forward(input_ids)?,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }

    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut xs = input_embeds;
        let mut cache = self.cache.lock();
        let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            self.sliding_window,
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
                (&self.rope_parameters.0, &self.rope_parameters.1),
            )?;
        }
        xs = xs.to_device(&self.device)?;
        xs = xs.apply(&self.norm)?;
        if matches!(self.lm_head, QMatMul::QTensor(_)) {
            xs = xs.to_dtype(DType::F32)?;
        }
        extract_logits(&MatMul.qmatmul(&xs, &self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.down_proj, Some(i)));
            tensors.push((&mut layer.mlp.up_proj, Some(i)));
            tensors.push((&mut layer.mlp.gate_proj, Some(i)));
        }
        (tensors, &*self.mapper)
    }
}

impl LLaVALLM for Model {
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.get_input_embeddings(input_ids)
    }

    fn forward_input_embed(
        &self,
        input_ids: &Tensor,
        input_embed: Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            input_embed,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }
    fn cache(&self) -> &Cache {
        &self.cache
    }
}
