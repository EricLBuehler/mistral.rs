#![allow(clippy::cast_possible_truncation)]

use candle_core::{quantized::QMatMul, Device, Module, Result, Tensor};
use candle_nn::{layer_norm, linear_b, LayerNorm, Linear, VarBuilder};
use std::sync::Arc;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{CausalMasker, RotaryEmbedding, ScaledDotProductAttention},
    layers_utils::repeat_kv,
    pipeline::{extract_logits, Cache, IsqModel, NormalLoadingMetadata, NormalModel},
    utils::progress::NiceProgressBar,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: candle_nn::Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) norm_epsilon: f64,
    pub(crate) rope_theta: f64,
    pub(crate) use_bias: bool,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) use_flash_attn: bool,
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    act: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let (h_size, i_size) = (cfg.hidden_size, cfg.intermediate_size);
        let c_fc = linear_b(h_size, i_size, cfg.use_bias, vb.pp("c_fc"))?;
        let c_proj = linear_b(i_size, h_size, cfg.use_bias, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            act: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.c_fc)?.apply(&self.act)?.apply(&self.c_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
    sliding_window: Option<usize>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let b = cfg.use_bias;
        let q_proj = linear_b(hidden_sz, num_heads * head_dim, b, vb.pp("q_proj"))?;
        let k_proj = linear_b(hidden_sz, num_kv_heads * head_dim, b, vb.pp("k_proj"))?;
        let v_proj = linear_b(hidden_sz, num_kv_heads * head_dim, b, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, hidden_sz, b, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            use_flash_attn: cfg.use_flash_attn,
            sliding_window: cfg.sliding_window,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

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

        let attn_output = ScaledDotProductAttention.run_attention(
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

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            layer_norm(cfg.hidden_size, cfg.norm_epsilon, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_epsilon,
            vb.pp("post_attention_layernorm"),
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
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            kv_cache,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    sliding_window: Option<usize>,
    pub device: Device,
    pub cache: Cache,
    pub max_seq_len: usize,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        let vb = vb.set_dtype(mapper.get_min_dtype()?);
        let vb_m = vb.pp("model");

        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let rotary_emb = Arc::new(RotaryEmbedding::new(
                cfg.rope_theta as f32,
                head_dim,
                cfg.max_position_embeddings,
                mapper
                    .device_for(layer_idx, false)
                    .unwrap_or(&normal_loading_metadata.real_device),
                is_gptx,
                vb_m.dtype(),
            )?);
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
            )?)
        }
        let norm = layer_norm(cfg.hidden_size, cfg.norm_epsilon, vb_m.pp("norm"))?;
        let lm_head = candle_nn::Linear::new(embed_tokens.embeddings().clone(), None);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            cache: Cache::new(cfg.num_hidden_layers, false),
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        let mut cache = self.cache.lock();
        let attention_mask = CausalMasker.make_causal_mask_with_sliding_window_as_attn_bias(
            input_ids,
            &cache,
            self.sliding_window,
            xs.dtype(),
            self.layers[0].self_attn.num_heads,
        )?;

        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                &mut cache[i],
            )?
        }
        extract_logits(&xs.apply(&self.norm)?.apply(&self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        todo!()
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
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
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
}

// TODO
impl AnyMoeBaseModelMixin for Model {}
