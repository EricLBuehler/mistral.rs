#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{quantized::QMatMul, DType, Device, Result, Tensor};
use candle_nn::{
    embedding, linear_no_bias as linear, Embedding, Module, RotaryEmbedding, VarBuilder,
};
use mistralrs_quant::{gptq_linear_no_bias, QuantMethod, QuantizedConfig};
use serde::Deserialize;
use std::sync::Arc;

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeTrainableLayer},
    device_map::DeviceMapper,
    layers::{CausalMasker, MatMul, RmsNorm, ScaledDotProductAttention},
    layers_utils::repeat_kv,
    paged_attention::ModelConfigMetadata,
    pipeline::{extract_logits, IsqModel, NormalLoadingMetadata, NormalModel},
    utils::progress::NiceProgressBar,
};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub quantization_config: QuantizedConfig,
}

#[derive(Clone)]
struct CausalSelfAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    rotary_emb: Arc<RotaryEmbedding>,
    max_seq_len: usize,
}

impl CausalSelfAttention {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut crate::pipeline::LayerCaches,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let q = self.q_proj.matmul(x)?;
        let k = self.k_proj.matmul(x)?;
        let v = self.v_proj.matmul(x)?;

        let mut q = q.reshape((b_sz * seq_len, self.num_attention_heads, self.head_dim))?;
        let mut k = k.reshape((b_sz * seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = if seq_len != 1 {
            v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
        } else {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            v.reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?
        };

        self.rotary_emb
            .forward(seqlen_offsets, &start_offsets_kernel, &mut q, &mut k, b_sz)?;

        if q.rank() == 3 && seq_len != 1 {
            q = q
                .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        } else if q.rank() == 3 {
            // Optimization for seqlen = 1, avoid transpose and just modify reshape dims
            q = q
                .reshape((b_sz, self.num_attention_heads, seq_len, self.head_dim))?
                .contiguous()?;
            k = k
                .reshape((b_sz, self.num_key_value_heads, seq_len, self.head_dim))?
                .contiguous()?;
        }

        let (k, v) =
            crate::pipeline::Cache::update_kv_cache(&mut kv_cache[block_idx], k, v, false)?;

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;

        let y = ScaledDotProductAttention.run_attention(
            &q,
            &k,
            &v,
            self.num_attention_heads,
            self.head_dim,
            attention_mask.clone().as_ref(),
            self.use_flash_attn,
            b_sz,
            seq_len,
        )?;

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        self.o_proj.matmul(&y)
    }

    fn load(vb: VarBuilder, cfg: &Config, rope: Arc<RotaryEmbedding>) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj =
            gptq_linear_no_bias(size_in, size_q, &cfg.quantization_config, vb.pp("q_proj"))?;
        let k_proj =
            gptq_linear_no_bias(size_in, size_kv, &cfg.quantization_config, vb.pp("k_proj"))?;
        let v_proj =
            gptq_linear_no_bias(size_in, size_kv, &cfg.quantization_config, vb.pp("v_proj"))?;
        let o_proj =
            gptq_linear_no_bias(size_q, size_in, &cfg.quantization_config, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            rotary_emb: rope,
            max_seq_len: cfg.max_position_embeddings,
        })
    }
}

#[derive(Clone)]
struct Mlp {
    c_fc1: Arc<dyn QuantMethod>,
    c_fc2: Arc<dyn QuantMethod>,
    c_proj: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 =
            gptq_linear_no_bias(h_size, i_size, &cfg.quantization_config, vb.pp("gate_proj"))?;
        let c_fc2 =
            gptq_linear_no_bias(h_size, i_size, &cfg.quantization_config, vb.pp("up_proj"))?;
        let c_proj =
            gptq_linear_no_bias(i_size, h_size, &cfg.quantization_config, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (candle_nn::ops::silu(&self.c_fc1.matmul(x)?)? * self.c_fc2.matmul(x)?)?;
        self.c_proj.matmul(&x)
    }
}

impl AnyMoeTrainableLayer for Mlp {}

struct Block {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut crate::pipeline::LayerCaches,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(
            &x,
            attention_mask,
            seqlen_offsets,
            start_offsets_kernel,
            block_idx,
            kv_cache,
        )? + residual)?;
        let residual = &x;
        self.mlp.forward(&self.rms_2.forward(&x)?)? + residual
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        rope: Arc<RotaryEmbedding>,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
            rope,
        )?;
        let mlp = Mlp::load(mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq), cfg)?;
        let rms_1 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), loading_isq),
        )?;
        let rms_2 = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), loading_isq),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }
}

pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: QMatMul,
    pub kv_cache: crate::pipeline::Cache,
    pub device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
}

impl Llama {
    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(input_ids)?;
        let mut cache = self.kv_cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias(
            input_ids,
            &cache,
            x.dtype(),
            self.blocks[0].attn.num_attention_heads,
        )?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = self.mapper.map(x, block_idx)?;
            x = block.forward(
                &x,
                &mask.clone().map(|m| m.to_device(x.device()).unwrap()),
                seqlen_offsets,
                start_offsets_kernel.clone(),
                block_idx,
                &mut cache,
            )?;
        }
        let x = x.to_device(&self.device)?;
        let mut x = self.ln_f.forward(&x)?;
        if matches!(self.lm_head, QMatMul::QTensor(_)) {
            x = x.to_dtype(DType::F32)?;
        }
        let logits = MatMul.qmatmul(&x, &self.lm_head)?;
        extract_logits(&logits, context_lens)
    }

    pub fn new(
        cfg: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;

        let wte = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb.pp("model.embed_tokens"), false),
        )?;
        let lm_head = linear(
            cfg.hidden_size,
            cfg.vocab_size,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;
        let ln_f = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb.pp("model.norm"), false),
        )?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let blocks: Vec<_> =
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
                .into_iter()
                .map(|i| {
                    let rotary_emb = Arc::new(
                        RotaryEmbedding::new(
                            cfg.rope_theta,
                            head_dim,
                            cfg.max_position_embeddings,
                            mapper
                                .device_for(i, false)
                                .unwrap_or(&normal_loading_metadata.real_device),
                            is_gptx,
                            vb.dtype(),
                        )
                        .expect("Failed to create RoPE"),
                    );
                    Block::load(
                        vb.pp(&format!("model.layers.{i}")),
                        cfg,
                        &*mapper,
                        i,
                        normal_loading_metadata.loading_isq,
                        rotary_emb,
                    )
                    .expect("Failed to load block.")
                })
                .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            kv_cache: crate::pipeline::Cache::new(cfg.num_hidden_layers, false),
            device: normal_loading_metadata.real_device,
            mapper,
        })
    }
}

impl IsqModel for Llama {
    fn get_matmuls(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        unreachable!()
    }
    fn get_biases(&mut self) -> (Vec<(Option<&mut Tensor>, Option<usize>)>, &dyn DeviceMapper) {
        unreachable!()
    }
}

impl NormalModel for Llama {
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
    fn cache(&self) -> &crate::pipeline::Cache {
        &self.kv_cache
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
        todo!()
    }
}

impl AnyMoeBaseModelMixin for Llama {}
