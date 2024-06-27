//LLaMA without fused RoPE
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{quantized::QMatMul, DType, Device, Result, Tensor};
use candle_nn::{embedding, linear_no_bias as linear, Embedding, Module, VarBuilder};

use crate::{
    device_map::DeviceMapper,
    layers::{repeat_kv, CausalMasker, MatMul, RmsNorm, ScaledDotProductAttention},
    models::llama::Config,
    pipeline::{extract_logits, IsqModel, NormalLoadingMetadata, NormalModel},
    utils::progress::NiceProgressBar,
};

use super::{LLaVALLM, OrdinaryRoPE};

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    max_seq_len: usize,
}

impl CausalSelfAttention {
    fn forward(
        &self,
        x: &Tensor,
        attention_mask: &Option<Tensor>,
        seqlen_offsets: &[usize],
        _start_offsets_kernel: Tensor,
        block_idx: usize,
        kv_cache: &mut crate::pipeline::LayerCaches,
        rope_parameter: (&Tensor, &Tensor),
    ) -> Result<Tensor> {
        let (b_sz, seq_len, hidden_size) = x.dims3()?;

        let original_dtype = x.dtype();
        let mut x = x.clone();
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            x = x.to_dtype(DType::F32)?;
        }
        let mut q = MatMul.qmatmul(&x, &self.q_proj)?;
        let mut k = MatMul.qmatmul(&x, &self.k_proj)?;
        let mut v = MatMul.qmatmul(&x, &self.v_proj)?;
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }
        let mut q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let mut k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        q = OrdinaryRoPE::forward(&q, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        k = OrdinaryRoPE::forward(&k, seqlen_offsets[0], rope_parameter.0, rope_parameter.1)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (k, v) =
            crate::pipeline::Cache::update_kv_cache(&mut kv_cache[block_idx], k, v, false)?;

        let k = repeat_kv(k, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;
        let v = repeat_kv(v, self.num_attention_heads / self.num_key_value_heads)?.contiguous()?;

        let mut y = ScaledDotProductAttention.run_attention(
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

        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            y = y.to_dtype(DType::F32)?;
        }
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let mut y = MatMul.qmatmul(&y, &self.o_proj)?;
        if matches!(self.q_proj, QMatMul::QTensor(_)) {
            y = y.to_dtype(original_dtype)?;
        }
        Ok(y)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = linear(size_in, size_q, vb.pp("q_proj"))?;
        let k_proj = linear(size_in, size_kv, vb.pp("k_proj"))?;
        let v_proj = linear(size_in, size_kv, vb.pp("v_proj"))?;
        let o_proj = linear(size_q, size_in, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj: QMatMul::Tensor(q_proj.weight().clone()),
            k_proj: QMatMul::Tensor(k_proj.weight().clone()),
            v_proj: QMatMul::Tensor(v_proj.weight().clone()),
            o_proj: QMatMul::Tensor(o_proj.weight().clone()),
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            max_seq_len: cfg.max_position_embeddings,
        })
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    c_fc1: QMatMul,
    c_fc2: QMatMul,
    c_proj: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let original_dtype = x.dtype();
        let mut x = x.clone();
        if matches!(self.c_fc1, QMatMul::QTensor(_)) {
            x = x.to_dtype(DType::F32)?;
        }
        let x = (candle_nn::ops::silu(&MatMul.qmatmul(&x, &self.c_fc1)?)?
            * MatMul.qmatmul(&x, &self.c_fc2)?)?;
        let mut res = MatMul.qmatmul(&x, &self.c_proj)?;
        if matches!(self.c_fc1, QMatMul::QTensor(_)) {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        let c_fc1 = linear(h_size, i_size, vb.pp("gate_proj"))?;
        let c_fc2 = linear(h_size, i_size, vb.pp("up_proj"))?;
        let c_proj = linear(i_size, h_size, vb.pp("down_proj"))?;
        Ok(Self {
            c_fc1: QMatMul::Tensor(c_fc1.weight().clone()),
            c_fc2: QMatMul::Tensor(c_fc2.weight().clone()),
            c_proj: QMatMul::Tensor(c_proj.weight().clone()),
        })
    }
}

#[derive(Debug, Clone)]
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
        rope_parameters: (&Tensor, &Tensor),
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
            rope_parameters,
        )? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(
        vb: VarBuilder,
        cfg: &Config,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
    ) -> Result<Self> {
        let attn = CausalSelfAttention::load(
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            cfg,
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

#[derive(Debug)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: QMatMul,
    pub kv_cache: crate::pipeline::Cache,
    pub device: Device,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    rope_parameters: (Tensor, Tensor),
}

impl Llama {
    pub fn forward_input(
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
                (&self.rope_parameters.0, &self.rope_parameters.1),
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
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata
            .mapper
            .into_mapper(cfg.num_hidden_layers, &normal_loading_metadata.real_device)?;
        //  let vb = vb.set_dtype(mapper.get_min_dtype()?);

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
        
        let blocks: Vec<_> = NiceProgressBar(0..cfg.num_hidden_layers, "Loading repeating layers")
            .into_iter()
            .map(|i| {
                Block::load(
                    vb.pp(&format!("model.layers.{i}")),
                    cfg,
                    &*mapper,
                    i,
                    normal_loading_metadata.loading_isq,
                )
                .expect("Failed to load block.")
            })
            .collect();
        let rope_parameters = OrdinaryRoPE::create_parameters(
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            vb.dtype(),
            &normal_loading_metadata.real_device,
        )?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head: QMatMul::Tensor(lm_head.weight().clone()),
            kv_cache: crate::pipeline::Cache::new(cfg.num_hidden_layers, false),
            device: normal_loading_metadata.real_device,
            mapper,
            rope_parameters,
        })
    }
}

impl IsqModel for Llama {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.blocks.iter_mut().enumerate() {
            tensors.push((&mut layer.attn.q_proj, Some(i)));
            tensors.push((&mut layer.attn.k_proj, Some(i)));
            tensors.push((&mut layer.attn.v_proj, Some(i)));
            tensors.push((&mut layer.attn.o_proj, Some(i)));
            tensors.push((&mut layer.mlp.c_fc1, Some(i)));
            tensors.push((&mut layer.mlp.c_fc2, Some(i)));
            tensors.push((&mut layer.mlp.c_proj, Some(i)));
        }
        (tensors, &*self.mapper)
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
        self.forward_input(
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
}

impl LLaVALLM for Llama {
    fn embed(&self, x: &Tensor) -> Result<Tensor> {
        self.wte.forward(x)
    }
    fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let mut x = input_embed.clone();
        let mut cache = self.kv_cache.lock();
        let mask = CausalMasker.make_causal_mask_as_attn_bias_with_embed_tensor(
            &x,
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
                (&self.rope_parameters.0, &self.rope_parameters.1),
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
}
