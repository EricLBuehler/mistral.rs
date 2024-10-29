use candle_core::{DType, Result, Tensor, D};
use candle_nn::{layer_norm, Activation, LayerNorm, Linear, Module, Sequential, VarBuilder};

use super::config::VisionConfig;

struct PatchEmbed {
    cfg: VisionConfig,
    proj: (),
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L272
impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        todo!("Need conv3d!");
        Ok(Self {
            cfg: cfg.clone(),
            proj: (),
        })
    }
}

// https://github.com/huggingface/transformers/blob/a769ed45e17c44fd17b85c025863c4e4f2f73634/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L314
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl VisionMlp {
    fn new(dim: usize, hidden_dim: usize, act: Activation, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(dim, hidden_dim, vb.pp("fc1"))?,
            fc2: candle_nn::linear(hidden_dim, dim, vb.pp("fc2"))?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(xs)?)?)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(xs: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let orig_ty = xs.dtype();
    let xs = xs.to_dtype(DType::F32)?;
    let cos = freqs
        .cos()?
        .unsqueeze(1)?
        .repeat((1, 1, 2))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;
    let sin = freqs
        .sin()?
        .unsqueeze(1)?
        .repeat((1, 1, 2))?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;
    ((&xs * cos) + (rotate_half(&xs)? * sin)?)?.to_dtype(orig_ty)
}

// https://github.com/huggingface/transformers/blob/a769ed45e17c44fd17b85c025863c4e4f2f73634/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L325
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: candle_nn::linear(dim, dim * 3, vb.pp("qkv"))?,
            proj: candle_nn::linear_no_bias(dim, dim, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }
    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: Vec<u32>,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let (q, k, v) = {
            let qkv = self
                .qkv
                .forward(xs)?
                .reshape((seq_len, 3, self.num_heads, ()))?
                .permute((1, 0, 2, 3))?
                .chunk(3, 0)?;
            (qkv[0].clone(), qkv[1].clone(), qkv[2].clone())
        };

        let q = apply_rotary_pos_emb_vision(&q.unsqueeze(0)?, rotary_pos_emb)?.squeeze(0)?;
        let k = apply_rotary_pos_emb_vision(&k.unsqueeze(0)?, rotary_pos_emb)?.squeeze(0)?;

        let mut attention_mask =
            Tensor::full(f32::MIN, (1, seq_len, seq_len), q.device())?.to_dtype(q.dtype())?;
        for i in 1..cu_seqlens.len() {
            let a = cu_seqlens[i - 1] as usize;
            let b = cu_seqlens[i] as usize;
            attention_mask = attention_mask.slice_assign(
                &[&.., &(a..b), &(a..b)],
                &Tensor::zeros((1, b - a, b - a), q.dtype(), q.device())?,
            )?;
        }

        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;

        let att = {
            let att = (q.matmul(&k.transpose(1, 2)?)? / (self.head_dim as f64).sqrt())?;
            let att = (att + attention_mask)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v)?.transpose(0, 1)?.reshape((seq_len, ()))?
        };

        self.proj.forward(&att)
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L418
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    mlp: VisionMlp,
    attn: VisionAttention,
}

impl VisionBlock {
    fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?;
        let norm2 = layer_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?;

        let mlp_hidden_dim = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
        let mlp = VisionMlp::new(cfg.embed_dim, mlp_hidden_dim, cfg.hidden_act, vb.pp("mlp"))?;
        let attn = VisionAttention::new(cfg.embed_dim, cfg.num_heads, vb.pp("attn"))?;

        Ok(Self {
            norm1,
            norm2,
            mlp,
            attn,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: Vec<u32>,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let xs = (xs
            + self
                .attn
                .forward(&self.norm1.forward(&xs)?, cu_seqlens, rotary_pos_emb)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }
}

struct PatchMerger {
    ln_q: LayerNorm,
    mlp: Sequential,
    hidden_size: usize,
}

impl PatchMerger {
    pub fn new(
        dim: usize,
        context_dim: usize,
        spatial_merge_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_size = context_dim * spatial_merge_size.pow(2);
        let mut mlp = candle_nn::seq();
        mlp = mlp.add(candle_nn::linear_no_bias(
            hidden_size,
            hidden_size,
            vb.pp("mlp.0"),
        )?);
        mlp = mlp.add(candle_nn::Activation::Gelu);
        mlp = mlp.add(candle_nn::linear_no_bias(hidden_size, dim, vb.pp("mlp.2"))?);
        Ok(Self {
            ln_q: layer_norm(context_dim, 1e-6, vb.pp("ln_q"))?,
            mlp,
            hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.ln_q)?
            .reshape(((), self.hidden_size))?
            .apply(&self.mlp)
    }
}

pub struct Qwen2VLVisionModel {
    blocks: Vec<VisionBlock>,
    patch_merger: PatchMerger,
}

impl Qwen2VLVisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")))?);
        }
        let patch_merger = PatchMerger::new(
            cfg.hidden_size,
            cfg.embed_dim,
            cfg.spatial_merge_size,
            vb.pp("patch_merger"),
        )?;
        todo!()
    }
}
