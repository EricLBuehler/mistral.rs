use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::{ColumnParallelLayer, QuantMethod, RowParallelLayer, ShardedVarBuilder};

use crate::{
    layers::{self, Activation, Conv3dConfig, Conv3dNoBias, MatMul, RmsNorm},
    ops::RepeatInterleaveOp,
};

use super::config::VisionConfig;

struct PatchEmbed {
    proj: Conv3dNoBias,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L272
impl PatchEmbed {
    fn new(cfg: &VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        if cfg.temporal_patch_size != 2 {
            candle_core::bail!("Only support temporal patch size of 2");
        }
        Ok(Self {
            proj: Conv3dNoBias::new(
                cfg.in_chans,
                cfg.hidden_size,
                [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size],
                Conv3dConfig {
                    stride: cfg.patch_size,
                    ..Default::default()
                },
                vb.pp("proj"),
            )?,
            in_channels: cfg.in_chans,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        xs.apply(&self.proj)?.reshape(((), self.hidden_size))
    }
}

// https://github.com/huggingface/transformers/blob/6a1ab634b6886b6560b0502e7a305c8cd881732e/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L75
struct VisionMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act: Activation,
}

impl VisionMlp {
    fn new(
        dim: usize,
        hidden_dim: usize,
        act: Activation,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: ColumnParallelLayer::new(
                dim,
                hidden_dim,
                &None,
                true,
                comm,
                vb.pp("gate_proj"),
            )?,
            up_proj: ColumnParallelLayer::new(
                dim,
                hidden_dim,
                &None,
                true,
                comm,
                vb.pp("up_proj"),
            )?,
            down_proj: RowParallelLayer::new(
                hidden_dim,
                dim,
                &None,
                true,
                comm,
                vb.pp("down_proj"),
            )?,
            act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let lhs = self
            .gate_proj
            .forward(&xs.unsqueeze(0)?)?
            .apply(&self.act)?;
        let rhs = self.up_proj.forward(&xs.unsqueeze(0)?)?;
        let mut res = self.down_proj.forward(&(lhs * rhs)?)?;

        res = res.squeeze(0)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn apply_rotary_pos_emb_vision(xs: &Tensor, freqs: &Tensor) -> Result<Tensor> {
    let cos = freqs.cos()?.unsqueeze(D::Minus2)?.to_dtype(xs.dtype())?;
    let sin = freqs.sin()?.unsqueeze(D::Minus2)?.to_dtype(xs.dtype())?;

    xs.broadcast_mul(&cos)? + rotate_half(xs)?.broadcast_mul(&sin)
}

// https://github.com/huggingface/transformers/blob/a769ed45e17c44fd17b85c025863c4e4f2f73634/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L325
struct VisionAttention {
    qkv: Arc<dyn QuantMethod>,
    proj: Arc<dyn QuantMethod>,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn new(dim: usize, num_heads: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            qkv: mistralrs_quant::linear(dim, dim * 3, &None, vb.pp("qkv"))?,
            proj: mistralrs_quant::linear(dim, dim, &None, vb.pp("proj"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let (mut q, mut k, mut v) = {
            let qkv = self
                .qkv
                .forward(&xs.unsqueeze(0)?)?
                .reshape((seq_len, 3, self.num_heads, ()))?
                .permute((1, 0, 2, 3))?
                .chunk(3, 0)?;
            (qkv[0].squeeze(0)?, qkv[1].squeeze(0)?, qkv[2].squeeze(0)?)
        };

        q = apply_rotary_pos_emb_vision(&q.unsqueeze(0)?, rotary_pos_emb)?
            .squeeze(0)?
            .to_dtype(q.dtype())?;
        k = apply_rotary_pos_emb_vision(&k.unsqueeze(0)?, rotary_pos_emb)?
            .squeeze(0)?
            .to_dtype(q.dtype())?;

        q = q.transpose(0, 1)?.contiguous()?;
        k = k.transpose(0, 1)?.contiguous()?;
        v = v.transpose(0, 1)?.contiguous()?;

        let att = {
            let mut att =
                (MatMul.matmul(&q, &k.transpose(1, 2)?)? / (self.head_dim as f64).sqrt())?;
            att = match attention_mask {
                Some(m) => att.broadcast_add(m)?,
                None => att,
            };
            att = candle_nn::ops::softmax_last_dim(&att)?;
            MatMul
                .matmul(&att, &v)?
                .transpose(0, 1)?
                .reshape((seq_len, ()))?
                .to_dtype(xs.dtype())?
        };

        self.proj.forward(&att.unsqueeze(0)?)?.squeeze(0)
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L418
struct VisionBlock {
    norm1: RmsNorm,
    norm2: RmsNorm,
    mlp: VisionMlp,
    attn: VisionAttention,
}

impl VisionBlock {
    fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let norm1 = RmsNorm::new(cfg.hidden_size, 1e-6, vb.pp("norm1"))?;
        let norm2 = RmsNorm::new(cfg.hidden_size, 1e-6, vb.pp("norm2"))?;

        let mlp = VisionMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_act,
            vb.pp("mlp"),
            comm,
        )?;
        let attn = VisionAttention::new(cfg.hidden_size, cfg.num_heads, vb.pp("attn"))?;

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
        attention_mask: Option<&Tensor>,
        rotary_pos_emb: &Tensor,
    ) -> Result<Tensor> {
        let xs = (xs
            + self
                .attn
                .forward(&self.norm1.forward(xs)?, attention_mask, rotary_pos_emb)?)?;
        &xs + self.mlp.forward(&self.norm2.forward(&xs)?)?
    }
}

struct PatchMerger {
    ln_q: RmsNorm,
    mlp0: Linear,
    mlp2: Linear,
    out_hidden_size: usize,
}

impl PatchMerger {
    pub fn new(
        dim: usize,
        context_dim: usize,
        spatial_merge_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let out_hidden_size = context_dim * spatial_merge_size.pow(2);
        let mlp0 = layers::linear(out_hidden_size, out_hidden_size, vb.pp("mlp.0"))?;
        let mlp2 = layers::linear(out_hidden_size, dim, vb.pp("mlp.2"))?;
        Ok(Self {
            ln_q: RmsNorm::new(context_dim, 1e-6, vb.pp("ln_q"))?,
            mlp0,
            mlp2,
            out_hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.unsqueeze(0)?
            .apply(&self.ln_q)?
            .reshape(((), self.out_hidden_size))?
            .apply(&self.mlp0)?
            .gelu()?
            .apply(&self.mlp2)?
            .squeeze(0)
    }
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq)
    }
}

pub struct Qwen2_5VLVisionModel {
    blocks: Vec<VisionBlock>,
    patch_merger: PatchMerger,
    patch_embed: PatchEmbed,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indices: Vec<usize>,
}

impl Qwen2_5VLVisionModel {
    pub fn new(
        cfg: &VisionConfig,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.depth {
            blocks.push(VisionBlock::new(cfg, vb.pp(format!("blocks.{i}")), comm)?);
        }

        let patch_merger = PatchMerger::new(
            cfg.out_hidden_size,
            cfg.hidden_size,
            cfg.spatial_merge_size,
            vb.pp("merger"),
        )?;

        let patch_embed = PatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let head_dim = cfg.hidden_size / cfg.num_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, vb.device())?;

        Ok(Self {
            blocks,
            patch_embed,
            patch_merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_merge_unit: cfg.spatial_merge_size * cfg.spatial_merge_size,
            window_size: cfg.window_size,
            patch_size: cfg.patch_size,
            fullatt_block_indices: cfg.fullatt_block_indexes.clone(),
        })
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor, device: &Device) -> Result<Tensor> {
        let mut pos_ids = Vec::new();
        for i_thw in grid_thw.to_vec2::<u32>()? {
            let (t, h, w) = (i_thw[0], i_thw[1], i_thw[2]);
            let mut hpos_ids = Tensor::arange(0, h, device)?
                .unsqueeze(1)?
                .repeat((1, w as usize))?;
            hpos_ids = hpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            hpos_ids = hpos_ids.permute((0, 2, 1, 3))?;
            hpos_ids = hpos_ids.flatten_all()?;

            let mut wpos_ids = Tensor::arange(0, w, device)?
                .unsqueeze(0)?
                .repeat((h as usize, 1))?;
            wpos_ids = wpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            wpos_ids = wpos_ids.permute((0, 2, 1, 3))?;
            wpos_ids = wpos_ids.flatten_all()?;

            pos_ids.push(Tensor::stack(&[hpos_ids, wpos_ids], D::Minus1)?.repeat((t as usize, 1))?);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0)?;
        let max_grid_size = grid_thw.i((.., 1..))?.max(0)?.max(0)?.to_scalar::<u32>()?;
        let rotary_pos_emb_full = self.rotary_pos_emb.make_embeds(max_grid_size as usize)?;

        assert_eq!(pos_ids.rank(), 2);
        rotary_pos_emb_full
            .index_select(&pos_ids.flatten_all()?, 0)?
            .reshape((pos_ids.dim(0)?, pos_ids.dim(1)?, ()))?
            .flatten_from(1)
    }

    fn get_window_index(&self, grid_thw: &Tensor, device: &Device) -> Result<(Tensor, Vec<i64>)> {
        const PADDING_VALUE: i32 = -100;
        let mut window_index = Vec::new();
        let mut cu_window_seqlens = vec![0];
        let mut window_index_id = 0;
        let vit_merger_window_size = self.window_size / self.spatial_merge_size / self.patch_size;

        for i_thw in grid_thw.to_vec2::<u32>()? {
            let (t, h, w) = (i_thw[0] as usize, i_thw[1] as usize, i_thw[2] as usize);
            let llm_grid_h = h / self.spatial_merge_size;
            let llm_grid_w = w / self.spatial_merge_size;
            let index = Tensor::arange(0i32, (t * llm_grid_h * llm_grid_w) as i32, &Device::Cpu)?
                .reshape((t, llm_grid_h, llm_grid_w))?;
            let pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
            let pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
            let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            let index_padded = {
                let h = Tensor::full(PADDING_VALUE, (t, pad_h, llm_grid_w), &Device::Cpu)?;
                let w = Tensor::full(PADDING_VALUE, (t, pad_h + llm_grid_h, pad_w), &Device::Cpu)?;
                let mut index = Tensor::cat(&[index, h], D::Minus2)?;
                index = Tensor::cat(&[index, w], D::Minus1)?;
                index = index.reshape((
                    t,
                    num_windows_h,
                    vit_merger_window_size,
                    num_windows_w,
                    vit_merger_window_size,
                ))?;
                index = index.permute((0, 1, 3, 2, 4))?.reshape((
                    t,
                    num_windows_h * num_windows_w,
                    vit_merger_window_size,
                    vit_merger_window_size,
                ))?;
                index
            };
            let seqlens = index_padded
                .ne(PADDING_VALUE)?
                .to_dtype(index_padded.dtype())?
                .sum((2, 3))?
                .flatten_all()?;
            let index_new = index_padded
                .flatten_all()?
                .to_vec1::<i32>()?
                .into_iter()
                .filter(|x| *x != PADDING_VALUE)
                .collect::<Vec<_>>();
            window_index.push(Tensor::new(
                index_new
                    .iter()
                    .map(|x| (x + window_index_id) as u32)
                    .collect::<Vec<_>>(),
                device,
            )?);
            let cu_seqlens_tmp = ((seqlens
                .to_dtype(DType::F32)?
                .cumsum(0)?
                .to_dtype(seqlens.dtype())?
                * self.spatial_merge_unit as f64)?
                + cu_window_seqlens[cu_window_seqlens.len() - 1] as f64)?;
            cu_window_seqlens.extend(
                cu_seqlens_tmp
                    .to_vec1::<i32>()?
                    .into_iter()
                    .map(|x| x as i64)
                    .collect::<Vec<_>>(),
            );
            window_index_id += (t * llm_grid_h * llm_grid_w) as i32;
        }

        Ok((Tensor::cat(&window_index, 0)?, cu_window_seqlens))
    }

    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let xs = self
            .patch_embed
            .forward(&xs.to_dtype(self.patch_merger.mlp0.weight().dtype())?)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw, xs.device())?;
        let (window_index, mut cu_window_seqlens) = self.get_window_index(grid_thw, xs.device())?;
        cu_window_seqlens.dedup();

        let seq_len = xs.dims2()?.0;
        let mut xs = xs.reshape((
            seq_len / self.spatial_merge_unit,
            self.spatial_merge_unit,
            (),
        ))?;
        xs = xs.index_select(&window_index, 0)?;
        xs = xs.reshape((seq_len, ()))?;
        let mut rotary_pos_emb = rotary_pos_emb.reshape((
            seq_len / self.spatial_merge_unit,
            self.spatial_merge_unit,
            (),
        ))?;
        rotary_pos_emb = rotary_pos_emb.index_select(&window_index, 0)?;
        rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        rotary_pos_emb = Tensor::cat(&[&rotary_pos_emb; 2], D::Minus1)?;
        rotary_pos_emb = rotary_pos_emb.to_dtype(xs.dtype())?;

        let grid_thw = grid_thw.to_device(&Device::Cpu)?;
        let cu_seqlens = (grid_thw.i((.., 1))? * grid_thw.i((.., 2))?)?
            .repeat_interleave_flat(grid_thw.i((.., 0))?.to_vec1::<u32>()?)?
            .to_dtype(DType::F32)?
            .cumsum(0)?
            .to_dtype(DType::U32)?
            .pad_with_zeros(0, 1, 0)?
            .to_vec1::<u32>()?;

        let seq_len = xs.dim(0)?;
        let attention_mask_full = match &cu_seqlens[..] {
            &[0, len] if len == seq_len as u32 => None,
            cu_seqlens => {
                let mut attention_mask =
                    Tensor::full(f32::MIN, (1, seq_len, seq_len), xs.device())?
                        .to_dtype(xs.dtype())?;
                for i in 1..cu_seqlens.len() {
                    let a = cu_seqlens[i - 1] as usize;
                    let b = cu_seqlens[i] as usize;
                    attention_mask = attention_mask.slice_assign(
                        &[0..attention_mask.dim(0)?, a..b, a..b],
                        &Tensor::zeros((1, b - a, b - a), xs.dtype(), xs.device())?,
                    )?;
                }
                Some(attention_mask)
            }
        };
        let attention_mask_window = match &cu_window_seqlens[..] {
            &[0, len] if len == seq_len as i64 => None,
            cu_seqlens => {
                let mut attention_mask =
                    Tensor::full(f32::MIN, (1, seq_len, seq_len), xs.device())?
                        .to_dtype(xs.dtype())?;
                for i in 1..cu_seqlens.len() {
                    let a = cu_seqlens[i - 1] as usize;
                    let b = cu_seqlens[i] as usize;
                    attention_mask = attention_mask.slice_assign(
                        &[0..attention_mask.dim(0)?, a..b, a..b],
                        &Tensor::zeros((1, b - a, b - a), xs.dtype(), xs.device())?,
                    )?;
                }
                Some(attention_mask)
            }
        };

        for (i, blk) in self.blocks.iter().enumerate() {
            let attention_mask = if self.fullatt_block_indices.contains(&i) {
                attention_mask_full.as_ref()
            } else {
                attention_mask_window.as_ref()
            };
            xs = blk.forward(&xs, attention_mask, &rotary_pos_emb)?;
        }

        xs = self.patch_merger.forward(&xs)?;
        let reverse_indices = window_index.arg_sort_last_dim(true)?;
        xs.index_select(&reverse_indices, 0)
    }
}
