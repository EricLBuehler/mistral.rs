#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, VarBuilder};
use mistralrs_quant::MatMul;

use crate::{layers::GetFloatInfo, layers_masker::masked_fill, utils::unvarbuilder::UnVarBuilder};

const DEFAULT_MAX_SIZE: (usize, usize) = (70, 70);

fn get_2d_sincos_pos_embed(
    embed_dim: usize,
    image_size: (usize, usize),
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let (grid_h_size, grid_w_size) = image_size;
    let grid_h = Tensor::arange(0f32, grid_h_size as f32, device)?;
    let grid_w = Tensor::arange(0f32, grid_w_size as f32, device)?;
    // Original code uses np.meshgrid, xy is default
    let grid = Tensor::meshgrid(&[grid_w, grid_h], true)?;
    let grid = Tensor::stack(&grid, 0)?;

    get_2d_sincos_pos_embed_from_grid(embed_dim, &grid)?.to_dtype(dtype)
}

fn get_2d_sincos_pos_embed_from_grid(embed_dim: usize, grid: &Tensor) -> Result<Tensor> {
    assert_eq!(embed_dim % 2, 0);

    let emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, &grid.i(0)?)?;
    let emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, &grid.i(1)?)?;

    Tensor::cat(&[emb_h, emb_w], D::Minus1)
}

fn get_1d_sincos_pos_embed_from_grid_new(embed_dim: usize, pos: &Tensor) -> Result<Tensor> {
    let inv_freq: Vec<_> = (0..embed_dim)
        .step_by(2)
        .map(|i| 1f32 / 10_000f32.powf(i as f32 / embed_dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    let omega = Tensor::from_vec(inv_freq, (1, inv_freq_len), pos.device())?;

    let (h, w) = pos.dims2()?;

    let mut out = pos
        .reshape(((), 1))?
        .matmul(&omega.reshape((1, ()))?)
        .unwrap();

    out = out.reshape((h, w, ()))?;

    let emb_sin = out.sin()?;
    let emb_cos = out.cos()?;

    Tensor::cat(&[emb_sin, emb_cos], D::Minus1)
}

struct SinCos2dPosEmbed {
    pos_embed: Tensor,
    max_size: (usize, usize),
}

pub struct Resampler {
    query: Tensor,
    kv_proj: Option<Linear>,
    proj: Tensor,
    ln_q: LayerNorm,
    ln_kv: LayerNorm,
    ln_post: LayerNorm,
    attn: MultiheadAttention,
    sincos_pos_embed: Arc<Mutex<SinCos2dPosEmbed>>,
    embed_dim: usize,
}

impl Resampler {
    pub fn new(
        num_queries: usize,
        embed_dim: usize,
        num_heads: usize,
        kv_dim: usize,
        _adaptive: bool,
        max_size: Option<(usize, usize)>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let max_size = max_size.unwrap_or(DEFAULT_MAX_SIZE);

        let query = vb.get((num_queries, embed_dim), "query")?;
        let kv_proj = if kv_dim != embed_dim {
            Some(candle_nn::linear_no_bias(
                kv_dim,
                embed_dim,
                vb.pp("kv_proj"),
            )?)
        } else {
            None
        };
        let ln_q = layer_norm(embed_dim, 1e-6, vb.pp("ln_q"))?;
        let ln_kv = layer_norm(embed_dim, 1e-6, vb.pp("ln_kv"))?;
        let ln_post = layer_norm(embed_dim, 1e-6, vb.pp("ln_post"))?;
        let proj = vb.get((embed_dim, embed_dim), "proj")?;
        let attn = MultiheadAttention::new(embed_dim, num_heads, vb.pp("attn"))?;

        let pos_embed = Arc::new(Mutex::new(SinCos2dPosEmbed {
            pos_embed: get_2d_sincos_pos_embed(embed_dim, max_size, vb.device(), vb.dtype())?,
            max_size,
        }));

        Ok(Self {
            query,
            kv_proj,
            proj,
            ln_q,
            ln_kv,
            ln_post,
            attn,
            sincos_pos_embed: pos_embed,
            embed_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, tgt_sizes_vec: &[Vec<u32>]) -> Result<Tensor> {
        let mut pos_embed_cache = self.sincos_pos_embed.lock().unwrap();

        let bs = x.dim(0)?;
        let device = x.device();

        assert_eq!(bs, tgt_sizes_vec.len());

        let tgt_sizes_vec_0 = tgt_sizes_vec.iter().map(|x| x[0]).collect::<Vec<_>>();
        let tgt_sizes_vec_1 = tgt_sizes_vec.iter().map(|x| x[1]).collect::<Vec<_>>();
        let patch_len = tgt_sizes_vec_0
            .iter()
            .zip(&tgt_sizes_vec_1)
            .map(|(x, y)| x * y)
            .collect::<Vec<_>>();

        // Adjust/recompute pos embeds
        {
            let max_h = *tgt_sizes_vec_0.iter().max().unwrap() as usize;
            let max_w = *tgt_sizes_vec_1.iter().max().unwrap() as usize;

            if max_h > pos_embed_cache.max_size.0 || max_w > pos_embed_cache.max_size.1 {
                pos_embed_cache.max_size = (
                    max_h.max(pos_embed_cache.max_size.0),
                    max_w.max(pos_embed_cache.max_size.1),
                );
                pos_embed_cache.pos_embed = get_2d_sincos_pos_embed(
                    self.embed_dim,
                    pos_embed_cache.max_size,
                    device,
                    x.dtype(),
                )?;
            }
        }

        let max_patch_len = *patch_len.iter().max().unwrap() as usize;

        let mut key_padding_mask = Tensor::zeros((bs, max_patch_len), DType::U8, device)?;

        let mut pos_embed = Vec::new();
        for (i, tgt_sizes_vec_i) in tgt_sizes_vec.iter().enumerate().take(bs) {
            let (tgt_h, tgt_w) = (tgt_sizes_vec_i[0] as usize, tgt_sizes_vec_i[1] as usize);
            pos_embed.push(
                pos_embed_cache
                    .pos_embed
                    .i((..tgt_h, ..tgt_w, ..))?
                    .reshape((tgt_h * tgt_w, ()))?,
            );

            let n = patch_len[i] as usize;
            if n != max_patch_len {
                key_padding_mask = key_padding_mask.slice_assign(
                    &[&i, &(n..)],
                    &Tensor::ones((1, max_patch_len - n), DType::U8, device)?,
                )?;
            }
        }

        let lens = pos_embed
            .iter()
            .map(|emb| emb.dim(0))
            .collect::<Result<Vec<_>>>()?;
        let max_len = lens.into_iter().max().expect("No pixe values somehow?");
        pos_embed = pos_embed
            .into_iter()
            .map(|emb| emb.pad_with_zeros(0, 0, max_len - emb.dim(0)?))
            .collect::<Result<Vec<_>>>()?;
        let pos_embed = Tensor::stack(&pos_embed, 0)?;

        let mut x = if let Some(kv_proj) = &self.kv_proj {
            x.apply(kv_proj)?
        } else {
            x.clone()
        };
        x = x.apply(&self.ln_kv)?;

        let q = self.query.apply(&self.ln_q)?;

        let mut out = self.attn.forward(
            &self.repeat_q_bs(&q, bs)?,
            &(&x + &pos_embed)?,
            &x,
            Some(key_padding_mask),
            None,
        )?;

        out = out.apply(&self.ln_post)?;
        out.broadcast_matmul(&self.proj)
    }

    fn repeat_q_bs(&self, q: &Tensor, n: usize) -> Result<Tensor> {
        q.unsqueeze(0)?.repeat((n, 1, 1))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_attn = uvb.pp("attn");
        uvb_attn.pp("out_proj").add(&self.attn.out_proj);
        uvb_attn.add_tensor("in_proj_weight", self.attn.in_proj_weight.clone());
        uvb_attn.add_tensor("in_proj_bias", self.attn.in_proj_bias.clone());

        uvb.pp("ln_kv").add(&self.ln_kv);
        uvb.pp("ln_post").add(&self.ln_post);
        uvb.pp("ln_q").add(&self.ln_q);
        uvb.add_tensor("proj", self.proj.clone());
        uvb.add_tensor("query", self.query.clone());

        uvb.to_safetensors()
    }
}

struct MultiheadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    in_proj_weight: Tensor,
    in_proj_bias: Tensor,
}

impl MultiheadAttention {
    fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj_bias = vb.get(embed_dim * 3, "in_proj_bias")?;
        let in_proj_weight = vb.get((embed_dim * 3, embed_dim), "in_proj_weight")?;
        let q_proj = Linear::new(
            in_proj_weight.i(0..embed_dim)?,
            Some(in_proj_bias.i(0..embed_dim)?),
        );
        let k_proj = Linear::new(
            in_proj_weight.i(embed_dim..embed_dim * 2)?,
            Some(in_proj_bias.i(embed_dim..embed_dim * 2)?),
        );
        let v_proj = Linear::new(
            in_proj_weight.i(embed_dim * 2..embed_dim * 3)?,
            Some(in_proj_bias.i(embed_dim * 2..embed_dim * 3)?),
        );
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim: embed_dim / num_heads,
            in_proj_weight,
            in_proj_bias,
        })
    }

    fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        key_padding_mask: Option<Tensor>,
        mut attn_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let (bs, q_seq, _) = q.dims3()?;
        let (_, kv_seq, _) = k.dims3()?;

        let mut q = q.apply(&self.q_proj)?;
        let mut k = k.apply(&self.k_proj)?;
        let mut v = v.apply(&self.v_proj)?;

        // Merge key padding and attention masks
        if let Some(mut key_padding_mask) = key_padding_mask {
            key_padding_mask = key_padding_mask
                .reshape((bs, 1, 1, kv_seq))?
                .repeat((1, self.num_heads, 1, 1))?
                .reshape((bs * self.num_heads, 1, kv_seq))?;
            if let Some(attn_mask) = attn_mask.as_mut() {
                *attn_mask = attn_mask.broadcast_add(&key_padding_mask)?;
            } else {
                attn_mask = Some(key_padding_mask);
            }
        }

        q = q
            .reshape((bs, q_seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        k = k
            .reshape((bs, kv_seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        v = v
            .reshape((bs, kv_seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut y = {
            let mut att =
                MatMul.matmul_affine_mul(&q, &k.t()?, (1. / self.head_dim as f64).sqrt())?;

            att = match attn_mask {
                Some(mask) => {
                    let mask = mask.reshape((bs, self.num_heads, (), kv_seq))?;
                    masked_fill(&att, &mask, att.dtype().finfo()?.min as f32)?
                }
                None => att,
            };
            candle_nn::ops::inplace_softmax_last_dim(&mut att)?;
            MatMul.matmul(&att, &v)?
        };

        y = y.transpose(1, 2)?.reshape((bs, q_seq, ()))?;
        y.apply(&self.out_proj)
    }
}
