//! Adaptive-MLP connector `mlp_AR` (class `Projector`), bridges the SigLIP vision tower to the
//! ERNIE text hidden size. Faithful to the list branch of the native transformers 5.13 forward
//! (`vision_return_embed_list=True`).
//!
//! Per image, features `[t*h*w, 1152]` (already through vision `post_layernorm`) go through
//! `pre_norm` LayerNorm(1152, eps 1e-5, NOT the tower's 1e-6) per patch before the merge, then a
//! 2x2 spatial merge via einops `(t h p1 w p2) d -> (t h w) (p1 p2 d)` giving `[t*(h/2)*(w/2), 4608]`
//! (the 4608 = concat [TL | TR | BL | BR] x 1152, p1=row-in-block outer, p2=col-in-block, d inner),
//! then `linear_1[4608->4608]` -> erf GELU (`.gelu_erf()`, exact, the vision MLP's tanh GELU is a
//! different activation) -> `linear_2[4608->1024]`.

use crate::layers::{layer_norm, linear};
use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

/// The `mlp_AR` connector. `merge_size` is the per-axis spatial merge (2); `vision_hidden` (1152)
/// is the SigLIP channel dim; `text_hidden` (1024) the ERNIE hidden the output lands in.
pub struct Connector {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
    merge_size: usize,
    vision_hidden: usize,
}

impl Connector {
    /// `vb` is the connector root (`mlp_AR.*`).
    pub fn load(
        vb: ShardedVarBuilder,
        vision_hidden: usize,
        merge_size: usize,
        text_hidden: usize,
    ) -> Result<Self> {
        let merged = vision_hidden * merge_size * merge_size; // 1152*2*2 = 4608
        Ok(Self {
            // eps 1e-5, deliberately different from the vision tower's 1e-6.
            pre_norm: layer_norm(vision_hidden, 1e-5, vb.pp("pre_norm"))?,
            linear_1: linear(merged, merged, vb.pp("linear_1"))?,
            linear_2: linear(merged, text_hidden, vb.pp("linear_2"))?,
            merge_size,
            vision_hidden,
        })
    }

    /// `x`: vision post-LN features `[t*h*w, vision_hidden]`, grid `(t, h, w)` (still images: t=1).
    /// Returns `[t*(h/2)*(w/2), text_hidden]`, one row per merged 2x2 patch block.
    pub fn forward(&self, x: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?; // per-patch LayerNorm, BEFORE the merge
        let x = self.merge(&x, t, h, w)?; // [t*(h/2)*(w/2), 4608]
        let x = self.linear_1.forward(&x)?.gelu_erf()?; // exact erf GELU, not tanh
        self.linear_2.forward(&x)
    }

    /// einops `(t h p1 w p2) d -> (t h w) (p1 p2 d)` via reshape->permute->reshape.
    ///
    /// Row-major `[t*h*w, d]` flattens as nesting `(t, h/2, p1, w/2, p2, d)`, so a plain reshape
    /// splits the axes with no data movement. Permute to `(t, h/2, w/2, p1, p2, d)` groups the
    /// output-token dims first and the 4-patch block `(p1, p2, d)` last, then reshape collapses
    /// each to `[t*(h/2)*(w/2), 4*d]` in [TL|TR|BL|BR] order. `.contiguous()` before the final
    /// reshape because the permute left the tensor non-contiguous.
    fn merge(&self, x: &Tensor, t: usize, h: usize, w: usize) -> Result<Tensor> {
        let m = self.merge_size;
        let d = self.vision_hidden;
        x.reshape((t, h / m, m, w / m, m, d))? // (t, hb, p1, wb, p2, d)
            .permute((0, 1, 3, 2, 4, 5))? // (t, hb, wb, p1, p2, d)
            .contiguous()?
            .reshape((t * (h / m) * (w / m), m * m * d))
    }
}
