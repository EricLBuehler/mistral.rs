use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Module};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::{
    layers, utils::unvarbuilder::UnVarBuilder, vision_models::gemma4::config::Gemma4VisionConfig,
};

pub struct Gemma4UnifiedVisionEmbedder {
    patch_ln1: LayerNorm,
    patch_dense: Arc<dyn QuantMethod>,
    patch_ln2: LayerNorm,
    pos_embedding: Tensor,
    pos_norm: LayerNorm,
}

impl Gemma4UnifiedVisionEmbedder {
    pub fn new(cfg: &Gemma4VisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let patch_dim = cfg.unified_patch_dim();
        let embed_dim = cfg.mm_embed_dim;
        let patch_ln1 = layers::layer_norm(patch_dim, 1e-5, vb.pp("patch_ln1"))?;
        let patch_dense =
            mistralrs_quant::linear_b(patch_dim, embed_dim, true, &None, vb.pp("patch_dense"))?;
        let patch_ln2 = layers::layer_norm(embed_dim, 1e-5, vb.pp("patch_ln2"))?;
        let pos_embedding = vb.get((cfg.mm_posemb_size, 2, embed_dim), "pos_embedding")?;
        let pos_norm = layers::layer_norm(embed_dim, 1e-5, vb.pp("pos_norm"))?;

        Ok(Self {
            patch_ln1,
            patch_dense,
            patch_ln2,
            pos_embedding,
            pos_norm,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let dense_dtype = self.patch_dense.dtype_and_device().0;
        let mut hidden_states = pixel_values.clone();
        if hidden_states.dtype() != dense_dtype {
            hidden_states = hidden_states.to_dtype(dense_dtype)?;
        }
        let hidden_states = self.patch_ln1.forward(&hidden_states)?;
        let hidden_states = self.patch_dense.forward(&hidden_states)?;
        let hidden_states = self.patch_ln2.forward(&hidden_states)?;

        let valid = position_ids
            .to_dtype(DType::F32)?
            .eq(-1.0)?
            .sum_keepdim(D::Minus1)?
            .eq(0.0)?;
        let clamped = position_ids.clamp(0i64, self.pos_embedding.dim(0)? as i64 - 1)?;
        let (b, n, _) = clamped.dims3()?;
        let flat_x = clamped
            .i((.., .., 0usize))?
            .flatten_all()?
            .to_dtype(DType::U32)?;
        let flat_y = clamped
            .i((.., .., 1usize))?
            .flatten_all()?
            .to_dtype(DType::U32)?;
        let pos_x = self
            .pos_embedding
            .i((.., 0usize, ..))?
            .index_select(&flat_x, 0)?
            .reshape((b, n, ()))?;
        let pos_y = self
            .pos_embedding
            .i((.., 1usize, ..))?
            .index_select(&flat_y, 0)?
            .reshape((b, n, ()))?;
        let pos_emb = (pos_x + pos_y)?;
        let zeros = Tensor::zeros_like(&pos_emb)?;
        let valid = valid.broadcast_as(pos_emb.shape())?.to_dtype(DType::U8)?;
        let pos_emb = valid.where_cond(&pos_emb, &zeros)?;
        let hidden_states = (hidden_states + pos_emb)?;
        self.pos_norm.forward(&hidden_states)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("patch_ln1").add(&self.patch_ln1);
        uvb.pp("patch_dense").add(&self.patch_dense);
        uvb.pp("patch_ln2").add(&self.patch_ln2);
        uvb.add_tensor("pos_embedding", self.pos_embedding.clone());
        uvb.pp("pos_norm").add(&self.pos_norm);
        uvb.to_safetensors()
    }
}
