#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::layers::RmsNorm;

/// Gemma4 multimodal embedder that projects modality features into language model space.
///
/// Unlike Gemma3n's embedder, this does NOT have an embedding table, hard/soft norms.
/// It's simply: linear projection + RMSNorm (no learnable scale).
pub struct Gemma4MultimodalEmbedder {
    pub(crate) embedding_projection: Arc<dyn QuantMethod>,
    pub(crate) embedding_post_projection_norm: RmsNorm,
}

impl Gemma4MultimodalEmbedder {
    pub fn new(
        multimodal_hidden_size: usize,
        text_hidden_size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let embedding_projection = mistralrs_quant::linear_no_bias(
            multimodal_hidden_size,
            text_hidden_size,
            &None,
            vb.pp("embedding_projection"),
        )?;

        // Post-projection normalization without learnable scale (with_scale = false)
        let embedding_post_projection_norm = RmsNorm::new_gemma_3n(
            text_hidden_size,
            eps,
            false,
            vb.pp("embedding_post_projection_norm"),
        )?;

        Ok(Self {
            embedding_projection,
            embedding_post_projection_norm,
        })
    }

    /// Project soft features (from vision or audio encoder) into language model space.
    pub fn forward(&self, soft_features: &Tensor) -> Result<Tensor> {
        let mut projected = self.embedding_projection.forward_autocast(soft_features)?;
        let norm_dtype = self.embedding_post_projection_norm.weight().dtype();
        if projected.dtype() != norm_dtype {
            projected = projected.to_dtype(norm_dtype)?;
        }
        self.embedding_post_projection_norm.forward(&projected)
    }
}
