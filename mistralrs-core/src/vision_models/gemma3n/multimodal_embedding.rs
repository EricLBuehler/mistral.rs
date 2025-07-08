use candle_core::{DType, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::layers::{embedding, RmsNorm, ScaledEmbedding};

use super::config::Gemma3nTextConfig;

/// Multimodal embedder for Gemma3n that handles both text tokens and vision embeddings
pub struct Gemma3nMultimodalEmbedder {
    /// Embedding layer for vocabulary tokens
    pub(crate) embedding: ScaledEmbedding,
    /// RMS normalization for hard embeddings (text tokens)
    pub(crate) hard_embedding_norm: RmsNorm,
    /// RMS normalization for soft embeddings (vision features)
    pub(crate) soft_embedding_norm: RmsNorm,
    /// Linear projection from multimodal hidden size to text hidden size
    pub(crate) embedding_projection: Arc<dyn QuantMethod>,
    /// Post-projection normalization (without scale)
    pub(crate) embedding_post_projection_norm: RmsNorm,
    /// The vocabulary offset to subtract from input IDs
    vocab_offset: i64,
}

impl Gemma3nMultimodalEmbedder {
    pub fn new(
        cfg: &Gemma3nTextConfig,
        multimodal_vocab_size: usize,
        multimodal_hidden_size: usize,
        vocab_offset: i64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Create the embedding layer with proper VarBuilder prefix
        // The embedding layer uses the multimodal vocab size (e.g., 128 for vision)
        let embed_tokens = embedding(
            multimodal_vocab_size,
            multimodal_hidden_size,
            vb.pp("embedding"),
            &cfg.quantization_config,
        )?;

        // Scale the embeddings by sqrt(multimodal_hidden_size)
        let embedding = ScaledEmbedding::new((multimodal_hidden_size as f64).sqrt(), embed_tokens);

        // Create normalization layers with proper prefixes
        let hard_embedding_norm = RmsNorm::new_gemma_3n(
            multimodal_hidden_size,
            cfg.rms_norm_eps,
            true, // with_scale = true
            vb.pp("hard_embedding_norm"),
        )?;

        let soft_embedding_norm = RmsNorm::new_gemma_3n(
            multimodal_hidden_size,
            cfg.rms_norm_eps,
            true, // with_scale = true
            vb.pp("soft_embedding_norm"),
        )?;

        // Linear projection from multimodal to text hidden size
        let embedding_projection = mistralrs_quant::linear_no_bias(
            multimodal_hidden_size,
            cfg.hidden_size,
            &None,
            vb.pp("embedding_projection"),
        )?;

        // Post-projection normalization without scale
        let embedding_post_projection_norm = RmsNorm::new_gemma_3n(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            false, // with_scale = false
            vb.pp("embedding_post_projection_norm"),
        )?;

        Ok(Self {
            embedding,
            hard_embedding_norm,
            soft_embedding_norm,
            embedding_projection,
            embedding_post_projection_norm,
            vocab_offset,
        })
    }

    /// Forward pass for text input IDs
    pub fn forward_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Subtract vocab_offset from input_ids
        let adjusted_ids = if self.vocab_offset != 0 {
            let adjusted = (input_ids.to_dtype(DType::F32)? - self.vocab_offset as f64)?;
            adjusted.to_dtype(input_ids.dtype())?
        } else {
            input_ids.clone()
        };

        // Embed the tokens
        let embeddings = self.embedding.forward(&adjusted_ids)?;

        // Apply hard embedding normalization
        let normalized = self.hard_embedding_norm.forward(&embeddings)?;

        // Project to text hidden size
        let projected = self
            .embedding_projection
            .forward_autocast(&normalized.unsqueeze(0)?)?
            .squeeze(0)?;

        // Apply post-projection normalization
        self.embedding_post_projection_norm.forward(&projected)
    }

    /// Forward pass for vision embeddings (soft features)
    pub fn forward_vision(&self, soft_features: &Tensor) -> Result<Tensor> {
        // Apply soft embedding normalization
        let normalized = self.soft_embedding_norm.forward(soft_features)?;

        // Project to text hidden size
        let projected = self.embedding_projection.forward_autocast(&normalized)?;

        // Apply post-projection normalization
        self.embedding_post_projection_norm.forward(&projected)
    }
}
