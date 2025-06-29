#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{linear_no_bias, RmsNorm};

use super::config::Gemma3nConfig;

/// Gemma3n Multimodal Embedder for vision tokens
/// Embeds token ids or soft tokens into language model space
pub struct Gemma3nMultimodalEmbedder {
    embedding: Embedding,
    hard_embedding_norm: RmsNorm,
    soft_embedding_norm: RmsNorm,
    embedding_projection: candle_nn::Linear,
    embedding_post_projection_norm: RmsNorm,
    vocab_offset: usize,
    vocab_size: usize,
}

impl Gemma3nMultimodalEmbedder {
    pub fn new(cfg: &Gemma3nConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let vision_hidden_size = cfg.vision_config.hidden_size;
        let text_hidden_size = cfg.text_config.hidden_size;
        let eps = cfg.vision_config.layer_norm_eps;
        
        // Vision vocab offset and size from config
        let vocab_offset = cfg.vision_vocab_offset;
        let vocab_size = 128; // Vision vocab size (from weights shape)
        
        // Embedding for vision token IDs
        let embedding = Embedding::new(
            vb.get((vocab_size, vision_hidden_size), "embedding.weight")?,
            vision_hidden_size,
        );
        
        // Normalization layers
        let hard_embedding_norm = RmsNorm::new(vision_hidden_size, eps, vb.pp("hard_embedding_norm"))?;
        let soft_embedding_norm = RmsNorm::new(vision_hidden_size, eps, vb.pp("soft_embedding_norm"))?;
        
        // Projection to text space
        let embedding_projection = linear_no_bias(
            vision_hidden_size,
            text_hidden_size,
            vb.pp("embedding_projection"),
        )?;
        
        // Post-projection normalization (without scale)
        let embedding_post_projection_norm = RmsNorm::new_gemma_3n(
            text_hidden_size,
            eps,
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
            vocab_size,
        })
    }
    
    /// Forward pass for the embedder
    /// Can take either input_ids (hard embeddings) or inputs_embeds (soft embeddings from vision tower)
    pub fn forward(
        &self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let emb_norm = match (input_ids, inputs_embeds) {
            (None, Some(embeds)) => {
                // Soft embeddings from vision tower
                self.soft_embedding_norm.forward(embeds)?
            }
            (Some(ids), None) => {
                // Hard embeddings from token IDs
                // Subtract vocab_offset to get indices into embedding table
                let adjusted_ids = (ids - self.vocab_offset as f64)?;
                let hard_emb = self.embedding.forward(&adjusted_ids)?;
                self.hard_embedding_norm.forward(&hard_emb)?
            }
            _ => {
                return Err(candle_core::Error::Msg(
                    "Must specify exactly one of input_ids or inputs_embeds".to_string(),
                ));
            }
        };
        
        // Project to text space
        let emb_norm_proj = self.embedding_projection.forward(&emb_norm)?;
        
        // Post-projection normalization
        self.embedding_post_projection_norm.forward(&emb_norm_proj)
    }
    
    pub fn vocab_offset(&self) -> usize {
        self.vocab_offset
    }
    
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}