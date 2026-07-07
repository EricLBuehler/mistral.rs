//! Merge step: text token embeddings + masked-scatter of the connector output into the
//! image-placeholder positions. This is `merged_input_embeds`, the tensor fed to the LM as
//! layer-0 input (HF reference `hs[0]`).
//!
//! Reference (native transformers 5.13): `inputs_embeds = embed_tokens(input_ids)` then
//! `inputs_embeds.masked_scatter(input_ids == image_token_id, image_embeds)`. torch
//! `masked_scatter` fills the True positions in row-major order with successive rows of
//! `image_embeds`, so with the 161 contiguous image placeholders the connector rows drop in
//! in their natural (t, h/2, w/2) order.
//!
//! We implement it without a scatter primitive: concat `[text ; image_embeds]` -> `[S+K, D]`,
//! then one `index_select` with a per-position index that points text rows at themselves and
//! image rows at the running image counter. General over any mask layout (multi-image too).

use crate::layers::embedding;
use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Module};
use mistralrs_quant::ShardedVarBuilder;

pub struct Merger {
    embed_tokens: Embedding,
    image_token_id: i64,
}

impl Merger {
    /// `vb` is the `model.*` root; the embedding weight is `model.embed_tokens.weight`.
    pub fn load(
        vb: ShardedVarBuilder,
        vocab: usize,
        hidden: usize,
        image_token_id: i64,
    ) -> Result<Self> {
        Ok(Self {
            embed_tokens: embedding(vocab, hidden, vb.pp("embed_tokens"), &None)?,
            image_token_id,
        })
    }

    /// Embed a slice of token ids `[n]` -> `[n, D]`. Used by the autoregressive greedy loop to
    /// embed each newly generated token (pure text tokens, no scatter). Device is taken from the
    /// embedding weight so callers don't have to thread one through.
    pub fn embed(&self, ids: &[i64]) -> Result<Tensor> {
        let idx: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        let n = idx.len();
        let idx = Tensor::from_vec(idx, n, self.embed_tokens.embeddings().device())?;
        self.embed_tokens.forward(&idx)
    }

    /// `input_ids` `[S]` (i64), `image_embeds` `[K, D]` (connector output). Returns `[S, D]`.
    pub fn forward(&self, input_ids: &Tensor, image_embeds: &Tensor) -> Result<Tensor> {
        let ids = input_ids.to_dtype(DType::I64)?.to_vec1::<i64>()?;
        let s = ids.len();
        // candle Embedding gathers with u32 indices.
        let ids_u32: Vec<u32> = ids.iter().map(|&v| v as u32).collect();
        let idx_emb = Tensor::from_vec(ids_u32, s, input_ids.device())?;
        let text = self.embed_tokens.forward(&idx_emb)?; // [S, D]

        // Gather index into cat([text ; image_embeds]): text rows -> self, image rows -> S+counter.
        let mut gather = Vec::with_capacity(s);
        let mut img = 0u32;
        for (j, &id) in ids.iter().enumerate() {
            if id == self.image_token_id {
                gather.push(s as u32 + img);
                img += 1;
            } else {
                gather.push(j as u32);
            }
        }
        let combined = Tensor::cat(&[&text, image_embeds], 0)?; // [S+K, D]
        let gather = Tensor::from_vec(gather, s, input_ids.device())?;
        combined.index_select(&gather, 0)
    }
}
