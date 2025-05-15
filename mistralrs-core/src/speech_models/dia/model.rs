use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    attention::SdpaParams,
    layers::{self, DiaRotaryEmbedding, RmsNorm, Sdpa},
    layers_masker::masked_fill,
};

use super::{cache::DiaKvCache, config::DiaConfig};

pub fn dense_general_column(
    in_features: usize,
    out_features: Vec<usize>,
    vb: ShardedVarBuilder,
) -> Result<Linear> {
    let kernel_shape = [vec![in_features], out_features.clone()].concat();
    let weight = vb.get(kernel_shape, "weight")?.flatten_from(1)?.t()?;

    Ok(Linear::new(weight, None))
}

pub fn dense_general_row(
    in_features: Vec<usize>,
    out_features: usize,
    vb: ShardedVarBuilder,
) -> Result<Linear> {
    let kernel_shape = [in_features.clone(), vec![out_features]].concat();
    let weight = vb.get(kernel_shape, "weight")?.flatten_to(D::Minus2)?.t()?;

    Ok(Linear::new(weight, None))
}

struct DiaAttention<const CROSS_ATTN: bool> {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: DiaRotaryEmbedding,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    num_gqa_groups: usize,
}

impl<const CROSS_ATTN: bool> DiaAttention<CROSS_ATTN> {
    fn new(
        cfg: &DiaConfig,
        vb: ShardedVarBuilder,
        num_q_heads: usize,
        num_kv_heads: usize,
        q_embed_dim: usize,
        kv_embed_dim: usize,
        head_dim: usize,
        output_dim: usize,
    ) -> Result<Self> {
        let q_proj =
            dense_general_column(q_embed_dim, vec![num_q_heads, head_dim], vb.pp("q_proj"))?;
        let k_proj =
            dense_general_column(kv_embed_dim, vec![num_kv_heads, head_dim], vb.pp("k_proj"))?;
        let v_proj =
            dense_general_column(kv_embed_dim, vec![num_kv_heads, head_dim], vb.pp("v_proj"))?;
        let o_proj = dense_general_row(vec![num_q_heads, head_dim], output_dim, vb.pp("o_proj"))?;

        let rope = DiaRotaryEmbedding::new(
            cfg.model.rope_min_timescale,
            cfg.model.rope_max_timescale,
            head_dim,
            vb.device(),
            vb.dtype(),
            cfg.data.text_length,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            num_q_heads,
            num_kv_heads,
            head_dim,
            num_gqa_groups: num_q_heads / num_kv_heads,
        })
    }

    fn forward(
        &self,
        xq: &Tensor,
        xkv: &Tensor,
        q_positions: &Tensor,
        kv_positions: &Tensor,
        attn_mask: Option<&Tensor>,
        cached_kv: Option<&mut DiaKvCache>,
        prefill: bool,
        current_index: usize,
    ) -> Result<Tensor> {
        let (b, t, d) = xq.dims3()?;

        let mut xq = self
            .q_proj
            .forward(xq)?
            .reshape((b, t, self.num_q_heads, self.head_dim))?;
        xq = self.rope.forward(&xq, q_positions)?;
        xq = xq.transpose(1, 2)?;

        // ---- K‒V computation & cache handling --------------------------------
        let (k, v) = if CROSS_ATTN {
            // Cross‑attention re‑uses a pre‑computed immutable cache.
            cached_kv
                .expect("cross-attention requires cached KV tensors")
                .k_v()
        } else {
            // Compute fresh K and V for self‑attention.
            let mut k =
                self.k_proj
                    .forward(xkv)?
                    .reshape((b, t, self.num_kv_heads, self.head_dim))?;
            let mut v =
                self.v_proj
                    .forward(xkv)?
                    .reshape((b, t, self.num_kv_heads, self.head_dim))?;

            // Apply RoPE to K and put heads first.
            k = self.rope.forward(&k, kv_positions)?;
            k = k.transpose(1, 2)?;
            v = v.transpose(1, 2)?;

            // Update / create cache when provided.
            match cached_kv {
                Some(kv_cache) => {
                    if prefill {
                        kv_cache.prefill(&k, &v)?
                    } else {
                        kv_cache.update(&k, &v, current_index)?
                    }
                }
                // No cache supplied – just use freshly computed tensors.
                None => (k, v),
            }
        };

        fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
            if n_rep == 1 {
                Ok(x)
            } else {
                let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
                Tensor::cat(&vec![&x; n_rep], 2)?.reshape((
                    b_sz,
                    n_kv_head * n_rep,
                    seq_len,
                    head_dim,
                ))
            }
        }
        let k = repeat_kv(k, self.num_gqa_groups)?;
        let v = repeat_kv(v, self.num_gqa_groups)?;

        let mut att = xq.contiguous()?.matmul(&k.t()?.contiguous()?)?;

        if let Some(mask) = attn_mask {
            att = masked_fill(&att, &(1. - mask)?, f32::NEG_INFINITY)?;
        }
        att = candle_nn::ops::softmax_last_dim(&att)?;

        let mut attn_output = att.matmul(&v.contiguous()?)?;
        // let attn_mask = match attn_mask {
        //     Some(attn_mask) => {
        //         let should_attend = attn_mask.eq(1.)?;
        //         let should_not_attend = attn_mask.eq(0.)?;
        //         let mask = masked_fill(
        //             &attn_mask.to_dtype(DType::F32)?,
        //             &should_not_attend,
        //             f32::NEG_INFINITY,
        //         )?;
        //         dbg!(&mask.mean_all()?);
        //         Some(mask)
        //     }
        //     None => None,
        // };
        // // TODO: flash attention
        // let mut attn_output = Sdpa.run_attention(
        //     &xq,
        //     &k,
        //     &v,
        //     attn_mask.as_ref(),
        //     None,
        //     &SdpaParams {
        //         n_kv_groups: self.num_gqa_groups,
        //         sliding_window: None,
        //         softcap: None,
        //         softmax_scale: 1.,
        //     },
        // )?;

        attn_output = attn_output.transpose(1, 2)?.reshape((b, t, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct DiaMlp {
    wi: Linear,
    wo: Linear,
}

impl DiaMlp {
    fn new(vb: ShardedVarBuilder, embed_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let wi = dense_general_column(embed_dim, vec![2, intermediate_dim], vb.pp("wi_fused"))?;
        let wo = dense_general_row(vec![intermediate_dim], embed_dim, vb.pp("wo"))?;

        Ok(Self { wi, wo })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seqlen, _dim) = xs.dims3()?;
        let fused_x = self.wi.forward(xs)?.reshape((bs, seqlen, 2, ()))?;
        let gate = fused_x.i((.., .., 0, ..))?;
        let up = fused_x.i((.., .., 1, ..))?;
        let hidden = (gate.silu()? * up)?;
        self.wo.forward(&hidden)
    }
}

struct DiaEncoderLayer {
    pre_sa_norm: RmsNorm,
    post_sa_norm: RmsNorm,
    self_attn: DiaAttention<false>,
    mlp: DiaMlp,
}

impl DiaEncoderLayer {
    fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_sa_norm = RmsNorm::new(
            cfg.model.encoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("pre_sa_norm"),
        )?;
        let post_sa_norm = RmsNorm::new(
            cfg.model.encoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("post_sa_norm"),
        )?;
        let self_attn = DiaAttention::new(
            cfg,
            vb.pp("self_attention"),
            cfg.model.encoder.n_head,
            cfg.model.encoder.n_head,
            cfg.model.encoder.n_embd,
            cfg.model.encoder.n_embd,
            cfg.model.encoder.head_dim,
            cfg.model.encoder.n_embd,
        )?;
        let mlp = DiaMlp::new(
            vb.pp("mlp"),
            cfg.model.encoder.n_embd,
            cfg.model.encoder.n_hidden,
        )?;

        Ok(Self {
            pre_sa_norm,
            post_sa_norm,
            self_attn,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        attn_mask: Option<&Tensor>,
        current_index: usize,
    ) -> Result<Tensor> {
        let mut residual = x;
        let mut x_norm = self.pre_sa_norm.forward(x)?;

        let sa_out = self.self_attn.forward(
            &x_norm,
            &x_norm,
            positions,
            positions,
            attn_mask,
            None,
            false,
            current_index,
        )?;
        let x = (residual + sa_out)?;

        residual = &x;
        x_norm = self.post_sa_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;

        residual + mlp_out
    }
}

pub struct DiaEncoder {
    embedding: Embedding,
    norm: RmsNorm,
    layers: Vec<DiaEncoderLayer>,
}

impl DiaEncoder {
    fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embedding = layers::embedding(
            cfg.model.src_vocab_size,
            cfg.model.encoder.n_embd,
            vb.pp("embedding"),
            &None,
        )?;
        let norm = RmsNorm::new(
            cfg.model.encoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("norm"),
        )?;
        let mut layers = Vec::new();
        for i in 0..cfg.model.encoder.n_layer {
            layers.push(DiaEncoderLayer::new(cfg, vb.pp("layers").pp(i))?)
        }

        Ok(Self {
            embedding,
            norm,
            layers,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(x)?;

        for layer in &self.layers {
            x = layer.forward(&x, positions, attn_mask, 0)?;
        }

        self.norm.forward(&x)
    }
}

struct DiaDecoderLayer {
    pre_sa_norm: RmsNorm,
    pre_ca_norm: RmsNorm,
    pre_mlp_norm: RmsNorm,
    self_attn: DiaAttention<false>,
    cross_attn: DiaAttention<true>,
    mlp: DiaMlp,
}

impl DiaDecoderLayer {
    fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_sa_norm = RmsNorm::new(
            cfg.model.decoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("pre_sa_norm"),
        )?;
        let pre_ca_norm = RmsNorm::new(
            cfg.model.decoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("pre_ca_norm"),
        )?;
        let pre_mlp_norm = RmsNorm::new(
            cfg.model.decoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("pre_mlp_norm"),
        )?;
        // Self-Attention (GQA) with Causal Masking
        let self_attn = DiaAttention::new(
            cfg,
            vb.pp("self_attention"),
            cfg.model.decoder.gqa_query_heads,
            cfg.model.decoder.kv_heads,
            cfg.model.decoder.n_embd,
            cfg.model.decoder.n_embd,
            cfg.model.decoder.gqa_head_dim,
            cfg.model.decoder.n_embd,
        )?;
        // Cross-Attention (MHA)
        let cross_attn = DiaAttention::new(
            cfg,
            vb.pp("cross_attention"),
            cfg.model.decoder.cross_query_heads,
            cfg.model.decoder.cross_query_heads,
            cfg.model.decoder.n_embd,
            cfg.model.encoder.n_embd,
            cfg.model.decoder.cross_head_dim,
            cfg.model.decoder.n_embd,
        )?;
        let mlp = DiaMlp::new(
            vb.pp("mlp"),
            cfg.model.decoder.n_embd,
            cfg.model.decoder.n_hidden,
        )?;

        Ok(Self {
            pre_sa_norm,
            pre_ca_norm,
            pre_mlp_norm,
            self_attn,
            cross_attn,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        encoder_out: &Tensor,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        self_attn_cache: Option<&mut DiaKvCache>,
        cross_attn_cache: Option<&mut DiaKvCache>,
        prefill: bool,
        current_idx: usize,
    ) -> Result<Tensor> {
        let mut residual = x;
        let mut x_norm = self.pre_sa_norm.forward(x)?;

        let self_attn_mask = match self_attn_mask {
            Some(self_attn_mask) => Some(
                self_attn_mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .i((.., .., current_idx))?
                    .unsqueeze(2)?,
            ),
            None => None,
        };

        let sa_out = self.self_attn.forward(
            &x_norm,
            &x_norm,
            decoder_positions,
            decoder_positions,
            self_attn_mask.as_ref(),
            self_attn_cache,
            prefill,
            current_idx,
        )?;
        let x = (residual + sa_out)?;

        residual = &x;
        x_norm = self.pre_ca_norm.forward(&x)?;

        let ca_out = self.cross_attn.forward(
            &x_norm,
            &encoder_out,
            decoder_positions,
            encoder_positions,
            cross_attn_mask,
            cross_attn_cache,
            false,
            current_idx,
        )?;

        let x = (residual + ca_out)?;
        residual = &x;

        x_norm = self.pre_mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&x_norm)?;

        let res = (residual + mlp_out)?;
        Ok(res)
    }
}

pub struct DiaDecoder {
    embeddings: Vec<Embedding>,
    norm: RmsNorm,
    layers: Vec<DiaDecoderLayer>,
    logits_dense: Linear,
    channels: usize,
    vocab_size: usize,
}

impl DiaDecoder {
    fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut embeddings = Vec::new();
        for i in 0..cfg.data.channels {
            let embedding = layers::embedding(
                cfg.model.tgt_vocab_size,
                cfg.model.decoder.n_embd,
                vb.pp("embeddings").pp(i),
                &None,
            )?;
            embeddings.push(embedding);
        }

        let norm = RmsNorm::new(
            cfg.model.decoder.n_embd,
            cfg.model.normalization_layer_epsilon,
            vb.pp("norm"),
        )?;

        let mut layers = Vec::new();
        for i in 0..cfg.model.decoder.n_layer {
            layers.push(DiaDecoderLayer::new(cfg, vb.pp("layers").pp(i))?)
        }

        let logits_dense = dense_general_column(
            cfg.model.decoder.n_embd,
            vec![cfg.data.channels, cfg.model.tgt_vocab_size],
            vb.pp("logits_dense"),
        )?;

        Ok(Self {
            embeddings,
            norm,
            layers,
            logits_dense,
            channels: cfg.data.channels,
            vocab_size: cfg.model.tgt_vocab_size,
        })
    }

    /// Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
    pub fn precompute_cross_attn_cache(
        &self,
        encoder_out: &Tensor,
        encoder_positions: &Tensor,
        encoder_padding_mask: &Tensor,
    ) -> Result<Vec<Option<DiaKvCache>>> {
        let (b, t, _d) = encoder_out.dims3()?;

        let mut per_layer_kv_cache = Vec::new();

        for layer in &self.layers {
            let ca = &layer.cross_attn;

            let mut k_proj =
                ca.k_proj
                    .forward(encoder_out)?
                    .reshape((b, t, ca.num_kv_heads, ca.head_dim))?;
            k_proj = ca.rope.forward(&k_proj, encoder_positions)?;
            k_proj = k_proj.transpose(1, 2)?;
            // k_proj = masked_fill(
            //     &k_proj,
            //     &(1. - encoder_padding_mask.unsqueeze(1)?.unsqueeze(3)?)?,
            //     0f32,
            // )?;

            let mut v_proj =
                ca.v_proj
                    .forward(encoder_out)?
                    .reshape((b, t, ca.num_kv_heads, ca.head_dim))?;
            v_proj = v_proj.transpose(1, 2)?;

            per_layer_kv_cache.push(Some(DiaKvCache::from_kv(k_proj, v_proj)));
        }

        Ok(per_layer_kv_cache)
    }

    /// Performs a single decoding step, managing KV caches layer by layer.
    pub fn decode_step(
        &self,
        tgt_ids: &Tensor,
        encoder_out: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        self_attn_cache: &mut Vec<Option<DiaKvCache>>,
        cross_attn_cache: &mut Vec<Option<DiaKvCache>>,
        current_idx: usize,
    ) -> Result<Tensor> {
        let mut x: Option<Tensor> = None;
        for (i, embedding) in self.embeddings.iter().enumerate() {
            let channel_tokens = tgt_ids.narrow(D::Minus1, i, 1)?.squeeze(D::Minus1)?;
            let channel_embed = embedding.forward(&channel_tokens)?;
            x = match x {
                Some(x) => Some((x + channel_embed)?),
                None => Some(channel_embed),
            };
        }

        let mut x = x.unwrap();

        for (i, layer) in self.layers.iter().enumerate() {
            let self_cache = &mut self_attn_cache[i];
            let cross_cache = &mut cross_attn_cache[i];
            x = layer.forward(
                &x,
                encoder_out,
                encoder_positions,
                decoder_positions,
                self_attn_mask,
                cross_attn_mask,
                self_cache.as_mut(),
                cross_cache.as_mut(),
                false,
                current_idx,
            )?;
        }

        x = self.norm.forward(&x)?;

        x = self.logits_dense.forward(&x)?;

        x.reshape((x.dim(0)?, x.dim(1)?, self.channels, self.vocab_size))
    }

    // /// Forward pass for the Decoder stack, managing KV caches.
    // pub fn forward(
    //     &self,
    //     tgt_ids: &Tensor,
    //     encoder_out: &Tensor,
    //     self_attn_mask: Option<&Tensor>,
    //     cross_attn_mask: Option<&Tensor>,
    //     encoder_positions: &Tensor,
    //     decoder_positions: &Tensor,
    //     self_attn_cache: &mut Vec<Option<DiaKvCache>>,
    //     cross_attn_cache: &mut Vec<Option<DiaKvCache>>,
    // ) -> Result<Tensor> {
    //     let mut x: Option<Tensor> = None;
    //     for (i, embedding) in self.embeddings.iter().enumerate() {
    //         let channel_tokens = tgt_ids.narrow(D::Minus1, i, 1)?.squeeze(D::Minus1)?;
    //         let channel_embed = embedding.forward(&channel_tokens)?;
    //         x = match x {
    //             Some(x) => Some((x + channel_embed)?),
    //             None => Some(channel_embed),
    //         };
    //     }

    //     let mut x = x.unwrap();

    //     for (i, layer) in self.layers.iter().enumerate() {
    //         let self_cache = &mut self_attn_cache[i];
    //         let cross_cache = &mut cross_attn_cache[i];
    //         x = layer.forward(
    //             &x,
    //             encoder_out,
    //             encoder_positions,
    //             decoder_positions,
    //             None,
    //             cross_attn_mask,
    //             self_cache.as_mut(),
    //             cross_cache.as_mut(),
    //             true,
    //             0,
    //         )?;
    //     }

    //     x = self.norm.forward(&x)?;

    //     self.logits_dense.forward(&x)
    // }
}

pub struct DiaModel {
    pub encoder: DiaEncoder,
    pub decoder: DiaDecoder,
}

impl DiaModel {
    pub fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = DiaEncoder::new(cfg, vb.pp("encoder"))?;
        let decoder = DiaDecoder::new(cfg, vb.pp("decoder"))?;

        Ok(Self { encoder, decoder })
    }
}
