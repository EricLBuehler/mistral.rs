use std::sync::Arc;

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::{
    apply_immediate_isq, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear,
};

use crate::{
    attention::{naive_sdpa, SdpaParams},
    layers::{self, repeat_kv, DiaRotaryEmbedding, RmsNorm},
    utils::progress::{new_multi_progress, NiceProgressBar},
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
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    rope: DiaRotaryEmbedding,
    head_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    sdpa_params: SdpaParams,
}

impl<const CROSS_ATTN: bool> DiaAttention<CROSS_ATTN> {
    #[allow(clippy::too_many_arguments)]
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

        let mut q_proj: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(q_proj))?);
        q_proj = apply_immediate_isq(q_proj, vb.clone())?;
        let mut k_proj: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(k_proj))?);
        k_proj = apply_immediate_isq(k_proj, vb.clone())?;
        let mut v_proj: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(v_proj))?);
        v_proj = apply_immediate_isq(v_proj, vb.clone())?;
        let mut o_proj: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(o_proj))?);
        o_proj = apply_immediate_isq(o_proj, vb.clone())?;

        let rope = DiaRotaryEmbedding::new(
            cfg.model.rope_min_timescale,
            cfg.model.rope_max_timescale,
            head_dim,
            vb.device(),
            vb.dtype(),
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
            sdpa_params: SdpaParams {
                n_kv_groups: num_q_heads / num_kv_heads,
                sliding_window: None,
                softcap: None,
                softmax_scale: 1.,
                sinks: None,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        let (b, t, _d) = xq.dims3()?;

        let mut xq =
            self.q_proj
                .forward_autocast(xq)?
                .reshape((b, t, self.num_q_heads, self.head_dim))?;
        xq = self.rope.forward(&xq, q_positions)?;
        xq = xq.transpose(1, 2)?;

        // ---- K‒V computation & cache handling --------------------------------
        let (mut k, mut v) = if CROSS_ATTN {
            // Cross‑attention re‑uses a pre‑computed immutable cache.
            cached_kv
                .expect("cross-attention requires cached KV tensors")
                .k_v()
        } else {
            // Compute fresh K and V for self‑attention.
            let mut k = self.k_proj.forward_autocast(xkv)?.reshape((
                b,
                t,
                self.num_kv_heads,
                self.head_dim,
            ))?;
            let mut v = self.v_proj.forward_autocast(xkv)?.reshape((
                b,
                t,
                self.num_kv_heads,
                self.head_dim,
            ))?;

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

        k = repeat_kv(k.clone(), self.sdpa_params.n_kv_groups)?;
        v = repeat_kv(v.clone(), self.sdpa_params.n_kv_groups)?;

        let mut attn_output = naive_sdpa(
            &xq.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            attn_mask,
            &self.sdpa_params,
        )?;

        attn_output = attn_output.transpose(1, 2)?.reshape((b, t, ()))?;

        self.o_proj.forward_autocast(&attn_output)
    }
}

struct DiaMlp {
    wi: Arc<dyn QuantMethod>,
    wo: Arc<dyn QuantMethod>,
}

impl DiaMlp {
    fn new(vb: ShardedVarBuilder, embed_dim: usize, intermediate_dim: usize) -> Result<Self> {
        let wi = dense_general_column(embed_dim, vec![2, intermediate_dim], vb.pp("wi_fused"))?;
        let wo = dense_general_row(vec![intermediate_dim], embed_dim, vb.pp("wo"))?;

        let mut wi: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(wi))?);
        wi = apply_immediate_isq(wi, vb.clone())?;
        let mut wo: Arc<dyn QuantMethod> =
            Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(wo))?);
        wo = apply_immediate_isq(wo, vb.clone())?;

        Ok(Self { wi, wo })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seqlen, _dim) = xs.dims3()?;
        let fused_x = self.wi.forward_autocast(xs)?.reshape((bs, seqlen, 2, ()))?;
        let gate = fused_x.i((.., .., 0, ..))?;
        let up = fused_x.i((.., .., 1, ..))?;
        let hidden = (gate.silu()? * up)?;
        self.wo.forward_autocast(&hidden)
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

        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.model.encoder.n_layer,
            "Loading encoder",
            &new_multi_progress(),
        )
        .run(false, |i| DiaEncoderLayer::new(cfg, vb.pp("layers").pp(i)))?;

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

        let attn_mask = match attn_mask {
            Some(mask) => {
                let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.to_dtype(x.dtype())?;
                let dims = mask.dims();
                let mask = mask.to_dtype(DType::U8)?.where_cond(
                    &Tensor::zeros(dims, neg_inf.dtype(), neg_inf.device())?,
                    &neg_inf.to_device(mask.device())?.broadcast_as(dims)?,
                )?;
                Some(mask)
            }
            None => None,
        };

        for layer in &self.layers {
            x = layer.forward(&x, positions, attn_mask.as_ref(), 0)?;
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

    #[allow(clippy::too_many_arguments)]
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
            encoder_out,
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

        let layers = NiceProgressBar::<_, 'b'>(
            0..cfg.model.decoder.n_layer,
            "Loading decoder",
            &new_multi_progress(),
        )
        .run(false, |i| DiaDecoderLayer::new(cfg, vb.pp("layers").pp(i)))?;

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
    ) -> Result<Vec<Option<DiaKvCache>>> {
        let (b, t, _d) = encoder_out.dims3()?;

        let mut per_layer_kv_cache = Vec::new();

        for layer in &self.layers {
            let ca = &layer.cross_attn;

            let mut k_proj = ca.k_proj.forward_autocast(encoder_out)?.reshape((
                b,
                t,
                ca.num_kv_heads,
                ca.head_dim,
            ))?;
            k_proj = ca.rope.forward(&k_proj, encoder_positions)?;
            k_proj = k_proj.transpose(1, 2)?;

            let mut v_proj = ca.v_proj.forward_autocast(encoder_out)?.reshape((
                b,
                t,
                ca.num_kv_heads,
                ca.head_dim,
            ))?;
            v_proj = v_proj.transpose(1, 2)?;

            per_layer_kv_cache.push(Some(DiaKvCache::from_kv(k_proj, v_proj)));
        }

        Ok(per_layer_kv_cache)
    }

    /// Performs a single decoding step, managing KV caches layer by layer.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &self,
        tgt_ids: &Tensor,
        encoder_out: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        encoder_positions: &Tensor,
        decoder_positions: &Tensor,
        self_attn_cache: &mut [Option<DiaKvCache>],
        cross_attn_cache: &mut [Option<DiaKvCache>],
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

        let self_attn_mask = match self_attn_mask {
            Some(mask) => {
                let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.to_dtype(x.dtype())?;
                let dims = mask.dims();
                let mask = mask.to_dtype(DType::U8)?.where_cond(
                    &Tensor::zeros(dims, neg_inf.dtype(), neg_inf.device())?,
                    &neg_inf.to_device(mask.device())?.broadcast_as(dims)?,
                )?;
                Some(mask)
            }
            None => None,
        };

        let cross_attn_mask = match cross_attn_mask {
            Some(mask) => {
                let neg_inf = Tensor::new(f32::NEG_INFINITY, x.device())?.to_dtype(x.dtype())?;
                let dims = mask.dims();
                let mask = mask.to_dtype(DType::U8)?.where_cond(
                    &Tensor::zeros(dims, neg_inf.dtype(), neg_inf.device())?,
                    &neg_inf.to_device(mask.device())?.broadcast_as(dims)?,
                )?;
                Some(mask)
            }
            None => None,
        };

        for (i, layer) in self.layers.iter().enumerate() {
            let self_cache = &mut self_attn_cache[i];
            let cross_cache = &mut cross_attn_cache[i];
            x = layer.forward(
                &x,
                encoder_out,
                encoder_positions,
                decoder_positions,
                self_attn_mask.as_ref(),
                cross_attn_mask.as_ref(),
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
