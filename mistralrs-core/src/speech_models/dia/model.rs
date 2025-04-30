use candle_core::Result;
use candle_nn::{Embedding, Linear};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{self, RmsNorm};

use super::config::DiaConfig;

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
    let weight = vb.get(kernel_shape, "weight")?.flatten_from(1)?.t()?;

    Ok(Linear::new(weight, None))
}

struct DiaAttention<const CROSS_ATTN: bool> {
    pre_sa_norm: RmsNorm,
    post_sa_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
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

        let q_proj =
            dense_general_column(q_embed_dim, vec![num_q_heads, head_dim], vb.pp("q_proj"))?;
        let k_proj =
            dense_general_column(kv_embed_dim, vec![num_kv_heads, head_dim], vb.pp("k_proj"))?;
        let v_proj =
            dense_general_column(kv_embed_dim, vec![num_kv_heads, head_dim], vb.pp("v_proj"))?;
        let o_proj = dense_general_row(vec![num_q_heads, head_dim], output_dim, vb.pp("v_proj"))?;

        Ok(Self {
            pre_sa_norm,
            post_sa_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }
}

struct DiaMlp {
    wi: Linear,
    wo: Linear,
}

impl DiaMlp {
    fn new(
        cfg: &DiaConfig,
        vb: ShardedVarBuilder,
        embed_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Self> {
        let wi = dense_general_column(embed_dim, vec![2, intermediate_dim], vb.pp("wi"))?;
        let wo = dense_general_row(vec![intermediate_dim], embed_dim, vb.pp("v_proj"))?;

        Ok(Self { wi, wo })
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
            cfg.model.encoder.head_dim,
        )?;
        let mlp = DiaMlp::new(
            cfg,
            vb.pp("self_attention"),
            cfg.model.encoder.head_dim,
            cfg.model.encoder.n_hidden,
        )?;

        Ok(Self {
            pre_sa_norm,
            post_sa_norm,
            self_attn,
            mlp,
        })
    }
}

struct DiaEncoder {
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
            vb.pp("self_attention"),
            cfg.model.decoder.cross_query_heads,
            cfg.model.decoder.cross_query_heads,
            cfg.model.decoder.n_embd,
            cfg.model.encoder.n_embd,
            cfg.model.decoder.cross_query_heads,
            cfg.model.decoder.n_embd,
        )?;
        let mlp = DiaMlp::new(
            cfg,
            vb.pp("self_attention"),
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
}

struct DiaDecoder {
    embeddings: Vec<Embedding>,
    norm: RmsNorm,
    layers: Vec<DiaDecoderLayer>,
    logits_dense: Linear,
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
        })
    }
}

pub struct DiaModel {
    encoder: DiaEncoder,
    decoder: DiaDecoder,
}

impl DiaModel {
    pub fn new(cfg: &DiaConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = DiaEncoder::new(cfg, vb.pp("encoder"))?;
        let decoder = DiaDecoder::new(cfg, vb.pp("decoder"))?;
        todo!()
    }
}
