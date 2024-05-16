// Based on: https://github.com/huggingface/text-embeddings-inference/blob/cc1c510e8d8af8447c01e6b14c417473cf2dfda9/backends/candle/src/models/nomic.rs#L297

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, Activation, Embedding, LayerNorm, Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct NomicConfig {
    pub prenorm: bool,
    pub rotary_emb_fraction: f32,
    pub qkv_proj_bias: bool,
    pub rotary_emb_base: f32,
    pub rotary_emb_interleaved: bool,
    pub mlp_fc1_bias: bool,
    pub mlp_fc2_bias: bool,
    pub rotary_scaling_factor: Option<f32>,
    #[serde(default = "default_max_trained_positions")]
    pub max_trained_positions: usize,

    pub n_embd: usize,
    pub n_head: usize,
    pub n_inner: usize,
    pub n_layer: usize,
    pub n_positions: usize,

    pub activation_function: Activation,

    pub vocab_size: usize,
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f32,
}

fn default_max_trained_positions() -> usize {
    2048
}

impl NomicConfig {
    // For now, we only support these parameters
    pub fn valid(&self) -> bool {
        !self.prenorm
            && self.rotary_emb_fraction == 1.0
            && !self.qkv_proj_bias
            && !self.rotary_emb_interleaved
            && !self.mlp_fc1_bias
            && !self.mlp_fc2_bias
            && self.type_vocab_size > 0
            && self.activation_function == Activation::Swiglu
    }
}

#[derive(Debug)]
pub struct NomicBertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl NomicBertEmbeddings {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(
                vb.pp("embeddings.word_embeddings")
                    .get((config.vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            token_type_embeddings: Embedding::new(
                vb.pp("embeddings.token_type_embeddings")
                    .get((config.type_vocab_size, config.n_embd), "weight")?,
                config.n_embd,
            ),
            layer_norm: layer_norm(
                config.n_embd,
                config.layer_norm_epsilon as f64,
                vb.pp("emb_ln"),
            )?,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        self.layer_norm.forward(&input_embeddings)? + &token_type_embeddings
    }
}

pub struct NomicBertGatedMLP {
    gate_up_proj: Linear,
    gate_up_proj_act: Activation,
    down_proj: Linear,
}

impl NomicBertGatedMLP {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let intermediate_size = config.n_inner;

        let gate_proj_weight = vb
            .pp("fc12")
            .get((intermediate_size, config.n_embd), "weight")?;

        let up_proj_weight = vb
            .pp("fc11")
            .get((intermediate_size, config.n_embd), "weight")?;

        let gate_up_proj_weight = Tensor::cat(&[&gate_proj_weight, &up_proj_weight], 0)?;
        let gate_up_proj = Linear::new(gate_up_proj_weight, None);

        let down_proj_weight = vb
            .pp("fc2")
            .get((config.n_embd, intermediate_size), "weight")?;
        let down_proj = Linear::new(down_proj_weight, None);

        Ok(Self {
            gate_up_proj,
            down_proj,
            gate_up_proj_act: config.activation_function.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let gate_up_states = self.gate_up_proj.forward(hidden_states)?;
        self.down_proj.forward(&gate_up_states)
    }
}

struct NomicAttention {
    qkv_linear: Linear,
    out_proj: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,

    softmax_scale: f64,
}

impl NomicAttention {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let num_attention_heads = config.n_head;
        let attention_head_size = config.n_embd / config.n_head;
        let hidden_size = config.n_embd;

        let qkv_weight = vb.pp("Wqkv").get(
            (3 * num_attention_heads * attention_head_size, hidden_size),
            "weight",
        )?;
        let qkv_linear = Linear::new(qkv_weight, None);

        let out_proj_weight = vb
            .pp("out_proj")
            .get((hidden_size, hidden_size), "weight")?;
        let out_proj = Linear::new(out_proj_weight, None);

        let softmax_scale = 1. / (attention_head_size as f64).sqrt();

        Ok(Self {
            qkv_linear,
            out_proj,
            num_attention_heads,
            attention_head_size,
            softmax_scale,
        })
    }

    fn apply_rotary(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let dim = self.attention_head_size / 2;
        let x1 = x.narrow(D::Minus1, 0, dim)?;
        let x2 = x.narrow(D::Minus1, dim, dim)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(cos)? + rotate_x.broadcast_mul(sin)?)?;
        Ok(rope)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let qkv = self.qkv_linear.forward(hidden_states)?;

        let mut new_qkv_shape = qkv.dims().to_vec();
        new_qkv_shape.pop();
        new_qkv_shape.push(self.num_attention_heads * 3);
        new_qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(new_qkv_shape.as_slice())?.transpose(1, 2)?;

        let qkv = qkv.chunk(3, 1)?;
        let query_layer = &qkv[0].contiguous()?;
        let key_layer = &qkv[1].contiguous()?;
        let value_layer = &qkv[2];

        let query_layer = self.apply_rotary(query_layer, cos, sin)?;
        let key_layer = self.apply_rotary(key_layer, cos, sin)?;

        let context_layer = {
            let attention_scores = query_layer.matmul(&key_layer.t()?)?;
            let mut attention_scores = (attention_scores * self.softmax_scale)?;

            if let Some(attention_bias) = attention_bias {
                attention_scores = attention_scores.add(attention_bias)?;
            }

            let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
            attention_probs.matmul(&value_layer.contiguous()?)
        }?;

        let context_layer = context_layer.transpose(1, 2)?.flatten_from(D::Minus2)?;

        let hidden_states = self.out_proj.forward(&context_layer)?;

        Ok(hidden_states)
    }
}

struct NomicBertBlock {
    attention: NomicAttention,
    mlp: NomicBertGatedMLP,
    post_attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
}

impl NomicBertBlock {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let attention = NomicAttention::load(vb.pp("attn"), config)?;
        let mlp = NomicBertGatedMLP::load(vb.pp("mlp"), config)?;

        let post_attention_layer_norm = layer_norm(
            config.n_embd,
            config.layer_norm_epsilon as f64,
            vb.pp("norm1"),
        )?;
        let output_layer_norm = layer_norm(
            config.n_embd,
            config.layer_norm_epsilon as f64,
            vb.pp("norm2"),
        )?;

        Ok(Self {
            attention,
            mlp,
            post_attention_layer_norm,
            output_layer_norm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let attn_output = self
            .attention
            .forward(hidden_states, attention_bias, cos, sin)?;
        let hidden_states =
            (self.post_attention_layer_norm.forward(hidden_states)? + &attn_output)?;

        let mlp_out = self.mlp.forward(&hidden_states)?;

        self.output_layer_norm.forward(&hidden_states) + &mlp_out
    }
}

struct NomicBertEncoder {
    layers: Vec<NomicBertBlock>,
}

impl NomicBertEncoder {
    pub fn load(vb: VarBuilder, config: &NomicConfig) -> Result<Self> {
        let layers = (0..config.n_layer)
            .map(|index| NomicBertBlock::load(vb.pp(format!("layers.{index}")), config))
            .collect::<Result<Vec<_>>>()?;

        Ok(NomicBertEncoder { layers })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_bias: Option<&Tensor>,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_bias, cos, sin)?
        }

        Ok(hidden_states)
    }
}

pub struct NomicBertModel {
    embeddings: NomicBertEmbeddings,
    encoder: NomicBertEncoder,
    pool: Pool,
    pub device: Device,
    dtype: DType,

    rotary_dim: usize,
    max_trained_positions: usize,
    rotary_cache: (Tensor, Tensor),
    scaled_rotary_cache: Option<(Tensor, Tensor)>,

    num_attention_heads: usize,
}

impl NomicBertModel {
    pub fn load(vb: VarBuilder, config: &NomicConfig, model_type: ModelType) -> Result<Self> {
        if !config.valid() {
            candle_core::bail!("config is not supported")
        }

        let pool = {
            if pool == Pool::Splade {
                candle_core::bail!("`splade` is not supported for Nomic")
            }
            pool
        };

        let embeddings = NomicBertEmbeddings::load(vb.clone(), config)?;
        let encoder = NomicBertEncoder::load(vb.pp("encoder"), config)?;

        let rotary_dim = encoder.layers[0].attention.attention_head_size;
        let inv_freqs_tensor = inv_freqs(rotary_dim, config.rotary_emb_base, vb.device())?;
        let rotary_cache = cos_sin(config.n_positions, &inv_freqs_tensor, vb.dtype())?;

        let scaled_rotary_cache = if let Some(scaling_factor) = config.rotary_scaling_factor {
            let new_base = (config.rotary_emb_base
                * ((scaling_factor * config.n_positions as f32
                    / config.max_trained_positions as f32)
                    - (scaling_factor - 1.0)))
                .powi((rotary_dim as f32 / (rotary_dim as f32 - 2.0)) as i32);
            let inv_freqs_tensor = inv_freqs(rotary_dim, new_base, vb.device())?;
            Some(cos_sin(config.n_positions, &inv_freqs_tensor, vb.dtype())?)
        } else {
            None
        };

        Ok(Self {
            embeddings,
            encoder,
            pool,
            rotary_dim,
            max_trained_positions: config.max_trained_positions,
            rotary_cache,
            scaled_rotary_cache,
            num_attention_heads: config.n_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        position_ids: &[usize],
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (batch_size, max_length) = input_ids.dims2()?;
        let (cos, sin) =
            if self.scaled_rotary_cache.is_some() && max_length > self.max_trained_positions {
                let cos = self
                    .scaled_rotary_cache
                    .as_ref()
                    .unwrap()
                    .0
                    .index_select(&position_ids, 0)?;
                let sin = self
                    .scaled_rotary_cache
                    .as_ref()
                    .unwrap()
                    .1
                    .index_select(&position_ids, 0)?;
                (cos, sin)
            } else {
                let cos = self.rotary_cache.0.index_select(&position_ids, 0)?;
                let sin = self.rotary_cache.1.index_select(&position_ids, 0)?;
                (cos, sin)
            };

        let cos = cos.reshape((batch_size, 1, max_length, self.rotary_dim))?;
        let sin = sin.reshape((batch_size, 1, max_length, self.rotary_dim))?;

        let embedding_output = self.embeddings.forward(&input_ids, &type_ids)?;

        let outputs =
            self.encoder
                .forward(&embedding_output, attention_bias.as_ref(), &cos, &sin)?;

        let has_pooling_requests = !batch.pooled_indices.is_empty();
        let has_raw_requests = !batch.raw_indices.is_empty();

        let pooled_embeddings = if has_pooling_requests {
            let pooled_indices_length = batch.pooled_indices.len();
            let mut outputs = outputs.clone();

            // Only use pooled_indices if at least one member of the batch ask for raw embeddings
            let pooled_indices = if has_raw_requests {
                let pooled_indices =
                    Tensor::from_vec(batch.pooled_indices, pooled_indices_length, &self.device)?;

                // Select values in the batch
                outputs = outputs.index_select(&pooled_indices, 0)?;
                Some(pooled_indices)
            } else {
                None
            };

            let pooled_embeddings = match self.pool {
                // CLS pooling
                Pool::Cls => outputs.i((.., 0))?,
                // Mean pooling
                Pool::Mean => {
                    if let Some(ref attention_mask) = attention_mask {
                        let mut attention_mask = attention_mask.clone();

                        if let Some(pooled_indices) = pooled_indices {
                            // Select values in the batch
                            attention_mask = attention_mask.index_select(&pooled_indices, 0)?;
                        };

                        // Mask padded values
                        outputs = outputs.broadcast_mul(&attention_mask)?;
                    }

                    (outputs.sum(1)?.broadcast_div(&input_lengths))?
                }
                Pool::Splade => unreachable!(),
            };
            Some(pooled_embeddings)
        } else {
            None
        };

        let raw_embeddings = if has_raw_requests {
            // Reshape outputs
            let (b, l, h) = outputs.shape().dims3()?;
            let outputs = outputs.reshape((b * l, h))?;

            // We need to remove the padding tokens only if batch_size > 1 and there are some
            // member of the batch that require pooling
            // or if batch_size > 1 and the members of the batch have different lengths
            if (attention_mask.is_some() || has_pooling_requests) && batch_size > 1 {
                let mut final_indices: Vec<u32> = Vec::with_capacity(batch_size * max_length);

                for i in batch.raw_indices.into_iter() {
                    let start = i * batch.max_length;
                    let i = i as usize;
                    let length =
                        batch.cumulative_seq_lengths[i + 1] - batch.cumulative_seq_lengths[i];

                    for j in start..start + length {
                        // Add indices for the tokens of this specific member of the batch
                        final_indices.push(j);
                    }
                }

                let final_indices_length = final_indices.len();
                let final_indices =
                    Tensor::from_vec(final_indices, final_indices_length, &self.device)?;

                // Select the tokens with final indices
                Some(outputs.index_select(&final_indices, 0)?)
            } else {
                Some(outputs)
            }
        } else {
            None
        };

        Ok((pooled_embeddings, raw_embeddings))
    }
}

pub fn inv_freqs(dim: usize, base: f32, device: &Device) -> Result<Tensor> {
    let inv_freq: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    Tensor::from_vec(inv_freq, (1, inv_freq_len), device)
}

pub fn cos_sin(length: usize, inv_freqs: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
    let t = Tensor::arange(0u32, length as u32, inv_freqs.device())?
        .to_dtype(DType::F32)?
        .reshape((length, 1))?;
    let freqs = t.matmul(inv_freqs)?;
    let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

    let cos = freqs.cos()?.to_dtype(dtype)?;
    let sin = freqs.sin()?.to_dtype(dtype)?;
    Ok((cos, sin))
}
