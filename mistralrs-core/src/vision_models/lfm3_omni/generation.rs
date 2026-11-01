use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Linear, Module};
use mistralrs_quant::{lfm3_causal_conv1d, ShardedVarBuilder};
use rand::{seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{linear, linear_b, linear_no_bias, Sdpa},
    models::lfm3::{LayerType, RopeParameters},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

const PROJECTION_PARTS: usize = 3;
const DEFAULT_AUDIO_TEMPERATURE: f64 = 0.4;
const DEFAULT_AUDIO_REPETITION_PENALTY: f32 = 1.2;
const DEFAULT_AUDIO_REPETITION_WINDOW: usize = 64;
const STANDALONE_AUDIO_TEMPERATURE: f64 = 1.0;
const STANDALONE_AUDIO_TOP_K: usize = 128;
const STANDALONE_AUDIO_REPETITION_PENALTY: f32 = 1.1;
const STANDALONE_AUDIO_REPETITION_WINDOW: usize = 4096;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct AuxiliaryConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub use_sandwich_norm: bool,
    pub layer_types: Vec<LayerType>,
    pub sliding_window: usize,
    pub rope_parameters: HashMap<String, RopeParameters>,
}

pub(crate) type DepthConfig = AuxiliaryConfig;

impl AuxiliaryConfig {
    pub(crate) fn parameter_count(&self) -> usize {
        let hidden = self.hidden_size;
        let ffn = PROJECTION_PARTS * hidden * self.intermediate_size;
        let norms = (2 + 2 * usize::from(self.use_sandwich_norm)) * hidden;
        hidden
            + self
                .layer_types
                .iter()
                .map(|kind| {
                    let operator = match kind {
                        LayerType::Conv => {
                            4 * hidden * hidden
                                + hidden * self.conv_l_cache
                                + usize::from(self.conv_bias) * 5 * hidden
                        }
                        _ => {
                            2 * hidden
                                * self.head_dim
                                * (self.num_attention_heads + self.num_key_value_heads)
                                + hidden * self.num_attention_heads
                                + 2 * self.head_dim
                        }
                    };
                    operator + norms + ffn
                })
                .sum::<usize>()
    }

    pub(crate) fn depth_parameter_count(
        &self,
        text_hidden: usize,
        codebooks: usize,
        codebook_size: usize,
    ) -> usize {
        self.parameter_count()
            + self.hidden_size
            + codebooks * self.hidden_size * (text_hidden + codebook_size + 2)
    }
}

struct Norm {
    weight: Tensor,
    eps: f64,
    scale_in_float32: bool,
}

impl Norm {
    fn new(size: usize, eps: f64, vb: ShardedVarBuilder, scale_in_float32: bool) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
            scale_in_float32,
        })
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.add_tensor("weight", self.weight.clone());
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let normalized =
            xs.broadcast_div(&(xs.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?)?;
        if self.scale_in_float32 {
            normalized
                .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
                .to_dtype(dtype)
        } else {
            normalized.to_dtype(dtype)?.broadcast_mul(&self.weight)
        }
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    gate_proj: Linear,
    q_norm: Norm,
    k_norm: Norm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope: RopeParameters,
    window: Option<usize>,
}

impl Attention {
    fn new(config: &AuxiliaryConfig, kind: LayerType, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim;
        let sliding = kind == LayerType::SlidingAttention;
        let rope_key = if sliding {
            "sliding_attention"
        } else {
            "full_attention"
        };
        let rope = config
            .rope_parameters
            .get(rope_key)
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "Missing LFM3 auxiliary {rope_key} RoPE configuration"
                ))
            })?
            .clone();
        Ok(Self {
            q_proj: linear_no_bias(
                hidden,
                config.num_attention_heads * head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                hidden,
                config.num_key_value_heads * head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                hidden,
                config.num_key_value_heads * head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                config.num_attention_heads * head_dim,
                hidden,
                vb.pp("o_proj"),
            )?,
            gate_proj: linear_no_bias(hidden, config.num_attention_heads, vb.pp("gate_proj"))?,
            q_norm: Norm::new(head_dim, config.norm_eps, vb.pp("q_norm"), false)?,
            k_norm: Norm::new(head_dim, config.norm_eps, vb.pp("k_norm"), false)?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim,
            rope,
            window: sliding.then_some(config.sliding_window),
        })
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("o_proj").add(&self.o_proj);
        uvb.pp("gate_proj").add(&self.gate_proj);
        self.q_norm.residual_tensors(uvb.pp("q_norm"));
        self.k_norm.residual_tensors(uvb.pp("k_norm"));
    }

    fn rotate(&self, xs: &Tensor, float32_rope: bool) -> Result<Tensor> {
        let seq = xs.dim(2)?;
        let half = self.head_dim / 2;
        // RoPE follows the reference's integer rotary dimension and float32 frequency calculation.
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let frequencies = {
            let rotated = (self.rope.partial_rotary_factor * self.head_dim as f64) as usize / 2;
            (0..half)
                .map(|index| {
                    if index < rotated {
                        (self.rope.rope_theta as f32)
                            .powf(-((2 * index) as f32) / self.head_dim as f32)
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>()
        };
        let frequencies = Tensor::from_vec(frequencies, (1, half), xs.device())?;
        let positions = Tensor::arange(
            0u32,
            u32::try_from(seq).map_err(candle_core::Error::wrap)?,
            xs.device(),
        )?
        .to_dtype(DType::F32)?
        .reshape((seq, 1))?;
        let angles = positions.matmul(&frequencies)?;
        let angles = Tensor::cat(&[&angles, &angles], 1)?.reshape((1, 1, seq, self.head_dim))?;
        let cos = angles.cos()?;
        let sin = angles.sin()?;
        let (cos, sin) = if float32_rope {
            (cos, sin)
        } else {
            (
                cos.to_dtype(xs.dtype())?.to_dtype(DType::F32)?,
                sin.to_dtype(xs.dtype())?.to_dtype(DType::F32)?,
            )
        };
        let float = xs.to_dtype(DType::F32)?;
        let rotated = Tensor::cat(
            &[
                float.narrow(3, half, half)?.neg()?,
                float.narrow(3, 0, half)?,
            ],
            3,
        )?;
        (float.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?.to_dtype(xs.dtype())
    }

    fn forward(&self, xs: &Tensor, causal: bool, float32_rope: bool) -> Result<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let query = self
            .q_norm
            .forward(
                &self
                    .q_proj
                    .forward(xs)?
                    .reshape((batch, seq, self.heads, self.head_dim))?,
            )?
            .transpose(1, 2)?;
        let key = self
            .k_norm
            .forward(&self.k_proj.forward(xs)?.reshape((
                batch,
                seq,
                self.kv_heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?;
        let value = self
            .v_proj
            .forward(xs)?
            .reshape((batch, seq, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let query = self.rotate(&query, float32_rope)?.contiguous()?;
        let key = self.rotate(&key, float32_rope)?.contiguous()?;
        let mask = if causal {
            AttentionMask::CausalFlash
        } else {
            AttentionMask::Bidirectional
        };
        #[allow(clippy::cast_precision_loss)]
        let softmax_scale = 1.0 / (self.head_dim as f32).sqrt();
        let result = Sdpa
            .run_attention(
                &query,
                &key,
                &value,
                &mask,
                Some(&FlashParams::empty(causal)),
                &SdpaParams {
                    n_kv_groups: self.heads / self.kv_heads,
                    sliding_window: self.window,
                    softcap: None,
                    softmax_scale,
                    sinks: None,
                },
            )?
            .transpose(1, 2)?;
        let gate = candle_nn::ops::sigmoid(&self.gate_proj.forward(xs)?)?.unsqueeze(D::Minus1)?;
        self.o_proj.forward(&result.broadcast_mul(&gate)?.reshape((
            batch,
            seq,
            self.heads * self.head_dim,
        ))?)
    }
}

struct ShortConv {
    input: Linear,
    output: Linear,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl ShortConv {
    fn new(config: &AuxiliaryConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            input: linear_b(
                config.hidden_size,
                PROJECTION_PARTS * config.hidden_size,
                config.conv_bias,
                vb.pp("in_proj"),
            )?,
            output: linear_b(
                config.hidden_size,
                config.hidden_size,
                config.conv_bias,
                vb.pp("out_proj"),
            )?,
            weight: vb
                .pp("conv")
                .get((config.hidden_size, 1, config.conv_l_cache), "weight")?,
            bias: config
                .conv_bias
                .then(|| vb.pp("conv").get(config.hidden_size, "bias"))
                .transpose()?,
        })
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("in_proj").add(&self.input);
        uvb.pp("out_proj").add(&self.output);
        uvb.pp("conv").add_tensor("weight", self.weight.clone());
        if let Some(bias) = &self.bias {
            uvb.pp("conv").add_tensor("bias", bias.clone());
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, _, hidden) = xs.dims3()?;
        let state = Tensor::zeros(
            (batch, hidden, self.weight.dim(2)?),
            xs.dtype(),
            xs.device(),
        )?;
        let (output, _) = lfm3_causal_conv1d(
            &self.input.forward(xs)?,
            &self.weight,
            self.bias.as_ref(),
            &state,
            false,
        )?;
        self.output.forward(&output)
    }
}

enum Operator {
    Attention(Attention),
    Conv(ShortConv),
}

struct Layer {
    operator: Operator,
    operator_norm: Norm,
    ffn_norm: Norm,
    post_operator_norm: Option<Norm>,
    post_ffn_norm: Option<Norm>,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Layer {
    fn new(config: &AuxiliaryConfig, kind: LayerType, vb: ShardedVarBuilder) -> Result<Self> {
        let norm = |name| Norm::new(config.hidden_size, config.norm_eps, vb.pp(name), false);
        let ff = vb.pp("feed_forward");
        Ok(Self {
            operator: match kind {
                LayerType::Conv => Operator::Conv(ShortConv::new(config, vb.pp("conv"))?),
                kind => Operator::Attention(Attention::new(config, kind, vb.pp("self_attn"))?),
            },
            operator_norm: norm("operator_norm")?,
            ffn_norm: norm("ffn_norm")?,
            post_operator_norm: config
                .use_sandwich_norm
                .then(|| norm("post_operator_norm"))
                .transpose()?,
            post_ffn_norm: config
                .use_sandwich_norm
                .then(|| norm("post_ffn_norm"))
                .transpose()?,
            w1: linear_no_bias(config.hidden_size, config.intermediate_size, ff.pp("w1"))?,
            w2: linear_no_bias(config.intermediate_size, config.hidden_size, ff.pp("w2"))?,
            w3: linear_no_bias(config.hidden_size, config.intermediate_size, ff.pp("w3"))?,
        })
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        match &self.operator {
            Operator::Attention(attention) => attention.residual_tensors(uvb.pp("self_attn")),
            Operator::Conv(conv) => conv.residual_tensors(uvb.pp("conv")),
        }
        self.operator_norm.residual_tensors(uvb.pp("operator_norm"));
        self.ffn_norm.residual_tensors(uvb.pp("ffn_norm"));
        if let Some(norm) = &self.post_operator_norm {
            norm.residual_tensors(uvb.pp("post_operator_norm"));
        }
        if let Some(norm) = &self.post_ffn_norm {
            norm.residual_tensors(uvb.pp("post_ffn_norm"));
        }
        uvb.pp("feed_forward.w1").add(&self.w1);
        uvb.pp("feed_forward.w2").add(&self.w2);
        uvb.pp("feed_forward.w3").add(&self.w3);
    }

    fn forward(&self, xs: &Tensor, causal: bool, float32_rope: bool) -> Result<Tensor> {
        let input = self.operator_norm.forward(xs)?;
        let mut output = match &self.operator {
            Operator::Attention(attention) => attention.forward(&input, causal, float32_rope)?,
            Operator::Conv(conv) => conv.forward(&input)?,
        };
        if let Some(norm) = &self.post_operator_norm {
            output = norm.forward(&output)?;
        }
        let xs = (xs + output)?;
        let input = self.ffn_norm.forward(&xs)?;
        let mut output = self.w2.forward(
            &(candle_nn::ops::silu(&self.w1.forward(&input)?)? * self.w3.forward(&input)?)?,
        )?;
        if let Some(norm) = &self.post_ffn_norm {
            output = norm.forward(&output)?;
        }
        xs + output
    }
}

pub(crate) struct AuxiliaryLfm3 {
    layers: Vec<Layer>,
    norm: Norm,
    causal: bool,
    float32_rope: bool,
}

impl AuxiliaryLfm3 {
    pub(crate) fn new(
        config: &AuxiliaryConfig,
        vb: ShardedVarBuilder,
        causal: bool,
        float32_rope: bool,
    ) -> Result<Self> {
        if config.layer_types.len() != config.num_hidden_layers {
            candle_core::bail!("LFM3 auxiliary layer count does not match its configuration");
        }
        Ok(Self {
            layers: config
                .layer_types
                .iter()
                .enumerate()
                .map(|(index, &kind)| Layer::new(config, kind, vb.pp("layers").pp(index)))
                .collect::<Result<Vec<_>>>()?,
            norm: Norm::new(
                config.hidden_size,
                config.norm_eps,
                vb.pp("embedding_norm"),
                false,
            )?,
            causal,
            float32_rope,
        })
    }

    pub(crate) fn residual_tensors(&self, uvb: UnVarBuilder) {
        for (index, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(uvb.pp("layers").pp(index));
        }
        self.norm.residual_tensors(uvb.pp("embedding_norm"));
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, self.causal, self.float32_rope)?;
        }
        self.norm.forward(&xs)
    }
}

pub(crate) struct DepthDecoder {
    linear: Linear,
    mask: Tensor,
    model: AuxiliaryLfm3,
    embeddings: Vec<Embedding>,
    norms: Vec<Norm>,
    hidden_size: usize,
}

impl DepthDecoder {
    pub(crate) fn new(
        config: &AuxiliaryConfig,
        text_hidden: usize,
        codebooks: usize,
        codebook_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let embeddings = vb.pp("depth_embeddings");
        Ok(Self {
            linear: linear(
                text_hidden,
                codebooks * config.hidden_size,
                vb.pp("depth_linear"),
            )?,
            mask: vb.get(config.hidden_size, "mask_embedding")?,
            model: AuxiliaryLfm3::new(config, vb.pp("model"), false, false)?,
            embeddings: (0..codebooks)
                .map(|i| {
                    Ok(Embedding::new(
                        embeddings
                            .pp(i)
                            .pp("embedding")
                            .get((codebook_size, config.hidden_size), "weight")?,
                        config.hidden_size,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            norms: (0..codebooks)
                .map(|i| {
                    Norm::new(
                        config.hidden_size,
                        config.norm_eps,
                        embeddings.pp(i).pp("embedding_norm"),
                        true,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            hidden_size: config.hidden_size,
        })
    }

    pub(crate) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("depth_linear").add(&self.linear);
        uvb.add_tensor("mask_embedding", self.mask.clone());
        self.model.residual_tensors(uvb.pp("model"));
        for (index, (embedding, norm)) in self.embeddings.iter().zip(&self.norms).enumerate() {
            let base = uvb.pp("depth_embeddings").pp(index);
            base.pp("embedding")
                .add_tensor("weight", embedding.embeddings().clone());
            norm.residual_tensors(base.pp("embedding_norm"));
        }
        uvb.to_safetensors()
    }

    pub(crate) fn forward(
        &self,
        hidden: &Tensor,
        codes: &Tensor,
        masked: &Tensor,
    ) -> Result<Tensor> {
        let batch = hidden.dim(0)?;
        let books = self.embeddings.len();
        let embedded = self
            .embeddings
            .iter()
            .enumerate()
            .map(|(i, embedding)| embedding.forward(&codes.narrow(1, i, 1)?.squeeze(1)?))
            .collect::<Result<Vec<_>>>()?;
        let embedded = Tensor::stack(&embedded, 1)?;
        let mask = masked
            .unsqueeze(2)?
            .broadcast_as((batch, books, self.hidden_size))?;
        let embedded = mask.where_cond(
            &self.mask.broadcast_as((batch, books, self.hidden_size))?,
            &embedded,
        )?;
        let projected = self
            .linear
            .forward(hidden)?
            .reshape((batch, books, self.hidden_size))?;
        let states = self.model.forward(&(projected + embedded)?)?;
        let logits = self
            .embeddings
            .iter()
            .zip(&self.norms)
            .enumerate()
            .map(|(index, (embedding, norm))| {
                norm.forward(&states.narrow(1, index, 1)?.squeeze(1)?)?
                    .matmul(&embedding.embeddings().t()?)
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::stack(&logits, 1)
    }

    pub(crate) fn sample_frame<R: Rng + ?Sized>(
        &self,
        hidden: &Tensor,
        schedule: &[usize],
        sampling: &AudioSampling,
        history: &[Vec<u32>],
        rng: &mut R,
    ) -> Result<Vec<u32>> {
        let books = self.embeddings.len();
        if schedule.iter().sum::<usize>() != books {
            candle_core::bail!("MaskGIT schedule must reveal exactly {books} codebooks");
        }
        let mut order = (0..books).collect::<Vec<_>>();
        order.shuffle(rng);
        let mut codes = vec![0u32; books];
        let mut masked = vec![1u8; books];
        let hidden = hidden.reshape((1, ()))?;
        let mut offset = 0;
        for &count in schedule {
            let logits = self
                .forward(
                    &hidden,
                    &Tensor::from_slice(&codes, (1, books), hidden.device())?,
                    &Tensor::from_slice(&masked, (1, books), hidden.device())?,
                )?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .squeeze(0)?
                .to_vec2::<f32>()?;
            for &index in &order[offset..offset + count] {
                let previous = history
                    .iter()
                    .rev()
                    .take(sampling.repetition_window)
                    .map(|frame| frame[index])
                    .collect::<Vec<_>>();
                codes[index] = sample_logits(&logits[index], sampling, &previous, rng)?;
                masked[index] = 0;
            }
            offset += count;
        }
        Ok(codes)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AudioSampling {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

impl Default for AudioSampling {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_AUDIO_TEMPERATURE,
            top_k: None,
            repetition_penalty: DEFAULT_AUDIO_REPETITION_PENALTY,
            repetition_window: DEFAULT_AUDIO_REPETITION_WINDOW,
        }
    }
}

impl AudioSampling {
    pub(crate) fn standalone() -> Self {
        Self {
            temperature: STANDALONE_AUDIO_TEMPERATURE,
            top_k: Some(STANDALONE_AUDIO_TOP_K),
            repetition_penalty: STANDALONE_AUDIO_REPETITION_PENALTY,
            repetition_window: STANDALONE_AUDIO_REPETITION_WINDOW,
        }
    }
}

fn sample_logits<R: Rng + ?Sized>(
    logits: &[f32],
    config: &AudioSampling,
    history: &[u32],
    rng: &mut R,
) -> Result<u32> {
    if !config.temperature.is_finite()
        || config.temperature < 0.0
        || !config.repetition_penalty.is_finite()
        || config.repetition_penalty <= 0.0
    {
        candle_core::bail!(
            "Audio sampling requires a nonnegative temperature and positive repetition penalty"
        );
    }
    let mut scores = logits.to_vec();
    let mut seen = vec![false; scores.len()];
    for &token in history {
        let index = token as usize;
        if !seen[index] {
            scores[index] = if scores[index] < 0.0 {
                scores[index] * config.repetition_penalty
            } else {
                scores[index] / config.repetition_penalty
            };
            seen[index] = true;
        }
    }
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if config.temperature == 0.0 {
        return u32::try_from(ranked[0].0).map_err(candle_core::Error::wrap);
    }
    if let Some(k) = config.top_k.filter(|&k| k > 0) {
        let threshold = ranked[k.min(ranked.len()) - 1].1;
        ranked.retain(|(_, score)| *score >= threshold);
    }
    let maximum = ranked[0].1 as f64;
    let probabilities = ranked
        .iter()
        .map(|(_, score)| ((*score as f64 - maximum) / config.temperature).exp())
        .collect::<Vec<_>>();
    let mut draw = rng.random::<f64>() * probabilities.iter().sum::<f64>();
    for ((token, _), probability) in ranked.iter().zip(probabilities) {
        draw -= probability;
        if draw <= 0.0 {
            return u32::try_from(*token).map_err(candle_core::Error::wrap);
        }
    }
    u32::try_from(ranked.last().unwrap().0).map_err(candle_core::Error::wrap)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationMode {
    Text,
    Audio,
    Interleaved,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GenerationMarkers {
    pub audio_start: u32,
    pub mixed_start: u32,
    pub text_end: u32,
    pub audio_eos: u32,
    pub text_slots: usize,
    pub audio_slots: usize,
}

pub(crate) struct GenerationState {
    mode: GenerationMode,
    markers: GenerationMarkers,
    speaking: bool,
    text_done: bool,
    slots_left: usize,
}

impl GenerationState {
    pub(crate) fn new(mode: GenerationMode, markers: GenerationMarkers) -> Self {
        Self {
            mode,
            markers,
            speaking: false,
            text_done: false,
            slots_left: markers.text_slots,
        }
    }

    pub(crate) fn is_audio(&self) -> bool {
        self.mode == GenerationMode::Audio
            || (self.mode == GenerationMode::Interleaved && self.speaking)
    }

    pub(crate) fn is_interleaved(&self) -> bool {
        self.mode == GenerationMode::Interleaved
    }

    pub(crate) fn accept_text(&mut self, token: u32) {
        let interleaved = self.mode == GenerationMode::Interleaved;
        if token == self.markers.text_end && interleaved {
            self.text_done = true;
            self.speaking = true;
        } else if !interleaved {
            if token == self.markers.audio_start {
                self.mode = GenerationMode::Audio;
            } else if token == self.markers.mixed_start {
                self.mode = GenerationMode::Interleaved;
                self.slots_left = self.markers.text_slots;
                self.speaking = false;
                self.text_done = false;
            }
        }
        self.advance(interleaved);
    }

    pub(crate) fn accept_audio(&mut self, codes: &mut [u32]) -> bool {
        let interleaved = self.mode == GenerationMode::Interleaved;
        let terminal = codes.contains(&self.markers.audio_eos);
        if terminal {
            codes.fill(self.markers.audio_eos);
            self.mode = GenerationMode::Text;
        }
        self.advance(interleaved);
        terminal
    }

    fn advance(&mut self, interleaved: bool) {
        if interleaved && !self.text_done {
            self.slots_left -= 1;
            if self.slots_left == 0 {
                self.speaking = !self.speaking;
                self.slots_left = if self.speaking {
                    self.markers.audio_slots
                } else {
                    self.markers.text_slots
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn interleaved_generation_finishes_text_before_audio_tail() {
        let markers = GenerationMarkers {
            audio_start: 10,
            mixed_start: 11,
            text_end: 12,
            audio_eos: 100,
            text_slots: 2,
            audio_slots: 3,
        };
        let mut state = GenerationState::new(GenerationMode::Text, markers);
        state.accept_text(markers.mixed_start);
        assert!(!state.is_audio());
        state.accept_text(1);
        assert!(!state.is_audio());
        state.accept_text(2);
        assert!(state.is_audio());
        for _ in 0..markers.audio_slots {
            assert!(!state.accept_audio(&mut [1, 2]));
        }
        assert!(!state.is_audio());
        state.accept_text(markers.text_end);
        assert!(state.is_audio());
        for _ in 0..markers.audio_slots + 1 {
            assert!(!state.accept_audio(&mut [1, 2]));
        }
        assert!(state.is_audio());
        let mut end = [1, markers.audio_eos];
        assert!(state.accept_audio(&mut end));
        assert_eq!(end, [markers.audio_eos; 2]);
        assert!(!state.is_audio());
    }

    #[test]
    fn audio_repetition_penalty_applies_once_per_code() -> Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let config = AudioSampling {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Default::default()
        };
        assert_eq!(
            sample_logits(&[3.0, 1.0, -1.0], &config, &[0, 0, 0], &mut rng)?,
            0
        );
        assert_eq!(sample_logits(&[-1.0, -1.5], &config, &[0], &mut rng)?, 1);
        Ok(())
    }
}
