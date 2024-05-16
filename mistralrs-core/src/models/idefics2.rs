use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    conv2d, embedding, layer_norm, linear_no_bias, Activation, Conv2d, Conv2dConfig, Embedding,
    LayerNorm, Linear, Module, VarBuilder,
};
use serde::Deserialize;

use crate::{
    layers::{repeat_kv, CausalMasker, RmsNorm},
    pipeline::Cache,
};

use super::mistral;

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py

fn default_32000() -> usize {
    32000
}
fn default_32001() -> usize {
    32001
}
fn default_4096() -> usize {
    4006
}
fn default_14336() -> usize {
    14336
}
fn default_32() -> usize {
    32
}
fn default_8() -> usize {
    8
}
fn default_act() -> Activation {
    Activation::Silu
}
fn default_131072() -> usize {
    131072
}
fn default_eps() -> f64 {
    1e-6
}
fn default_rope() -> f64 {
    10000.0
}
fn default_false() -> bool {
    false
}
fn default_sliding() -> Option<usize> {
    Some(4096)
}
fn default_gelu() -> Activation {
    Activation::GeluPytorchTanh
}
fn default_64() -> usize {
    64
}
fn default_3() -> usize {
    3
}
fn default_16() -> usize {
    16
}
fn default_96() -> usize {
    96
}
fn default_4() -> usize {
    4
}
fn default_0_0() -> f32 {
    0.0
}
fn default_0_02() -> f32 {
    0.02
}
fn default_768() -> usize {
    768
}
fn default_3072() -> usize {
    3072
}
fn default_12() -> usize {
    12
}
fn default_224() -> usize {
    224
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct PerceiverConfig {
    #[serde(default = "default_act")]
    hidden_act: Activation,
    #[serde(default = "default_64")]
    resampler_n_latents: usize,
    #[serde(default = "default_3")]
    resampler_depth: usize,
    #[serde(default = "default_16")]
    resampler_n_heads: usize,
    #[serde(default = "default_96")]
    resampler_head_dim: usize,
    #[serde(default = "default_4")]
    num_kv_heads: usize,
    #[serde(default = "default_0_0")]
    attn_dropout: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisionConfig {
    #[serde(default = "default_768")]
    hidden_size: usize,
    #[serde(default = "default_3072")]
    intermediate_size: usize,
    #[serde(default = "default_12")]
    num_hidden_layers: usize,
    #[serde(default = "default_12")]
    num_attn_heads: usize,
    #[serde(default = "default_3")]
    num_channels: usize,
    #[serde(default = "default_224")]
    image_size: usize,
    #[serde(default = "default_32")]
    patch_size: usize,
    #[serde(default = "default_gelu")]
    hidden_act: Activation,
    #[serde(default = "default_eps")]
    layer_norm_eps: f64,
    #[serde(default = "default_0_0")]
    attn_dropout: f32,
    #[serde(default = "default_0_02")]
    initiailizer_range: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct TextConfig {
    #[serde(default = "default_32000")]
    vocab_size: usize,
    #[serde(default = "default_4096")]
    hidden_size: usize,
    #[serde(default = "default_14336")]
    intermediate_size: usize,
    #[serde(default = "default_32")]
    num_hidden_layers: usize,
    #[serde(default = "default_32")]
    num_attention_heads: usize,
    #[serde(default = "default_8")]
    num_key_value_heads: usize,
    #[serde(default = "default_act")]
    hidden_act: Activation,
    #[serde(default = "default_131072")]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    rope_theta: f64,
    #[serde(default = "default_sliding")]
    sliding_window: Option<usize>,

    #[serde(default = "default_false")]
    use_flash_attn: bool,
    model_type: String, // Must be mistral for now
}

impl Into<mistral::Config> for TextConfig {
    fn into(self) -> mistral::Config {
        mistral::Config {
            vocab_size: self.vocab_size,
            hidden_act: self.hidden_act,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: self.sliding_window,
            use_flash_attn: self.use_flash_attn,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct Config {
    perceiver_config: PerceiverConfig,
    vision_config: VisionConfig,
    text_config: TextConfig,
    #[serde(default = "default_32001")]
    image_token_id: usize,
    #[serde(default = "default_false")]
    tie_word_embeddings: bool,
}

// == START VISION MODEL ==

struct VisionEmbeddings {
    embed_dim: usize,
    image_size: usize,
    patch_size: usize,
    patch_embedding: Conv2d,
    num_patches_per_side: usize,
    num_patches: usize,
    num_positions: usize,
    position_embedding: Embedding,
}

impl VisionEmbeddings {
    fn new(config: &VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut conv_config = Conv2dConfig::default();
        conv_config.stride = config.patch_size;
        let patch_embedding = conv2d(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            conv_config,
            vb.pp("patch_embedding"),
        )?;
        let num_patches_per_side = config.image_size / config.patch_size;
        let num_patches = num_patches_per_side.pow(2);
        Ok(Self {
            embed_dim: config.hidden_size,
            image_size: config.image_size,
            patch_size: config.patch_size,
            patch_embedding,
            num_patches_per_side,
            num_patches,
            num_positions: num_patches,
            position_embedding: embedding(
                num_patches,
                config.hidden_size,
                vb.pp("position_embedding"),
            )?,
        })
    }

    fn forward(&self, pixel_values: &Tensor, patch_attention_mask: &Tensor) -> Result<Tensor> {
        let (bs, _, max_im_h, max_im_w) = pixel_values.dims4()?;

        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let embeddings = patch_embeds.flatten(2, D::Minus1)?.transpose(1, 2)?;

        let (max_nb_patches_h, max_nb_patches_w) =
            (max_im_h / self.patch_size, max_im_w / self.patch_size);
        let boundaries = Tensor::arange_step(
            1.0 / self.num_patches_per_side as f64,
            1.0,
            1.0 / self.num_patches_per_side as f64,
            pixel_values.device(),
        )?;
        let position_ids = Tensor::full(
            0u32,
            (bs, max_nb_patches_h * max_nb_patches_w),
            pixel_values.device(),
        )?;

        for (b_idx, p_attn_mask) in patch_attention_mask.chunk(bs, 0)?.iter().enumerate() {
            let nb_patches_h = p_attn_mask.i((.., 0))?.sum_all()?;
            let nb_patches_w = p_attn_mask.i((0,))?.sum_all()?;

            let fractional_coords_h = Tensor::arange_step(
                0.0,
                1.0 - 1e-6,
                1.0 / nb_patches_h.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                pixel_values.device(),
            )?;
            let fractional_coords_w = Tensor::arange_step(
                0.0,
                1.0 - 1e-6,
                1.0 / nb_patches_w.to_dtype(DType::F32)?.to_scalar::<f32>()?,
                pixel_values.device(),
            )?;

            // TODO(EricLBuehler): https://github.com/huggingface/candle/issues/2185
            /*
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
            */
        }
        let position_ids = position_ids.to_device(self.position_embedding.embeddings().device())?;
        embeddings + self.position_embedding.forward(&position_ids)?
    }
}

struct Attention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    neg_inf: Tensor,
}

impl Attention {
    fn new(config: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attn_heads;
        let head_dim = embed_dim / num_heads;
        let scale = (head_dim as f64).sqrt();

        let q_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("o_proj"))?;

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            neg_inf: Tensor::new(f32::NEG_INFINITY, vb.device())?.to_dtype(vb.dtype())?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        vision_transformer_kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(&xs)?;
        let k = self.k_proj.forward(&xs)?;
        let v = self.v_proj.forward(&xs)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (k, v) = Cache::update_kv_cache(vision_transformer_kv_cache, k, v, false)?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;

        let attn_weights =
            CausalMasker.apply_mask(&attention_mask.cloned(), attn_weights, &self.neg_inf)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.embed_dim))?
            .apply(&self.o_proj)
    }
}

struct VisionMLP {
    activation: Activation,
    fc1: Linear,
    fc2: Linear,
}

impl VisionMLP {
    fn new(config: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self {
            activation: config.hidden_act,
            fc1,
            fc2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = self.activation.forward(&x)?;
        self.fc2.forward(&x)
    }
}

struct EncoderLayer {
    mlp: VisionMLP,
    attn: Attention,
    layer_norm_1: LayerNorm,
    layer_norm_2: LayerNorm,
}

impl EncoderLayer {
    fn new(config: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mlp = VisionMLP::new(config.clone(), vb.pp("mlp"))?;
        let attn = Attention::new(config.clone(), vb.pp("self_attn"))?;
        let layer_norm_1 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm1"),
        )?;
        let layer_norm_2 = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm2"),
        )?;
        Ok(Self {
            mlp,
            attn,
            layer_norm_1,
            layer_norm_2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        vision_transformer_kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = xs.clone();

        let hidden_states = self.layer_norm_1.forward(xs)?;
        let hidden_states =
            self.attn
                .forward(&hidden_states, attention_mask, vision_transformer_kv_cache)?;
        let hidden_states = (hidden_states + residual)?;

        let residual = &hidden_states;
        let hidden_states = self.layer_norm_2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        hidden_states + residual
    }
}

struct Encoder {
    layers: Vec<EncoderLayer>,
    cache: Cache,
}

impl Encoder {
    fn new(config: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let vb_l = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::new(config.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            layers,
            cache: Cache::new(config.num_hidden_layers, false),
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        let mut cache = self.cache.lock();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, attention_mask, &mut cache[i])?;
        }
        Ok(hidden_states)
    }
}

struct VisionTransformer {
    embeddings: VisionEmbeddings,
    encoder: Encoder,
    post_layernorm: LayerNorm,
    config: VisionConfig,
}

impl VisionTransformer {
    fn new(config: VisionConfig, vb: VarBuilder) -> Result<Self> {
        let embeddings = VisionEmbeddings::new(&config, vb.pp("embeddings"))?;
        let post_layernorm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("post_layernorm"),
        )?;
        let encoder = Encoder::new(config.clone(), vb.pp("encoder"))?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
            config,
        })
    }

    fn forward(&self, pixel_values: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let bs = pixel_values.dim(0)?;
        let patch_attention_mask = if let Some(attn_mask) = attention_mask {
            attn_mask.clone()
        } else {
            let patch_size = self.config.patch_size;
            Tensor::ones(
                (
                    bs,
                    pixel_values.dim(2)? / patch_size,
                    pixel_values.dim(3)? / patch_size,
                ),
                DType::U8,
                pixel_values.device(),
            )?
        };

        let hidden_states = self
            .embeddings
            .forward(pixel_values, &patch_attention_mask)?;

        let attention_mask = if attention_mask.is_none() {
            None
        } else {
            let mask = patch_attention_mask.reshape((patch_attention_mask.dim(0)?, ()))?;
            Some(CausalMasker.expand_mask(&mask, patch_attention_mask.dtype(), None)?)
        };
        let hidden_states = self
            .encoder
            .forward(&hidden_states, attention_mask.as_ref())?;
        hidden_states.apply(&self.post_layernorm)
    }
}

// == END VISION MODEL ==

// == START CONNECTOR ==
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl MLP {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        output_size: usize,
        activation: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        let gate_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_size, output_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.down_proj
            .forward(&self.activation.forward(&self.gate_proj.forward(x)?)?)?
            * self.up_proj.forward(x)?
    }
}

struct PerceiverAttention {
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    neg_inf: Tensor,
}

impl PerceiverAttention {
    fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.text_config.hidden_size;
        let num_heads = config.perceiver_config.resampler_n_heads;
        let head_dim = config.perceiver_config.resampler_head_dim;
        let num_key_value_heads = config.perceiver_config.num_kv_heads;
        let num_key_value_groups = num_heads / num_key_value_heads;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            neg_inf: Tensor::new(f32::NEG_INFINITY, vb.device())?.to_dtype(vb.dtype())?,
            num_kv_heads: num_key_value_heads,
            num_kv_groups: num_key_value_groups,
        })
    }

    fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        attention_mask: &Tensor,
        perceiver_kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = latents.dims3()?;

        let hidden_states = Tensor::cat(&[context, latents], D::Minus2)?;

        let q = self.q_proj.forward(&latents)?;
        let k = self.k_proj.forward(&hidden_states)?;
        let v = self.v_proj.forward(&hidden_states)?;

        let q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (k, v) = Cache::update_kv_cache(perceiver_kv_cache, k, v, false)?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * (self.head_dim as f64).sqrt())?;

        let attn_weights =
            CausalMasker.apply_mask(&Some(attention_mask.clone()), attn_weights, &self.neg_inf)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }
}

struct PerceiverLayer {
    input_latents_norm: RmsNorm,
    input_context_norm: RmsNorm,
    self_attn: PerceiverAttention,
    post_attn_norm: RmsNorm,
    mlp: MLP,
}

impl PerceiverLayer {
    fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.text_config.hidden_size;
        let mlp_act = config.perceiver_config.hidden_act;
        let rms_eps = config.text_config.rms_norm_eps;

        Ok(Self {
            input_latents_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("input_latents_norm"))?,
            input_context_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("input_context_norm"))?,
            self_attn: PerceiverAttention::new(config, vb.pp("self_attn"))?,
            post_attn_norm: RmsNorm::new(hidden_size, rms_eps, vb.pp("post_attention_layernorm"))?,
            mlp: MLP::new(
                hidden_size,
                hidden_size * 4,
                hidden_size,
                mlp_act,
                vb.pp("mlp"),
            )?,
        })
    }

    fn forward(
        &self,
        latents: &Tensor,
        context: &Tensor,
        attention_mask: &Tensor,
        perceiver_kv_cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let residual = latents;

        let latents = self.input_latents_norm.forward(latents)?;
        let context = self.input_latents_norm.forward(context)?;

        let latents =
            self.self_attn
                .forward(&latents, &context, attention_mask, perceiver_kv_cache)?;
        let latents = (residual + latents)?;
        let residual = &latents;

        let latents = self.post_attn_norm.forward(&latents)?;
        let latents = self.mlp.forward(&latents)?;
        residual + latents
    }
}

struct PerceiverResampler {
    latents: Tensor,
    layers: Vec<PerceiverLayer>,
    norm: RmsNorm,
    n_latents: usize,
    cache: Cache,
}

impl PerceiverResampler {
    fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let n_latents = config.perceiver_config.resampler_n_latents;
        let hidden_size = config.text_config.hidden_size;
        let depth = config.perceiver_config.resampler_depth;

        let latents = vb.get((n_latents, hidden_size), "latents")?;
        let mut layers = Vec::new();
        let vb_l = vb.pp("layers");
        for i in 0..depth {
            layers.push(PerceiverLayer::new(config.clone(), vb_l.pp(i))?);
        }
        let norm = RmsNorm::new(hidden_size, config.text_config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            latents,
            layers,
            norm,
            n_latents,
            cache: Cache::new(depth, false),
        })
    }

    fn forward(&self, context: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let latents = self
            .latents
            .unsqueeze(0)?
            .expand(vec![context.dim(0)?].extend(self.latents.dims()))?;

        let latent_attention_mask = Tensor::ones(
            (attention_mask.dim(0)?, latents.dim(1)?),
            attention_mask.dtype(),
            attention_mask.device(),
        )?;
        let attention_mask = Tensor::cat(&[attention_mask, &latent_attention_mask], D::Minus1)?;
        let attention_mask =
            CausalMasker.expand_mask(&attention_mask, latents.dtype(), Some(self.n_latents))?;

        let mut compressed_context = latents;
        let mut cache = self.cache.lock();
        for (i, perceiver_layer) in self.layers.iter().enumerate() {
            compressed_context = perceiver_layer.forward(
                &compressed_context,
                context,
                &attention_mask,
                &mut cache[i],
            )?;
        }
        self.norm.forward(&compressed_context)
    }
}

struct Connector {
    modality_projection: MLP,
    perceiver_resampler: PerceiverResampler,
}

impl Connector {
    fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let modality_projection = MLP::new(
            config.vision_config.hidden_size,
            config.text_config.intermediate_size,
            config.text_config.hidden_size,
            config.text_config.hidden_act,
            vb.pp("modality_projection"),
        )?;
        let perceiver_resampler = PerceiverResampler::new(config, vb.pp("perceiver_resampler"))?;
        Ok(Self {
            modality_projection,
            perceiver_resampler,
        })
    }

    fn forward(&self, image_hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let image_hidden_states = self.modality_projection.forward(image_hidden_states)?;
        self.perceiver_resampler
            .forward(&image_hidden_states, attention_mask)
    }
}

// == END CONNECTOR ==
