use candle_core::{Result, Tensor, D};
use candle_nn::{
    conv2d, embedding, Activation, Conv2d, Conv2dConfig, Embedding, Module, VarBuilder,
};
use serde::Deserialize;

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
pub struct PerceiverConfig {
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
pub struct VisionConfig {
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
pub struct TextConfig {
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    perceiver_config: PerceiverConfig,
    vision_config: VisionConfig,
    text_config: TextConfig,
    #[serde(default = "default_32001")]
    image_token_id: usize,
    #[serde(default = "default_false")]
    tie_word_embeddings: bool,
}

pub struct VisionEmbeddings {
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
        todo!()
    }
}
