use mistralrs_quant::QuantizedConfig;
use serde::Deserialize;

use crate::layers::{Activation, DeepSeekV2RopeScaling};
use crate::serde_default_fn;

// ── Vision config defaults ──

serde_default_fn!(usize, default_vt_hidden_size, 1152);
serde_default_fn!(usize, default_vt_intermediate_size, 4304);
serde_default_fn!(usize, default_vt_num_attention_heads, 16);
serde_default_fn!(usize, default_vt_num_hidden_layers, 27);
serde_default_fn!(usize, default_patch_size, 14);
serde_default_fn!(usize, default_init_pos_emb_height, 64);
serde_default_fn!(usize, default_init_pos_emb_width, 64);
serde_default_fn!(usize, default_init_pos_emb_time, 4);
serde_default_fn!(Vec<usize>, default_merge_kernel_size, vec![2, 2]);
serde_default_fn!(usize, default_mm_hidden_size, 1152);
serde_default_fn!(usize, default_text_hidden_size_vision, 7168);
serde_default_fn!(f64, default_projector_ln_eps, 1e-5);

// ── Text config defaults ──

serde_default_fn!(f64, default_routed_scaling_factor, 1.0);
serde_default_fn!(usize, default_moe_layer_freq, 1);
serde_default_fn!(usize, default_first_k_dense_replace, 0);
serde_default_fn!(Activation, default_hidden_act, Activation::Silu);
serde_default_fn!(bool, default_tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
enum TopkMethod {
    #[serde(rename = "noaux_tc")]
    NoAuxTc,
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
    #[serde(rename = "sigmoid")]
    Sigmoid,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vt_hidden_size")]
    pub vt_hidden_size: usize,
    #[serde(default = "default_vt_intermediate_size")]
    pub vt_intermediate_size: usize,
    #[serde(default = "default_vt_num_attention_heads")]
    pub vt_num_attention_heads: usize,
    #[serde(default = "default_vt_num_hidden_layers")]
    pub vt_num_hidden_layers: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_init_pos_emb_height")]
    pub init_pos_emb_height: usize,
    #[serde(default = "default_init_pos_emb_width")]
    pub init_pos_emb_width: usize,
    #[serde(default = "default_init_pos_emb_time")]
    pub init_pos_emb_time: usize,
    #[serde(default = "default_merge_kernel_size")]
    pub merge_kernel_size: Vec<usize>,
    #[serde(default = "default_mm_hidden_size")]
    pub mm_hidden_size: usize,
    #[serde(default = "default_text_hidden_size_vision")]
    pub text_hidden_size: usize,
    #[serde(default = "default_projector_ln_eps")]
    pub projector_ln_eps: f64,
}

impl VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.vt_hidden_size / self.vt_num_attention_heads
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub n_shared_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    #[serde(default = "default_routed_scaling_factor")]
    pub routed_scaling_factor: f64,
    #[serde(default)]
    topk_method: TopkMethod,
    pub num_experts_per_tok: Option<usize>,
    #[serde(default = "default_moe_layer_freq")]
    pub moe_layer_freq: usize,
    #[serde(default = "default_first_k_dense_replace")]
    pub first_k_dense_replace: usize,
    #[serde(default)]
    scoring_func: ScoringFunc,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub rope_theta: f32,
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub attention_bias: bool,
    pub q_lora_rank: Option<usize>,
    pub qk_rope_head_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: usize,
    pub qk_nope_head_dim: usize,
    #[serde(alias = "quantization")]
    pub quantization_config: Option<QuantizedConfig>,
    pub n_group: usize,
    pub topk_group: usize,
}

impl Default for TopkMethod {
    fn default() -> Self {
        TopkMethod::Greedy
    }
}

impl Default for ScoringFunc {
    fn default() -> Self {
        ScoringFunc::Softmax
    }
}

impl TextConfig {
    pub fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    pub fn softmax_scale(&self) -> f32 {
        let mut softmax_scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(DeepSeekV2RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale =
                crate::layers::DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }
        softmax_scale
    }

    pub fn use_sigmoid_scoring(&self) -> bool {
        matches!(self.scoring_func, ScoringFunc::Sigmoid)
    }

    pub fn use_noaux_tc(&self) -> bool {
        matches!(self.topk_method, TopkMethod::NoAuxTc)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub media_placeholder_token_id: usize,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

impl Config {
    /// Returns the effective quantization config, with top-level text_config's
    /// quantization_config taking precedence.
    #[allow(dead_code)]
    pub fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.text_config.quantization_config
    }
}
