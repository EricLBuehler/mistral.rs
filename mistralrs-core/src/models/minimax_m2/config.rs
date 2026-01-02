use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(usize, block_size, 256);
serde_default_fn!(Activation, hidden_act, Activation::Silu);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub(crate) attn_type_list: Vec<usize>,
    pub(crate) head_dim: Option<usize>,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) mtp_transformer_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) num_local_experts: usize,
    pub(crate) num_mtp_modules: usize,
    pub(crate) qk_norm_type: String,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) rotary_dim: usize,
    pub(crate) scoring_func: String,
    pub(crate) shared_intermediate_size: usize,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) use_qk_norm: bool,
    pub(crate) use_routing_bias: bool,
    pub(crate) vocab_size: usize,
    #[serde(default = "block_size")]
    pub(crate) block_size: usize,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
}

impl Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Into<crate::models::mixtral::Config> for Config {
    fn into(self) -> crate::models::mixtral::Config {
        crate::models::mixtral::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            hidden_act: self.hidden_act,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            sliding_window: None,
            num_experts_per_tok: self.num_experts_per_tok,
            num_local_experts: self.num_local_experts,
            quantization_config: self.quantization_config,
            tie_word_embeddings: self.tie_word_embeddings,
        }
    }
}
