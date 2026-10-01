use mistralrs_quant::QuantizedConfig;

use crate::gdn::{GdnGateActivation, GdnStateDType, GdnVHeadLayout};
use crate::layers::{Activation, YarnRopeConfig};
use crate::serde_default_fn;
use crate::vision_models::qwen3_5::config::RopeParameters;

pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(bool, default_norm_topk_prob, true);
serde_default_fn!(usize, default_hc_count, 4);
serde_default_fn!(usize, default_hc_lowrank, 320);
serde_default_fn!(usize, default_ple_conv_kernel_size, 4);
serde_default_fn!(usize, default_ngram_size, 3);
serde_default_fn!(usize, default_heads_per_ngram, 8);
serde_default_fn!(u64, default_ngram_vocab_size_base, 20_000_000);
serde_default_fn!(u64, default_seed, 1234);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    #[serde(alias = "indexed_attention")]
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIds {
    pub fn first(&self) -> Option<u32> {
        match self {
            Self::Single(id) => Some(*id),
            Self::Multiple(ids) => ids.first().copied(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
    #[serde(default)]
    pub output_gate_type: Option<GdnGateActivation>,
    #[serde(default = "default_hc_count")]
    pub hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv_kernel_size")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub indexer_n_heads: Option<usize>,
    #[serde(default)]
    pub indexer_kv_heads: Option<usize>,
    #[serde(default)]
    pub indexer_head_dim: Option<usize>,
    #[serde(default)]
    pub indexer_budget: Option<usize>,
    #[serde(default)]
    pub indexer_compress_ratio: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIds>,
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_gdn_v_head_layout")]
    pub(crate) gdn_v_head_layout: GdnVHeadLayout,
    #[serde(default, rename = "_mistralrs_ple_hash")]
    pub(crate) ple_hash: Option<PleHashConstants>,
}

/// PLE hash constants stated outright (GGUF stores these rather than the seed they derive from).
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub(crate) struct PleHashConstants {
    pub layer_multipliers: Vec<u64>,
    pub head_vocab_sizes: Vec<u64>,
    pub head_offsets: Vec<u64>,
}

/// Block-sparse attention (QSA) parameters shared by every indexed attention layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QsaConfig {
    pub n_heads: usize,
    pub head_dim: usize,
    pub budget: usize,
    pub compress_ratio: usize,
}

impl QsaConfig {
    pub fn block_topk(&self) -> usize {
        self.budget / self.compress_ratio
    }

    /// Most tokens any query attends to: whole selected blocks plus the incomplete tail.
    pub fn max_selected_tokens(&self) -> usize {
        self.budget + self.compress_ratio - 1
    }

    pub(crate) fn aux_cache_elements_per_token(&self, rot_dim: usize) -> usize {
        self.head_dim + rot_dim + self.head_dim.div_ceil(self.compress_ratio)
    }
}

/// Hashed n-gram PLE parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleConfig {
    pub layer_idx: usize,
    pub embed_dim: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel_size: usize,
    pub eos_token_id: u32,
}

impl PleConfig {
    pub fn num_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads()
    }

    pub fn conv_dilation(&self) -> usize {
        self.ngram_size
    }

    pub fn conv_state_len(&self) -> usize {
        (self.conv_kernel_size - 1) * self.conv_dilation()
    }

    pub fn context_len(&self) -> usize {
        self.ngram_size - 1
    }
}

impl TextConfig {
    pub fn validate(&self) -> candle_core::Result<()> {
        if self.num_hidden_layers == 0 || self.max_position_embeddings == 0 {
            candle_core::bail!("Qwen4-Exp requires at least one layer and a positive context");
        }
        if self.hc_count <= 1 || self.hc_lowrank == 0 {
            candle_core::bail!(
                "Qwen4-Exp requires hc_count > 1 and a positive hc_lowrank, got {} and {}",
                self.hc_count,
                self.hc_lowrank
            );
        }
        let layer_types = self.layer_types();
        if layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Qwen4-Exp layer_types lists {} layers, expected {}",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        if !layer_types.contains(&LayerType::FullAttention) {
            candle_core::bail!("Qwen4-Exp needs at least one attention layer");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            candle_core::bail!(
                "Qwen4-Exp has incompatible attention head counts: {} query and {} KV",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.linear_num_key_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            candle_core::bail!("Qwen4-Exp has incompatible GDN head counts");
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            candle_core::bail!(
                "Qwen4-Exp has invalid MoE routing: {} of {} experts",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        let rot_dim = self.rot_dim();
        if rot_dim == 0 || rot_dim > self.head_dim || !rot_dim.is_multiple_of(2) {
            candle_core::bail!("Qwen4-Exp rotary dimension {rot_dim} is invalid");
        }
        if !self.rope_parameters.mrope_interleaved || self.mrope_section().len() != 3 {
            candle_core::bail!("Qwen4-Exp requires interleaved three-section MRoPE");
        }
        if let Some(qsa) = self.qsa()? {
            if rot_dim > qsa.head_dim {
                candle_core::bail!(
                    "Qwen4-Exp rotary dim {rot_dim} exceeds indexer head dim {}",
                    qsa.head_dim
                );
            }
        }
        if let Some(ple) = self.ple()? {
            if layer_types[ple.layer_idx] != LayerType::LinearAttention {
                candle_core::bail!("Qwen4-Exp PLE is only supported on linear attention layers");
            }
        }
        self.rope_parameters
            .validate_scaling(self.max_position_embeddings)
    }

    pub fn qsa(&self) -> candle_core::Result<Option<QsaConfig>> {
        let fields = [
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        ];
        if fields.iter().all(Option::is_none) {
            return Ok(None);
        }
        let [Some(n_heads), Some(kv_heads), Some(head_dim), Some(budget), Some(compress_ratio)] =
            fields
        else {
            candle_core::bail!("Qwen4-Exp QSA config is missing fields: {fields:?}");
        };
        if kv_heads != 1 {
            candle_core::bail!("Qwen4-Exp QSA requires indexer_kv_heads=1, got {kv_heads}");
        }
        if n_heads == 0
            || head_dim == 0
            || compress_ratio == 0
            || budget == 0
            || !budget.is_multiple_of(compress_ratio)
        {
            candle_core::bail!("Qwen4-Exp QSA config is invalid: {fields:?}");
        }
        Ok(Some(QsaConfig {
            n_heads,
            head_dim,
            budget,
            compress_ratio,
        }))
    }

    pub fn ple(&self) -> candle_core::Result<Option<PleConfig>> {
        let mut ids = self.ple_layer_ids.clone();
        ids.sort_unstable();
        ids.dedup();
        match ids.as_slice() {
            [] => Ok(None),
            [layer_id] => {
                if *layer_id == 0 || *layer_id > self.num_hidden_layers {
                    candle_core::bail!("Qwen4-Exp ple_layer_ids are one-indexed, got {layer_id}");
                }
                let eos_token_id = self
                    .eos_token_id
                    .as_ref()
                    .and_then(TokenIds::first)
                    .ok_or_else(|| {
                        candle_core::Error::msg("Qwen4-Exp PLE requires text_config.eos_token_id")
                    })?;
                let ple = PleConfig {
                    layer_idx: layer_id - 1,
                    embed_dim: self.ple_embed_dim.unwrap_or(self.hidden_size),
                    ngram_size: self.ngram_size,
                    heads_per_ngram: self.heads_per_ngram,
                    conv_kernel_size: self.ple_conv_kernel_size,
                    eos_token_id,
                };
                let explicit_hash_mismatch = self.ple_hash.as_ref().is_some_and(|hash| {
                    hash.layer_multipliers.len() != ple.ngram_size
                        || hash.head_vocab_sizes.len() != ple.num_heads()
                        || hash.head_offsets.len() != ple.num_heads()
                });
                if ple.ngram_size < 2
                    || ple.heads_per_ngram == 0
                    || ple.conv_kernel_size == 0
                    || !ple.embed_dim.is_multiple_of(ple.num_heads())
                    || explicit_hash_mismatch
                {
                    candle_core::bail!("Qwen4-Exp PLE config is invalid: {ple:?}");
                }
                Ok(Some(ple))
            }
            _ => candle_core::bail!("Qwen4-Exp supports a single PLE layer, got {ids:?}"),
        }
    }

    pub fn output_gate_activation(&self) -> candle_core::Result<GdnGateActivation> {
        if let Some(activation) = self.output_gate_type {
            return Ok(activation);
        }
        match self.hidden_act {
            Activation::Silu => Ok(GdnGateActivation::Silu),
            Activation::Sigmoid => Ok(GdnGateActivation::Sigmoid),
            other => {
                candle_core::bail!("Qwen4-Exp output gate activation {other:?} is unsupported")
            }
        }
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    pub fn yarn_rope_config(&self) -> candle_core::Result<Option<YarnRopeConfig>> {
        self.rope_parameters
            .yarn_rope_config(self.max_position_embeddings, self.rot_dim())
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if let Some(layer_types) = &self.layer_types {
            return layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim + self.linear_value_dim()
    }

    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }

    pub fn hc_hidden_size(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    pub fn mtp_layers(&self, mtp: bool) -> usize {
        if mtp {
            self.mtp_num_hidden_layers
        } else {
            0
        }
    }

    /// Paged-KV layer kinds: the main stack, then any MTP blocks (all QSA attention) after it.
    pub fn paged_layer_types(&self, mtp: bool) -> Vec<LayerType> {
        let mut layers = self.layer_types();
        layers.extend(std::iter::repeat_n(
            LayerType::FullAttention,
            self.mtp_layers(mtp),
        ));
        layers
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    #[serde(default)]
    pub vision_config: Option<VisionConfig>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    /// Injected by the loader when the built-in MTP head should be loaded (see `MTP_CONFIG_KEY`).
    #[serde(default, rename = "_mistralrs_mtp")]
    pub mtp: bool,
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The text config of `Qwen/Qwen3.8-Flash-Next`, with the layer list shortened.
    pub(crate) fn flash_next_text_config(num_layers: usize) -> TextConfig {
        let layer_types = (0..num_layers)
            .map(|i| {
                if (i + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        serde_json::from_value(serde_json::json!({
            "eos_token_id": 248044,
            "hc_count": 4,
            "hc_lowrank": 320,
            "head_dim": 256,
            "heads_per_ngram": 8,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "indexer_head_dim": 128,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 4,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "make_ngram_vocab_size_divisible_by": 128,
            "mamba_ssm_dtype": "float32",
            "max_position_embeddings": 262144,
            "moe_intermediate_size": 640,
            "ngram_size": 3,
            "ngram_vocab_size_base": 20000000,
            "num_attention_heads": 24,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "num_hidden_layers": num_layers,
            "num_key_value_heads": 2,
            "output_gate_type": "sigmoid",
            "ple_conv_kernel_size": 4,
            "ple_embed_dim": 2560,
            "ple_layer_ids": [2],
            "rms_norm_eps": 1e-06,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default"
            },
            "shared_expert_intermediate_size": 640,
            "vocab_size": 248320
        }))
        .unwrap()
    }

    #[test]
    fn flash_next_config_parses_and_validates() {
        let cfg = flash_next_text_config(48);
        cfg.validate().unwrap();
        assert_eq!(cfg.rot_dim(), 64);
        assert_eq!(
            cfg.output_gate_activation().unwrap(),
            GdnGateActivation::Sigmoid
        );
        let qsa = cfg.qsa().unwrap().unwrap();
        assert_eq!((qsa.block_topk(), qsa.max_selected_tokens()), (512, 2051));
        let ple = cfg.ple().unwrap().unwrap();
        assert_eq!(ple.layer_idx, 1);
        assert_eq!(
            (ple.num_heads(), ple.head_dim(), ple.conv_state_len()),
            (16, 160, 9)
        );
        assert_eq!(cfg.layer_types()[3], LayerType::FullAttention);
    }
}
