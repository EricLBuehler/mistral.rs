#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::RmsNorm,
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::FlashParams, EitherCache, IsqModel, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
};

serde_default_fn!(bool, tie_word_embeddings, false);
serde_default_fn!(f64, rms_norm_eps, 1e-6);
serde_default_fn!(usize, max_position_embeddings, 1_048_576);
serde_default_fn!(f64, rope_theta, 10_000.0);
serde_default_fn!(f64, final_logit_softcapping, 30.0);
serde_default_fn!(f64, situ_beta, 4.0);
serde_default_fn!(f64, situ_linear_beta, 25.0);
serde_default_fn!(f64, qk_l2_clip_threshold, 22.627416997969522);
serde_default_fn!(f64, rel_l2_clip_threshold, 8.0);
serde_default_fn!(usize, n_log_scale_floor, 8192);
serde_default_fn!(f64, f_log_scale_alpha, 0.1);
serde_default_fn!(usize, attn_res_block_size, 6);
serde_default_fn!(bool, attn_res_normalize_values, true);
serde_default_fn!(usize, n_shared_experts, 0);
serde_default_fn!(f64, route_scale, 8.0);
serde_default_fn!(bool, use_gate_bias, true);
serde_default_fn!(usize, sconv_kernel_size, 4);
serde_default_fn!(usize, gate_rank, 64);

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    #[serde(default = "gate_rank")]
    pub gate_rank: usize,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "final_logit_softcapping")]
    pub final_logit_softcapping: f64,
    #[serde(default = "situ_beta")]
    pub situ_beta: f64,
    #[serde(default = "situ_linear_beta")]
    pub situ_linear_beta: f64,
    #[serde(default = "qk_l2_clip_threshold")]
    pub qk_l2_clip_threshold: f64,
    #[serde(default = "rel_l2_clip_threshold")]
    pub rel_l2_clip_threshold: f64,
    #[serde(default = "n_log_scale_floor")]
    pub n_log_scale_floor: usize,
    #[serde(default = "f_log_scale_alpha")]
    pub f_log_scale_alpha: f64,
    #[serde(default = "attn_res_block_size")]
    pub attn_res_block_size: usize,
    #[serde(default = "attn_res_normalize_values")]
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "n_shared_experts")]
    pub n_shared_experts: usize,
    #[serde(default = "route_scale")]
    pub route_scale: f64,
    #[serde(default = "use_gate_bias")]
    pub use_gate_bias: bool,
    #[serde(default = "sconv_kernel_size")]
    pub sconv_kernel_size: usize,
    pub loop_config: LoopConfig,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub sliding_window_pattern: Option<Vec<bool>>,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    #[serde(default = "tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<mistralrs_quant::QuantizedConfig>,
}

impl Config {
    pub fn is_swa(&self, layer: usize) -> bool {
        if let Some(pattern) = &self.sliding_window_pattern {
            if let Some(is_swa) = pattern.get(layer) {
                return *is_swa;
            }
        }
        if let Some(local) = &self.local_layer_ids {
            return local.contains(&layer);
        }
        layer % 6 != 5
    }
}

pub struct Model {
    cfg: ModelConfigMetadata,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    lm_head: Arc<dyn QuantMethod>,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mapper = normal_loading_metadata.mapper;
        let vb_m = vb.pp("model");
        let _embed_tokens = crate::layers::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &None,
        )?;
        let _norm = RmsNorm::new_gemma(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            &None,
            mapper.set_nm_device(vb.pp("lm_head"), normal_loading_metadata.loading_isq),
        )?;

        let cache = EitherCache::Normal(crate::pipeline::NormalCache::new(
            cfg.num_hidden_layers,
            cfg.max_position_embeddings,
        ));

        Ok(Self {
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim,
                v_head_dim: cfg.head_dim,
                kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
            },
            device: normal_loading_metadata.real_device.clone(),
            cache,
            max_seq_len: cfg.max_position_embeddings,
            mapper,
            lm_head,
        })
    }
}

impl IsqModel for Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        _input_ids: &Tensor,
        _ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        candle_core::bail!("inkling2_loop forward not implemented yet")
    }

    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        candle_core::bail!("inkling2_loop xlora not supported")
    }

    fn is_xlora(&self) -> bool {
        false
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl crate::speculative::SpeculativeTargetMixin for Model {}

impl AnyMoeBaseModelMixin for Model {}
