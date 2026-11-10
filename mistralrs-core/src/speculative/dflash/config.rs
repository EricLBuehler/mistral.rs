use super::*;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct DFlashSpecificConfig {
    pub target_layer_ids: Option<Vec<usize>>,
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    pub input_embedding_scale: Option<f64>,
    pub output_multiplier: Option<f64>,
    pub selector_rank: Option<usize>,
    pub selector_top_k: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub conv_group_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub rope_theta: Option<f64>,
    pub rope_parameters: Option<serde_json::Value>,
    pub num_target_layers: Option<usize>,
    pub sliding_window: Option<usize>,
    pub layer_types: Option<Vec<String>>,
    pub is_causal: Option<bool>,
    // v1 checkpoints keep some fields at the top level instead of inside `dflash_config`
    pub block_size: Option<usize>,
    pub mask_token_id: Option<u32>,
    #[serde(default)]
    pub dflash_config: DFlashSpecificConfig,
}

pub struct DFlashLoadTarget<'a> {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub yarn_rope_config: Option<&'a YarnRopeConfig>,
    pub device: &'a Device,
    pub dtype: DType,
}

impl DFlashConfig {
    pub fn is_dflash(&self) -> bool {
        self.architectures
            .iter()
            .any(|a| a.contains("DFlash") || a.contains("Dflash"))
    }

    pub(super) fn is_v2(&self) -> bool {
        self.architectures.iter().any(|a| a.contains("DFlash2"))
    }

    pub(super) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub(super) fn rope_theta(&self) -> f64 {
        self.rope_theta
            .or_else(|| {
                self.rope_parameters
                    .as_ref()
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .unwrap_or(1e7)
    }

    pub(super) fn validate_rope_type(&self) -> Result<()> {
        let Some(parameters) = self.rope_parameters.as_ref() else {
            return Ok(());
        };
        let parameters = parameters.as_object().ok_or_else(|| {
            candle_core::Error::msg("DFlash rope_parameters must be a JSON object")
        })?;
        let rope_type = parameters
            .get("rope_type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    candle_core::Error::msg("DFlash rope_parameters.rope_type must be a string")
                })
            })
            .transpose()?;
        let legacy_type = parameters
            .get("type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    candle_core::Error::msg("DFlash rope_parameters.type must be a string")
                })
            })
            .transpose()?;
        if let (Some(rope_type), Some(legacy_type)) = (rope_type, legacy_type) {
            if rope_type != legacy_type {
                candle_core::bail!(
                    "DFlash rope_parameters.rope_type `{rope_type}` conflicts with legacy type `{legacy_type}`"
                );
            }
        }
        let rope_type = rope_type.or(legacy_type).unwrap_or("default");
        if rope_type != "default" {
            candle_core::bail!(
                "DFlash draft-side RoPE type `{rope_type}` is unsupported; configure RoPE scaling on the target model"
            );
        }
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    pub(super) fn yarn_rope_config(
        &self,
        target: Option<&YarnRopeConfig>,
    ) -> Result<Option<YarnRopeConfig>> {
        let Some(target) = target else {
            return Ok(None);
        };
        let max_position_embeddings = self.max_position_embeddings.ok_or_else(|| {
            candle_core::Error::msg(
                "DFlash max_position_embeddings is required when the target uses YaRN",
            )
        })?;
        if max_position_embeddings != target.original_max_position_embeddings {
            candle_core::bail!(
                "DFlash native context length {} does not match target YaRN original context length {}",
                max_position_embeddings,
                target.original_max_position_embeddings
            );
        }
        let mut config = target.clone();
        config.base = self.rope_theta() as f32;
        config.head_dim = self.head_dim();
        Ok(Some(config))
    }

    pub fn block_size(&self) -> usize {
        self.dflash_config
            .block_size
            .or(self.block_size)
            .unwrap_or(DEFAULT_BLOCK_SIZE)
    }

    pub(super) fn mask_token_id(&self) -> Result<u32> {
        self.dflash_config
            .mask_token_id
            .or(self.mask_token_id)
            .ok_or_else(|| candle_core::Error::msg("DFlash config has no mask_token_id"))
    }

    pub fn target_layer_ids(&self) -> Result<Vec<usize>> {
        if let Some(ids) = &self.dflash_config.target_layer_ids {
            return Ok(ids.clone());
        }
        let Some(num_target_layers) = self.num_target_layers else {
            candle_core::bail!("DFlash config has neither target_layer_ids nor num_target_layers");
        };
        let n = self.num_hidden_layers;
        if n == 1 {
            return Ok(vec![num_target_layers / 2]);
        }
        // Mirrors the reference build_target_layer_ids: evenly spaced over [1, num_target_layers - 3]
        let (start, end) = (1, num_target_layers - 3);
        Ok((0..n)
            .map(|i| start + (i * (end - start) + (n - 1) / 2) / (n - 1))
            .collect())
    }

    /// Per-layer (is_causal, sliding_window), following the reference: a layer is causal exactly
    /// when its type is `sliding_attention`, unless the config pins `is_causal` explicitly.
    pub(super) fn layer_attention(&self, layer_idx: usize) -> (bool, Option<usize>) {
        let layer_type = self
            .layer_types
            .as_ref()
            .and_then(|t| t.get(layer_idx))
            .map(String::as_str)
            .unwrap_or("full_attention");
        let sliding = layer_type == "sliding_attention";
        let is_causal = self.is_causal.unwrap_or(sliding);
        (is_causal, sliding.then_some(self.sliding_window).flatten())
    }
}
