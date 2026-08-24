use candle_core::DType;
use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

pub const GDN_V_HEAD_LAYOUT_CONFIG_KEY: &str = "_mistralrs_gdn_v_head_layout";

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum GdnStateDType {
    #[serde(rename = "float16", alias = "f16", alias = "half")]
    F16,
    #[serde(rename = "bfloat16", alias = "bf16")]
    BF16,
    #[default]
    #[serde(rename = "float32", alias = "f32", alias = "float")]
    F32,
}

impl GdnStateDType {
    pub fn dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F32 => DType::F32,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GdnVHeadLayout {
    #[default]
    Grouped,
    Tiled,
}

impl GdnVHeadLayout {
    pub fn k_head_for_v_head(self, v_head: usize, num_k_heads: usize, v_per_group: usize) -> usize {
        match self {
            Self::Grouped => v_head / v_per_group,
            Self::Tiled => v_head % num_k_heads,
        }
    }
}

#[allow(dead_code)]
pub trait GdnConfig {
    fn hidden_size(&self) -> usize;
    fn rms_norm_eps(&self) -> f64;
    fn linear_conv_kernel_dim(&self) -> usize;
    fn linear_key_head_dim(&self) -> usize;
    fn linear_value_head_dim(&self) -> usize;
    fn linear_num_key_heads(&self) -> usize;
    fn linear_num_value_heads(&self) -> usize;
    fn quantization_config(&self) -> &Option<QuantizedConfig>;
    fn v_head_layout(&self) -> GdnVHeadLayout {
        GdnVHeadLayout::Grouped
    }

    fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads() * self.linear_key_head_dim()
    }

    fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads() * self.linear_value_head_dim()
    }

    fn linear_conv_dim(&self) -> usize {
        self.linear_key_dim() * 2 + self.linear_value_dim()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GdnDims {
    pub hidden_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_dim: usize,
    pub v_per_group: usize,
    pub v_head_layout: GdnVHeadLayout,
}

impl GdnDims {
    pub fn new(cfg: &dyn GdnConfig) -> Self {
        let hidden_size = cfg.hidden_size();
        let num_k_heads = cfg.linear_num_key_heads();
        let num_v_heads = cfg.linear_num_value_heads();
        let head_k_dim = cfg.linear_key_head_dim();
        let head_v_dim = cfg.linear_value_head_dim();
        let conv_kernel_size = cfg.linear_conv_kernel_dim();
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = key_dim * 2 + value_dim;
        let v_per_group = num_v_heads / num_k_heads;

        Self {
            hidden_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel_size,
            key_dim,
            value_dim,
            conv_dim,
            v_per_group,
            v_head_layout: cfg.v_head_layout(),
        }
    }

    pub fn qkvz_out_dim(&self) -> usize {
        self.key_dim * 2 + self.value_dim * 2
    }

    pub fn ba_out_dim(&self) -> usize {
        self.num_v_heads * 2
    }

    pub fn k_head_for_v_head(&self, v_head: usize) -> usize {
        self.v_head_layout
            .k_head_for_v_head(v_head, self.num_k_heads, self.v_per_group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct StateDTypeConfig {
        #[serde(default)]
        mamba_ssm_dtype: GdnStateDType,
    }

    #[test]
    fn recurrent_state_dtype_honors_checkpoint_metadata() {
        let config: StateDTypeConfig =
            serde_json::from_str(r#"{"mamba_ssm_dtype":"float32"}"#).unwrap();
        assert_eq!(config.mamba_ssm_dtype.dtype(), DType::F32);
        let config: StateDTypeConfig =
            serde_json::from_str(r#"{"mamba_ssm_dtype":"bfloat16"}"#).unwrap();
        assert_eq!(config.mamba_ssm_dtype.dtype(), DType::BF16);
        let config: StateDTypeConfig =
            serde_json::from_str(r#"{"mamba_ssm_dtype":"float16"}"#).unwrap();
        assert_eq!(config.mamba_ssm_dtype.dtype(), DType::F16);
        let config: StateDTypeConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config.mamba_ssm_dtype.dtype(), DType::F32);
    }

    #[test]
    fn qwen35_tiled_head_mapping_matches_converter_order() {
        let num_k_heads = 3;
        let v_per_group = 4;
        let num_v_heads = num_k_heads * v_per_group;
        let grouped = (0..num_v_heads)
            .map(|head| head / v_per_group)
            .collect::<Vec<_>>();
        let converter_order = (0..v_per_group)
            .flat_map(|within_group| {
                (0..num_k_heads).map(move |k_head| k_head * v_per_group + within_group)
            })
            .collect::<Vec<_>>();
        let expected = converter_order
            .iter()
            .map(|grouped_head| grouped[*grouped_head])
            .collect::<Vec<_>>();
        let actual = (0..num_v_heads)
            .map(|v_head| GdnVHeadLayout::Tiled.k_head_for_v_head(v_head, num_k_heads, v_per_group))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
