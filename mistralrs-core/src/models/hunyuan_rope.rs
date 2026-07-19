use candle_core::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub beta_fast: Option<f64>,
    #[serde(default)]
    pub beta_slow: Option<f64>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(rename = "type", alias = "rope_type")]
    pub rope_type: String,
}

impl RopeScalingConfig {
    pub fn effective_theta(&self, base_theta: f64, head_dim: usize) -> Result<f64> {
        match self.rope_type.as_str() {
            "default" => Ok(base_theta),
            "dynamic" => {
                let Some(alpha) = self.alpha.filter(|alpha| alpha.is_finite() && *alpha > 0.0)
                else {
                    candle_core::bail!(
                        "HunYuan dynamic RoPE scaling without a positive alpha is not implemented"
                    )
                };
                if head_dim <= 2 {
                    candle_core::bail!(
                        "HunYuan dynamic-alpha RoPE requires head_dim greater than 2"
                    )
                }
                let head_dim = f64::from(u32::try_from(head_dim)?);
                let exponent = head_dim / (head_dim - 2.0);
                Ok(base_theta * alpha.powf(exponent))
            }
            "linear" => candle_core::bail!("HunYuan linear RoPE scaling is not implemented"),
            rope_type => candle_core::bail!("Unsupported HunYuan RoPE scaling type {rope_type}"),
        }
    }
}

pub fn effective_rope_theta(
    base_theta: f64,
    head_dim: usize,
    scaling: Option<&RopeScalingConfig>,
) -> Result<f64> {
    match scaling {
        Some(scaling) => scaling.effective_theta(base_theta, head_dim),
        None => Ok(base_theta),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaling(rope_type: &str, alpha: Option<f64>, factor: Option<f64>) -> RopeScalingConfig {
        RopeScalingConfig {
            alpha,
            original_max_position_embeddings: None,
            beta_fast: None,
            beta_slow: None,
            factor,
            mscale: None,
            mscale_all_dim: None,
            rope_type: rope_type.to_string(),
        }
    }

    #[test]
    fn dynamic_alpha_matches_hunyuan_theta() -> Result<()> {
        let theta = scaling("dynamic", Some(1_000.0), Some(1.0)).effective_theta(10_000.0, 128)?;
        let expected = 10_000.0 * 1_000.0f64.powf(128.0 / 126.0);

        assert!((theta - expected).abs() <= expected * f64::EPSILON);
        Ok(())
    }

    #[test]
    fn linear_scaling_is_rejected() {
        let err = scaling("linear", Some(1.0), Some(2.0))
            .effective_theta(10_000.0, 128)
            .unwrap_err();

        assert!(err.to_string().contains("linear RoPE scaling"));
    }

    #[test]
    fn dynamic_factor_scaling_is_rejected() {
        let err = scaling("dynamic", None, Some(2.0))
            .effective_theta(10_000.0, 128)
            .unwrap_err();

        assert!(err.to_string().contains("without a positive alpha"));
    }
}
