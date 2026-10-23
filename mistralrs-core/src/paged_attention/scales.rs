use std::sync::Once;

use candle_core::{DType, Result};
use mistralrs_quant::ShardedVarBuilder;

const Q_SCALE_NAME: &str = "q_scale";
const K_SCALE_NAME: &str = "k_scale";
const V_SCALE_NAME: &str = "v_scale";
const LEGACY_KV_SCALE_NAME: &str = "kv_scale";

static LEGACY_KV_SCALE_WARNING: Once = Once::new();
static MISSING_Q_SCALE_WARNING: Once = Once::new();

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fp8AttentionScales {
    pub q: f32,
    pub k: f32,
    pub v: f32,
}

impl Fp8AttentionScales {
    pub const UNIT: Self = Self {
        q: 1.0,
        k: 1.0,
        v: 1.0,
    };

    pub fn validate(self) -> Result<Self> {
        for (name, value) in [
            (Q_SCALE_NAME, self.q),
            (K_SCALE_NAME, self.k),
            (V_SCALE_NAME, self.v),
        ] {
            if !value.is_finite() || value <= 0.0 {
                candle_core::bail!("FP8 attention {name} must be finite and positive, got {value}");
            }
        }
        Ok(self)
    }
}

impl Default for Fp8AttentionScales {
    fn default() -> Self {
        Self::UNIT
    }
}

fn load_scalar(vb: &ShardedVarBuilder, name: &str) -> Result<f32> {
    let tensor = vb.get_unchecked(name)?;
    if tensor.elem_count() != 1 {
        candle_core::bail!(
            "FP8 attention scale `{}.{name}` must be scalar, got shape {:?}",
            vb.prefix(),
            tensor.shape()
        );
    }
    tensor.to_dtype(DType::F32)?.reshape(())?.to_scalar::<f32>()
}

pub fn load_fp8_attention_scales(
    attention_vb: &ShardedVarBuilder,
) -> Result<Option<Fp8AttentionScales>> {
    let has_q = attention_vb.contains_tensor(Q_SCALE_NAME);
    let has_k = attention_vb.contains_tensor(K_SCALE_NAME);
    let has_v = attention_vb.contains_tensor(V_SCALE_NAME);
    let has_legacy = attention_vb.contains_tensor(LEGACY_KV_SCALE_NAME);

    match (has_q, has_k, has_v, has_legacy) {
        (false, false, false, false) => Ok(None),
        (true, true, true, false) => Ok(Some(
            Fp8AttentionScales {
                q: load_scalar(attention_vb, Q_SCALE_NAME)?,
                k: load_scalar(attention_vb, K_SCALE_NAME)?,
                v: load_scalar(attention_vb, V_SCALE_NAME)?,
            }
            .validate()?,
        )),
        (false, true, true, false) => {
            MISSING_Q_SCALE_WARNING.call_once(|| {
                tracing::warn!(
                    "FP8 attention q_scale is missing; using k_scale for query quantization"
                );
            });
            let k = load_scalar(attention_vb, K_SCALE_NAME)?;
            Ok(Some(
                Fp8AttentionScales {
                    q: k,
                    k,
                    v: load_scalar(attention_vb, V_SCALE_NAME)?,
                }
                .validate()?,
            ))
        }
        (false, false, false, true) => {
            LEGACY_KV_SCALE_WARNING.call_once(|| {
                tracing::warn!(
                    "loading deprecated FP8 attention `kv_scale`; use q_scale, k_scale, and v_scale"
                );
            });
            let scale = load_scalar(attention_vb, LEGACY_KV_SCALE_NAME)?;
            Ok(Some(
                Fp8AttentionScales {
                    q: scale,
                    k: scale,
                    v: scale,
                }
                .validate()?,
            ))
        }
        _ => candle_core::bail!(
            "FP8 attention scales under `{}` must define k_scale and v_scale together, with optional q_scale, or only deprecated kv_scale",
            attention_vb.prefix()
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device, Tensor};
    use mistralrs_quant::ShardedSafeTensors;

    use super::{load_fp8_attention_scales, Fp8AttentionScales};

    fn attention_vb(values: &[(&str, Tensor)]) -> mistralrs_quant::ShardedVarBuilder {
        let tensors = values
            .iter()
            .map(|(name, tensor)| (format!("model.layers.0.self_attn.{name}"), tensor.clone()))
            .collect::<HashMap<_, _>>();
        ShardedSafeTensors::wrap(tensors, DType::F32, Device::Cpu)
            .pp("model")
            .pp("layers")
            .pp(0)
            .pp("self_attn")
    }

    #[test]
    fn missing_scales_use_uncalibrated_fallback() {
        let vb = attention_vb(&[]);
        assert_eq!(load_fp8_attention_scales(&vb).unwrap(), None);
    }

    #[test]
    fn loads_scalar_and_single_element_scales() {
        let vb = attention_vb(&[
            ("q_scale", Tensor::new(0.25f32, &Device::Cpu).unwrap()),
            ("k_scale", Tensor::new(&[0.5f32], &Device::Cpu).unwrap()),
            ("v_scale", Tensor::new(0.75f32, &Device::Cpu).unwrap()),
        ]);
        assert_eq!(
            load_fp8_attention_scales(&vb).unwrap(),
            Some(Fp8AttentionScales {
                q: 0.25,
                k: 0.5,
                v: 0.75,
            })
        );
    }

    #[test]
    fn duplicates_legacy_kv_scale() {
        let vb = attention_vb(&[("kv_scale", Tensor::new(0.125f32, &Device::Cpu).unwrap())]);
        assert_eq!(
            load_fp8_attention_scales(&vb).unwrap(),
            Some(Fp8AttentionScales {
                q: 0.125,
                k: 0.125,
                v: 0.125,
            })
        );
    }

    #[test]
    fn falls_back_from_missing_q_scale_to_k_scale() {
        let vb = attention_vb(&[
            ("k_scale", Tensor::new(0.5f32, &Device::Cpu).unwrap()),
            ("v_scale", Tensor::new(0.75f32, &Device::Cpu).unwrap()),
        ]);
        assert_eq!(
            load_fp8_attention_scales(&vb).unwrap(),
            Some(Fp8AttentionScales {
                q: 0.5,
                k: 0.5,
                v: 0.75,
            })
        );
    }

    #[test]
    fn rejects_partial_or_mixed_scale_sets() {
        let q = Tensor::new(0.25f32, &Device::Cpu).unwrap();
        let k = Tensor::new(0.5f32, &Device::Cpu).unwrap();
        assert!(load_fp8_attention_scales(&attention_vb(&[("q_scale", q.clone())])).is_err());
        assert!(load_fp8_attention_scales(&attention_vb(&[("k_scale", k.clone())])).is_err());
        assert!(load_fp8_attention_scales(&attention_vb(&[("v_scale", k.clone())])).is_err());
        assert!(load_fp8_attention_scales(&attention_vb(&[
            ("q_scale", q.clone()),
            ("k_scale", k.clone()),
        ]))
        .is_err());
        assert!(load_fp8_attention_scales(&attention_vb(&[
            ("q_scale", q.clone()),
            ("k_scale", k.clone()),
            ("v_scale", q.clone()),
            ("kv_scale", q),
        ]))
        .is_err());
    }

    #[test]
    fn rejects_invalid_values_and_shapes() {
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let scalar = Tensor::new(value, &Device::Cpu).unwrap();
            let vb = attention_vb(&[
                ("q_scale", scalar.clone()),
                ("k_scale", Tensor::new(0.5f32, &Device::Cpu).unwrap()),
                ("v_scale", Tensor::new(0.75f32, &Device::Cpu).unwrap()),
            ]);
            assert!(load_fp8_attention_scales(&vb).is_err());
        }

        let vector = Tensor::new(&[0.25f32, 0.5], &Device::Cpu).unwrap();
        let vb = attention_vb(&[
            ("q_scale", vector),
            ("k_scale", Tensor::new(0.5f32, &Device::Cpu).unwrap()),
            ("v_scale", Tensor::new(0.75f32, &Device::Cpu).unwrap()),
        ]);
        assert!(load_fp8_attention_scales(&vb).is_err());
    }

    #[test]
    fn validates_programmatic_scales() {
        assert_eq!(
            Fp8AttentionScales {
                q: 0.25,
                k: 0.5,
                v: 0.75,
            }
            .validate()
            .unwrap(),
            Fp8AttentionScales {
                q: 0.25,
                k: 0.5,
                v: 0.75,
            }
        );
        assert!(Fp8AttentionScales {
            q: 1.0,
            k: 0.0,
            v: 1.0,
        }
        .validate()
        .is_err());
    }
}
