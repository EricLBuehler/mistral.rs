use std::collections::HashMap;

use candle_core::DType;
use mistralrs_quant::{Comm, Id, IsqCaptureMode, IsqType, ShardedSafeTensors};
use regex::Regex;

use super::*;

const EXPERTS: usize = 2;
const FEATURES: usize = 128;

struct ClearIsq;

impl Drop for ClearIsq {
    fn drop(&mut self) {
        mistralrs_quant::clear_immediate_isq();
    }
}

#[test]
fn hqq_and_fp8_isq_experts_load_and_forward() -> Result<()> {
    isq_experts_on_device(Device::Cpu)?;
    #[cfg(feature = "cuda")]
    isq_experts_on_device(Device::new_cuda_with_stream(0)?)?;
    Ok(())
}

fn isq_experts_on_device(device: Device) -> Result<()> {
    let comm = Arc::new(Comm::from_device(Id::new(), &device, 0, 1)?);
    let cfg = MoEExpertsConfig {
        num_experts: EXPERTS,
        num_experts_per_tok: EXPERTS,
        hidden_size: FEATURES,
        moe_intermediate_size: FEATURES,
        expert_proj_names: ExpertProjNames::DEFAULT,
    };
    let mut tensors = HashMap::new();
    for expert in 0..EXPERTS {
        let mut values = vec![0f32; FEATURES * FEATURES];
        let scale = f32::from(u16::try_from(expert + 1).unwrap()) / 8.0;
        for diagonal in 0..FEATURES {
            values[diagonal * FEATURES + diagonal] = scale;
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            tensors.insert(
                format!("experts.{expert}.{projection}.weight"),
                Tensor::from_vec(values.clone(), (FEATURES, FEATURES), &Device::Cpu)?,
            );
        }
    }
    let input = Tensor::ones((1, 2, FEATURES), DType::F32, &device)?;
    let ids = Tensor::new(&[[1u32, 0], [0, 1]], &device)?;
    let weights = Tensor::new(&[[0.75f32, 0.25], [0.25, 0.75]], &device)?;
    let expert_weights =
        Regex::new(r"^experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$").unwrap();
    for ty in [IsqType::HQQ4, IsqType::HQQ8, IsqType::F8E4M3, IsqType::F8Q8] {
        mistralrs_quant::set_immediate_isq(
            Some(ty),
            vec![expert_weights.clone()],
            IsqCaptureMode::Immediate,
        );
        let _clear = ClearIsq;
        let vb = ShardedSafeTensors::wrap(tensors.clone(), DType::F32, device.clone());
        let loaded = MoEExperts::new(
            &cfg,
            vb,
            device.clone(),
            &comm,
            true,
            &None,
            Activation::Relu,
        );
        if ty == IsqType::F8Q8 {
            let error = loaded.err().expect("unsupported gather format must fail");
            assert!(error.to_string().contains("not supported for MoE experts"));
            continue;
        }
        let loaded = loaded?;
        let MoEExpertsBackendImpl::Fast(ref fast) = loaded.backend else {
            panic!("ISQ experts must use quantized projections");
        };
        let projections = [
            &fast.fused_gate_proj,
            &fast.fused_up_proj,
            &fast.fused_down_proj,
        ];
        let mut dequantized = Vec::new();
        for projection in projections {
            assert_ne!(projection.name(), "unquant-linear");
            dequantized.push(
                projection
                    .dequantize_w()?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .to_vec3::<f32>()?,
            );
        }
        let expert_outputs = (0..EXPERTS)
            .map(|expert| {
                let intermediate = (0..FEATURES)
                    .map(|row| {
                        dequantized[0][expert][row].iter().sum::<f32>().max(0.0)
                            * dequantized[1][expert][row].iter().sum::<f32>()
                    })
                    .collect::<Vec<_>>();
                dequantized[2][expert]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .zip(&intermediate)
                            .map(|(w, x)| w * x)
                            .sum::<f32>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let route_ids = ids.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let route_weights = weights.to_device(&Device::Cpu)?.to_vec2::<f32>()?;
        let actual = loaded.forward(&input, weights.clone(), &ids)?;
        for (token, values) in actual
            .reshape((2, FEATURES))?
            .to_vec2::<f32>()?
            .iter()
            .enumerate()
        {
            for (row, value) in values.iter().enumerate() {
                let expected = route_ids[token]
                    .iter()
                    .zip(&route_weights[token])
                    .map(|(&expert, weight)| expert_outputs[expert as usize][row] * weight)
                    .sum::<f32>();
                assert!(
                    (value - expected).abs() < 0.0002,
                    "{ty}: {value} != {expected}"
                );
            }
        }
    }
    Ok(())
}
