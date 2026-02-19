use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

mod ops;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    utils::{serialize_tensor, UQFF_VERSION},
    AfqBits, AfqGroupSize, AfqLayer, DummyLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits,
    HqqConfig, HqqLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, QuantizedSerdeType, Shard, ShardedVarBuilder, UnquantLinear,
};

/// Per-tensor FP8 Linear layer with static activation scaling.
///
/// This is used for models that have per-tensor FP8 quantization (weight_block_size = null)
/// with static activation scales. Each linear layer has:
/// - `<layer>.weight` (FP8 E4M3)
/// - `<layer>.weight_scale_inv` (F32 scalar) - dequantization scale for weights
/// - `<layer>.activation_scale` (F32 scalar) - quantization scale for activations
#[derive(Debug)]
pub struct PerTensorFP8Linear {
    weight: Tensor,
    #[allow(dead_code)]
    weight_scale_inv: Tensor,
    #[allow(dead_code)]
    activation_scale: Option<Tensor>,
    bias: Option<Tensor>,
    #[allow(dead_code)]
    dequant_dtype: DType,
}

impl QuantMethod for PerTensorFP8Linear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::PerTensorFP8 {
                weight,
                weight_scale_inv,
                activation_scale,
                bias,
                dequant_dtype,
            } => {
                // Dequantize immediately since Candle FP8 is storage-only (no ops)
                let dequant_weight =
                    ops::fp8_pertensor_dequantize(&weight, &weight_scale_inv, dequant_dtype)?;
                Ok(Self {
                    weight: dequant_weight,
                    weight_scale_inv,
                    activation_scale,
                    bias,
                    dequant_dtype,
                })
            }
            _ => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        // Weight is already dequantized on load
        Ok(self.weight.clone())
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Weight is already dequantized, use standard matmul
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.weight.clone(),
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("PerTensorFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F8E4M3, self.weight.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.dequantize_w()?;
        match dtype {
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    _ => unreachable!(),
                };
                let cfg = HqqConfig {
                    bits,
                    group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                    axis: HqqAxis::Zero,
                    optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                    round_zeros: false,
                    channel_wise: true,
                };
                let res = HqqLayer::quantize(&weight.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.bias {
                    let bias = bias
                        .to_device(&device)?
                        .to_dtype(res.dtype_and_device().0)?;
                    Ok(Arc::new(res.with_bias(bias)))
                } else {
                    Ok(Arc::new(res))
                }
            }
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("AFQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::AFQ8 => AfqBits::Eight,
                    IsqType::AFQ6 => AfqBits::Six,
                    IsqType::AFQ4 => AfqBits::Four,
                    IsqType::AFQ3 => AfqBits::Three,
                    IsqType::AFQ2 => AfqBits::Two,
                    _ => unreachable!(),
                };

                Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: weight.to_device(&device)?,
                    bias: self.bias.as_ref().map(|b| b.to_device(&device).unwrap()),
                    bits,
                    group_size: AfqGroupSize::default(),
                })?))
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(weight, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .bias
                        .as_ref()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8E4M3 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                    lin: Linear::new(w, b),
                    dtype: DType::F8E4M3,
                })?))
            }
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8Q8 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }
}

// Serialization structure (same as UnquantLinear):
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (1 for unquantized), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for PerTensorFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "pertensor-fp8-linear"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        // Serialize as unquantized since weights are already dequantized
        let mut buffer = Vec::new();

        // Version is always first!
        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for unquant is 1 (same as UnquantLinear)
        buffer.push(QuantizedSerdeType::Unquant as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight (already dequantized)
        serialize_tensor(&mut buffer, &self.weight)?;

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }
}

/// Load a per-tensor FP8 linear layer from the VarBuilder.
///
/// This handles models with per-tensor FP8 quantization where:
/// - `weight_block_size` is null (per-tensor, not blockwise)
/// - Each layer has: weight (FP8), weight_scale_inv (F32), activation_scale (F32)
pub fn pertensor_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    _config: &QuantizedConfig,
    bias: bool,
    _hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    // Handle the case where we actually have unquantized weights
    if vb.contains_tensor("weight") && !vb.contains_tensor("weight_scale_inv") {
        return crate::linear_b(in_dim, out_dim, bias, &None, vb);
    }

    // Handle the case where the layer is dummy (no tensors)
    if !vb.contains_tensor("weight") {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    // Load FP8 weight tensor
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;

    // Load per-tensor weight scale (scalar)
    let weight_scale_inv =
        vb.get_with_hints_dtype((), "weight_scale_inv", Default::default(), DType::F32)?;

    // Load activation scale if present (optional - some models may not have it)
    let activation_scale = if vb.contains_tensor("activation_scale") {
        Some(vb.get_with_hints_dtype((), "activation_scale", Default::default(), DType::F32)?)
    } else {
        None
    };

    let bias = if bias && vb.contains_tensor("bias") {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    // Determine the output dtype for dequantization.
    // We can't use vb.dtype() as that returns F8E4M3 (the storage type).
    // Use the bias dtype if available, otherwise default to BF16.
    let dequant_dtype = bias.as_ref().map(|b| b.dtype()).unwrap_or(DType::BF16);

    // Use new() which handles dequantization (Candle FP8 is storage-only)
    Ok(Arc::new(PerTensorFP8Linear::new(
        QuantMethodConfig::PerTensorFP8 {
            weight,
            weight_scale_inv,
            activation_scale,
            bias,
            dequant_dtype,
        },
    )?))
}
