use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

mod ops;

#[cfg(feature = "cuda")]
mod ffi;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    DummyLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits, HqqConfig, HqqLayer, IsqType, QuantMethod,
    QuantMethodConfig, QuantizedConfig, QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
};

#[derive(Debug)]
pub struct BlockwiseFP8Linear {
    weight: Tensor,
    weight_scale_inv: Tensor,
    bias: Option<Tensor>,
    dequant_dtype: DType,
    weight_block_size: Vec<usize>,
}

impl QuantMethod for BlockwiseFP8Linear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::FP8 { .. } => unreachable!(),
            QuantMethodConfig::BlockwiseFP8 {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
            } => Ok(Self {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
            }),
        }
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Dequantize matmul always.
        // TODO: add a specific kernel?
        let weight = self.dequantize_w()?;
        let lin = Linear::new(weight, self.bias.clone());
        lin.forward(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("BlockwiseFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::F8E4M3, self.weight.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )?;
        match dtype {
            /*Some(IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    candle_core::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    // IsqType::HQQ3 => HqqBits::Three,
                    // IsqType::HQQ2 => HqqBits::Two,
                    // IsqType::HQQ1 => HqqBits::One,
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
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype, n_quantized)
                } else {
                    generate_isq!(weight, device, dtype, n_quantized)
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
                if imatrix_weight.is_some() {
                    // TODO just warn?
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
            None => {
                // Ignore imatrix altogether

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

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        match dtype {
            IsqType::F8E4M3 => None,
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
            | IsqType::Q8_1
            | IsqType::HQQ4
            | IsqType::HQQ8 => None,
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (3 for fp8), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// Dequant W scalar, f32, little endian
// -----------------------
// Dequant X scalar, f32, little endian
// -----------------------
// Quant scalar, f32, little endian
// -----------------------
// Quantization type, u32, little endian
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for BlockwiseFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "blockwise-fp8-linear"
    }
}

pub fn blockwise_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    bias: bool,
    hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    // Handle the case where the layer is dummy (no tensors)
    if !(vb.contains_tensor("weight") && vb.contains_tensor("weight_scale_inv")) {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    let weight_block_size = config
        .weight_block_size
        .as_ref()
        .expect("Blockwise FP8 requires weight_block_size in config");
    if weight_block_size.len() != 2 {
        candle_core::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        hints,
        DType::F32,
    )?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_block_size: weight_block_size.clone(),
        weight_scale_inv,
        bias,
        dequant_dtype: vb.dtype(),
    }))
}
