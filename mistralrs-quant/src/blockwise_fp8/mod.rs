use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, DType, Device, IndexOp, Result, Tensor};
use candle_nn::Linear;

mod ops;
pub use ops::fp8_blockwise_dequantize;

#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub use ops::{blockwise_fp8_gemm, blockwise_fp8_moe_gemm, fp8_blockwise_quantize};

#[cfg(not(feature = "cuda"))]
pub use ops::fp8_blockwise_quantize;

#[cfg(feature = "cuda")]
mod ffi;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    AfqBits, AfqGroupSize, AfqLayer, DummyLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits,
    HqqConfig, HqqLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
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
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
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
        let weight_rank = self.weight.dims().len();
        if weight_rank == 2 {
            // Standard 2D weight: [N, K]
            ops::fp8_blockwise_dequantize(
                &self.weight,
                &self.weight_scale_inv,
                self.weight_block_size.to_vec(),
                self.dequant_dtype,
            )
        } else if weight_rank == 3 {
            // Stacked MoE weight: [E, N, K] - dequantize each expert
            let num_experts = self.weight.dim(0)?;
            let mut dequantized = Vec::with_capacity(num_experts);
            for i in 0..num_experts {
                let w = self.weight.i(i)?;
                let s = self.weight_scale_inv.i(i)?;
                let dq = ops::fp8_blockwise_dequantize(
                    &w,
                    &s,
                    self.weight_block_size.clone(),
                    self.dequant_dtype,
                )?;
                dequantized.push(dq);
            }
            Tensor::stack(&dequantized, 0)
        } else {
            candle_core::bail!(
                "BlockwiseFP8Linear: expected weight rank 2 or 3, got {}",
                weight_rank
            )
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda() {
            // Use the custom blockwise FP8 GEMM kernel
            let x_rank = x.dims().len();
            let (batch_shape, x_2d) = if x_rank > 2 {
                let batch_dims = &x.dims()[..x_rank - 1];
                let k = x.dim(x_rank - 1)?;
                let batch_size: usize = batch_dims.iter().product();
                (Some(batch_dims.to_vec()), x.reshape((batch_size, k))?)
            } else {
                (None, x.clone())
            };

            let mut output = ops::blockwise_fp8_gemm(
                &x_2d,
                &self.weight,
                &self.weight_scale_inv,
                &self.weight_block_size,
            )?;

            if let Some(bias) = &self.bias {
                output = output.broadcast_add(bias)?;
            }

            if let Some(batch_dims) = batch_shape {
                let n = output.dim(1)?;
                let mut out_shape = batch_dims;
                out_shape.push(n);
                output = output.reshape(out_shape)?;
            }

            return Ok(output);
        }

        // Fallback: Dequantize matmul
        let weight = self.dequantize_w()?;
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight,
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // For MoE gather_forward, we need stacked weights [E, N, K]
        // This requires loading with stacked format - see BlockwiseFP8Stacked
        // For single-expert BlockwiseFP8Linear, fall back to dequantize path
        let weight = self.dequantize_w()?;
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight,
            self.bias.clone(),
        )))?;
        unquant.gather_forward(a, indices)
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
        guard: QuantizeOntoGuard,
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
                let _acquired_quantize_guard = guard.acquire(&device);
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
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
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
                let _acquired_quantize_guard = guard.acquire(&device);
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
    let QuantizedConfig::Fp8 { weight_block_size } = config else {
        candle_core::bail!("Unexpected quantization config.")
    };

    // Handle the case where we actually have an unqiantzed
    if vb.contains_tensor("weight") && !vb.contains_tensor("weight_scale_inv") {
        return crate::linear_b(in_dim, out_dim, bias, &None, vb);
    }

    // Handle the case where the layer is dummy (no tensors)
    if !(vb.contains_tensor("weight") && vb.contains_tensor("weight_scale_inv")) {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

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

impl BlockwiseFP8Linear {
    /// Create a BlockwiseFP8Linear from pre-loaded tensors.
    /// Supports both 2D weights [N, K] and stacked 3D weights [E, N, K] for MoE.
    pub fn from_tensors(
        weight: Tensor,
        weight_scale_inv: Tensor,
        weight_block_size: Vec<usize>,
        dequant_dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            weight,
            weight_scale_inv,
            bias: None,
            dequant_dtype,
            weight_block_size,
        })
    }

    /// Create a stacked BlockwiseFP8Linear by stacking per-expert FP8 weights.
    /// Takes vectors of weight and scale tensors (one per expert) and stacks them.
    pub fn from_per_expert(
        weights: Vec<Tensor>,
        scales: Vec<Tensor>,
        weight_block_size: Vec<usize>,
        dequant_dtype: DType,
    ) -> Result<Self> {
        let num_experts = weights.len();
        if num_experts == 0 {
            candle_core::bail!("Cannot create BlockwiseFP8Linear with 0 experts");
        }
        if weights.len() != scales.len() {
            candle_core::bail!(
                "Mismatch between weights ({}) and scales ({}) count",
                weights.len(),
                scales.len()
            );
        }

        let weight = Tensor::stack(&weights, 0)?;
        let weight_scale_inv = Tensor::stack(&scales, 0)?;

        Ok(Self {
            weight,
            weight_scale_inv,
            bias: None,
            dequant_dtype,
            weight_block_size,
        })
    }
}

/// Create a stacked blockwise FP8 linear layer for MoE experts.
/// Loads weights of shape [num_experts, N, K] and scales [num_experts, N/block_y, K/block_x].
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub fn blockwise_fp8_stacked_linear(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    weight_block_size: &[usize],
    vb: ShardedVarBuilder,
    weight_name: &str,
    scale_name: &str,
    transposed: bool,
) -> Result<Arc<dyn QuantMethod>> {
    if weight_block_size.len() != 2 {
        candle_core::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }

    // Determine weight shape based on transposed flag
    let weight_shape = if transposed {
        (num_experts, in_dim, out_dim) // [E, K, N]
    } else {
        (num_experts, out_dim, in_dim) // [E, N, K]
    };

    // Determine scale shape based on transposed flag
    let scale_shape = if transposed {
        (
            num_experts,
            in_dim.div_ceil(weight_block_size[0]),
            out_dim.div_ceil(weight_block_size[1]),
        )
    } else {
        (
            num_experts,
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        )
    };

    let weight =
        vb.get_with_hints_dtype(weight_shape, weight_name, Default::default(), DType::F8E4M3)?;
    let weight_scale_inv =
        vb.get_with_hints_dtype(scale_shape, scale_name, Default::default(), DType::F32)?;

    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype: vb.dtype(),
        weight_block_size: weight_block_size.to_vec(),
    }))
}
