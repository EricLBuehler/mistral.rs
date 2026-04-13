use std::{
    borrow::Cow,
    io::Cursor,
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{DType, Device, Result, Tensor};

use crate::{
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, ShardedVarBuilder,
};

#[cfg(feature = "cuda")]
pub(crate) mod ffi;
#[cfg(feature = "metal")]
pub(crate) mod metal_ops;
#[cfg(feature = "cuda")]
pub(crate) mod ops;

/// MXFP4 block size (32 elements per scale)
pub const MXFP4_BLOCK_SIZE: usize = 32;

pub(crate) const N_BITS: usize = 4;

#[derive(Debug)]
pub struct MXFP4Layer {
    /// Packed FP4 weights: [N, K/2] or [num_experts, N, K/2]
    /// Each byte contains 2 FP4 values (low nibble = k, high nibble = k+1)
    #[allow(dead_code)]
    blocks: Tensor,
    /// E8M0 scales: [N, K/32] or [num_experts, N, K/32]
    /// Each byte is an 8-bit exponent with bias 127
    scales: Tensor,
    /// Optional bias: [N] or [num_experts, N]
    #[allow(dead_code)]
    bias: Option<Tensor>,
}

impl QuantMethod for MXFP4Layer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Afq { .. } => unreachable!(),
            QuantMethodConfig::MXFP4 {
                blocks,
                scales,
                bias,
            } => Ok(Self {
                blocks,
                scales,
                bias,
            }),
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        #[cfg(feature = "metal")]
        if self.blocks.device().is_metal() {
            use crate::afq::ops;
            use crate::{AfqBits, AfqGroupSize};
            return ops::afq_dequantize_op(
                &self.blocks,
                &self.scales,
                &self.scales.clone(),
                AfqGroupSize::Low,
                AfqBits::Mxfp4,
            );
        }
        // CPU fallback
        self.dequantize_weights()
    }

    #[allow(unused_variables)]
    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            let orig_dims = x.dims().to_vec();
            let x_2d = if orig_dims.len() > 2 {
                let features = orig_dims[orig_dims.len() - 1];
                let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                x.reshape((batch_size, features))?
            } else {
                x.clone()
            };

            let result = ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

            if orig_dims.len() > 2 {
                let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                new_dims.push(result.dim(1)?);
                return result.reshape(new_dims);
            }
            return Ok(result);
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                let orig_dims = x.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    x.reshape((batch_size, features))?
                } else {
                    x.clone()
                };

                let result =
                    metal_ops::mxfp4_matmul(&x_2d, &self.blocks, &self.scales, self.bias.as_ref())?;

                if orig_dims.len() > 2 {
                    let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                    new_dims.push(result.dim(1)?);
                    return result.reshape(new_dims);
                }
                return Ok(result);
            }
        }

        self.forward_dequantize(x)
    }

    #[allow(unused_variables)]
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if matches!(x.device(), Device::Cuda(_)) && ffi::HAVE_MXFP4_GEMM_KERNELS {
            return ops::mxfp4_indexed_moe_gemm(
                x,
                &self.blocks,
                &self.scales,
                self.bias.as_ref(),
                indices,
            );
        }

        #[cfg(feature = "metal")]
        {
            if x.device().is_metal() {
                return metal_ops::mxfp4_indexed_moe_gemm(
                    x,
                    &self.blocks,
                    &self.scales,
                    self.bias.as_ref(),
                    indices,
                );
            }
        }

        self.gather_forward_dequantize(x, indices)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("MXFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::BF16, self.scales.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("MXFP4Layer does not support ISQ")
    }
}

impl MXFP4Layer {
    /// Check if the device supports MXFP4 operations
    fn device_supported(_device: &Device) -> bool {
        #[cfg(feature = "cuda")]
        if matches!(_device, Device::Cuda(_)) {
            return ffi::HAVE_MXFP4_GEMM_KERNELS;
        }
        #[cfg(feature = "metal")]
        if _device.is_metal() {
            return true;
        }
        false
    }

    /// Quantize an unquantized weight tensor to MXFP4 format.
    /// weight shape: [N, K], bias shape: [N] (optional)
    pub fn quantize(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight_f32 = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let dims = weight_f32.dims2()?;
        let (n, k) = (dims.0, dims.1);

        if k % MXFP4_BLOCK_SIZE != 0 {
            candle_core::bail!(
                "MXFP4 quantization requires K ({k}) divisible by block size ({MXFP4_BLOCK_SIZE})"
            );
        }

        let weight_data: Vec<f32> = weight_f32.flatten_all()?.to_vec1()?;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let k_half = k / 2;

        // Parallelize quantization across rows with rayon
        use rayon::prelude::*;
        let row_results: Vec<(Vec<u8>, Vec<u8>)> = (0..n)
            .into_par_iter()
            .map(|row| {
                let row_offset = row * k;
                let mut row_packed = vec![0u8; k_half];
                let mut row_scales = vec![0u8; num_blocks_per_row];

                for (blk, row_scale) in row_scales.iter_mut().enumerate() {
                    let blk_start = row_offset + blk * MXFP4_BLOCK_SIZE;
                    let block = &weight_data[blk_start..blk_start + MXFP4_BLOCK_SIZE];

                    let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));

                    let scale = if max_abs == 0.0 {
                        127u8
                    } else {
                        let raw = (max_abs / 6.0).log2().floor() as i32 + 127;
                        raw.clamp(0, 254) as u8
                    };
                    *row_scale = scale;

                    let scale_factor = 2.0f32.powi(scale as i32 - 127);
                    let inv_scale = if scale_factor == 0.0 {
                        0.0
                    } else {
                        1.0 / scale_factor
                    };

                    for (elem, &val) in block.iter().enumerate() {
                        let nibble = Self::quantize_to_fp4(val * inv_scale);
                        let k_idx = blk * MXFP4_BLOCK_SIZE + elem;
                        let byte_idx = k_idx / 2;
                        if k_idx.is_multiple_of(2) {
                            row_packed[byte_idx] |= nibble;
                        } else {
                            row_packed[byte_idx] |= nibble << 4;
                        }
                    }
                }
                (row_packed, row_scales)
            })
            .collect();

        let mut packed = Vec::with_capacity(n * k_half);
        let mut scales = Vec::with_capacity(n * num_blocks_per_row);
        for (row_packed, row_scales) in row_results {
            packed.extend_from_slice(&row_packed);
            scales.extend_from_slice(&row_scales);
        }

        let blocks = Tensor::from_vec(packed, (n, k / 2), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let scales = Tensor::from_vec(scales, (n, num_blocks_per_row), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(device)?;
        let bias = bias.map(|b| b.to_device(device)).transpose()?;

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Quantize a single scaled value to the nearest FP4 E2M1 nibble (0..15).
    fn quantize_to_fp4(val: f32) -> u8 {
        // FP4 E2M1 positive values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        // Negative values are the same with sign bit set (indices 8..15)
        let sign = val < 0.0;
        let abs_val = val.abs();

        // Decision boundaries (midpoints between consecutive FP4 values)
        let nibble = if abs_val < 0.25 {
            0 // 0.0
        } else if abs_val < 0.75 {
            1 // 0.5
        } else if abs_val < 1.25 {
            2 // 1.0
        } else if abs_val < 1.75 {
            3 // 1.5
        } else if abs_val < 2.5 {
            4 // 2.0
        } else if abs_val < 3.5 {
            5 // 3.0
        } else if abs_val < 5.0 {
            6 // 4.0
        } else {
            7 // 6.0
        };

        if sign {
            nibble | 0x08
        } else {
            nibble
        }
    }

    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    pub fn packed_linear_b(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        config: &QuantizedConfig,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let QuantizedConfig::MXFP4 {} = config else {
            candle_core::bail!("Unexpected quantization config.")
        };

        let blocks = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / 2),
            "blocks",
            Default::default(),
            DType::U8,
        )?;
        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, in_dim / MXFP4_BLOCK_SIZE),
            "scales",
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), "bias")?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Load GPT-OSS style MXFP4 experts (combined gate_up_proj format).
    ///
    /// GPT-OSS stores tensors as:
    /// - `{name}_blocks`: [num_experts, out_dim, num_blocks, 16] where 16 bytes = 32 FP4 values
    /// - `{name}_scales`: [num_experts, out_dim, num_blocks]
    /// - `{name}_bias`: [num_experts, out_dim]
    ///
    /// This function loads and reshapes the 4D blocks tensor to 3D [num_experts, out_dim, in_dim/2].
    pub fn packed_gptoss_linear(
        num_local_experts: usize,
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        name: &str,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        if !Self::device_supported(vb.device()) {
            candle_core::bail!("MXFP4Layer requires CUDA or Metal device.");
        }

        let num_blocks = in_dim / MXFP4_BLOCK_SIZE;

        let blocks_4d = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks, 16),
            &format!("{name}_blocks"),
            Default::default(),
            DType::U8,
        )?;

        let blocks = blocks_4d.reshape((num_local_experts, out_dim, num_blocks * 16))?;

        let scales = vb.get_with_hints_dtype(
            (num_local_experts, out_dim, num_blocks),
            &format!("{name}_scales"),
            Default::default(),
            DType::U8,
        )?;

        let bias = if bias {
            Some(vb.get((num_local_experts, out_dim), &format!("{name}_bias"))?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            blocks,
            scales,
            bias,
        }))
    }

    /// Combined FP4 × E8M0 dequant table: `DEQUANT_LUT[scale][nibble]`.
    /// For each of the 256 possible E8M0 scale values, stores the 16 possible
    /// dequantized values (FP4_LUT[nibble] * 2^(scale - 127)).
    /// This turns dequantization into a single table lookup per element.
    const DEQUANT_LUT: [[f32; 16]; 256] = {
        let mut lut = [[0.0f32; 16]; 256];
        let fp4: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut s = 0u32;
        while s < 256 {
            let scale_factor = f32::from_bits(s << 23);
            let mut n = 0;
            while n < 16 {
                lut[s as usize][n] = fp4[n] * scale_factor;
                n += 1;
            }
            s += 1;
        }
        lut
    };

    /// Dequantize MXFP4 weights to f32
    /// blocks: [num_experts, N, K/2] packed bytes
    /// scales: [num_experts, N, K/32] E8M0 scales
    /// Returns: [num_experts, N, K] f32 weights
    fn dequantize_weights(&self) -> Result<Tensor> {
        let blocks_dims = self.blocks.dims();

        let (num_experts, n, k_half) = if blocks_dims.len() == 3 {
            (blocks_dims[0], blocks_dims[1], blocks_dims[2])
        } else {
            (1, blocks_dims[0], blocks_dims[1])
        };
        let k = k_half * 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;

        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;

        let mut weights = vec![0f32; num_experts * n * k];
        let half_block = MXFP4_BLOCK_SIZE / 2; // 16 packed bytes per block

        for expert in 0..num_experts {
            for row in 0..n {
                let blocks_row = expert * n * k_half + row * k_half;
                let scales_row = expert * n * num_blocks_per_row + row * num_blocks_per_row;
                let weights_row = expert * n * k + row * k;

                for blk in 0..num_blocks_per_row {
                    let scale = scales_data[scales_row + blk] as usize;
                    let dequant = &Self::DEQUANT_LUT[scale];
                    let blk_bytes = &blocks_data[blocks_row + blk * half_block..];
                    let w_out = &mut weights[weights_row + blk * MXFP4_BLOCK_SIZE..];

                    for byte_i in 0..half_block {
                        let packed = blk_bytes[byte_i];
                        w_out[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                        w_out[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                    }
                }
            }
        }

        let shape = if blocks_dims.len() == 3 {
            vec![num_experts, n, k]
        } else {
            vec![n, k]
        };

        Tensor::from_vec(weights, shape.as_slice(), &Device::Cpu)?
            .to_device(self.blocks.device())?
            .to_dtype(DType::BF16)
    }

    /// CPU forward pass: blocked dequant + matmul to avoid full weight allocation.
    /// Processes MXFP4_BLOCK_SIZE (32) input columns at a time, dequantizing only
    /// the needed weight slice before accumulating partial results.
    fn forward_dequantize(&self, x: &Tensor) -> Result<Tensor> {
        let orig_dims = x.dims().to_vec();

        let x_2d = if orig_dims.len() > 2 {
            let features = orig_dims[orig_dims.len() - 1];
            let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
            x.reshape((batch_size, features))?
        } else {
            x.clone()
        };

        let x_f32 = x_2d.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let (m, k) = x_f32.dims2()?;

        let blocks_dims = self.blocks.dims();
        let n = if blocks_dims.len() == 3 {
            blocks_dims[1]
        } else {
            blocks_dims[0]
        };
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        // output: [m, n], accumulate x @ W^T in blocks of 32 columns
        let mut output = vec![0f32; m * n];
        let k_half = k / 2;

        for blk in 0..num_blocks_per_row {
            let col_start = blk * MXFP4_BLOCK_SIZE;

            for row in 0..n {
                let scale = scales_data[row * num_blocks_per_row + blk] as usize;
                let dequant = &Self::DEQUANT_LUT[scale];
                let blk_bytes = &blocks_data[row * k_half + blk * half_block..];

                // Dequantize this block of 32 weights for this output row
                let mut w_block = [0f32; MXFP4_BLOCK_SIZE];
                for byte_i in 0..half_block {
                    let packed = blk_bytes[byte_i];
                    w_block[byte_i * 2] = dequant[(packed & 0x0F) as usize];
                    w_block[byte_i * 2 + 1] = dequant[((packed >> 4) & 0x0F) as usize];
                }

                // Accumulate dot product for all tokens against this weight block
                for token in 0..m {
                    let x_row = &x_data[token * k + col_start..];
                    let mut acc = 0f32;
                    for i in 0..MXFP4_BLOCK_SIZE {
                        acc += x_row[i] * w_block[i];
                    }
                    output[token * n + row] += acc;
                }
            }
        }

        let mut result = Tensor::from_vec(output, (m, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;

        if let Some(bias) = &self.bias {
            result = result.broadcast_add(bias)?;
        }

        if orig_dims.len() > 2 {
            let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
            new_dims.push(result.dim(1)?);
            result = result.reshape(new_dims)?;
        }

        Ok(result)
    }

    /// CPU MoE forward: blocked dequant per (token, expert) pair.
    /// Avoids dequantizing all experts, only touches the needed weight blocks.
    fn gather_forward_dequantize(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let indices_dims = indices.dims();

        let (num_tokens, topk, k, x_has_topk) = if x_dims.len() == 2 {
            (x_dims[0], indices_dims[1], x_dims[1], false)
        } else {
            (x_dims[0], x_dims[1], x_dims[2], true)
        };

        let blocks_dims = self.blocks.dims();
        let n = blocks_dims[1];
        let k_half = k / 2;
        let num_blocks_per_row = k / MXFP4_BLOCK_SIZE;
        let half_block = MXFP4_BLOCK_SIZE / 2;

        let blocks_cpu = self.blocks.to_device(&Device::Cpu)?;
        let scales_cpu = self.scales.to_device(&Device::Cpu)?;
        let blocks_data: Vec<u8> = blocks_cpu.flatten_all()?.to_vec1()?;
        let scales_data: Vec<u8> = scales_cpu.flatten_all()?.to_vec1()?;

        let x_f32 = x.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let x_data: Vec<f32> = x_f32.flatten_all()?.to_vec1()?;

        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::U32)?;
        let indices_data: Vec<u32> = indices_cpu.flatten_all()?.to_vec1()?;

        let bias_data: Option<Vec<f32>> = self
            .bias
            .as_ref()
            .map(|b| {
                b.to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1()
            })
            .transpose()?;

        // output: [num_tokens * topk, n]
        let mut output = vec![0f32; num_tokens * topk * n];

        for token_idx in 0..num_tokens {
            for slot_idx in 0..topk {
                let expert_idx = indices_data[token_idx * topk + slot_idx] as usize;
                let out_row = token_idx * topk + slot_idx;

                // Get input row
                let x_offset = if x_has_topk {
                    (token_idx * topk + slot_idx) * k
                } else {
                    token_idx * k
                };

                // Blocked dequant + matmul for this (token, expert) pair
                let expert_blocks_base = expert_idx * n * k_half;
                let expert_scales_base = expert_idx * n * num_blocks_per_row;

                for blk in 0..num_blocks_per_row {
                    let col_start = blk * MXFP4_BLOCK_SIZE;

                    // Load input block
                    let x_blk =
                        &x_data[x_offset + col_start..x_offset + col_start + MXFP4_BLOCK_SIZE];

                    for row in 0..n {
                        let scale = scales_data[expert_scales_base + row * num_blocks_per_row + blk]
                            as usize;
                        let dequant = &Self::DEQUANT_LUT[scale];
                        let blk_bytes =
                            &blocks_data[expert_blocks_base + row * k_half + blk * half_block..];

                        let mut dot = 0f32;
                        for byte_i in 0..half_block {
                            let packed = blk_bytes[byte_i];
                            let w0 = dequant[(packed & 0x0F) as usize];
                            let w1 = dequant[((packed >> 4) & 0x0F) as usize];
                            dot += x_blk[byte_i * 2] * w0 + x_blk[byte_i * 2 + 1] * w1;
                        }
                        output[out_row * n + row] += dot;
                    }
                }

                // Add bias
                if let Some(ref bias) = bias_data {
                    let bias_offset = expert_idx * n;
                    for row in 0..n {
                        output[out_row * n + row] += bias[bias_offset + row];
                    }
                }
            }
        }

        let result = Tensor::from_vec(output, (num_tokens * topk, n), &Device::Cpu)?
            .to_device(x.device())?
            .to_dtype(x.dtype())?;
        result.reshape((num_tokens, topk, n))
    }
}

// UQFF binary layout for MXFP4Layer:
// -----------------------
// [u32 LE] UQFF version
// [u8]     QuantizedSerdeType::Mxfp4 (6)
// [u8]     has_bias (0 or 1)
// -----------------------
// Blocks tensor data via serialize_tensor
// -----------------------
// Scales tensor data via serialize_tensor
// -----------------------
// [OPTIONAL] Bias tensor data via serialize_tensor
// -----------------------

impl QuantizedSerde for MXFP4Layer {
    fn name(&self) -> &'static str {
        "mxfp4-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();

        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Mxfp4 as u8);
        buffer.push(bias.is_some() as u8);

        serialize_tensor(&mut buffer, &self.blocks)?;
        serialize_tensor(&mut buffer, &self.scales)?;

        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        let (layer, _bias) = Self::deserialize_ext_bias(data, device, guard)?;
        Ok(layer)
    }

    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.to_vec());

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Mxfp4 as usize {
            candle_core::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Mxfp4 as usize
            );
        }

        let has_bias = buffer.read_u8()? != 0;

        let _acquired_load_guard = guard.acquire(device);
        let blocks = deserialize_tensor(&mut buffer, device)?;
        let scales = deserialize_tensor(&mut buffer, device)?;

        let bias = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let ext_bias = bias.clone();

        Ok((
            Arc::new(Self {
                blocks,
                scales,
                bias,
            }),
            ext_bias,
        ))
    }
}
