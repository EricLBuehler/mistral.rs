use candle_core::Result;
use candle_nn::Linear;
use std::sync::Arc;

use crate::{
    DummyLayer, QuantMethod, QuantMethodConfig, QuantizedConfig, ShardedVarBuilder, UnquantLinear,
};

#[cfg(feature = "cuda")]
use candle_core::{DType, Tensor};
#[cfg(feature = "cuda")]
use crate::GptqLayer;

/// Load a linear layer from compressed-tensors format.
///
/// If `weight_packed` is present, the layer is quantized:
/// - Repacks from compressed-tensors layout to GPTQ layout (transpose)
/// - Creates a GptqLayer that reuses existing GPTQ/exllama CUDA kernels
///
/// If `weight_packed` is absent (layer is in the `ignore` list),
/// falls back to loading a regular unquantized weight tensor.
pub fn compressed_tensors_linear(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::CompressedTensors { bits, group_size } = config else {
        candle_core::bail!("Expected CompressedTensors quantization config");
    };

    // If this layer doesn't have weight_packed, it's not quantized (in the ignore list)
    if !vb.contains_tensor("weight_packed") {
        if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
        }
        // Load as unquantized
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get_with_hints((out_dim,), "bias", Default::default())?)
        } else {
            None
        };
        let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
            Linear::new(weight, bias),
        ))?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (in_dim, out_dim, bits, group_size);
        candle_core::bail!(
            "compressed-tensors quantization requires the `cuda` feature. \
             The model's routed expert weights are pre-quantized as INT4 packed tensors \
             which require CUDA kernels for efficient inference."
        );
    }

    #[cfg(feature = "cuda")]
    {
        let bits = *bits;
        let group_size = *group_size;
        let pack_factor = 32 / bits;

        // Load compressed-tensors weights
        // weight_packed: (out_features, in_features/pack_factor) as INT32
        let weight_packed = vb.get_with_hints_dtype(
            (out_dim, in_dim / pack_factor),
            "weight_packed",
            Default::default(),
            DType::I32,
        )?;

        // weight_scale: (out_features, in_features/group_size) as F16
        let weight_scale = vb.get_with_hints_dtype(
            (out_dim, in_dim / group_size),
            "weight_scale",
            Default::default(),
            DType::F16,
        )?;

        let bias = if vb.contains_tensor("bias") {
            Some(vb.get_with_hints_dtype(
                (out_dim,),
                "bias",
                Default::default(),
                DType::F16,
            )?)
        } else {
            None
        };

        // Repack from compressed-tensors layout to GPTQ layout.
        //
        // compressed-tensors packs along dim=1 (columns/input features):
        //   weight_packed shape: (out_features, in_features/pack_factor)
        //
        // GPTQ packs along dim=0 (rows/input features):
        //   qweight shape: (in_features/pack_factor, out_features)
        //
        // Within each INT32, both formats pack consecutive input feature values
        // for a single output feature, so a simple transpose preserves the
        // internal bit layout.
        let qweight = weight_packed.t()?.contiguous()?;

        // scales: transpose from (out, in/group_size) to (in/group_size, out)
        let scales = weight_scale.t()?.contiguous()?;

        // Create constant qzeros for symmetric quantization.
        // For unsigned 4-bit with symmetric quantization, zero_point = 8.
        // Pack 8 zero points of value 8 into each INT32:
        //   8 | (8<<4) | (8<<8) | ... | (8<<28) = 0x88888888
        let n_groups = in_dim / group_size;
        let packed_zp_value: i32 = {
            let mut val: i32 = 0;
            for i in 0..pack_factor {
                val |= (1i32 << (bits - 1)) << (bits * i);
            }
            val
        };
        let qzeros_numel = n_groups * (out_dim / pack_factor);
        let qzeros_data: Vec<i32> = vec![packed_zp_value; qzeros_numel];
        let qzeros = Tensor::from_vec(
            qzeros_data,
            (n_groups, out_dim / pack_factor),
            qweight.device(),
        )?;

        // Create sequential g_idx: g_idx[i] = i / group_size
        // This represents simple contiguous grouping (no reordering).
        let g_idx_data: Vec<i32> = (0..in_dim).map(|i| (i / group_size) as i32).collect();
        let g_idx = Tensor::from_vec(g_idx_data, (in_dim,), qweight.device())?;

        let config = QuantMethodConfig::GptqAwq {
            bits: bits as i32,
            use_exllama: false,
            q_weight: qweight,
            qzeros: Some(qzeros),
            scales,
            g_idx: Some(g_idx),
            bias,
            workspace: None,
            is_marlin: false,
            is_awq: false,
        };

        Ok(Arc::new(GptqLayer::new(config)?))
    }
}
