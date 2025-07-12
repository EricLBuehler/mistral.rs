use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::Linear;

mod ops;

#[cfg(feature = "cuda")]
mod ffi;

use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    utils::{
        deserialize_tensor, read_dtype, serialize_tensor, version_is_compatible, write_dtype,
        UQFF_VERSION,
    },
    AfqBits, AfqGroupSize, AfqLayer, DummyLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits,
    HqqConfig, HqqLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, QuantizedSerdeType, Shard, ShardedVarBuilder, UnquantLinear
    // Removed Comm, distributed::Id from here; they are used in tests only
};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::borrow::Cow;
use std::io::Cursor;

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
            | QuantMethodConfig::CutlassFP8PTQ { .. } => {
                unreachable!("Cannot create BlockwiseFP8Linear from this config")
            }
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
        let weight = self.dequantize_w()?;
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight,
            self.bias.clone(),
        )))?;
        unquant.forward(x)
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
                IsqType::Q2K | IsqType::Q3K | IsqType::Q4K | IsqType::Q4_0 | IsqType::Q4_1
                | IsqType::Q5K | IsqType::Q5_0 | IsqType::Q5_1 | IsqType::Q6K | IsqType::Q8K
                | IsqType::Q8_0 | IsqType::Q8_1,
            ) => {
                let dtype_ggml: GgmlDType = dtype.unwrap().try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype_ggml, n_quantized, guard)
                } else {
                    generate_isq!(weight, device, dtype_ggml, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self.bias.as_ref().map(|b| {
                        b.to_dtype(DType::F32)
                            .unwrap()
                            .to_device(&device)
                            .unwrap()
                    }),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    candle_core::bail!("F8E4M3 does not support imatrix.");
                }
                let w = weight.to_device(&device)?;
                let b = if let Some(b_val) = &self.bias {
                    Some(b_val.to_device(&device)?)
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
                let w = weight.to_device(&device)?;
                let b = if let Some(b_val) = &self.bias {
                    Some(b_val.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl QuantizedSerde for BlockwiseFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "blockwise-fp8-linear"
    }

    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<[u8]>> {
        let mut buffer = Vec::new();
        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Fp8 as u8);
        buffer.push(bias.is_some() as u8);
        write_dtype(self.dequant_dtype, &mut buffer);
        buffer.write_u8(self.weight_block_size.len() as u8)?;
        for dim in &self.weight_block_size {
            buffer.write_u64::<LittleEndian>(*dim as u64)?;
        }
        serialize_tensor(&mut buffer, &self.weight)?;
        serialize_tensor(&mut buffer, &self.weight_scale_inv)?;
        if let Some(bias_tensor) = &bias {
            serialize_tensor(&mut buffer, bias_tensor)?;
        }
        Ok(Cow::from(buffer))
    }

    fn serialize(&self) -> Result<Cow<[u8]>> {
        self.serialize_with_bias(self.bias.clone())
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
        let mut buffer = Cursor::new(data.as_ref());
        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }
        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Fp8 as usize {
            candle_core::bail!("ISQ type ({isq_type}) doesn't match expected type {}", QuantizedSerdeType::Fp8 as usize);
        }
        let has_bias = buffer.read_u8()? != 0;
        let _acquired_load_guard = guard.acquire(device);
        let dequant_dtype = read_dtype(&mut buffer)?;
        let num_block_dims = buffer.read_u8()?;
        let mut weight_block_size = Vec::with_capacity(num_block_dims as usize);
        for _ in 0..num_block_dims {
            weight_block_size.push(buffer.read_u64::<LittleEndian>()? as usize);
        }
        let weight = deserialize_tensor(&mut buffer, device)?;
        let weight_scale_inv = deserialize_tensor(&mut buffer, device)?;
        let bias = if has_bias { Some(deserialize_tensor(&mut buffer, device)?) } else { None };
        Ok(Arc::new(Self { weight, weight_scale_inv, bias, dequant_dtype, weight_block_size }))
    }

    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        let mut buffer = Cursor::new(data.as_ref());
        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(candle_core::Error::wrap(e));
        }
        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Fp8 as usize {
            candle_core::bail!("ISQ type ({isq_type}) doesn't match expected type {}", QuantizedSerdeType::Fp8 as usize);
        }
        let has_bias = buffer.read_u8()? != 0;
        let _acquired_load_guard = guard.acquire(device);
        let dequant_dtype = read_dtype(&mut buffer)?;
        let num_block_dims = buffer.read_u8()?;
        let mut weight_block_size = Vec::with_capacity(num_block_dims as usize);
        for _ in 0..num_block_dims {
            weight_block_size.push(buffer.read_u64::<LittleEndian>()? as usize);
        }
        let weight = deserialize_tensor(&mut buffer, device)?;
        let weight_scale_inv = deserialize_tensor(&mut buffer, device)?;
        let bias = if has_bias { Some(deserialize_tensor(&mut buffer, device)?) } else { None };
        Ok((Arc::new(Self { weight, weight_scale_inv, bias: None, dequant_dtype, weight_block_size }), bias))
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
    if vb.contains_tensor("weight") && !vb.contains_tensor("weight_scale_inv") {
        return crate::linear_b(in_dim, out_dim, bias, &None, vb);
    }
    if !(vb.contains_tensor("weight") && vb.contains_tensor("weight_scale_inv")) {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }
    if weight_block_size.len() != 2 {
        candle_core::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (out_dim.div_ceil(weight_block_size[0]), in_dim.div_ceil(weight_block_size[1])),
        "weight_scale_inv", hints, DType::F32,
    )?;
    let bias_tensor = if bias { Some(vb.get((out_dim,), "bias")?) } else { None };
    Ok(Arc::new(BlockwiseFP8Linear {
        weight, weight_block_size: weight_block_size.clone(), weight_scale_inv, bias: bias_tensor, dequant_dtype: vb.dtype(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Comm, QuantizeOntoGuard, QuantizedSerde, distributed::Id};
    use candle_core::{DType, Device, Result, Tensor};
    use std::sync::Arc;

    fn create_sample_blockwise_fp8_linear(
        device: &Device,
        with_bias: bool,
    ) -> Result<BlockwiseFP8Linear> {
        let weight = Tensor::ones((4, 8), DType::F8E4M3, device)?;
        let weight_scale_inv = Tensor::ones((2, 2), DType::F32, device)?;
        let bias = if with_bias { Some(Tensor::zeros(4, DType::F32, device)?) } else { None };
        let dequant_dtype = DType::F32;
        let weight_block_size = vec![2, 4];
        Ok(BlockwiseFP8Linear { weight, weight_scale_inv, bias, dequant_dtype, weight_block_size })
    }

    #[test]
    fn test_blockwise_fp8_serde_no_bias() -> Result<()> {
        let device = Device::Cpu;
        let original_layer = create_sample_blockwise_fp8_linear(&device, false)?;
        let serialized_data = original_layer.serialize()?;
        let guard = QuantizeOntoGuard::new();
        let comm = Arc::new(Comm::from_device(Id::default(), &device, 0, 1).unwrap());
        let deserialized_layer_dyn = BlockwiseFP8Linear::deserialize(serialized_data, &device, &comm, guard)?;
        let deserialized_layer_concrete = deserialized_layer_dyn.as_any().downcast_ref::<BlockwiseFP8Linear>().expect("Failed to downcast");
        assert_eq!(original_layer.weight.to_vec2::<f32>()?, deserialized_layer_concrete.weight.to_vec2::<f32>()?);
        assert_eq!(original_layer.weight_scale_inv.to_vec2::<f32>()?, deserialized_layer_concrete.weight_scale_inv.to_vec2::<f32>()?);
        assert!(deserialized_layer_concrete.bias.is_none());
        assert_eq!(original_layer.dequant_dtype, deserialized_layer_concrete.dequant_dtype);
        assert_eq!(original_layer.weight_block_size, deserialized_layer_concrete.weight_block_size);
        Ok(())
    }

    #[test]
    fn test_blockwise_fp8_serde_with_bias() -> Result<()> {
        let device = Device::Cpu;
        let original_layer = create_sample_blockwise_fp8_linear(&device, true)?;
        let serialized_data = original_layer.serialize()?;
        let guard = QuantizeOntoGuard::new();
        let comm = Arc::new(Comm::from_device(Id::default(), &device, 0, 1).unwrap());
        let deserialized_layer_dyn = BlockwiseFP8Linear::deserialize(serialized_data, &device, &comm, guard)?;
        let deserialized_layer_concrete = deserialized_layer_dyn.as_any().downcast_ref::<BlockwiseFP8Linear>().expect("Failed to downcast");
        assert_eq!(original_layer.weight.to_vec2::<f32>()?, deserialized_layer_concrete.weight.to_vec2::<f32>()?);
        assert_eq!(original_layer.weight_scale_inv.to_vec2::<f32>()?, deserialized_layer_concrete.weight_scale_inv.to_vec2::<f32>()?);
        assert!(deserialized_layer_concrete.bias.is_some());
        assert_eq!(original_layer.bias.as_ref().unwrap().to_vec1::<f32>()?, deserialized_layer_concrete.bias.as_ref().unwrap().to_vec1::<f32>()?);
        assert_eq!(original_layer.dequant_dtype, deserialized_layer_concrete.dequant_dtype);
        assert_eq!(original_layer.weight_block_size, deserialized_layer_concrete.weight_block_size);
        Ok(())
    }

    #[test]
    fn test_blockwise_fp8_serde_ext_bias() -> Result<()> {
        let device = Device::Cpu;
        let original_layer_with_bias = create_sample_blockwise_fp8_linear(&device, true)?;
        let original_bias = original_layer_with_bias.bias.clone().unwrap();
        let serialized_data = original_layer_with_bias.serialize()?;
        let guard = QuantizeOntoGuard::new();
        let (deserialized_arc_no_bias, deserialized_bias_option) = BlockwiseFP8Linear::deserialize_ext_bias(serialized_data, &device, guard)?;
        let deserialized_layer_no_bias_concrete = deserialized_arc_no_bias.as_any().downcast_ref::<BlockwiseFP8Linear>().expect("Failed to downcast");
        assert!(deserialized_layer_no_bias_concrete.bias.is_none());
        assert!(deserialized_bias_option.is_some());
        let deserialized_bias = deserialized_bias_option.unwrap();
        assert_eq!(original_bias.to_vec1::<f32>()?, deserialized_bias.to_vec1::<f32>()?);
        assert_eq!(original_layer_with_bias.weight.to_vec2::<f32>()?, deserialized_layer_no_bias_concrete.weight.to_vec2::<f32>()?);
        assert_eq!(original_layer_with_bias.weight_scale_inv.to_vec2::<f32>()?, deserialized_layer_no_bias_concrete.weight_scale_inv.to_vec2::<f32>()?);
        assert_eq!(original_layer_with_bias.dequant_dtype, deserialized_layer_no_bias_concrete.dequant_dtype);
        assert_eq!(original_layer_with_bias.weight_block_size, deserialized_layer_no_bias_concrete.weight_block_size);
        Ok(())
    }
}
