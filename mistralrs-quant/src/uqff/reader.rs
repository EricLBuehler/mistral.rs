use std::{num::NonZeroUsize, path::PathBuf, sync::Arc};

use candle_core::{Device, Result, Shape, Tensor};
use safetensors::tensor::Dtype;

use crate::{
    hqq::{HqqAxis, HqqBits, HqqConfig},
    safetensors::MmapedSafetensors,
    utils::uqff_code_to_dtype,
    AfqBits, AfqGroupSize, AfqLayer, F8Q8Linear, FP8Linear, GgufMatMul, HqqLayer, IsqType,
    MXFP4Layer, QuantMethod, QuantizedSerdeType,
};

pub struct UqffReader {
    artifacts: MmapedSafetensors,
}

impl UqffReader {
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let artifacts = unsafe { MmapedSafetensors::multi(paths)? };
        let names = artifacts
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        if !names.is_empty() && names.iter().all(|name| name.parse::<usize>().is_ok()) {
            candle_core::bail!(
                "UQFF v1 numeric artifacts are no longer supported; expected named UQFF v2 artifacts."
            );
        }
        Ok(Self { artifacts })
    }

    pub fn pack_factor(&self, dtype: candle_core::DType) -> Result<usize> {
        Ok(self
            .first_isq_type()?
            .map(|isq| isq.pack_factor(dtype))
            .unwrap_or(1))
    }

    pub fn load_linear(&self, key: &str, device: &Device) -> Result<Option<Arc<dyn QuantMethod>>> {
        let weight_key = format!("{key}.weight");
        match self.artifacts.get(&weight_key) {
            Ok(_) => (),
            Err(candle_core::Error::CannotFindTensor { .. }) => return Ok(None),
            Err(err) => return Err(err),
        }

        let format = self.load_format(key)?;
        match format {
            QuantizedSerdeType::Gguf => self
                .load_gguf(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
            QuantizedSerdeType::Unquant => {
                candle_core::bail!("UQFF v2 does not support unquantized linear artifacts.")
            }
            QuantizedSerdeType::Hqq => self
                .load_hqq(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
            QuantizedSerdeType::Fp8 => self
                .load_fp8(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
            QuantizedSerdeType::Afq => self
                .load_afq(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
            QuantizedSerdeType::F8Q8 => self
                .load_f8q8(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
            QuantizedSerdeType::Mxfp4 => self
                .load_mxfp4(key, device)
                .map(|layer| Some(Arc::new(layer) as Arc<dyn QuantMethod>)),
        }
    }

    fn first_isq_type(&self) -> Result<Option<IsqType>> {
        for (name, _) in self.artifacts.tensors() {
            if let Some(prefix) = name.strip_suffix(".weight.format") {
                return self.isq_type_for_prefix(prefix).map(Some);
            }
        }
        Ok(None)
    }

    fn isq_type_for_prefix(&self, prefix: &str) -> Result<IsqType> {
        match self.load_format(prefix)? {
            QuantizedSerdeType::Gguf => GgufMatMul::isq_type_from_uqff_dtype(
                self.load_u32_scalar(&format!("{prefix}.weight.dtype"))?,
            ),
            QuantizedSerdeType::Unquant => {
                candle_core::bail!("UQFF v2 does not support unquantized linear artifacts.")
            }
            QuantizedSerdeType::Hqq => match HqqBits::try_from(
                self.load_u8_scalar(&format!("{prefix}.weight.bits"))? as usize,
            )? {
                HqqBits::Eight => Ok(IsqType::HQQ8),
                HqqBits::Four => Ok(IsqType::HQQ4),
                HqqBits::One | HqqBits::Two | HqqBits::Three => {
                    candle_core::bail!("Cannot convert HQQ bit width to an ISQ type.")
                }
            },
            QuantizedSerdeType::Fp8 => Ok(IsqType::F8E4M3),
            QuantizedSerdeType::Afq => match AfqBits::try_from(
                self.load_u8_scalar(&format!("{prefix}.weight.bits"))? as usize,
            )? {
                AfqBits::Two => Ok(IsqType::AFQ2),
                AfqBits::Three => Ok(IsqType::AFQ3),
                AfqBits::Four => Ok(IsqType::AFQ4),
                AfqBits::Six => Ok(IsqType::AFQ6),
                AfqBits::Eight => Ok(IsqType::AFQ8),
                AfqBits::Mxfp4 => Ok(IsqType::MXFP4),
            },
            QuantizedSerdeType::F8Q8 => Ok(IsqType::F8Q8),
            QuantizedSerdeType::Mxfp4 => Ok(IsqType::MXFP4),
        }
    }

    fn load_format(&self, key: &str) -> Result<QuantizedSerdeType> {
        QuantizedSerdeType::try_from(self.load_u8_scalar(&format!("{key}.weight.format"))? as usize)
    }

    fn load_gguf(&self, key: &str, device: &Device) -> Result<GgufMatMul> {
        let dtype = self.load_u32_scalar(&format!("{key}.weight.dtype"))?;
        let shape = self.load_u32_vec(&format!("{key}.weight.shape"))?;
        let weight = self.load_raw_u8(&format!("{key}.weight"))?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        GgufMatMul::from_raw_uqff(dtype, weight, shape, bias, device)
    }

    fn load_hqq(&self, key: &str, device: &Device) -> Result<HqqLayer> {
        let w_q = self.load_tensor(&format!("{key}.weight"), device)?;
        let scales = self.load_tensor(&format!("{key}.weight.scales"), device)?;
        let zeros = self.load_tensor(&format!("{key}.weight.zeros"), device)?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        let w_shape = Shape::from_dims(&self.load_u32_vec(&format!("{key}.weight.shape"))?);
        let optimization_steps =
            match self.load_u32_scalar(&format!("{key}.weight.optimization_steps"))? as usize {
                0 => None,
                steps => Some(steps),
            };
        let cfg = HqqConfig {
            bits: HqqBits::try_from(self.load_u8_scalar(&format!("{key}.weight.bits"))? as usize)?,
            group_size: NonZeroUsize::try_from(
                self.load_u32_scalar(&format!("{key}.weight.group_size"))? as usize,
            )?,
            axis: HqqAxis::try_from(self.load_u8_scalar(&format!("{key}.weight.axis"))? as usize)?,
            optimization_steps,
            round_zeros: self.load_u8_scalar(&format!("{key}.weight.round_zeros"))? != 0,
            channel_wise: self.load_u8_scalar(&format!("{key}.weight.channel_wise"))? != 0,
        };
        Ok(HqqLayer::from_parts(w_q, scales, zeros, bias, w_shape, cfg))
    }

    fn load_fp8(&self, key: &str, device: &Device) -> Result<FP8Linear> {
        let weight = self.load_tensor(&format!("{key}.weight"), device)?;
        let dequant_w_scale = self.load_tensor(&format!("{key}.weight.dequant_w_scale"), device)?;
        let dequant_x_scale = self.load_tensor(&format!("{key}.weight.dequant_x_scale"), device)?;
        let quant_scale = self.load_tensor(&format!("{key}.weight.quant_scale"), device)?;
        let dtype = uqff_code_to_dtype(self.load_u32_scalar(&format!("{key}.weight.dtype"))?)?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        Ok(FP8Linear::from_parts(
            weight,
            bias,
            dequant_w_scale,
            dequant_x_scale,
            quant_scale,
            dtype,
        ))
    }

    fn load_afq(&self, key: &str, device: &Device) -> Result<AfqLayer> {
        let w_q = self.load_tensor(&format!("{key}.weight"), device)?;
        let scales = self.load_tensor(&format!("{key}.weight.scales"), device)?;
        let biases = self.load_tensor(&format!("{key}.weight.biases"), device)?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        let bits = AfqBits::try_from(self.load_u8_scalar(&format!("{key}.weight.bits"))?)?;
        let group_size =
            AfqGroupSize::try_from(self.load_u8_scalar(&format!("{key}.weight.group_size"))?)?;
        Ok(AfqLayer::from_parts(
            w_q, scales, biases, bias, bits, group_size,
        ))
    }

    fn load_f8q8(&self, key: &str, device: &Device) -> Result<F8Q8Linear> {
        let weight = self.load_raw_u8(&format!("{key}.weight"))?;
        let shape = self.load_u32_vec(&format!("{key}.weight.shape"))?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        F8Q8Linear::from_raw_parts(weight, shape, bias)
    }

    fn load_mxfp4(&self, key: &str, device: &Device) -> Result<MXFP4Layer> {
        let blocks = self.load_tensor(&format!("{key}.weight"), device)?;
        let scales = self.load_tensor(&format!("{key}.weight.scales"), device)?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;
        Ok(MXFP4Layer::from_parts(blocks, scales, bias))
    }

    fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.artifacts.load(name, device, None)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        match self.load_tensor(name, device) {
            Ok(tensor) => Ok(Some(tensor)),
            Err(candle_core::Error::CannotFindTensor { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    fn load_raw_u8(&self, name: &str) -> Result<Vec<u8>> {
        let view = self.artifacts.get(name)?;
        if view.dtype() != Dtype::U8 {
            candle_core::bail!("Expected U8 UQFF tensor `{name}`, got {:?}.", view.dtype());
        }
        Ok(view.data().to_vec())
    }

    fn load_u8_scalar(&self, name: &str) -> Result<u8> {
        self.artifacts
            .load(name, &Device::Cpu, None)?
            .to_scalar::<u8>()
    }

    fn load_u32_scalar(&self, name: &str) -> Result<u32> {
        self.artifacts
            .load(name, &Device::Cpu, None)?
            .to_scalar::<u32>()
    }

    fn load_u32_vec(&self, name: &str) -> Result<Vec<usize>> {
        let values: Vec<u32> = self
            .artifacts
            .load(name, &Device::Cpu, None)?
            .flatten_all()?
            .to_vec1()?;
        Ok(values.into_iter().map(|value| value as usize).collect())
    }
}
