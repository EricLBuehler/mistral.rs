use std::{path::PathBuf, sync::Arc};

use candle_core::{Device, Result, Tensor};
use safetensors::tensor::Dtype;

use crate::{
    safetensors::MmapedSafetensors, AfqLayer, F8Q8Linear, FP8Linear, GgufMatMul, HqqLayer, IsqType,
    MXFP4Layer, QuantMethod, QuantizedSerde, QuantizedSerdeType,
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
            QuantizedSerdeType::Gguf => {
                GgufMatMul::deserialize_directly(self, key, device).map(Some)
            }
            QuantizedSerdeType::Unquant => {
                candle_core::bail!("UQFF v2 does not support unquantized linear artifacts.")
            }
            QuantizedSerdeType::Hqq => HqqLayer::deserialize_directly(self, key, device).map(Some),
            QuantizedSerdeType::Fp8 => FP8Linear::deserialize_directly(self, key, device).map(Some),
            QuantizedSerdeType::Afq => AfqLayer::deserialize_directly(self, key, device).map(Some),
            QuantizedSerdeType::F8Q8 => {
                F8Q8Linear::deserialize_directly(self, key, device).map(Some)
            }
            QuantizedSerdeType::Mxfp4 => {
                MXFP4Layer::deserialize_directly(self, key, device).map(Some)
            }
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
            QuantizedSerdeType::Gguf => GgufMatMul::isq_type_from_uqff_direct(self, prefix),
            QuantizedSerdeType::Unquant => {
                candle_core::bail!("UQFF v2 does not support unquantized linear artifacts.")
            }
            QuantizedSerdeType::Hqq => HqqLayer::isq_type_from_uqff_direct(self, prefix),
            QuantizedSerdeType::Fp8 => FP8Linear::isq_type_from_uqff_direct(self, prefix),
            QuantizedSerdeType::Afq => AfqLayer::isq_type_from_uqff_direct(self, prefix),
            QuantizedSerdeType::F8Q8 => F8Q8Linear::isq_type_from_uqff_direct(self, prefix),
            QuantizedSerdeType::Mxfp4 => MXFP4Layer::isq_type_from_uqff_direct(self, prefix),
        }
    }

    pub(crate) fn load_format(&self, key: &str) -> Result<QuantizedSerdeType> {
        QuantizedSerdeType::try_from(self.load_u8_scalar(&format!("{key}.weight.format"))? as usize)
    }

    pub(crate) fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        self.artifacts.load(name, device, None)
    }

    pub(crate) fn load_optional_tensor(
        &self,
        name: &str,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        match self.load_tensor(name, device) {
            Ok(tensor) => Ok(Some(tensor)),
            Err(candle_core::Error::CannotFindTensor { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }

    pub(crate) fn load_raw_u8(&self, name: &str) -> Result<Vec<u8>> {
        let view = self.artifacts.get(name)?;
        if view.dtype() != Dtype::U8 {
            candle_core::bail!("Expected U8 UQFF tensor `{name}`, got {:?}.", view.dtype());
        }
        Ok(view.data().to_vec())
    }

    pub(crate) fn load_u8_scalar(&self, name: &str) -> Result<u8> {
        self.artifacts
            .load(name, &Device::Cpu, None)?
            .to_scalar::<u8>()
    }

    pub(crate) fn load_u32_scalar(&self, name: &str) -> Result<u32> {
        self.artifacts
            .load(name, &Device::Cpu, None)?
            .to_scalar::<u32>()
    }

    pub(crate) fn load_u32_vec(&self, name: &str) -> Result<Vec<usize>> {
        let values: Vec<u32> = self
            .artifacts
            .load(name, &Device::Cpu, None)?
            .flatten_all()?
            .to_vec1()?;
        Ok(values.into_iter().map(|value| value as usize).collect())
    }
}
