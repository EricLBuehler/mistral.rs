use std::{collections::HashSet, path::PathBuf, sync::Arc};

use candle_core::{Device, Result, Tensor};
use safetensors::tensor::Dtype;

use super::{bias_shard, BiasShard};
use crate::{
    safetensors::MmapedSafetensors, AfqLayer, F8Q8Linear, FP8Linear, GgufMatMul, HqqLayer, IsqType,
    MXFP4Layer, QuantMethod, QuantizedSerde, QuantizedSerdeType, Shard, UnquantLinear,
};

pub struct UqffReader {
    artifacts: MmapedSafetensors,
    names: HashSet<String>,
}

impl UqffReader {
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let artifacts = unsafe { MmapedSafetensors::multi(paths)? };
        let names = artifacts
            .tensors()
            .into_iter()
            .map(|(name, _)| name)
            .collect::<HashSet<_>>();
        if !names.is_empty() && names.iter().all(|name| name.parse::<usize>().is_ok()) {
            candle_core::bail!(
                "Pre-1.0 UQFF artifacts are no longer supported; regenerate with `mistralrs quantize`."
            );
        }
        if names.contains(super::UQFF_VERSION_MAJOR_KEY) {
            let load_version = |key: &str| -> Result<u32> {
                artifacts.load(key, &Device::Cpu, None)?.to_scalar::<u32>()
            };
            let major = load_version(super::UQFF_VERSION_MAJOR_KEY)?;
            let minor = load_version(super::UQFF_VERSION_MINOR_KEY)?;
            let patch = load_version(super::UQFF_VERSION_PATCH_KEY)?;
            let ours = format!(
                "{}.{}.{}",
                super::UQFF_VERSION_MAJOR,
                super::UQFF_VERSION_MINOR,
                super::UQFF_VERSION_PATCH
            );
            if major != super::UQFF_VERSION_MAJOR {
                candle_core::bail!(
                    "UQFF version {major}.{minor}.{patch} is incompatible with this build ({ours}); regenerate with `mistralrs quantize`."
                );
            }
            // Same major, higher minor: the file may use additions this reader does not know.
            if minor > super::UQFF_VERSION_MINOR {
                candle_core::bail!(
                    "UQFF version {major}.{minor}.{patch} was written by a newer mistral.rs than this build ({ours}); upgrade mistral.rs."
                );
            }
        } else {
            candle_core::bail!(
                "UQFF artifact has no version tag (pre-1.0 file); regenerate with `mistralrs quantize`."
            );
        }
        Ok(Self { artifacts, names })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }

    pub fn pack_factor(&self, dtype: candle_core::DType) -> Result<usize> {
        Ok(self
            .first_isq_type()?
            .map(|isq| isq.pack_factor(dtype))
            .unwrap_or(1))
    }

    pub fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        if !self.contains(&format!("{key}.weight")) {
            return Ok(None);
        }

        let format = self.load_format(key)?;
        match format {
            QuantizedSerdeType::Gguf => {
                GgufMatMul::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::Unquant => {
                UnquantLinear::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::Hqq => {
                HqqLayer::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::Fp8 => {
                FP8Linear::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::Afq => {
                AfqLayer::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::F8Q8 => {
                F8Q8Linear::deserialize_uqff(self, key, device, shard).map(Some)
            }
            QuantizedSerdeType::Mxfp4 => {
                MXFP4Layer::deserialize_uqff(self, key, device, shard).map(Some)
            }
        }
    }

    /// Required element alignment for sharding this layer's packed (input) dim.
    pub fn shard_alignment(&self, key: &str) -> Result<usize> {
        match self.load_format(key)? {
            QuantizedSerdeType::Gguf => {
                let code = self.load_u32_scalar(&format!("{key}.weight.dtype"))?;
                Ok(GgufMatMul::block_size_from_uqff_dtype(code)?)
            }
            QuantizedSerdeType::Afq => {
                Ok(self.load_u8_scalar(&format!("{key}.weight.group_size"))? as usize)
            }
            QuantizedSerdeType::Mxfp4 | QuantizedSerdeType::F8Q8 => Ok(32),
            QuantizedSerdeType::Fp8 => Ok(1),
            QuantizedSerdeType::Hqq => {
                candle_core::bail!("HQQ UQFF artifacts do not support sharded loading.")
            }
            QuantizedSerdeType::Unquant => Ok(1),
        }
    }

    fn first_isq_type(&self) -> Result<Option<IsqType>> {
        for name in &self.names {
            if let Some(prefix) = name.strip_suffix(".weight.format") {
                // Fallback-quantized layers (e.g. GGUF F32 skips) have no ISQ type
                if let Ok(ty) = self.isq_type_for_prefix(prefix) {
                    return Ok(Some(ty));
                }
            }
        }
        Ok(None)
    }

    fn isq_type_for_prefix(&self, prefix: &str) -> Result<IsqType> {
        match self.load_format(prefix)? {
            QuantizedSerdeType::Gguf => GgufMatMul::isq_type_from_uqff(self, prefix),
            QuantizedSerdeType::Unquant => {
                candle_core::bail!("Unquantized UQFF layers do not have an ISQ type.")
            }
            QuantizedSerdeType::Hqq => HqqLayer::isq_type_from_uqff(self, prefix),
            QuantizedSerdeType::Fp8 => FP8Linear::isq_type_from_uqff(self, prefix),
            QuantizedSerdeType::Afq => AfqLayer::isq_type_from_uqff(self, prefix),
            QuantizedSerdeType::F8Q8 => F8Q8Linear::isq_type_from_uqff(self, prefix),
            QuantizedSerdeType::Mxfp4 => MXFP4Layer::isq_type_from_uqff(self, prefix),
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
        if !self.contains(name) {
            return Ok(None);
        }
        self.load_tensor(name, device).map(Some)
    }

    /// Load a tensor, narrowing on CPU first so the device only ever sees the shard.
    pub(crate) fn load_tensor_sharded(
        &self,
        name: &str,
        device: &Device,
        range: Option<(usize, usize, usize)>,
    ) -> Result<Tensor> {
        match range {
            None => self.load_tensor(name, device),
            Some((dim, start, len)) => self
                .artifacts
                .load(name, &Device::Cpu, None)?
                .narrow(dim, start, len)?
                .contiguous()?
                .to_device(device),
        }
    }

    /// Load a layer's bias according to the shard semantics of its weight.
    pub(crate) fn load_bias(
        &self,
        key: &str,
        device: &Device,
        range: Option<(usize, usize, usize)>,
        weight_rank: usize,
    ) -> Result<Option<Tensor>> {
        match bias_shard(range, weight_rank) {
            BiasShard::Skip => Ok(None),
            BiasShard::Full => self.load_optional_tensor(&format!("{key}.bias"), device),
            BiasShard::Narrow(start, len) => {
                let name = format!("{key}.bias");
                if !self.contains(&name) {
                    return Ok(None);
                }
                Ok(Some(
                    self.artifacts
                        .load(&name, &Device::Cpu, None)?
                        .narrow(0, start, len)?
                        .contiguous()?
                        .to_device(device)?,
                ))
            }
        }
    }

    pub(crate) fn tensor_dims(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.artifacts.get(name)?.shape().to_vec())
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
