use std::{io::Cursor, path::PathBuf, sync::Arc};

use candle_core::{Device, Result, Tensor};

use crate::{
    safetensors::MmapedSafetensors, utils::deserialize_tensor, AfqBits, AfqGroupSize, AfqLayer,
    IsqType, QuantMethod,
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

    pub fn pack_factor(&self, dtype: candle_core::DType) -> usize {
        IsqType::AFQ4.pack_factor(dtype)
    }

    pub fn load_linear(&self, key: &str, device: &Device) -> Result<Option<Arc<dyn QuantMethod>>> {
        let weight_key = format!("{key}.weight");
        if self.artifacts.get(&weight_key).is_err() {
            return Ok(None);
        }

        let w_q = self.load_tensor(&weight_key, device)?;
        let scales = self.load_tensor(&format!("{key}.weight.scales"), device)?;
        let biases = self.load_tensor(&format!("{key}.weight.biases"), device)?;
        let bias = self.load_optional_tensor(&format!("{key}.bias"), device)?;

        Ok(Some(Arc::new(AfqLayer::from_parts(
            w_q,
            scales,
            biases,
            bias,
            AfqBits::Four,
            AfqGroupSize::Med,
        ))))
    }

    fn load_tensor(&self, name: &str, device: &Device) -> Result<Tensor> {
        let artifact = self.artifacts.get(name)?;
        let mut cursor = Cursor::new(artifact.data());
        deserialize_tensor(&mut cursor, device)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        match self.load_tensor(name, device) {
            Ok(tensor) => Ok(Some(tensor)),
            Err(candle_core::Error::CannotFindTensor { .. }) => Ok(None),
            Err(err) => Err(err),
        }
    }
}
