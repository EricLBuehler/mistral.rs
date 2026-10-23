use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
};

use candle_core::{Device, Result, Tensor};
use safetensors::tensor::Dtype;

use super::{bias_shard, BiasShard};
use crate::{
    block_pack_factor, safetensors::MmapedSafetensors, AfqLayer, F8Q8Linear, FP8Linear, GgufMatMul,
    HqqLayer, MXFP4Layer, QuantMethod, QuantizedSerde, QuantizedSerdeType, QuantizedWeightSource,
    Shard, UnquantLinear,
};

pub struct UqffReader {
    artifacts: MmapedSafetensors,
    names: HashSet<String>,
}

fn repeating_layer_key(prefix: &str) -> Option<String> {
    let segments = prefix.split('.').collect::<Vec<_>>();
    for index in 0..segments.len().saturating_sub(1) {
        if matches!(segments[index], "layers" | "blocks" | "h")
            && segments[index + 1].parse::<usize>().is_ok()
        {
            return Some(segments[..=index + 1].join("."));
        }
    }
    None
}

fn aggregate_pack_factor(layouts: &[(usize, usize)], dtype: candle_core::DType) -> Result<usize> {
    let logical_elements = layouts.iter().try_fold(0usize, |total, (logical, _)| {
        total
            .checked_add(*logical)
            .ok_or_else(|| candle_core::Error::Msg("UQFF logical element estimate overflow".into()))
    })?;
    let resident_bytes = layouts.iter().try_fold(0usize, |total, (_, resident)| {
        total
            .checked_add(*resident)
            .ok_or_else(|| candle_core::Error::Msg("UQFF resident byte estimate overflow".into()))
    })?;
    let dtype_bytes = dtype.size_in_bytes();
    let mut factor = block_pack_factor(logical_elements, dtype, resident_bytes);
    while factor > 1 {
        let estimated_bytes = layouts.iter().try_fold(0usize, |total, (logical, _)| {
            let bytes = (logical / factor).checked_mul(dtype_bytes).ok_or_else(|| {
                candle_core::Error::Msg("UQFF packed byte estimate overflow".into())
            })?;
            total
                .checked_add(bytes)
                .ok_or_else(|| candle_core::Error::Msg("UQFF packed byte estimate overflow".into()))
        })?;
        if estimated_bytes >= resident_bytes {
            break;
        }
        factor -= 1;
    }
    Ok(factor)
}

fn ensure_pack_factor_representable(
    prefix: &str,
    dtype: candle_core::DType,
    logical_elements: usize,
    resident_bytes: usize,
) -> Result<()> {
    let dense_bytes = logical_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| candle_core::Error::Msg("UQFF dense byte estimate overflow".into()))?;
    if resident_bytes > dense_bytes {
        candle_core::bail!(
            "UQFF layer `{prefix}` occupies {resident_bytes} bytes, more than its {dense_bytes}-byte {dtype:?} dense estimate; use an equal-or-wider model dtype or explicit device mapping."
        );
    }
    Ok(())
}

fn is_version_key(name: &str) -> bool {
    matches!(
        name,
        super::UQFF_VERSION_MAJOR_KEY
            | super::UQFF_VERSION_MINOR_KEY
            | super::UQFF_VERSION_PATCH_KEY
    )
}

fn version_scalar(
    name: &str,
    tensor: &safetensors::tensor::TensorView<'_>,
    path: &Path,
) -> Result<u32> {
    if tensor.dtype() != Dtype::U32 || !tensor.shape().is_empty() || tensor.data().len() != 4 {
        candle_core::bail!(
            "UQFF version tensor `{name}` in `{}` must be a scalar U32.",
            path.display()
        );
    }
    Ok(u32::from_le_bytes(
        tensor.data().try_into().expect("U32 scalar is 4 bytes"),
    ))
}

fn validate_shard_tensor_keys(paths: &[PathBuf], artifacts: &MmapedSafetensors) -> Result<()> {
    let mut seen = HashMap::<String, (PathBuf, Option<u32>)>::new();
    for (path, tensors) in paths.iter().zip(artifacts.tensors_by_file()) {
        for (name, tensor) in tensors {
            let value = is_version_key(&name)
                .then(|| version_scalar(&name, &tensor, path))
                .transpose()?;
            let Some((previous_path, previous_value)) = seen.get(&name) else {
                seen.insert(name, (path.clone(), value));
                continue;
            };
            match (*previous_value, value) {
                (Some(previous), Some(current)) if previous == current => {}
                (Some(previous), Some(current)) => candle_core::bail!(
                    "Conflicting UQFF version tensor `{name}` found in `{}` ({previous}) and `{}` ({current}).",
                    previous_path.display(),
                    path.display()
                ),
                _ => candle_core::bail!(
                    "Duplicate tensor key `{name}` found in `{}` and `{}`.",
                    previous_path.display(),
                    path.display()
                ),
            }
        }
    }
    Ok(())
}

impl UqffReader {
    pub fn open(paths: &[PathBuf]) -> Result<Self> {
        let artifacts = unsafe { MmapedSafetensors::multi(paths)? };
        validate_shard_tensor_keys(paths, &artifacts)?;
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
        let mut layers = HashMap::<String, Vec<(usize, usize)>>::new();
        let mut all_layouts = Vec::new();
        let mut non_repeating_layouts = Vec::new();
        for name in &self.names {
            if let Some(prefix) = name.strip_suffix(".weight.format") {
                let layout = self.stored_layout(prefix)?;
                ensure_pack_factor_representable(prefix, dtype, layout.0, layout.1)?;
                all_layouts.push(layout);
                if let Some(layer) = repeating_layer_key(prefix) {
                    layers.entry(layer).or_default().push(layout);
                } else {
                    non_repeating_layouts.push(layout);
                }
            }
        }
        let layouts = if layers.is_empty() {
            vec![all_layouts]
        } else {
            layers
                .into_values()
                .chain(non_repeating_layouts.into_iter().map(|layout| vec![layout]))
                .collect()
        };
        layouts
            .iter()
            .map(|layouts| aggregate_pack_factor(layouts, dtype))
            .collect::<Result<Vec<_>>>()
            .map(|factors| factors.into_iter().min().unwrap_or(1))
    }

    pub fn pack_factor_for(
        &self,
        prefix: &str,
        dtype: candle_core::DType,
    ) -> Result<Option<usize>> {
        let prefix = prefix.strip_suffix(".weight").unwrap_or(prefix);
        if !self.contains(&format!("{prefix}.weight.format")) {
            return Ok(None);
        }
        let (logical_elements, packed_bytes) = self.stored_layout(prefix)?;
        ensure_pack_factor_representable(prefix, dtype, logical_elements, packed_bytes)?;
        Ok(Some(block_pack_factor(
            logical_elements,
            dtype,
            packed_bytes,
        )))
    }

    fn stored_layout(&self, prefix: &str) -> Result<(usize, usize)> {
        let format = self.load_format(prefix)?;
        let (logical_elements, weight_suffixes): (usize, &[&str]) = match format {
            QuantizedSerdeType::Unquant => {
                let logical_elements = self.tensor_elem_count(&format!("{prefix}.weight"))?;
                (logical_elements, &["weight"])
            }
            QuantizedSerdeType::Gguf => {
                let logical_elements = self
                    .load_u32_vec(&format!("{prefix}.weight.shape"))?
                    .into_iter()
                    .try_fold(1usize, |elements, dim| elements.checked_mul(dim))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("UQFF logical element overflow".into())
                    })?;
                (logical_elements, &["weight"])
            }
            QuantizedSerdeType::Afq => {
                let group_size =
                    self.load_u8_scalar(&format!("{prefix}.weight.group_size"))? as usize;
                let groups = self.tensor_elem_count(&format!("{prefix}.weight.scales"))?;
                let logical_elements = groups.checked_mul(group_size).ok_or_else(|| {
                    candle_core::Error::Msg("UQFF AFQ logical element overflow".into())
                })?;
                (
                    logical_elements,
                    &["weight", "weight.scales", "weight.biases"],
                )
            }
            QuantizedSerdeType::Hqq => {
                let logical_elements = self
                    .load_u32_vec(&format!("{prefix}.weight.shape"))?
                    .into_iter()
                    .try_fold(1usize, |elements, dim| elements.checked_mul(dim))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("UQFF logical element overflow".into())
                    })?;
                (
                    logical_elements,
                    &["weight", "weight.scales", "weight.zeros"],
                )
            }
            QuantizedSerdeType::Fp8 => {
                let logical_elements = self.tensor_elem_count(&format!("{prefix}.weight"))?;
                (
                    logical_elements,
                    &[
                        "weight",
                        "weight.dequant_w_scale",
                        "weight.dequant_x_scale",
                        "weight.quant_scale",
                    ],
                )
            }
            QuantizedSerdeType::F8Q8 => {
                let logical_elements = self
                    .load_u32_vec(&format!("{prefix}.weight.shape"))?
                    .into_iter()
                    .try_fold(1usize, |elements, dim| elements.checked_mul(dim))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("UQFF logical element overflow".into())
                    })?;
                (logical_elements, &["weight"])
            }
            QuantizedSerdeType::Mxfp4 => {
                let packed_elements = self.tensor_elem_count(&format!("{prefix}.weight"))?;
                let logical_elements = packed_elements
                    .checked_mul(u8::BITS as usize / crate::mxfp4::N_BITS)
                    .ok_or_else(|| {
                        candle_core::Error::Msg("UQFF MXFP4 logical element overflow".into())
                    })?;
                (logical_elements, &["weight", "weight.scales"])
            }
        };
        let packed_bytes = weight_suffixes.iter().try_fold(0usize, |bytes, suffix| {
            bytes
                .checked_add(
                    self.artifacts
                        .get(&format!("{prefix}.{suffix}"))?
                        .data()
                        .len(),
                )
                .ok_or_else(|| candle_core::Error::Msg("UQFF resident byte overflow".into()))
        })?;
        Ok((logical_elements, packed_bytes))
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
        let shape_name = format!("{key}.weight.shape");
        let weight_rank = if self.contains(&shape_name) {
            self.load_u32_vec(&shape_name)?.len()
        } else {
            self.tensor_dims(&format!("{key}.weight"))?.len()
        };
        if weight_rank == 3
            && matches!(
                format,
                QuantizedSerdeType::Hqq | QuantizedSerdeType::Fp8 | QuantizedSerdeType::F8Q8
            )
        {
            candle_core::bail!(
                "UQFF layer `{key}` uses {format:?} for stacked expert weights, but that format does not support stacked expert gather."
            );
        }
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
            BiasShard::Narrow { dim, start, len } => {
                let name = format!("{key}.bias");
                if !self.contains(&name) {
                    return Ok(None);
                }
                Ok(Some(
                    self.artifacts
                        .load(&name, &Device::Cpu, None)?
                        .narrow(dim, start, len)?
                        .contiguous()?
                        .to_device(device)?,
                ))
            }
        }
    }

    pub(crate) fn tensor_dims(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.artifacts.get(name)?.shape().to_vec())
    }

    fn tensor_elem_count(&self, name: &str) -> Result<usize> {
        self.artifacts
            .get(name)?
            .shape()
            .iter()
            .try_fold(1usize, |elements, dim| elements.checked_mul(*dim))
            .ok_or_else(|| candle_core::Error::Msg("UQFF tensor element overflow".into()))
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

impl QuantizedWeightSource for UqffReader {
    fn contains(&self, name: &str) -> bool {
        UqffReader::contains(self, name)
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        UqffReader::load_linear(self, key, device, shard)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        UqffReader::load_optional_tensor(self, name, device)
    }

    fn shard_alignment(&self, key: &str) -> Result<usize> {
        UqffReader::shard_alignment(self, key)
    }

    fn pack_factor(&self, dtype: candle_core::DType) -> Result<usize> {
        UqffReader::pack_factor(self, dtype)
    }

    fn pack_factor_for(&self, key: &str, dtype: candle_core::DType) -> Result<Option<usize>> {
        UqffReader::pack_factor_for(self, key, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{uqff_version_tensors, IsqType, QuantizedSerdeType, UqffTensor};
    use candle_core::{
        quantized::{GgmlDType, QTensor},
        DType,
    };

    fn write_afq_layer_with(
        tensors: &mut Vec<UqffTensor>,
        prefix: &str,
        bits: u8,
        group_size: u8,
        metadata_dtype: DType,
    ) {
        let packed_elements = group_size as usize * bits as usize / u32::BITS as usize;
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Afq as u8,
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), bits),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.group_size"), group_size),
            UqffTensor::from_tensor(
                format!("{prefix}.weight"),
                &Tensor::zeros(packed_elements, DType::U32, &Device::Cpu).unwrap(),
            )
            .unwrap(),
            UqffTensor::from_tensor(
                format!("{prefix}.weight.scales"),
                &Tensor::zeros(1, metadata_dtype, &Device::Cpu).unwrap(),
            )
            .unwrap(),
            UqffTensor::from_tensor(
                format!("{prefix}.weight.biases"),
                &Tensor::zeros(1, metadata_dtype, &Device::Cpu).unwrap(),
            )
            .unwrap(),
        ]);
    }

    fn write_afq_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, bits: u8) {
        write_afq_layer_with(tensors, prefix, bits, 64, DType::BF16);
    }

    fn write_hqq_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, bits: u8, group_size: u32) {
        let packed_bytes = group_size as usize * bits as usize / u8::BITS as usize;
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Hqq as u8,
            ),
            UqffTensor::from_raw_u8(
                format!("{prefix}.weight"),
                vec![0; packed_bytes],
                vec![packed_bytes],
            ),
            UqffTensor::from_tensor(
                format!("{prefix}.weight.scales"),
                &Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap(),
            UqffTensor::from_tensor(
                format!("{prefix}.weight.zeros"),
                &Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap(),
            UqffTensor::from_u32_vec(
                format!("{prefix}.weight.shape"),
                vec![1, group_size],
                vec![2],
            ),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.bits"), bits),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.group_size"), group_size),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.axis"), 0),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.optimization_steps"), 10),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.round_zeros"), 0),
            UqffTensor::from_u8_scalar(format!("{prefix}.weight.channel_wise"), 1),
        ]);
    }

    fn write_gguf_float_layer(tensors: &mut Vec<UqffTensor>, prefix: &str) {
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Gguf as u8,
            ),
            UqffTensor::from_u32_scalar(format!("{prefix}.weight.dtype"), 1),
            UqffTensor::from_raw_u8(format!("{prefix}.weight"), vec![0; 4], vec![4]),
            UqffTensor::from_u32_vec(format!("{prefix}.weight.shape"), vec![1, 2], vec![2]),
        ]);
    }

    fn write_gguf_quant_layer(tensors: &mut Vec<UqffTensor>, prefix: &str, dtype: GgmlDType) {
        let weight = Tensor::zeros((1, dtype.block_size()), DType::F32, &Device::Cpu).unwrap();
        let weight = QTensor::quantize(&weight, dtype).unwrap();
        let layer = GgufMatMul::from_qtensor(weight, None);
        tensors.extend(
            layer
                .serialize_uqff(prefix, IsqType::try_from(dtype).unwrap())
                .unwrap(),
        );
    }

    fn write_unquant_layer(tensors: &mut Vec<UqffTensor>, prefix: &str) {
        tensors.extend([
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Unquant as u8,
            ),
            UqffTensor::from_tensor(
                format!("{prefix}.weight"),
                &Tensor::zeros((2, 32), DType::BF16, &Device::Cpu).unwrap(),
            )
            .unwrap(),
        ]);
    }

    #[test]
    fn uqff_reader_accepts_unique_tensors_split_across_shards() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("q4k-0.uqff");
        let second = dir.path().join("q4k-1.uqff");
        let mut first_tensors = uqff_version_tensors();
        first_tensors.push(UqffTensor::from_u8_scalar("first.tensor", 1));
        let mut second_tensors = uqff_version_tensors();
        second_tensors.push(UqffTensor::from_u8_scalar("second.tensor", 2));
        safetensors::serialize_to_file(
            first_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &first,
        )
        .unwrap();
        safetensors::serialize_to_file(
            second_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &second,
        )
        .unwrap();

        let reader = UqffReader::open(&[first, second]).unwrap();
        assert!(reader.contains("first.tensor"));
        assert!(reader.contains("second.tensor"));
    }

    #[test]
    fn uqff_reader_rejects_conflicting_versions_across_shards() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("q4k-0.uqff");
        let second = dir.path().join("q4k-1.uqff");
        let first_tensors = uqff_version_tensors();
        let second_tensors = [
            UqffTensor::from_u32_scalar(
                super::super::UQFF_VERSION_MAJOR_KEY,
                super::super::UQFF_VERSION_MAJOR + 1,
            ),
            UqffTensor::from_u32_scalar(
                super::super::UQFF_VERSION_MINOR_KEY,
                super::super::UQFF_VERSION_MINOR,
            ),
            UqffTensor::from_u32_scalar(
                super::super::UQFF_VERSION_PATCH_KEY,
                super::super::UQFF_VERSION_PATCH,
            ),
        ];
        safetensors::serialize_to_file(
            first_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &first,
        )
        .unwrap();
        safetensors::serialize_to_file(
            second_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &second,
        )
        .unwrap();

        let error = UqffReader::open(&[first, second])
            .err()
            .expect("conflicting UQFF versions must be rejected")
            .to_string();
        assert!(error.contains("Conflicting UQFF version tensor"), "{error}");
    }

    #[test]
    fn uqff_reader_rejects_malformed_version_copies() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("q4k-0.uqff");
        let second = dir.path().join("q4k-1.uqff");
        let first_tensors = uqff_version_tensors();
        let second_tensors = [
            UqffTensor::from_u8_scalar(super::super::UQFF_VERSION_MAJOR_KEY, 1),
            UqffTensor::from_u32_scalar(
                super::super::UQFF_VERSION_MINOR_KEY,
                super::super::UQFF_VERSION_MINOR,
            ),
            UqffTensor::from_u32_scalar(
                super::super::UQFF_VERSION_PATCH_KEY,
                super::super::UQFF_VERSION_PATCH,
            ),
        ];
        safetensors::serialize_to_file(
            first_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &first,
        )
        .unwrap();
        safetensors::serialize_to_file(
            second_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &second,
        )
        .unwrap();

        let error = UqffReader::open(&[first, second])
            .err()
            .expect("malformed UQFF version copies must be rejected")
            .to_string();
        assert!(error.contains("must be a scalar U32"), "{error}");
    }

    #[test]
    fn uqff_reader_rejects_duplicate_tensor_keys_across_shards() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("q4k-0.uqff");
        let second = dir.path().join("q4k-1.uqff");
        let mut first_tensors = uqff_version_tensors();
        first_tensors.push(UqffTensor::from_u8_scalar("duplicate.tensor", 1));
        let second_tensors = [UqffTensor::from_u8_scalar("duplicate.tensor", 2)];
        safetensors::serialize_to_file(
            first_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &first,
        )
        .unwrap();
        safetensors::serialize_to_file(
            second_tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &second,
        )
        .unwrap();

        let error = UqffReader::open(&[first.clone(), second.clone()])
            .err()
            .expect("duplicate UQFF keys must be rejected")
            .to_string();
        assert!(error.contains("duplicate.tensor"), "{error}");
        assert!(error.contains(&first.display().to_string()), "{error}");
        assert!(error.contains(&second.display().to_string()), "{error}");
    }

    #[test]
    fn ggml_uqff_reports_safe_block_pack_factors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("block-overhead.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_quant_layer(&mut tensors, "q4", GgmlDType::Q4K);
        write_gguf_quant_layer(&mut tensors, "q6", GgmlDType::Q6K);
        write_gguf_quant_layer(&mut tensors, "q8", GgmlDType::Q8_0);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        assert_eq!(reader.pack_factor_for("q4", DType::BF16).unwrap(), Some(3));
        assert_eq!(reader.pack_factor_for("q6", DType::BF16).unwrap(), Some(2));
        assert_eq!(reader.pack_factor_for("q8", DType::BF16).unwrap(), Some(1));
        assert_eq!(reader.pack_factor(DType::BF16).unwrap(), 2);
    }

    #[test]
    fn uqff_non_repeating_unquantized_weight_forces_dense_global_factor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed-text-projector.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.0.self_attn.q_proj",
            GgmlDType::Q4K,
        );
        write_unquant_layer(&mut tensors, "model.vision_tower.projector");
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        assert_eq!(reader.pack_factor(DType::BF16).unwrap(), 1);
        assert_eq!(
            reader
                .pack_factor_for("model.layers.0.self_attn.q_proj", DType::BF16)
                .unwrap(),
            Some(3)
        );
        assert_eq!(
            reader
                .pack_factor_for("model.vision_tower.projector", DType::BF16)
                .unwrap(),
            Some(1)
        );
    }

    #[test]
    fn uqff_pack_factors_use_exact_resident_storage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("resident-storage.uqff");
        let mut tensors = uqff_version_tensors();
        write_afq_layer_with(&mut tensors, "model.layers.0.afq", 4, 32, DType::F32);
        write_hqq_layer(&mut tensors, "model.layers.0.hqq", 4, 32);

        let fp8_scale = Tensor::zeros((), DType::F32, &Device::Cpu).unwrap();
        let fp8 = FP8Linear::from_parts(
            Tensor::zeros((1, 64), DType::F8E4M3, &Device::Cpu).unwrap(),
            None,
            fp8_scale.clone(),
            fp8_scale.clone(),
            fp8_scale,
            DType::F8E4M3,
        );
        tensors.extend(
            fp8.serialize_uqff("model.layers.0.fp8", IsqType::F8E4M3)
                .unwrap(),
        );

        let f8q8 = F8Q8Linear::from_weight(
            &Tensor::zeros((1, 32), DType::F32, &Device::Cpu).unwrap(),
            None,
        )
        .unwrap();
        tensors.extend(
            f8q8.serialize_uqff("model.layers.0.f8q8", IsqType::F8Q8)
                .unwrap(),
        );

        let mxfp4 = MXFP4Layer::from_parts(
            Tensor::zeros((1, 16), DType::U8, &Device::Cpu).unwrap(),
            Tensor::zeros((1, 1), DType::U8, &Device::Cpu).unwrap(),
            None,
        );
        tensors.extend(
            mxfp4
                .serialize_uqff("model.layers.0.mxfp4", IsqType::MXFP4)
                .unwrap(),
        );

        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();
        let reader = UqffReader::open(&[path]).unwrap();

        for (prefix, bf16_factor, f32_factor, logical_elements, resident_bytes) in [
            ("model.layers.0.afq", 2, 5, 32, 24),
            ("model.layers.0.hqq", 2, 5, 32, 24),
            ("model.layers.0.fp8", 1, 3, 64, 76),
            ("model.layers.0.f8q8", 1, 3, 32, 33),
            ("model.layers.0.mxfp4", 3, 6, 32, 17),
        ] {
            for (dtype, expected_factor) in [(DType::BF16, bf16_factor), (DType::F32, f32_factor)] {
                let factor = reader.pack_factor_for(prefix, dtype).unwrap().unwrap();
                assert_eq!(factor, expected_factor, "{prefix} {dtype:?}");
                assert!(
                    logical_elements / factor * dtype.size_in_bytes() >= resident_bytes,
                    "{prefix} {dtype:?}"
                );
            }
        }
        assert_eq!(reader.pack_factor(DType::BF16).unwrap(), 2);
    }

    #[test]
    fn mixed_uqff_uses_conservative_default_and_exact_per_key_factors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed.uqff");
        let mut tensors = uqff_version_tensors();
        write_afq_layer(&mut tensors, "model.layers.0.self_attn.q_proj", 4);
        write_afq_layer(&mut tensors, "model.layers.1.self_attn.q_proj", 4);
        write_afq_layer(&mut tensors, "model.embed_tokens", 6);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        assert_eq!(
            reader.pack_factor(candle_core::DType::BF16).unwrap(),
            IsqType::AFQ6.pack_factor(candle_core::DType::BF16)
        );
        assert_eq!(
            reader
                .pack_factor_for("model.embed_tokens.weight", candle_core::DType::BF16)
                .unwrap(),
            Some(IsqType::AFQ6.pack_factor(candle_core::DType::BF16))
        );
        assert_eq!(
            reader
                .pack_factor_for("missing", candle_core::DType::BF16)
                .unwrap(),
            None
        );

        let tied_path = dir.path().join("tied.uqff");
        let mut tensors = uqff_version_tensors();
        write_afq_layer(&mut tensors, "model.layers.0.self_attn.q_proj", 4);
        write_afq_layer(&mut tensors, "model.embed_tokens", 8);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &tied_path,
        )
        .unwrap();
        let reader = UqffReader::open(&[tied_path]).unwrap();
        assert_eq!(
            reader.pack_factor(candle_core::DType::BF16).unwrap(),
            IsqType::AFQ8.pack_factor(candle_core::DType::BF16)
        );
    }

    #[test]
    fn mixed_q_uqff_uses_conservative_default_and_exact_per_key_factors() {
        let dir = tempfile::tempdir().unwrap();
        let q4_path = dir.path().join("mixed-q4.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.0.self_attn.q_proj",
            GgmlDType::Q4K,
        );
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.1.self_attn.q_proj",
            GgmlDType::Q4K,
        );
        write_gguf_quant_layer(&mut tensors, "model.embed_tokens", GgmlDType::Q6K);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &q4_path,
        )
        .unwrap();

        let reader = UqffReader::open(&[q4_path]).unwrap();
        assert_eq!(
            reader.pack_factor(DType::BF16).unwrap(),
            IsqType::Q6K.pack_factor(DType::BF16)
        );
        assert_eq!(
            reader
                .pack_factor_for("model.embed_tokens.weight", DType::BF16)
                .unwrap(),
            Some(IsqType::Q6K.pack_factor(DType::BF16))
        );

        let q6_path = dir.path().join("mixed-q6.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.0.self_attn.q_proj",
            GgmlDType::Q6K,
        );
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.1.self_attn.q_proj",
            GgmlDType::Q6K,
        );
        write_gguf_quant_layer(&mut tensors, "lm_head", GgmlDType::Q8_0);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &q6_path,
        )
        .unwrap();

        let reader = UqffReader::open(&[q6_path]).unwrap();
        assert_eq!(
            reader.pack_factor(DType::BF16).unwrap(),
            IsqType::Q8_0.pack_factor(DType::BF16)
        );
        assert_eq!(
            reader
                .pack_factor_for("lm_head.weight", DType::BF16)
                .unwrap(),
            Some(IsqType::Q8_0.pack_factor(DType::BF16))
        );
    }

    #[test]
    fn mixed_q2_and_unquantized_uqff_sizes_for_dense_trunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mixed-q2-unquant.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_quant_layer(
            &mut tensors,
            "model.layers.0.mlp.experts.gate_proj",
            GgmlDType::Q2K,
        );
        write_unquant_layer(&mut tensors, "model.layers.0.self_attn.q_proj");
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        assert_eq!(reader.pack_factor(DType::BF16).unwrap(), 3);
        assert_eq!(
            reader
                .pack_factor_for("model.layers.0.mlp.experts.gate_proj", DType::BF16)
                .unwrap(),
            Some(IsqType::Q2K.pack_factor(DType::BF16))
        );
        assert_eq!(
            reader
                .pack_factor_for("model.layers.0.self_attn.q_proj", DType::BF16)
                .unwrap(),
            Some(1)
        );
    }

    #[test]
    fn uqff_rejects_stacked_backend_without_gather_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stacked-f8q8.uqff");
        let mut tensors = uqff_version_tensors();
        tensors.extend([
            UqffTensor::from_u8_scalar(
                "experts.gate.weight.format",
                QuantizedSerdeType::F8Q8 as u8,
            ),
            UqffTensor::from_raw_u8("experts.gate.weight", vec![0], vec![1]),
            UqffTensor::from_u32_scalar("experts.gate.weight.num_blocks", 1),
            UqffTensor::from_u32_vec("experts.gate.weight.shape", vec![2, 4, 32], vec![3]),
        ]);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        let error = match reader.load_linear("experts.gate", &Device::Cpu, Shard::default()) {
            Ok(_) => panic!("stacked F8Q8 must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            error.contains("does not support stacked expert gather"),
            "{error}"
        );
    }

    #[test]
    fn gguf_float_fallback_has_dense_pack_factor() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("float-fallback.uqff");
        let mut tensors = uqff_version_tensors();
        write_gguf_float_layer(&mut tensors, "model.embed_tokens");
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        assert_eq!(
            reader
                .pack_factor_for("model.embed_tokens.weight", candle_core::DType::BF16)
                .unwrap(),
            Some(1)
        );
    }

    #[test]
    fn uqff_rejects_storage_expansion_that_a_pack_factor_cannot_represent() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("expanded-storage.uqff");
        let mut tensors = uqff_version_tensors();
        tensors.extend([
            UqffTensor::from_u8_scalar(
                "model.layers.0.proj.weight.format",
                QuantizedSerdeType::Unquant as u8,
            ),
            UqffTensor::from_tensor(
                "model.layers.0.proj.weight",
                &Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap(),
            )
            .unwrap(),
        ]);
        safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .unwrap();

        let reader = UqffReader::open(&[path]).unwrap();
        for error in [
            reader
                .pack_factor_for("model.layers.0.proj", DType::BF16)
                .unwrap_err(),
            reader.pack_factor(DType::BF16).unwrap_err(),
        ] {
            assert!(error.to_string().contains("explicit device mapping"));
        }
    }
}
