use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::DType;
use safetensors::{tensor::Dtype as SafeDtype, tensor::TensorView};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CheckpointDeviceMapSizes {
    pub(crate) layer_sizes_in_bytes: Vec<usize>,
    pub(crate) non_mapped_size_in_bytes: usize,
    pub(crate) total_model_size_in_bytes: usize,
}

const NON_MAPPED_COMPONENTS: &[&str] = &[
    "audio",
    "audio_model",
    "audio_tower",
    "image",
    "mtp",
    "visual",
    "vision",
    "vision_encoder",
    "vision_model",
    "vision_tower",
];

pub(crate) fn standard_layer_index(tensor_name: &str) -> Option<usize> {
    let components = tensor_name.split('.').collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| NON_MAPPED_COMPONENTS.contains(component))
    {
        return None;
    }
    components.windows(2).find_map(|window| {
        (window[0] == "layers")
            .then(|| window[1].parse::<usize>().ok())
            .flatten()
    })
}

fn is_float(dtype: SafeDtype) -> bool {
    matches!(
        dtype,
        SafeDtype::F4
            | SafeDtype::F6_E2M3
            | SafeDtype::F6_E3M2
            | SafeDtype::F8_E4M3
            | SafeDtype::F8_E4M3FNUZ
            | SafeDtype::F8_E5M2
            | SafeDtype::F8_E5M2FNUZ
            | SafeDtype::F8_E8M0
            | SafeDtype::F16
            | SafeDtype::BF16
            | SafeDtype::F32
            | SafeDtype::F64
    )
}

fn is_sub_byte_or_fp8(dtype: SafeDtype) -> bool {
    matches!(
        dtype,
        SafeDtype::F4
            | SafeDtype::F6_E2M3
            | SafeDtype::F6_E3M2
            | SafeDtype::F8_E4M3
            | SafeDtype::F8_E4M3FNUZ
            | SafeDtype::F8_E5M2
            | SafeDtype::F8_E5M2FNUZ
            | SafeDtype::F8_E8M0
    )
}

fn runtime_tensor_bytes(name: &str, view: &TensorView<'_>, target_dtype: DType) -> Result<usize> {
    let stored_bytes = view.data().len();
    if is_sub_byte_or_fp8(view.dtype()) || !is_float(view.dtype()) {
        return Ok(stored_bytes);
    }
    let elements = view
        .shape()
        .iter()
        .try_fold(1usize, |elements, dim| elements.checked_mul(*dim))
        .context("checkpoint tensor element count overflow")?;
    let runtime_element_bytes = if name.ends_with(".weight_scale_inv") {
        DType::F32.size_in_bytes()
    } else {
        target_dtype.size_in_bytes()
    };
    let runtime_bytes = elements
        .checked_mul(runtime_element_bytes)
        .context("checkpoint tensor runtime size overflow")?;
    Ok(stored_bytes.max(runtime_bytes))
}

pub(crate) fn checkpoint_runtime_size(
    paths: &[PathBuf],
    target_dtype: DType,
) -> Result<Option<usize>> {
    if paths.is_empty()
        || paths
            .iter()
            .any(|path| path.extension().and_then(|ext| ext.to_str()) != Some("safetensors"))
    {
        return Ok(None);
    }

    let safetensors = unsafe {
        mistralrs_quant::safetensors::MmapedSafetensors::multi_unique(paths)
            .context("reading checkpoint tensor inventory")?
    };
    safetensors
        .tensors()
        .into_iter()
        .try_fold(0usize, |total, (name, view)| {
            total
                .checked_add(runtime_tensor_bytes(&name, &view, target_dtype)?)
                .context("checkpoint runtime size overflow")
        })
        .map(Some)
}

pub(crate) fn checkpoint_device_map_sizes(
    paths: &[PathBuf],
    num_layers: usize,
    target_dtype: DType,
    layer_index: impl Fn(&str) -> Option<usize>,
) -> Result<Option<CheckpointDeviceMapSizes>> {
    if paths.is_empty()
        || paths
            .iter()
            .any(|path| path.extension().and_then(|ext| ext.to_str()) != Some("safetensors"))
    {
        return Ok(None);
    }

    let safetensors = unsafe {
        mistralrs_quant::safetensors::MmapedSafetensors::multi_unique(paths)
            .context("reading checkpoint tensor inventory")?
    };
    let mut layer_sizes_in_bytes = vec![0usize; num_layers];
    let mut non_mapped_size_in_bytes = 0usize;

    for (name, view) in safetensors.tensors() {
        let bytes = runtime_tensor_bytes(&name, &view, target_dtype)?;
        if let Some(layer_idx) = layer_index(&name) {
            let Some(layer_size) = layer_sizes_in_bytes.get_mut(layer_idx) else {
                return Ok(None);
            };
            *layer_size = layer_size
                .checked_add(bytes)
                .context("checkpoint layer size overflow")?;
        } else {
            non_mapped_size_in_bytes = non_mapped_size_in_bytes
                .checked_add(bytes)
                .context("checkpoint non-mapped size overflow")?;
        }
    }

    if layer_sizes_in_bytes.iter().any(|size| *size == 0) {
        return Ok(None);
    }
    let mapped_size = layer_sizes_in_bytes
        .iter()
        .try_fold(0usize, |total, size| total.checked_add(*size))
        .context("checkpoint mapped size overflow")?;
    let total_model_size_in_bytes = mapped_size
        .checked_add(non_mapped_size_in_bytes)
        .context("checkpoint total size overflow")?;
    Ok(Some(CheckpointDeviceMapSizes {
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
    }))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safetensors::{serialize_to_file, tensor::TensorView};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn standard_layers_exclude_auxiliary_models() {
        assert_eq!(
            standard_layer_index("model.language_model.layers.63.mlp.up_proj.weight"),
            Some(63)
        );
        assert_eq!(
            standard_layer_index("model.vision_model.encoder.layers.0.self_attn.weight"),
            None
        );
        assert_eq!(standard_layer_index("mtp.layers.0.mlp.weight"), None);
    }

    #[test]
    fn mixed_fp8_checkpoint_inventory_accounts_runtime_scales() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("weights.safetensors");
        let fp8 = vec![0u8; 16];
        let scale = vec![0u8; 2];
        let norm = vec![0u8; 4];
        let vision = vec![0u8; 8];
        let tensors = HashMap::from([
            (
                "model.layers.0.proj.weight",
                TensorView::new(SafeDtype::F8_E4M3, vec![4, 4], &fp8)?,
            ),
            (
                "model.layers.0.proj.weight_scale_inv",
                TensorView::new(SafeDtype::BF16, vec![1], &scale)?,
            ),
            (
                "model.layers.0.norm.weight",
                TensorView::new(SafeDtype::BF16, vec![2], &norm)?,
            ),
            (
                "model.layers.1.proj.weight",
                TensorView::new(SafeDtype::F8_E4M3, vec![4, 4], &fp8)?,
            ),
            (
                "model.layers.1.proj.weight_scale_inv",
                TensorView::new(SafeDtype::BF16, vec![1], &scale)?,
            ),
            (
                "model.layers.1.norm.weight",
                TensorView::new(SafeDtype::BF16, vec![2], &norm)?,
            ),
            (
                "model.visual.proj.weight",
                TensorView::new(SafeDtype::BF16, vec![4], &vision)?,
            ),
        ]);
        serialize_to_file(tensors, None, &path)?;

        let sizes = checkpoint_device_map_sizes(
            std::slice::from_ref(&path),
            2,
            DType::BF16,
            standard_layer_index,
        )?
        .unwrap();
        assert_eq!(sizes.layer_sizes_in_bytes, [24, 24]);
        assert_eq!(sizes.non_mapped_size_in_bytes, 8);
        assert_eq!(sizes.total_model_size_in_bytes, 56);
        assert_eq!(
            checkpoint_runtime_size(std::slice::from_ref(&path), DType::BF16)?,
            Some(56)
        );
        Ok(())
    }

    #[test]
    fn incomplete_layer_inventory_falls_back() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("weights.safetensors");
        let bytes = vec![0u8; 16];
        let tensors = HashMap::from([(
            "model.layers.0.proj.weight",
            TensorView::new(SafeDtype::F8_E4M3, vec![4, 4], &bytes)?,
        )]);
        serialize_to_file(tensors, None, &path)?;

        assert!(
            checkpoint_device_map_sizes(&[path], 2, DType::BF16, standard_layer_index,)?.is_none()
        );
        Ok(())
    }
}
