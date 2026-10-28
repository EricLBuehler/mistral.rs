use anyhow::{bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::GgufArchive;
use serde_json::{json, Value as JsonValue};

use super::{
    multimodal_binding_utils::{
        metadata_string, metadata_usize, validate_architecture, validate_projector, TensorInventory,
    },
    normal_config::{synthesize_normal_config_value, validate_lfm3_metadata},
};
use crate::{
    pipeline::sanitize_quantized_weight_source_config, vision_models::lfm3_vl::Config,
    NormalLoaderType,
};

// llama.cpp's conversion/lfm3.py enforces these format constants instead of writing metadata.
const VISION_ROPE_THETA: f64 = 1_000_000.0;
const PROJECTOR_NORM_EPS: f64 = 1e-6;
const PROJECTOR_ACTIVATION: &str = "gelu";
const SPATIAL_CONV_KERNEL: usize = 3;
const PATCH_EMBEDDING: &str = "v.patch_embd.weight";
const SECOND_TEMPORAL_PATCH: &str = "v.patch_embd.weight.1";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";
const BILINEAR: usize = 2;
const BICUBIC: usize = 3;

pub(crate) fn prepare_lfm3_vl_config(
    external_config: Option<&str>,
    archive: &GgufArchive,
) -> Result<String> {
    validate_architecture(archive, "lfm3")?;
    validate_projector(archive, "lfm3vl")?;
    validate_lfm3_metadata(archive.metadata())?;
    let config = match external_config {
        Some(config) => sanitize_quantized_weight_source_config(config)?,
        None => {
            let tensor_names = archive.tensors().keys().cloned().collect::<Vec<_>>();
            let text = synthesize_normal_config_value(
                &NormalLoaderType::Lfm3,
                archive.metadata(),
                &tensor_names,
            )?;
            let vision = vision_config(archive)?;
            let inventory = TensorInventory::from_archive(archive);
            let [projector_hidden, merged_hidden] = inventory.shape("mm.0.weight")? else {
                bail!("LFM3-VL projector `mm.0.weight` must have rank 2");
            };
            let [text_hidden, projector_input] = inventory.shape("mm.2.weight")? else {
                bail!("LFM3-VL projector `mm.2.weight` must have rank 2");
            };
            let hidden = metadata_usize(archive, "clip.vision.embedding_length")?;
            let merge = metadata_usize(archive, "clip.vision.spatial_merge_size")?;
            let merged_size = hidden
                .checked_mul(merge)
                .and_then(|size| size.checked_mul(merge))
                .context("LFM3-VL merged vision dimension overflow")?;
            ensure!(
                *merged_hidden == merged_size
                    && projector_hidden == projector_input
                    && json!(text_hidden) == text["hidden_size"],
                "LFM3-VL projector tensor dimensions disagree with the vision/text configuration"
            );
            let projector_bias = inventory.contains("mm.0.bias");
            ensure!(
                projector_bias == inventory.contains("mm.2.bias"),
                "LFM3-VL projector linear layers must use matching bias settings"
            );
            serde_json::to_string(&json!({
                "architectures": ["Lfm3VlForConditionalGeneration"],
                "model_type": "lfm3_vl",
                "vision_config": vision,
                "image_token_id": token_id(archive, "<|image|>")?,
                "video_token_id": token_id(archive, "<|video|>")?,
                "vision_start_token_id": token_id(archive, "<|vision_start|>")?,
                "vision_end_token_id": token_id(archive, "<|vision_end|>")?,
                "projector_hidden_act": PROJECTOR_ACTIVATION,
                "projector_hidden_size": projector_hidden,
                "projector_bias": projector_bias,
                "projector_layer_norm_eps": PROJECTOR_NORM_EPS,
                "downsample_factor": merge,
                "tie_word_embeddings": text["tie_word_embeddings"],
                "text_config": text,
            }))?
        }
    };
    let _: Config = serde_json::from_str(&config)
        .context("LFM3-VL GGUF configuration is incompatible with the native loader")?;
    Ok(config)
}

fn vision_config(archive: &GgufArchive) -> Result<JsonValue> {
    let inventory = TensorInventory::from_archive(archive);
    let hidden = metadata_usize(archive, "clip.vision.embedding_length")?;
    let patch_size = metadata_usize(archive, "clip.vision.patch_size")?;
    let [patch_hidden, channels, patch_height, patch_width] = inventory.shape(PATCH_EMBEDDING)?
    else {
        bail!("LFM3-VL patch embedding must have rank 4");
    };
    ensure!(
        *patch_hidden == hidden && *patch_height == patch_size && *patch_width == patch_size,
        "LFM3-VL patch embedding dimensions disagree with vision metadata"
    );
    let temporal_patch_size = if inventory.contains(SECOND_TEMPORAL_PATCH) {
        ensure!(
            inventory.shape(SECOND_TEMPORAL_PATCH)? == inventory.shape(PATCH_EMBEDDING)?,
            "LFM3-VL temporal patch embedding shapes differ"
        );
        2
    } else {
        1
    };
    let layers = inventory.require_layers("v.blk.", "LFM3-VL")?;
    let num_hidden_layers = metadata_usize(archive, "clip.vision.block_count")?;
    ensure!(
        layers.len() == num_hidden_layers,
        "LFM3-VL vision layer count disagrees with tensor inventory"
    );
    let mut layer_types = Vec::with_capacity(num_hidden_layers);
    let mut conv_kernel_size = None;
    for layer in layers {
        let conv = format!("v.blk.{layer}.conv2d_spatial.weight");
        let attention = format!("v.blk.{layer}.attn_q.weight");
        match (inventory.contains(&conv), inventory.contains(&attention)) {
            (true, false) => {
                let [channels, groups, height, width] = inventory.shape(&conv)? else {
                    bail!("LFM3-VL spatial convolution `{conv}` must have rank 4");
                };
                ensure!(
                    *channels == hidden && *groups == 1 && height == width,
                    "LFM3-VL spatial convolution `{conv}` must be square and depthwise"
                );
                ensure!(
                    conv_kernel_size.is_none_or(|size| size == *height),
                    "LFM3-VL spatial convolutions must have a common kernel size"
                );
                conv_kernel_size = Some(*height);
                layer_types.push("conv");
            }
            (false, true) => layer_types.push("full_attention"),
            _ => bail!("LFM3-VL vision layer {layer} must contain exactly one operator"),
        }
    }
    Ok(json!({
        "hidden_size": hidden,
        "intermediate_size": metadata_usize(archive, "clip.vision.feed_forward_length")?,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": metadata_usize(archive, "clip.vision.attention.head_count")?,
        "num_key_value_heads": metadata_usize(archive, "clip.vision.attention.head_count_kv")?,
        "in_channels": channels,
        "patch_size": patch_size,
        "temporal_patch_size": temporal_patch_size,
        "spatial_merge_size": metadata_usize(archive, "clip.vision.spatial_merge_size")?,
        "conv_kernel_size": conv_kernel_size.unwrap_or(SPATIAL_CONV_KERNEL),
        "norm_eps": metadata_f64(archive, "clip.vision.attention.layer_norm_epsilon")?,
        "rope_parameters": {"rope_theta": VISION_ROPE_THETA},
        "layer_types": layer_types,
    }))
}

pub(crate) fn lfm3_vl_processor_config(archive: &GgufArchive) -> Result<String> {
    validate_projector(archive, "lfm3vl")?;
    let vision = vision_config(archive)?;
    let resample = match metadata_string(archive, "clip.vision.image_resize_algo")? {
        Some("bilinear") => BILINEAR,
        Some("bicubic") => BICUBIC,
        value => bail!("unsupported LFM3-VL GGUF image resize algorithm {value:?}"),
    };
    let shared = json!({
        "patch_size": vision["patch_size"],
        "merge_size": vision["spatial_merge_size"],
        "temporal_patch_size": vision["temporal_patch_size"],
        "image_mean": rgb_array(archive, "clip.vision.image_mean")?,
        "image_std": rgb_array(archive, "clip.vision.image_std")?,
        "resample": resample,
    });
    let mut image = shared.clone();
    image["min_pixels"] = json!(metadata_usize(archive, "clip.vision.image_min_pixels")?);
    image["max_pixels"] = json!(metadata_usize(archive, "clip.vision.image_max_pixels")?);
    // Video sizing and frame sampling use the same format defaults as the native LFM3-VL loader.
    Ok(serde_json::to_string(&json!({
        "processor_class": "Lfm3VlProcessor",
        "image_processor": image,
        "video_processor": shared,
    }))?)
}

fn metadata_f64(archive: &GgufArchive, key: &str) -> Result<f64> {
    match archive.metadata_value(key) {
        Some(Value::F32(value)) => Ok(f64::from(*value)),
        Some(Value::F64(value)) => Ok(*value),
        _ => bail!("GGUF metadata `{key}` must be a float"),
    }
}

fn rgb_array(archive: &GgufArchive, key: &str) -> Result<[f64; 3]> {
    let Some(Value::Array(values)) = archive.metadata_value(key) else {
        bail!("GGUF metadata `{key}` must be a float array");
    };
    let values = values
        .iter()
        .map(|value| match value {
            Value::F32(value) => Ok(f64::from(*value)),
            Value::F64(value) => Ok(*value),
            _ => bail!("GGUF metadata `{key}` must contain floats"),
        })
        .collect::<Result<Vec<_>>>()?;
    values
        .try_into()
        .map_err(|_| anyhow::anyhow!("GGUF metadata `{key}` must contain three channels"))
}

fn token_id(archive: &GgufArchive, token: &str) -> Result<usize> {
    let Some(Value::Array(tokens)) = archive.metadata_value(TOKENS_KEY) else {
        bail!("GGUF metadata `{TOKENS_KEY}` must contain the vocabulary");
    };
    tokens
        .iter()
        .position(|value| matches!(value, Value::String(value) if value == token))
        .with_context(|| format!("LFM3-VL GGUF vocabulary has no `{token}` token"))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use candle_core::{
        quantized::{gguf_file, GgmlDType, QTensor},
        DType, Device, Tensor,
    };
    use tempfile::NamedTempFile;

    use super::*;
    use crate::vision_models::{
        lfm3_vl::config::LayerType, preprocessor_config::PreProcessorConfig,
    };

    #[test]
    fn synthesizes_native_config_and_nested_processor_without_hf_assets() -> Result<()> {
        let (_file, archive) = archive("lfm3vl")?;
        let config: Config = serde_json::from_str(&prepare_lfm3_vl_config(None, &archive)?)?;
        assert_eq!(config.text_config.num_hidden_layers, 2);
        assert_eq!(config.vision_config.hidden_size, 8);
        assert_eq!(config.vision_config.temporal_patch_size, 2);
        assert_eq!(config.vision_config.conv_kernel_size, 3);
        assert_eq!(
            config.vision_config.layer_types,
            [LayerType::Conv, LayerType::Attention]
        );
        assert_eq!(config.projector_hidden_size, 16);
        assert!(config.projector_bias);
        assert_eq!(config.image_token_id, 3);
        assert_eq!(config.video_token_id, 1);
        assert_eq!(config.vision_start_token_id, 4);
        assert_eq!(config.vision_end_token_id, 2);
        let processor =
            PreProcessorConfig::from_processor_config_json(&lfm3_vl_processor_config(&archive)?)?;
        assert_eq!(processor.min_pixels, Some(64));
        assert_eq!(processor.max_pixels, Some(4096));
        assert_eq!(processor.image_mean, Some([0.25; 3]));
        assert_eq!(processor.resampling, Some(BILINEAR));
        let video = processor.video.context("missing video settings")?;
        assert_eq!(video.image_mean, processor.image_mean);
        assert_eq!(video.image_std, processor.image_std);
        assert_eq!(video.resampling, processor.resampling);
        assert_eq!(video.min_pixels, None);
        assert_eq!(video.max_pixels, None);
        Ok(())
    }

    #[test]
    fn preserves_explicit_config_and_removes_source_quantization() -> Result<()> {
        let (_file, archive) = archive("lfm3vl")?;
        let mut external: JsonValue =
            serde_json::from_str(&prepare_lfm3_vl_config(None, &archive)?)?;
        external["text_config"]["max_position_embeddings"] = json!(512);
        external["text_config"]["quantization_config"] = json!({"quant_method": "fp8"});
        let prepared = prepare_lfm3_vl_config(Some(&external.to_string()), &archive)?;
        let config: Config = serde_json::from_str(&prepared)?;
        assert_eq!(config.text_config.max_position_embeddings, 512);
        assert!(config.text_config.quantization_config.is_none());
        Ok(())
    }

    #[test]
    fn rejects_other_projector_formats() -> Result<()> {
        let (_file, archive) = archive("lfm2")?;
        assert!(prepare_lfm3_vl_config(None, &archive).is_err());
        assert!(lfm3_vl_processor_config(&archive).is_err());
        Ok(())
    }

    fn archive(projector: &str) -> Result<(NamedTempFile, GgufArchive)> {
        let metadata = [
            ("general.architecture", Value::String("lfm3".into())),
            ("lfm3.block_count", Value::U32(2)),
            ("lfm3.context_length", Value::U32(64)),
            ("lfm3.embedding_length", Value::U32(8)),
            ("lfm3.feed_forward_length", Value::U32(12)),
            ("lfm3.attention.head_count", Value::U32(2)),
            (
                "lfm3.attention.head_count_kv",
                Value::Array(vec![Value::U32(0), Value::U32(1)]),
            ),
            (
                "lfm3.attention.sliding_window_pattern",
                Value::Array(vec![Value::Bool(false), Value::Bool(false)]),
            ),
            ("lfm3.attention.sliding_window", Value::U32(8)),
            ("lfm3.attention.key_length", Value::U32(4)),
            ("lfm3.attention.value_length", Value::U32(4)),
            ("lfm3.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("lfm3.shortconv.l_cache", Value::U32(3)),
            ("lfm3.rope.dimension_count", Value::U32(4)),
            ("lfm3.rope.freq_base", Value::F32(1_000_000.0)),
            ("lfm3.rope.freq_base_swa", Value::F32(10_000.0)),
            (
                "clip.vision.projector_type",
                Value::String(projector.into()),
            ),
            ("clip.vision.embedding_length", Value::U32(8)),
            ("clip.vision.feed_forward_length", Value::U32(8)),
            ("clip.vision.block_count", Value::U32(2)),
            ("clip.vision.attention.head_count", Value::U32(2)),
            ("clip.vision.attention.head_count_kv", Value::U32(1)),
            ("clip.vision.attention.layer_norm_epsilon", Value::F32(1e-5)),
            ("clip.vision.patch_size", Value::U32(2)),
            ("clip.vision.spatial_merge_size", Value::U32(2)),
            ("clip.vision.image_min_pixels", Value::U32(64)),
            ("clip.vision.image_max_pixels", Value::U32(4096)),
            (
                "clip.vision.image_resize_algo",
                Value::String("bilinear".into()),
            ),
            (
                "clip.vision.image_mean",
                Value::Array(vec![Value::F32(0.25); 3]),
            ),
            (
                "clip.vision.image_std",
                Value::Array(vec![Value::F32(0.5); 3]),
            ),
            (
                TOKENS_KEY,
                Value::Array(
                    [
                        "unused",
                        "<|video|>",
                        "<|vision_end|>",
                        "<|image|>",
                        "<|vision_start|>",
                    ]
                    .map(|token| Value::String(token.into()))
                    .to_vec(),
                ),
            ),
        ];
        let tensors = [
            ("token_embd.weight", vec![5, 8]),
            ("blk.0.shortconv.conv.weight", vec![8, 3]),
            ("blk.0.shortconv.in_proj.weight", vec![24, 8]),
            ("blk.1.attn_q.weight", vec![8, 8]),
            (PATCH_EMBEDDING, vec![8, 3, 2, 2]),
            (SECOND_TEMPORAL_PATCH, vec![8, 3, 2, 2]),
            ("v.blk.0.conv2d_spatial.weight", vec![8, 1, 3, 3]),
            ("v.blk.1.attn_q.weight", vec![8, 8]),
            ("mm.0.weight", vec![16, 32]),
            ("mm.0.bias", vec![16]),
            ("mm.2.weight", vec![8, 16]),
            ("mm.2.bias", vec![8]),
        ]
        .into_iter()
        .map(|(name, shape)| {
            Ok((
                name,
                QTensor::quantize(
                    &Tensor::zeros(shape, DType::F32, &Device::Cpu)?,
                    GgmlDType::F32,
                )?,
            ))
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
        let mut file = NamedTempFile::new()?;
        gguf_file::write(
            file.as_file_mut(),
            &metadata
                .iter()
                .map(|(name, value)| (*name, value))
                .collect::<Vec<_>>(),
            &tensors
                .iter()
                .map(|(name, tensor)| (*name, tensor))
                .collect::<Vec<_>>(),
        )?;
        file.as_file_mut().flush()?;
        let archive = GgufArchive::open_file(file.path())?;
        Ok((file, archive))
    }
}
