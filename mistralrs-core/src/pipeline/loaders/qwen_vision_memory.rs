use anyhow::{Context, Result};
use candle_core::{DType, Device};

use super::{
    activation_memory::{attention_workspace_bytes, AttentionMemoryShape, AttentionWorkspaceKind},
    AutoDeviceMapParams,
};
use crate::vision_models::qwen3_vl::config::VisionConfig;

const RETAINED_HIDDEN_BUFFERS: usize = 3;
const POSITION_HIDDEN_BUFFERS: usize = 10;
const ROTARY_HIDDEN_BUFFERS: usize = 5;
const ROTARY_F32_BUFFERS: usize = 8;
const ROTARY_TABLE_BUFFERS: usize = 4;
const ATTENTION_HIDDEN_BUFFERS: usize = 7;
const MLP_HIDDEN_BUFFERS: usize = 5;
const MLP_INTERMEDIATE_BUFFERS: usize = 2;
const MERGER_HIDDEN_BUFFERS: usize = 3;
const MERGER_OUTPUT_BUFFERS: usize = 2;
const METAL_BUFFER_ROUNDING_BOUND: usize = 2;

fn product(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(1usize, |value, part| value.checked_mul(*part))
        .context("Qwen vision activation size overflow")
}

fn sum(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(0usize, |value, part| value.checked_add(*part))
        .context("Qwen vision activation size overflow")
}

pub(super) fn activation_memory_bytes(
    cfg: &VisionConfig,
    params: &AutoDeviceMapParams,
    dtype: DType,
    device: &Device,
) -> Result<usize> {
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let fused = if device.is_cuda() {
        crate::using_flash_attn()
            && matches!(dtype, DType::BF16 | DType::F16)
            && crate::attention::flash_backend_supports_sdpa(head_dim, false, false)
    } else {
        device.is_metal() && matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
    };
    let bytes = vision_activation_bytes(cfg, params, dtype, fused)?;
    product(&[
        bytes,
        if device.is_metal() {
            METAL_BUFFER_ROUNDING_BOUND
        } else {
            1
        },
    ])
}

fn vision_activation_bytes(
    cfg: &VisionConfig,
    params: &AutoDeviceMapParams,
    dtype: DType,
    fused: bool,
) -> Result<usize> {
    let AutoDeviceMapParams::Multimodal {
        max_batch_size,
        max_image_shape,
        max_num_images,
        ..
    } = params
    else {
        anyhow::bail!("Expected multimodal AutoDeviceMapParams for this model!")
    };
    let segment_tokens = product(&[
        max_image_shape.0 / cfg.patch_size,
        max_image_shape.1 / cfg.patch_size,
    ])?;
    let tokens = product(&[segment_tokens, *max_batch_size, *max_num_images])?;
    if tokens == 0 {
        return Ok(0);
    }
    let dtype_bytes = dtype.size_in_bytes();
    let hidden = product(&[tokens, cfg.hidden_size, dtype_bytes])?;
    let head_dim = cfg.hidden_size / cfg.num_heads;
    let input = product(&[
        tokens,
        cfg.in_chans,
        cfg.temporal_patch_size,
        cfg.patch_size,
        cfg.patch_size,
        DType::F32.size_in_bytes(),
    ])?;
    let output = product(&[
        tokens / product(&[cfg.spatial_merge_size, cfg.spatial_merge_size])?,
        cfg.out_hidden_size,
        dtype_bytes,
    ])?;
    let retained = sum(&[
        input,
        product(&[hidden, RETAINED_HIDDEN_BUFFERS])?,
        product(&[output, cfg.deepstack_visual_indexes.len()])?,
        product(&[
            tokens,
            head_dim,
            DType::F32.size_in_bytes(),
            ROTARY_TABLE_BUFFERS,
        ])?,
    ])?;
    let position = product(&[hidden, POSITION_HIDDEN_BUFFERS])?;
    let rotary = sum(&[
        product(&[hidden, ROTARY_HIDDEN_BUFFERS])?,
        product(&[
            tokens,
            cfg.hidden_size,
            DType::F32.size_in_bytes(),
            ROTARY_F32_BUFFERS,
        ])?,
    ])?;
    // VisionAttention evaluates image segments sequentially while retaining their outputs.
    let attention = sum(&[
        product(&[hidden, ATTENTION_HIDDEN_BUFFERS])?,
        attention_workspace_bytes(AttentionMemoryShape {
            batch: 1,
            query_len: segment_tokens,
            kv_len: segment_tokens,
            query_heads: cfg.num_heads,
            kv_heads: cfg.num_heads,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            dtype,
            backend: if fused {
                AttentionWorkspaceKind::Fused
            } else {
                AttentionWorkspaceKind::Eager
            },
            materialized_mask: false,
        })?,
    ])?;
    let mlp = sum(&[
        product(&[hidden, MLP_HIDDEN_BUFFERS])?,
        product(&[
            tokens,
            cfg.intermediate_size,
            dtype_bytes,
            MLP_INTERMEDIATE_BUFFERS,
        ])?,
    ])?;
    let merger = sum(&[
        product(&[hidden, MERGER_HIDDEN_BUFFERS])?,
        product(&[output, MERGER_OUTPUT_BUFFERS])?,
    ])?;
    sum(&[
        retained,
        position.max(rotary).max(attention).max(mlp).max(merger),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VisionConfig {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 5120,
            "num_heads": 16,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": []
        }))
        .unwrap()
    }

    fn params(batch: usize, images: usize) -> AutoDeviceMapParams {
        AutoDeviceMapParams::Multimodal {
            max_seq_len: 4096,
            max_batch_size: batch,
            max_image_shape: (1024, 1024),
            max_num_images: images,
        }
    }

    #[test]
    fn fused_vision_reserves_linear_buffers_instead_of_batched_scores() -> Result<()> {
        let cfg = config();
        let one = vision_activation_bytes(&cfg, &params(1, 1), DType::BF16, true)?;
        let batch = vision_activation_bytes(&cfg, &params(16, 1), DType::BF16, true)?;
        let legacy = 16usize * cfg.num_heads * 4096 * 4096 * DType::F32.size_in_bytes();
        assert_eq!(batch, one * 16);
        assert!(batch < legacy / 3);
        assert!(one > 4096 * cfg.hidden_size * DType::F32.size_in_bytes());
        Ok(())
    }

    #[test]
    fn eager_vision_reuses_score_workspace_between_images() -> Result<()> {
        let cfg = config();
        let one = vision_activation_bytes(&cfg, &params(1, 1), DType::BF16, false)?;
        let batch = vision_activation_bytes(&cfg, &params(16, 1), DType::BF16, false)?;
        let fused = vision_activation_bytes(&cfg, &params(1, 1), DType::BF16, true)?;
        assert!(one > fused);
        assert!(batch < one * 8);
        assert_eq!(
            batch,
            vision_activation_bytes(&cfg, &params(1, 16), DType::BF16, false)?
        );
        Ok(())
    }

    #[test]
    fn vision_limits_preserve_disabled_images_and_reject_overflow() -> Result<()> {
        let cfg = config();
        assert_eq!(
            vision_activation_bytes(&cfg, &params(16, 0), DType::BF16, true)?,
            0
        );
        assert!(vision_activation_bytes(&cfg, &params(usize::MAX, 1), DType::BF16, true).is_err());
        Ok(())
    }
}
