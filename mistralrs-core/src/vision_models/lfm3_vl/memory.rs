use candle_core::{DType, Device, Result};

use super::config::VisionConfig;

const VISION_HIDDEN_BUFFERS: usize = 10;
const VISION_INTERMEDIATE_BUFFERS: usize = 3;

pub(crate) struct EncoderMemoryEstimate {
    bytes_per_patch: usize,
    attention_bytes_per_pair: usize,
}

impl EncoderMemoryEstimate {
    pub(crate) fn new(config: &VisionConfig, dtype: DType, device: &Device) -> Self {
        let activations = (VISION_HIDDEN_BUFFERS * config.hidden_size
            + VISION_INTERMEDIATE_BUFFERS * config.intermediate_size)
            * dtype.size_in_bytes();
        let input = config.in_channels
            * config.temporal_patch_size
            * config.patch_size.pow(2)
            * DType::F32.size_in_bytes();
        let fused = device.is_metal()
            || device.is_cuda()
                && crate::using_flash_attn()
                && matches!(dtype, DType::F16 | DType::BF16);
        Self {
            bytes_per_patch: activations + input,
            attention_bytes_per_pair: if fused || device.is_cpu() {
                0
            } else {
                config.num_attention_heads * DType::F32.size_in_bytes()
            },
        }
    }

    pub(crate) fn estimate(&self, grids: impl IntoIterator<Item = [usize; 3]>) -> Result<usize> {
        grids
            .into_iter()
            .try_fold(0usize, |total, [time, height, width]| {
                let overflow =
                    || candle_core::Error::msg("LFM3-VL encoder workspace size overflow");
                let patches = height.checked_mul(width).ok_or_else(overflow)?;
                let per_patch = patches
                    .checked_mul(self.attention_bytes_per_pair)
                    .and_then(|attention| self.bytes_per_patch.checked_add(attention))
                    .ok_or_else(overflow)?;
                patches
                    .checked_mul(time)
                    .and_then(|tokens| tokens.checked_mul(per_patch))
                    .and_then(|bytes| total.checked_add(bytes))
                    .ok_or_else(overflow)
            })
    }

    pub(crate) fn for_sequence(&self, seq: &crate::sequence::Sequence) -> Result<Option<usize>> {
        let multimodal = &seq.multimodal;
        // Complete grids also bound prefix hits and chunked prompts without relying on mutable cache state.
        let grids = [
            multimodal
                .rope_img_grid_thw
                .as_ref()
                .or(multimodal.cached_img_thw.as_ref()),
            multimodal
                .rope_vid_grid_thw
                .as_ref()
                .or(multimodal.cached_vid_thw.as_ref()),
        ];
        if grids.iter().all(Option::is_none)
            || seq.has_images() && grids[0].is_none()
            || seq.has_videos() && grids[1].is_none()
        {
            return Ok(None);
        }
        let mut shapes = Vec::new();
        for grid in grids.into_iter().flatten() {
            for shape in grid.to_device(&Device::Cpu)?.to_vec2::<u32>()? {
                let [time, height, width] = shape.as_slice() else {
                    candle_core::bail!("LFM3-VL grid entries must have three dimensions");
                };
                shapes.push([*time as usize, *height as usize, *width as usize]);
            }
        }
        self.estimate(shapes).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_follows_actual_image_and_video_grids() -> Result<()> {
        let estimate = EncoderMemoryEstimate {
            bytes_per_patch: 100,
            attention_bytes_per_pair: 0,
        };
        assert_eq!(estimate.estimate([[1, 16, 16]])?, 25_600);
        assert_eq!(estimate.estimate([[1, 16, 16], [4, 8, 8]])?, 51_200);
        assert_eq!(estimate.estimate([[32, 64, 64]])?, 13_107_200);
        assert!(estimate.estimate([[usize::MAX, 2, 2]]).is_err());
        Ok(())
    }

    #[test]
    fn eager_attention_counts_independent_frames() -> Result<()> {
        let estimate = EncoderMemoryEstimate {
            bytes_per_patch: 100,
            attention_bytes_per_pair: 4,
        };
        assert_eq!(estimate.estimate([[3, 2, 4]])?, 3 * (8 * 100 + 8 * 8 * 4));
        Ok(())
    }
}
