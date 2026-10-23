use std::sync::{Arc, Mutex};

use candle_core::{DType, Result, Tensor};

use crate::{
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheManager},
    },
    vision_models::{
        multimodal_layout::{MultimodalEncoderOutputs, PackedMultimodalLayout},
        qwen3_vl::{insert_current_visual_outputs, vision::Qwen3VLVisionModel, VisualEncoder},
    },
};

pub(crate) struct PackedVisualInput<'a> {
    pub input_embeds: Tensor,
    pub pixel_values: Option<&'a Tensor>,
    pub pixel_values_videos: Option<&'a Tensor>,
    pub image_grid_thw: Option<&'a Tensor>,
    pub video_grid_thw: Option<&'a Tensor>,
    pub image_hashes: &'a [u64],
    pub video_hashes: &'a [u64],
    pub layout: &'a PackedMultimodalLayout,
}

#[derive(Debug)]
pub(crate) struct PackedVisualOutput {
    pub input_embeds: Tensor,
    pub visual_pos_mask: Option<Tensor>,
    pub deepstack_visual_embeds: Option<Vec<Tensor>>,
}

pub(crate) struct PackedVisualEncoder<'a> {
    vision: &'a Qwen3VLVisionModel,
    cache: &'a Arc<Mutex<EncoderCacheManager>>,
    spatial_merge_size: usize,
}

impl<'a> PackedVisualEncoder<'a> {
    pub(crate) fn new(
        vision: &'a Qwen3VLVisionModel,
        cache: &'a Arc<Mutex<EncoderCacheManager>>,
        spatial_merge_size: usize,
    ) -> Self {
        Self {
            vision,
            cache,
            spatial_merge_size,
        }
    }

    pub(crate) fn prepare(&self, input: PackedVisualInput<'_>) -> Result<PackedVisualOutput> {
        let mut encoder_outputs = MultimodalEncoderOutputs::new();
        self.encode_current(
            &mut encoder_outputs,
            MultimodalKind::Image,
            CacheModality::Image,
            input.pixel_values,
            input.image_grid_thw,
            input.image_hashes,
        )?;
        self.encode_current(
            &mut encoder_outputs,
            MultimodalKind::Video,
            CacheModality::Video,
            input.pixel_values_videos,
            input.video_grid_thw,
            input.video_hashes,
        )?;
        apply_packed_visual_layout(input.input_embeds, input.layout, &encoder_outputs)
    }

    fn encode_current(
        &self,
        encoder_outputs: &mut MultimodalEncoderOutputs,
        kind: MultimodalKind,
        modality: CacheModality,
        pixel_values: Option<&Tensor>,
        grid_thw: Option<&Tensor>,
        hashes: &[u64],
    ) -> Result<()> {
        let Some(pixel_values) = pixel_values else {
            if grid_thw.is_some() || !hashes.is_empty() {
                candle_core::bail!("packed Qwen {kind:?} metadata is missing pixel values");
            }
            return Ok(());
        };
        let grid_thw = grid_thw.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "packed Qwen {kind:?} pixel values are missing grid metadata"
            ))
        })?;
        let pixel_values = flatten_pixel_values(pixel_values)?;
        let outputs = VisualEncoder::new(self.vision, self.cache, self.spatial_merge_size).encode(
            &pixel_values,
            grid_thw,
            hashes,
            modality,
        )?;
        insert_current_visual_outputs(encoder_outputs, kind, hashes, outputs)
    }
}

fn flatten_pixel_values(pixel_values: &Tensor) -> Result<Tensor> {
    let rank = pixel_values.rank();
    if rank <= 2 {
        Ok(pixel_values.clone())
    } else {
        pixel_values.reshape(((), pixel_values.dim(rank - 1)?))
    }
}

fn encoder_output_count(encoder_outputs: &MultimodalEncoderOutputs) -> Result<usize> {
    let Some(first) = encoder_outputs.values().next() else {
        return Ok(1);
    };
    let count = first.len();
    if count == 0 {
        candle_core::bail!("packed Qwen encoder item has no outputs");
    }
    if encoder_outputs
        .values()
        .any(|outputs| outputs.len() != count)
    {
        candle_core::bail!("packed Qwen media items have different DeepStack output counts");
    }
    Ok(count)
}

fn apply_packed_visual_layout(
    input_embeds: Tensor,
    layout: &PackedMultimodalLayout,
    encoder_outputs: &MultimodalEncoderOutputs,
) -> Result<PackedVisualOutput> {
    let input_embeds = layout.splice_embeddings(&input_embeds, encoder_outputs)?;
    let destinations = layout.destination_positions(0);
    if destinations.is_empty() {
        return Ok(PackedVisualOutput {
            input_embeds,
            visual_pos_mask: None,
            deepstack_visual_embeds: None,
        });
    }

    let (batch_size, seq_len, _) = input_embeds.dims3()?;
    let indices = Tensor::from_vec(
        destinations
            .iter()
            .map(|position| u32::try_from(*position).map_err(candle_core::Error::wrap))
            .collect::<Result<Vec<_>>>()?,
        destinations.len(),
        input_embeds.device(),
    )?;
    let visual_pos_mask = Tensor::zeros(batch_size * seq_len, DType::F32, input_embeds.device())?
        .scatter_add(
            &indices,
            &Tensor::ones(destinations.len(), DType::F32, input_embeds.device())?,
            0,
        )?
        .reshape((batch_size, seq_len))?
        .to_dtype(DType::U8)?;

    let output_count = encoder_output_count(encoder_outputs)?;
    let mut deepstack = Vec::with_capacity(output_count - 1);
    for output in 1..output_count {
        let outputs = encoder_outputs
            .iter()
            .map(|(key, values)| (*key, vec![values[output].clone()]))
            .collect::<MultimodalEncoderOutputs>();
        deepstack.push(layout.gather_output_embeddings(0, &input_embeds, &outputs)?);
    }

    Ok(PackedVisualOutput {
        input_embeds,
        visual_pos_mask: Some(visual_pos_mask),
        deepstack_visual_embeds: Some(deepstack),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        paged_attention::block_hash::MultimodalAttentionPolicy,
        vision_models::multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            RequestMultimodalLayout,
        },
    };
    use candle_core::Device;

    fn item(
        kind: MultimodalKind,
        hash: u64,
        placeholder: std::ops::Range<usize>,
    ) -> MultimodalItemLayout {
        MultimodalItemLayout::new(
            MultimodalEncoderKey { kind, hash },
            0,
            placeholder.clone(),
            MultimodalAttentionPolicy::Causal,
            vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0).unwrap()],
        )
        .unwrap()
    }

    fn heterogeneous_layout() -> PackedMultimodalLayout {
        PackedMultimodalLayout::new(&[
            RequestMultimodalLayout {
                sequence_id: 7,
                query: 0..3,
                items: vec![item(MultimodalKind::Image, 11, 1..3)],
            },
            RequestMultimodalLayout {
                sequence_id: 8,
                query: 0..2,
                items: vec![item(MultimodalKind::Video, 22, 0..1)],
            },
        ])
        .unwrap()
    }

    #[test]
    fn heterogeneous_media_preserves_packed_destinations_and_deepstack_order() -> Result<()> {
        let layout = heterogeneous_layout();
        let input_embeds = Tensor::zeros((1, 5, 2), DType::F32, &Device::Cpu)?;
        let outputs = MultimodalEncoderOutputs::from([
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 11,
                },
                vec![
                    Tensor::new(&[[10f32, 11.], [12., 13.]], &Device::Cpu)?,
                    Tensor::new(&[[110f32, 111.], [112., 113.]], &Device::Cpu)?,
                ],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Video,
                    hash: 22,
                },
                vec![
                    Tensor::new(&[[20f32, 21.]], &Device::Cpu)?,
                    Tensor::new(&[[120f32, 121.]], &Device::Cpu)?,
                ],
            ),
        ]);

        let result = apply_packed_visual_layout(input_embeds, &layout, &outputs)?;
        assert_eq!(
            result.input_embeds.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 0., 10., 11., 12., 13., 20., 21., 0., 0.]
        );
        assert_eq!(
            result
                .visual_pos_mask
                .unwrap()
                .flatten_all()?
                .to_vec1::<u8>()?,
            vec![0, 1, 1, 1, 0]
        );
        assert_eq!(
            result.deepstack_visual_embeds.unwrap()[0]
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![110., 111., 112., 113., 120., 121.]
        );
        Ok(())
    }

    #[test]
    fn heterogeneous_media_rejects_deepstack_cardinality_mismatch() -> Result<()> {
        let layout = heterogeneous_layout();
        let input_embeds = Tensor::zeros((1, 5, 1), DType::F32, &Device::Cpu)?;
        let outputs = MultimodalEncoderOutputs::from([
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 11,
                },
                vec![
                    Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                    Tensor::zeros((2, 1), DType::F32, &Device::Cpu)?,
                ],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Video,
                    hash: 22,
                },
                vec![Tensor::zeros((1, 1), DType::F32, &Device::Cpu)?],
            ),
        ]);

        let error = apply_packed_visual_layout(input_embeds, &layout, &outputs).unwrap_err();
        assert!(error
            .to_string()
            .contains("different DeepStack output counts"));
        Ok(())
    }
}
