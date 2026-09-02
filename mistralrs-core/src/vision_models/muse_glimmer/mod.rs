#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{Context, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    amoe::{AnyMoeBaseModelMixin, AnyMoeConfig, AnyMoeExpertType, MlpLayer},
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::{
        multimodal_layout::{MultimodalEncoderOutputs, PackedMultimodalLayout},
        qwen2vl::insert_current_visual_outputs,
    },
};

pub(crate) mod config;
mod inputs_processor;
mod text;
mod vision;

pub(crate) use config::Config;
pub(crate) use inputs_processor::MuseGlimmerProcessor;
use text::TextModel;
use vision::VisionModel;

const ENCODER_CACHE_CAPACITY: usize = 32;

fn split_model_builders(
    vb: ShardedVarBuilder,
    vision_device: &Device,
) -> (ShardedVarBuilder, ShardedVarBuilder) {
    let vision = vb
        .clone()
        .without_lora_registry()
        .set_device(vision_device.clone());
    (vision, vb)
}

pub struct MuseGlimmerModel {
    text: TextModel,
    vision: VisionModel,
    merge_size: usize,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl MuseGlimmerModel {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let mut cfg = cfg.clone();
        if cfg.quantization_config.is_some() {
            cfg.text_config.quantization_config = cfg.quantization_config.clone();
        }
        cfg.validate()?;
        let comm = normal_loading_metadata.mapper.get_comm_for(0)?;
        let (vision_vb, text_vb) = split_model_builders(vb, &normal_loading_metadata.real_device);
        let vision = VisionModel::new(&cfg, vision_vb, &comm, is_gptx)?;
        let text = TextModel::new(
            &cfg.text_config,
            text_vb,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text,
            vision,
            merge_size: cfg.vision_config.merge_size,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_CAPACITY))),
        })
    }

    fn split_outputs(&self, encoded: &Tensor, grids: &[[u32; 3]]) -> Result<Vec<Tensor>> {
        let mut offset = 0usize;
        let mut outputs = Vec::with_capacity(grids.len());
        for [frames, height, width] in grids {
            let count = *frames as usize
                * (*height as usize / self.merge_size)
                * (*width as usize / self.merge_size);
            outputs.push(encoded.narrow(0, offset, count)?);
            offset += count;
        }
        if offset != encoded.dim(0)? {
            candle_core::bail!(
                "Muse-Glimmer vision produced {} tokens, expected {offset}",
                encoded.dim(0)?
            );
        }
        Ok(outputs)
    }

    fn encode_visual_items(
        &self,
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        hashes: &[u64],
        modality: CacheModality,
    ) -> Result<Vec<Tensor>> {
        let grids = grid_thw
            .to_vec2::<u32>()?
            .into_iter()
            .map(|grid| [grid[0], grid[1], grid[2]])
            .collect::<Vec<_>>();
        if hashes.is_empty() {
            let encoded = self.vision.forward(pixel_values, grid_thw)?;
            return self.split_outputs(&encoded, &grids);
        }
        if grids.len() != hashes.len() {
            candle_core::bail!(
                "Muse-Glimmer {modality:?} grid count {} does not match hash count {}",
                grids.len(),
                hashes.len()
            );
        }
        let patch_counts = grids
            .iter()
            .map(|[frames, height, width]| (*frames * *height * *width) as usize)
            .collect::<Vec<_>>();
        let mut outputs = vec![None; hashes.len()];
        let mut misses = Vec::new();
        {
            let mut cache = self
                .encoder_cache
                .lock()
                .expect("Muse-Glimmer encoder cache poisoned");
            for (index, &hash) in hashes.iter().enumerate() {
                if let Some(cached) = cache.get(modality, hash) {
                    outputs[index] = Some(cached[0].clone());
                } else {
                    misses.push(index);
                }
            }
        }
        if !misses.is_empty() {
            let mut pixel_offset = 0usize;
            let mut miss_pixels = Vec::with_capacity(misses.len());
            let mut miss_grids = Vec::with_capacity(misses.len());
            for (index, &patch_count) in patch_counts.iter().enumerate() {
                if misses.contains(&index) {
                    miss_pixels.push(pixel_values.narrow(0, pixel_offset, patch_count)?);
                    miss_grids.push(grid_thw.get(index)?);
                }
                pixel_offset += patch_count;
            }
            let encoded = self.vision.forward(
                &Tensor::cat(&miss_pixels, 0)?,
                &Tensor::stack(&miss_grids, 0)?,
            )?;
            let miss_grid_values = misses.iter().map(|&index| grids[index]).collect::<Vec<_>>();
            let miss_outputs = self.split_outputs(&encoded, &miss_grid_values)?;
            let mut cache = self
                .encoder_cache
                .lock()
                .expect("Muse-Glimmer encoder cache poisoned");
            for ((&index, output), grid) in misses.iter().zip(miss_outputs).zip(miss_grid_values) {
                let expected = grid[0] as usize
                    * (grid[1] as usize / self.merge_size)
                    * (grid[2] as usize / self.merge_size);
                debug_assert_eq!(output.dim(0)?, expected);
                cache.insert(modality, hashes[index], vec![output.clone()]);
                outputs[index] = Some(output);
            }
        }
        outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| candle_core::Error::msg("missing Muse-Glimmer vision output"))
            })
            .collect()
    }

    fn splice_ranges(
        mut input_embeds: Tensor,
        outputs: &[Tensor],
        ranges: &[Vec<(usize, usize)>],
        modality: &str,
    ) -> Result<Tensor> {
        let encoded = Tensor::cat(outputs, 0)?;
        let mut offset = 0usize;
        for (batch, batch_ranges) in ranges.iter().enumerate() {
            for &(start, end) in batch_ranges {
                if end < start {
                    candle_core::bail!("Muse-Glimmer {modality} placeholder range is reversed");
                }
                let len = end - start;
                if offset + len > encoded.dim(0)? {
                    candle_core::bail!(
                        "Muse-Glimmer {modality} placeholders require more encoded tokens than available"
                    );
                }
                input_embeds = input_embeds.slice_assign(
                    &[batch..batch + 1, start..end, 0..input_embeds.dim(2)?],
                    &encoded.narrow(0, offset, len)?.unsqueeze(0)?,
                )?;
                offset += len;
            }
        }
        if offset != encoded.dim(0)? {
            candle_core::bail!(
                "Muse-Glimmer has {} unused encoded {modality} tokens",
                encoded.dim(0)? - offset
            );
        }
        Ok(input_embeds)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        pixel_values_videos: Option<Tensor>,
        image_grid_thw: Option<Tensor>,
        video_grid_thw: Option<Tensor>,
        continuous_img_pad: Vec<Vec<(usize, usize)>>,
        continuous_vid_pad: Vec<Vec<(usize, usize)>>,
        image_hashes: &[u64],
        video_hashes: &[u64],
        packed_layout: Option<&PackedMultimodalLayout>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.text.embed_tokens(input_ids)?;
        let mut packed_outputs = MultimodalEncoderOutputs::new();

        if let Some(pixel_values) = pixel_values {
            let grid = image_grid_thw
                .as_ref()
                .context("Muse-Glimmer image pixels require image_grid_thw")?;
            let outputs =
                self.encode_visual_items(&pixel_values, grid, image_hashes, CacheModality::Image)?;
            if packed_layout.is_some() {
                insert_current_visual_outputs(
                    &mut packed_outputs,
                    MultimodalKind::Image,
                    image_hashes,
                    outputs,
                )?;
            } else {
                input_embeds =
                    Self::splice_ranges(input_embeds, &outputs, &continuous_img_pad, "image")?;
            }
        }

        if let Some(pixel_values_videos) = pixel_values_videos {
            if self.vision.collapsed_temporal() {
                candle_core::bail!(
                    "this Muse-Glimmer GGUF projector collapsed temporal patch weights and cannot process video"
                );
            }
            let grid = video_grid_thw
                .as_ref()
                .context("Muse-Glimmer video pixels require video_grid_thw")?;
            let outputs = self.encode_visual_items(
                &pixel_values_videos,
                grid,
                video_hashes,
                CacheModality::Video,
            )?;
            if packed_layout.is_some() {
                insert_current_visual_outputs(
                    &mut packed_outputs,
                    MultimodalKind::Video,
                    video_hashes,
                    outputs,
                )?;
            } else {
                input_embeds =
                    Self::splice_ranges(input_embeds, &outputs, &continuous_vid_pad, "video")?;
            }
        }

        if let Some(layout) = packed_layout {
            input_embeds = layout.splice_embeddings(&input_embeds, &packed_outputs)?;
        }
        self.text.forward_embeds(input_ids, input_embeds, ctx)
    }
}

#[derive(Default)]
pub(crate) struct MuseGlimmerSpecificArgs {
    pub(crate) pixel_values_videos: Option<Tensor>,
    pub(crate) image_grid_thw: Option<Tensor>,
    pub(crate) video_grid_thw: Option<Tensor>,
    pub(crate) continuous_img_pad: Vec<Vec<(usize, usize)>>,
    pub(crate) continuous_vid_pad: Vec<Vec<(usize, usize)>>,
    pub(crate) image_hashes: Vec<u64>,
    pub(crate) video_hashes: Vec<u64>,
    pub(crate) packed_layout: Option<PackedMultimodalLayout>,
}

impl crate::speculative::SpeculativeTargetMixin for MuseGlimmerModel {}
impl crate::block_diffusion::BlockDiffusionMixin for MuseGlimmerModel {}

impl MultimodalModel for MuseGlimmerModel {
    fn supports_packed_prefill(&self) -> bool {
        self.text.supports_packed_prefill()
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let MuseGlimmerSpecificArgs {
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            continuous_img_pad,
            continuous_vid_pad,
            image_hashes,
            video_hashes,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into MuseGlimmerSpecificArgs");
        let video_pixels = pixel_values_videos.or_else(|| {
            (image_grid_thw.is_none() && video_grid_thw.is_some())
                .then(|| pixel_values.clone())
                .flatten()
        });
        let image_pixels = image_grid_thw.is_some().then_some(pixel_values).flatten();
        self.forward_inner(
            input_ids,
            image_pixels,
            video_pixels,
            image_grid_thw,
            video_grid_thw,
            continuous_img_pad,
            continuous_vid_pad,
            &image_hashes,
            &video_hashes,
            packed_layout.as_ref(),
            ctx,
        )
    }

    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(MuseGlimmerSpecificArgs::default())
    }

    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        true
    }

    fn cache(&self) -> &EitherCache {
        &self.text.cache
    }

    fn device(&self) -> &Device {
        &self.text.device
    }

    fn max_seq_len(&self) -> usize {
        self.text.max_seq_len
    }

    fn config(&self) -> &ModelConfigMetadata {
        &self.text.cfg
    }

    fn encoder_cache(&self) -> Option<&Mutex<EncoderCacheManager>> {
        Some(&self.encoder_cache)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        Some(
            self.encoder_cache
                .lock()
                .expect("Muse-Glimmer encoder cache poisoned")
                .counters(),
        )
    }
}

impl IsqModel for MuseGlimmerModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(self.text.residual_tensors());
        uvb.extend(self.vision.residual_tensors());
        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for MuseGlimmerModel {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text.get_mlps()
    }

    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text.get_mlps_mut()
    }

    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        layer_prefix: (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.text.create_anymoe_layers(
            additional_vbs,
            config,
            layer_prefix,
            layers,
            expert_type,
            gate_vb,
        )
    }

    fn amoe_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Tensor};
    use mistralrs_quant::{LoraLayerRegistry, ShardedSafeTensors};

    use super::*;

    #[test]
    fn dynamic_lora_registry_is_limited_to_text_sites() {
        let registry = Arc::new(LoraLayerRegistry::new());
        let vb =
            ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), DType::F32, Device::Cpu)
                .with_lora_registry(registry.clone());
        let (vision_vb, text_vb) = split_model_builders(vb, &Device::Cpu);

        assert!(vision_vb.lora_registry().is_none());
        assert!(text_vb
            .lora_registry()
            .is_some_and(|text_registry| Arc::ptr_eq(text_registry, &registry)));
        assert_eq!(
            text_vb
                .pp("model")
                .pp("language_model")
                .pp("layers")
                .pp(0)
                .pp("self_attn")
                .pp("q_proj")
                .prefix(),
            "model.language_model.layers.0.self_attn.q_proj"
        );
        assert!(vision_vb
            .pp("model")
            .pp("vision_tower")
            .pp("layers")
            .pp(0)
            .pp("attn")
            .pp("q_proj")
            .lora_registry()
            .is_none());
    }
}
