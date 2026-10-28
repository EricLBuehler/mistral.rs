pub(crate) mod config;
mod inputs_processor;
pub(crate) mod memory;
mod vision;

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor};
use candle_nn::{LayerNorm, Module};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    layers::layer_norm,
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheBatchLookup, EncoderCacheManager},
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
        NormalModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::multimodal_layout::{MultimodalEncoderKey, MultimodalEncoderOutputs},
};

use crate::vision_models::grid_processor::GridVisionSpecificArgs as Lfm3VlSpecificArgs;
pub(crate) use config::Config;
pub(crate) use inputs_processor::Lfm3VlProcessor;
use vision::VisionModel;

const ENCODER_CACHE_CAPACITY: usize = 32;

struct MultiModalProjector {
    layer_norm: LayerNorm,
    linear_1: Arc<dyn QuantMethod>,
    linear_2: Arc<dyn QuantMethod>,
    activation: crate::layers::Activation,
    merged_size: usize,
}

impl MultiModalProjector {
    fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let merged_size = config.vision_config.hidden_size * config.downsample_factor.pow(2);
        Ok(Self {
            layer_norm: layer_norm(
                config.vision_config.hidden_size,
                config.projector_layer_norm_eps,
                vb.pp("layer_norm"),
            )?,
            linear_1: ReplicatedLayer::new(
                merged_size,
                config.projector_hidden_size,
                &None,
                config.projector_bias,
                vb.pp("linear_1"),
            )?,
            linear_2: ReplicatedLayer::new(
                config.projector_hidden_size,
                config.text_config.hidden_size,
                &None,
                config.projector_bias,
                vb.pp("linear_2"),
            )?,
            activation: config.projector_hidden_act,
            merged_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self
            .layer_norm
            .forward(xs)?
            .reshape(((), self.merged_size))?;
        self.linear_2
            .forward(&self.activation.forward(&self.linear_1.forward(&xs)?)?)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("layer_norm").add(&self.layer_norm);
        uvb.pp("linear_1").add(&self.linear_1);
        uvb.pp("linear_2").add(&self.linear_2);
        uvb.to_safetensors()
    }
}

pub struct Lfm3VlModel {
    language_model: crate::models::lfm3::Model,
    vision_tower: VisionModel,
    multi_modal_projector: MultiModalProjector,
    merge_size: usize,
    encoder_cache: Mutex<EncoderCacheManager>,
    encoder_memory: memory::EncoderMemoryEstimate,
}

impl Lfm3VlModel {
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let vision_vb = vb
            .clone()
            .without_lora_registry()
            .pp("model")
            .set_device(device);
        Ok(Self {
            encoder_memory: memory::EncoderMemoryEstimate::new(
                &config.vision_config,
                vision_vb.dtype(),
                vision_vb.device(),
            ),
            vision_tower: VisionModel::new(&config.vision_config, vision_vb.pp("vision_tower"))?,
            multi_modal_projector: MultiModalProjector::new(
                config,
                vision_vb.pp("multi_modal_projector"),
            )?,
            language_model: crate::models::lfm3::Model::new_inner(
                &config.text_config,
                vb.pp("model").pp("language_model"),
                vb.pp("lm_head"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            merge_size: config.downsample_factor,
            encoder_cache: Mutex::new(EncoderCacheManager::new(ENCODER_CACHE_CAPACITY)),
        })
    }

    fn encode(
        &self,
        pixels: &Tensor,
        grid: &Tensor,
        hashes: &[u64],
        modality: CacheModality,
    ) -> Result<Vec<Vec<Tensor>>> {
        let shapes = grid.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
        let patches = shapes
            .iter()
            .map(|shape| shape.iter().map(|&dim| dim as usize).product::<usize>())
            .collect::<Vec<_>>();
        let cache_hashes = hashes
            .iter()
            .zip(&shapes)
            .map(|(hash, shape)| {
                let mut hasher = DefaultHasher::new();
                hash.hash(&mut hasher);
                shape.hash(&mut hasher);
                self.merge_size.hash(&mut hasher);
                hasher.finish()
            })
            .collect::<Vec<_>>();
        let mut lookup = if hashes.is_empty() {
            EncoderCacheBatchLookup::uncached(shapes.len())
        } else {
            EncoderCacheBatchLookup::lookup_validated(
                modality,
                &cache_hashes,
                &self.encoder_cache,
                |i, outputs| {
                    outputs.len() == 1
                        && outputs[0].dims().first() == Some(&(patches[i] / self.merge_size.pow(2)))
                },
            )
        };
        if !lookup.miss_groups().is_empty() {
            let mut offsets = Vec::with_capacity(patches.len());
            let mut offset = 0;
            for &count in &patches {
                offsets.push(offset);
                offset += count;
            }
            let indices = lookup
                .miss_groups()
                .iter()
                .map(|group| group[0])
                .collect::<Vec<_>>();
            let pixels = indices
                .iter()
                .map(|&i| pixels.narrow(0, offsets[i], patches[i]))
                .collect::<Result<Vec<_>>>()?;
            let grids = indices
                .iter()
                .map(|&i| grid.get(i))
                .collect::<Result<Vec<_>>>()?;
            let features = self.multi_modal_projector.forward(
                &self
                    .vision_tower
                    .forward(&Tensor::cat(&pixels, 0)?, &Tensor::stack(&grids, 0)?)?,
            )?;
            let mut output_offset = 0;
            let mut cache = self.encoder_cache.lock().expect("encoder cache poisoned");
            for (miss, &index) in indices.iter().enumerate() {
                let count = patches[index] / self.merge_size.pow(2);
                let outputs = vec![features.narrow(0, output_offset, count)?];
                output_offset += count;
                if let Some(&hash) = cache_hashes.get(index) {
                    cache.insert(modality, hash, outputs.clone());
                }
                lookup.resolve_miss(miss, outputs);
            }
        }
        lookup.into_outputs()
    }

    fn splice(
        input: &Tensor,
        per_item: &[Vec<Tensor>],
        spans: &[Vec<(usize, usize)>],
    ) -> Result<Tensor> {
        let features = Tensor::cat(
            &per_item
                .iter()
                .map(|outputs| &outputs[0])
                .collect::<Vec<_>>(),
            0,
        )?
        .to_device(input.device())?
        .to_dtype(input.dtype())?;
        let hidden = input.dim(2)?;
        let expected: usize = spans.iter().flatten().map(|(start, end)| end - start).sum();
        if features.dim(0)? != expected {
            candle_core::bail!(
                "LFM3-VL produced {} visual tokens for {expected} placeholders",
                features.dim(0)?
            );
        }
        let mut input = input.clone();
        let mut offset = 0;
        for (batch, spans) in spans.iter().enumerate() {
            for &(start, end) in spans {
                input = input.slice_assign(
                    &[batch..batch + 1, start..end, 0..hidden],
                    &features.narrow(0, offset, end - start)?.unsqueeze(0)?,
                )?;
                offset += end - start;
            }
        }
        Ok(input)
    }
}

impl IsqModel for Lfm3VlModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.extend(
            self.language_model
                .residual_tensors_m(uvb.pp("model").pp("language_model")),
        );
        uvb.pp("model")
            .pp("vision_tower")
            .extend(self.vision_tower.residual_tensors());
        uvb.pp("model")
            .pp("multi_modal_projector")
            .extend(self.multi_modal_projector.residual_tensors());
        uvb.to_safetensors()
    }
}

impl crate::speculative::SpeculativeTargetMixin for Lfm3VlModel {}
impl crate::block_diffusion::BlockDiffusionMixin for Lfm3VlModel {}
impl AnyMoeBaseModelMixin for Lfm3VlModel {}

impl MultimodalModel for Lfm3VlModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let Lfm3VlSpecificArgs {
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            image_hashes,
            video_hashes,
            continuous_img_pad,
            continuous_vid_pad,
            packed_layout,
            ..
        } = *model_specific_args
            .downcast::<Lfm3VlSpecificArgs>()
            .expect("Cannot downcast LFM3-VL inputs");
        let video_pixels = pixel_values_videos.or_else(|| {
            (image_grid_thw.is_none() && video_grid_thw.is_some())
                .then(|| pixel_values.clone())
                .flatten()
        });
        let image_pixels = image_grid_thw.as_ref().and(pixel_values);
        let mut input = self.language_model.embed(input_ids)?;
        let mut packed_outputs: MultimodalEncoderOutputs = HashMap::new();
        for (pixels, grid, hashes, spans, modality, kind) in [
            (
                image_pixels,
                image_grid_thw,
                image_hashes,
                continuous_img_pad,
                CacheModality::Image,
                MultimodalKind::Image,
            ),
            (
                video_pixels,
                video_grid_thw,
                video_hashes,
                continuous_vid_pad,
                CacheModality::Video,
                MultimodalKind::Video,
            ),
        ] {
            if let Some(pixels) = pixels {
                let grid = grid.ok_or_else(|| {
                    candle_core::Error::msg("LFM3-VL visual inputs require grid dimensions")
                })?;
                let outputs = self.encode(&pixels, &grid, &hashes, modality)?;
                if packed_layout.is_some() {
                    for (hash, values) in hashes.into_iter().zip(outputs) {
                        packed_outputs.insert(MultimodalEncoderKey { kind, hash }, values);
                    }
                } else {
                    input = Self::splice(&input, &outputs, &spans)?;
                }
            }
        }
        if let Some(layout) = packed_layout {
            input = layout.splice_embeddings(&input, &packed_outputs)?;
        }
        self.language_model.forward_embeds(input_ids, input, ctx)
    }

    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        self.language_model.model_config()
    }
    fn supports_packed_prefill(&self) -> bool {
        self.language_model.supports_packed_prefill()
    }
    fn supports_mixed_media_batches(&self) -> bool {
        true
    }
    fn encoder_activation_memory_bytes(
        &self,
        seq: &crate::sequence::Sequence,
    ) -> Result<Option<usize>> {
        self.encoder_memory.for_sequence(seq)
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Lfm3VlSpecificArgs {
            input_ids_full: input_ids.clone(),
            pixel_values_videos: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rope_img_grid_thw: None,
            rope_vid_grid_thw: None,
            seqlens: vec![],
            continuous_img_pad: vec![],
            continuous_vid_pad: vec![],
            image_hashes: vec![],
            video_hashes: vec![],
            packed_layout: None,
            prompt_position_ids: None,
        })
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
                .expect("encoder cache poisoned")
                .counters(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn visual_spans_preserve_padding_and_video_timestamps() -> Result<()> {
        let input = Tensor::zeros((2, 5, 2), DType::F32, &Device::Cpu)?;
        let image = Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?;
        let video = Tensor::new(&[[5f32, 6.], [7., 8.]], &Device::Cpu)?;
        let output = Lfm3VlModel::splice(
            &input,
            &[vec![image], vec![video]],
            &[vec![(1, 3)], vec![(0, 1), (3, 4)]],
        )?;
        assert_eq!(
            output.to_vec3::<f32>()?,
            vec![
                vec![
                    vec![0., 0.],
                    vec![1., 2.],
                    vec![3., 4.],
                    vec![0., 0.],
                    vec![0., 0.]
                ],
                vec![
                    vec![5., 6.],
                    vec![0., 0.],
                    vec![0., 0.],
                    vec![7., 8.],
                    vec![0., 0.]
                ],
            ]
        );
        Ok(())
    }
}
