use std::{
    any::Any,
    ops::Range,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor};
pub use config::MiniCpmOConfig;
pub use inputs_processor::MiniCpmOProcessor;
use mistralrs_quant::ShardedVarBuilder;
use resampler::Resampler;

use crate::attention::AttentionMask;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    models::qwen2,
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
        NormalModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::multimodal_layout::{
        MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
    },
};

use self::siglip::SiglipVisionTransformer;

use super::siglip;

mod config;
mod inputs_processor;
mod resampler;

pub struct MiniCpmOModel {
    cfg: MiniCpmOConfig,
    llm: qwen2::Model,
    vpm: SiglipVisionTransformer,
    resampler: Resampler,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

pub(crate) struct MiniCpmOVisualInput {
    pub key: MultimodalEncoderKey,
    pub pixel_values: Vec<Tensor>,
    pub tgt_sizes: Tensor,
}

pub(crate) struct MiniCpmOLegacyMap {
    pub key: MultimodalEncoderKey,
    pub source_output: usize,
    pub destination: Range<usize>,
}

impl MiniCpmOModel {
    pub fn new(
        cfg: &MiniCpmOConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let real_device = normal_loading_metadata.real_device.clone();
        let llm = qwen2::Model::new(
            &cfg.text_config,
            vb.pp("llm"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        let vpm = SiglipVisionTransformer::new(
            &cfg.vision_config,
            vb.pp("vpm").set_device(real_device.clone()),
        )?;
        let resampler = Resampler::new(
            cfg.query_num,
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size / 128,
            cfg.vision_config.hidden_size,
            true,
            None,
            vb.pp("resampler").set_device(real_device.clone()),
        )?;
        Ok(Self {
            cfg: cfg.clone(),
            llm,
            vpm,
            resampler,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn encode_visual_input(
        &self,
        device: &Device,
        input: &MiniCpmOVisualInput,
    ) -> Result<Vec<Tensor>> {
        if input.key.kind != MultimodalKind::Image {
            candle_core::bail!("MiniCPMO received a non-image visual input");
        }
        if input.pixel_values.len() != input.tgt_sizes.dim(0)? {
            candle_core::bail!(
                "MiniCPMO visual input has {} slices but {} target sizes",
                input.pixel_values.len(),
                input.tgt_sizes.dim(0)?
            );
        }
        if let Some(cached) = self
            .encoder_cache
            .lock()
            .expect("encoder cache lock poisoned")
            .get(CacheModality::Image, input.key.hash)
        {
            if cached.len() == input.pixel_values.len() {
                return Ok(cached);
            }
        }

        let target_sizes = input.tgt_sizes.to_vec2::<u32>()?;
        let mut pixels = input
            .pixel_values
            .iter()
            .map(|pixel| pixel.flatten_to(1)?.permute((1, 0)))
            .collect::<Result<Vec<_>>>()?;
        let max_pixel_len = pixels
            .iter()
            .map(|pixel| pixel.dim(0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .ok_or_else(|| candle_core::Error::msg("MiniCPMO visual input has no slices"))?;
        pixels = pixels
            .into_iter()
            .map(|pixel| pixel.pad_with_zeros(0, 0, max_pixel_len - pixel.dim(0)?))
            .collect::<Result<Vec<_>>>()?;
        let batch_size = pixels.len();
        let pixels = Tensor::stack(&pixels, 0)?
            .permute((0, 2, 1))?
            .reshape((batch_size, 3, (), max_pixel_len))?
            .to_dtype(self.llm.embed_dtype())?;
        let max_patch_count = target_sizes
            .iter()
            .map(|size| (size[0] * size[1]) as usize)
            .max()
            .unwrap_or(0);
        let mut patch_mask = Tensor::zeros((batch_size, 1, max_patch_count), DType::U8, device)?;
        for (batch, size) in target_sizes.iter().enumerate() {
            let patch_count = (size[0] * size[1]) as usize;
            patch_mask = patch_mask.slice_assign(
                &[batch..batch + 1, 0..1, 0..patch_count],
                &Tensor::ones((1, 1, patch_count), DType::U8, device)?,
            )?;
        }

        let vision_batch_size = self.cfg.vision_batch_size.max(1);
        let mut vision_batches = Vec::with_capacity(batch_size.div_ceil(vision_batch_size));
        for start in (0..batch_size).step_by(vision_batch_size) {
            let len = vision_batch_size.min(batch_size - start);
            vision_batches.push(self.vpm.forward(
                &pixels.narrow(0, start, len)?,
                &AttentionMask::Custom(patch_mask.narrow(0, start, len)?),
                Some(&input.tgt_sizes.narrow(0, start, len)?),
            )?);
        }
        let vision = if vision_batches.len() == 1 {
            vision_batches.remove(0)
        } else {
            Tensor::cat(&vision_batches, 0)?
        };
        let outputs = self.resampler.forward(&vision, &target_sizes)?;
        let mut per_slice_outputs = Vec::with_capacity(batch_size);
        for slice_idx in 0..batch_size {
            let output = outputs.get(slice_idx)?;
            if output.dim(0)? != self.cfg.query_num {
                candle_core::bail!(
                    "MiniCPMO resampler produced {} rows, expected {}",
                    output.dim(0)?,
                    self.cfg.query_num
                );
            }
            per_slice_outputs.push(output);
        }
        self.encoder_cache
            .lock()
            .expect("encoder cache lock poisoned")
            .insert(
                CacheModality::Image,
                input.key.hash,
                per_slice_outputs.clone(),
            );
        Ok(per_slice_outputs)
    }

    fn get_vllm_embedding(
        &self,
        input_ids: &Tensor,
        visual_inputs: &[MiniCpmOVisualInput],
        legacy_maps: &[Vec<MiniCpmOLegacyMap>],
        packed_layout: Option<&PackedMultimodalLayout>,
    ) -> Result<Tensor> {
        let mut embedding = self.llm.get_input_embeddings(input_ids)?;
        let mut encoder_outputs = MultimodalEncoderOutputs::new();
        for input in visual_inputs {
            let outputs = self.encode_visual_input(self.llm.device(), input)?;
            encoder_outputs.insert(input.key, outputs);
        }

        if let Some(layout) = packed_layout {
            return layout.splice_embeddings(&embedding, &encoder_outputs);
        }
        if legacy_maps.len() != input_ids.dim(0)? {
            candle_core::bail!(
                "MiniCPMO legacy map count {} does not match batch {}",
                legacy_maps.len(),
                input_ids.dim(0)?
            );
        }
        let hidden_size = embedding.dim(2)?;
        for (batch, maps) in legacy_maps.iter().enumerate() {
            for map in maps {
                if map.destination.end > input_ids.dim(1)? {
                    candle_core::bail!("MiniCPMO image destination exceeds the input row");
                }
                let outputs = encoder_outputs.get(&map.key).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "missing MiniCPMO image output with hash {}",
                        map.key.hash
                    ))
                })?;
                let output = outputs.get(map.source_output).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "missing MiniCPMO slice output {} for hash {}",
                        map.source_output, map.key.hash
                    ))
                })?;
                if output.dims2()? != (map.destination.len(), hidden_size) {
                    candle_core::bail!(
                        "MiniCPMO slice output shape {:?} does not match destination {:?}",
                        output.shape(),
                        map.destination
                    );
                }
                embedding = embedding.slice_assign(
                    &[batch..batch + 1, map.destination.clone(), 0..hidden_size],
                    &output
                        .to_device(embedding.device())?
                        .to_dtype(embedding.dtype())?
                        .unsqueeze(0)?,
                )?;
            }
        }
        Ok(embedding)
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        visual_inputs: &[MiniCpmOVisualInput],
        legacy_maps: &[Vec<MiniCpmOLegacyMap>],
        packed_layout: Option<&PackedMultimodalLayout>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let vllm_embedding =
            self.get_vllm_embedding(input_ids, visual_inputs, legacy_maps, packed_layout)?;

        self.llm.forward_embed(input_ids, vllm_embedding, ctx)
    }
}

#[derive(Default)]
pub(crate) struct MiniCpmOSpecificArgs {
    pub(crate) visual_inputs: Vec<MiniCpmOVisualInput>,
    pub(crate) legacy_maps: Vec<Vec<MiniCpmOLegacyMap>>,
    pub(crate) packed_layout: Option<PackedMultimodalLayout>,
}

impl crate::speculative::SpeculativeTargetMixin for MiniCpmOModel {}

impl crate::block_diffusion::BlockDiffusionMixin for MiniCpmOModel {}

impl MultimodalModel for MiniCpmOModel {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn supports_mixed_media_batches(&self) -> bool {
        true
    }

    fn cache(&self) -> &EitherCache {
        self.llm.cache()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.llm.config()
    }
    fn device(&self) -> &Device {
        self.llm.device()
    }
    fn max_seq_len(&self) -> usize {
        self.llm.max_seq_len()
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let MiniCpmOSpecificArgs {
            visual_inputs,
            legacy_maps,
            packed_layout,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `MiniCpmOSpecificArgs`");
        self.forward(
            input_ids,
            &visual_inputs,
            &legacy_maps,
            packed_layout.as_ref(),
            ctx,
        )
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        let batch_size = input_ids.dim(0).unwrap_or(1);
        Box::new(MiniCpmOSpecificArgs {
            visual_inputs: Vec::new(),
            legacy_maps: (0..batch_size).map(|_| Vec::new()).collect(),
            packed_layout: None,
        })
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

impl IsqModel for MiniCpmOModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("llm").extend(self.llm.residual_tensors());
        uvb.pp("vpm").extend(self.vpm.residual_tensors());
        uvb.pp("resampler")
            .extend(self.resampler.residual_tensors());

        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for MiniCpmOModel {}
