#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod text;

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};
use text::TextModel;
use vision::Llama4VisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    layers::linear_no_bias,
    paged_attention::{
        block_hash::MultimodalKind,
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::FlashParams, EitherCache, IsqModel, ModelForwardContext,
        MultimodalModel, NormalLoadingMetadata, NormalModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    vision_models::multimodal_layout::{
        MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
    },
};

mod config;
mod inputs_processor;
mod vision;

pub(crate) use config::{Llama4Config, TextConfig};
pub(crate) use inputs_processor::{Llama4ImageProcessor, Llama4Processor, IMAGE_TOKEN};

struct Llama4MultiModalProjector {
    linear_1: Linear,
}

impl Llama4MultiModalProjector {
    fn new(cfg: &Llama4Config, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            linear_1: linear_no_bias(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("linear_1"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_1.forward(xs)
    }
}

pub struct Llama4Model {
    language_model: TextModel,
    vision_model: Llama4VisionModel,
    multi_modal_projector: Llama4MultiModalProjector,
    image_token_index: usize,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Llama4Model {
    pub fn new(
        cfg: &Llama4Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vision_model = Llama4VisionModel::new(
            &cfg.vision_config,
            vb.pp("vision_model"),
            &normal_loading_metadata.real_device,
            &normal_loading_metadata.mapper.get_comm_for(0)?,
            &normal_loading_metadata.multi_progress,
        )?;
        let multi_modal_projector = Llama4MultiModalProjector::new(
            cfg,
            vb.pp("multi_modal_projector")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let language_model = TextModel::new(
            &cfg.text_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        Ok(Self {
            language_model,
            vision_model,
            multi_modal_projector,
            image_token_index: cfg.image_token_index,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        args: &Llama4ModelSpecificArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        if args.packed_prefill && args.packed_layout.is_none() {
            candle_core::bail!("packed Llama4 prefill is missing its multimodal layout");
        }
        let mut input_embeds = self.language_model.get_input_embeddings(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            pixel_values.dims4()?;
            let special_image_mask = input_ids
                .eq(self.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            let (image_features, encoder_outputs) = if args.image_hashes.is_empty() {
                if args.packed_prefill {
                    candle_core::bail!("packed Llama4 media input has no image hashes");
                }
                let feats = self.vision_model.forward(&pixel_values)?;
                let flat = feats.reshape(((), feats.dim(D::Minus1)?))?;
                (self.multi_modal_projector.forward(&flat)?, None)
            } else {
                if args.image_hashes.len() != args.tile_counts.len()
                    || args.image_hashes.len() != args.image_token_counts.len()
                {
                    candle_core::bail!(
                        "Llama4 has {} image hashes, {} tile counts, and {} token counts",
                        args.image_hashes.len(),
                        args.tile_counts.len(),
                        args.image_token_counts.len()
                    );
                }
                let mut offsets = Vec::with_capacity(args.tile_counts.len() + 1);
                offsets.push(0usize);
                for &count in &args.tile_counts {
                    if count == 0 {
                        candle_core::bail!("Llama4 image has no tiles");
                    }
                    offsets.push(
                        offsets
                            .last()
                            .copied()
                            .unwrap()
                            .checked_add(count)
                            .ok_or_else(|| candle_core::Error::msg("Llama4 tile count overflow"))?,
                    );
                }
                if offsets.last().copied().unwrap_or_default() != pixel_values.dim(0)? {
                    candle_core::bail!(
                        "Llama4 has {} pixel tiles but tile counts total {}",
                        pixel_values.dim(0)?,
                        offsets.last().copied().unwrap_or_default()
                    );
                }
                let mut per_image = vec![None; args.image_hashes.len()];
                let mut miss_indices = Vec::new();
                {
                    let mut cache = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (index, &hash) in args.image_hashes.iter().enumerate() {
                        if let Some(outputs) = cache.get(CacheModality::Image, hash) {
                            let valid = outputs.len() == 1
                                && outputs[0].rank() == 2
                                && outputs[0].dim(0)? == args.image_token_counts[index];
                            if valid {
                                per_image[index] = Some(outputs);
                            } else {
                                miss_indices.push(index);
                            }
                        } else {
                            miss_indices.push(index);
                        }
                    }
                }
                for &index in &miss_indices {
                    let pv = pixel_values.narrow(0, offsets[index], args.tile_counts[index])?;
                    let feats = self.vision_model.forward(&pv)?;
                    let flat = feats.reshape(((), feats.dim(D::Minus1)?))?;
                    let output = self.multi_modal_projector.forward(&flat)?;
                    if output.dim(0)? != args.image_token_counts[index] {
                        candle_core::bail!(
                            "Llama4 image encoder returned {} rows for {} placeholders",
                            output.dim(0)?,
                            args.image_token_counts[index]
                        );
                    }
                    let outputs = vec![output];
                    self.encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned")
                        .insert(
                            CacheModality::Image,
                            args.image_hashes[index],
                            outputs.clone(),
                        );
                    per_image[index] = Some(outputs);
                }
                let per_image = per_image
                    .into_iter()
                    .map(|outputs| outputs.expect("all Llama4 images should be resolved"))
                    .collect::<Vec<_>>();
                let image_features = Tensor::cat(
                    &per_image
                        .iter()
                        .map(|outputs| outputs[0].clone())
                        .collect::<Vec<_>>(),
                    0,
                )?;
                let encoder_outputs = args
                    .image_hashes
                    .iter()
                    .copied()
                    .zip(per_image)
                    .map(|(hash, outputs)| {
                        (
                            MultimodalEncoderKey {
                                kind: MultimodalKind::Image,
                                hash,
                            },
                            outputs,
                        )
                    })
                    .collect::<MultimodalEncoderOutputs>();
                (image_features, Some(encoder_outputs))
            };

            if let Some(layout) = &args.packed_layout {
                input_embeds = layout.splice_embeddings(
                    &input_embeds,
                    &encoder_outputs.ok_or_else(|| {
                        candle_core::Error::msg(
                            "packed Llama4 input requires per-image encoder outputs",
                        )
                    })?,
                )?;
            } else {
                let mut x_flat = input_embeds.flatten_all()?;
                let src_flat = image_features.flatten_all()?;
                if src_flat.dim(0)? != indices.dim(0)? {
                    candle_core::bail!(
                        "Llama4 has {} image embedding values but {} placeholder values",
                        src_flat.dim(0)?,
                        indices.dim(0)?
                    );
                }
                let current_vals = x_flat.gather(&indices, 0)?;
                let diff = (src_flat - current_vals)?;
                x_flat = x_flat.scatter_add(&indices, &diff, 0)?;
                input_embeds = x_flat.reshape(input_embeds.shape())?;
            }
        }

        self.language_model
            .forward_embeds(input_ids, input_embeds, ctx)
    }
}

impl IsqModel for Llama4Model {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("multi_modal_projector")
            .pp("linear_1")
            .add(&self.multi_modal_projector.linear_1);
        uvb.pp("language_model")
            .extend(self.language_model.residual_tensors());
        uvb.pp("vision_model")
            .extend(self.vision_model.residual_tensors());

        uvb.to_safetensors()
    }
}

pub struct Llama4ModelSpecificArgs {
    pub image_hashes: Vec<u64>,
    pub tile_counts: Vec<usize>,
    pub image_token_counts: Vec<usize>,
    pub packed_layout: Option<PackedMultimodalLayout>,
    pub packed_prefill: bool,
}

impl crate::speculative::SpeculativeTargetMixin for Llama4Model {}

impl NormalModel for Llama4Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
            None,
            &Llama4ModelSpecificArgs {
                image_hashes: vec![],
                tile_counts: vec![],
                image_token_counts: vec![],
                packed_layout: None,
                packed_prefill: false,
            },
            ctx,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
}

impl crate::block_diffusion::BlockDiffusionMixin for Llama4Model {}

impl MultimodalModel for Llama4Model {
    fn supports_packed_prefill(&self) -> bool {
        true
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn std::any::Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        let args = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Llama4ModelSpecificArgs`");
        self.forward(input_ids, pixel_values, &args, ctx)
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Llama4ModelSpecificArgs {
            image_hashes: vec![],
            tile_counts: vec![],
            image_token_counts: vec![],
            packed_layout: None,
            packed_prefill: false,
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

impl AnyMoeBaseModelMixin for Llama4Model {}
