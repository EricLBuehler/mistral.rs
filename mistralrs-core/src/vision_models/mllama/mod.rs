#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod inputs_processor;
mod text;
mod vision;

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

pub(crate) use config::{MLlamaConfig, MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig};
use config::{MLlamaVisionConfig, VisionActivation};
pub(crate) use inputs_processor::MLlamaProcessor;
use text::MLlamaTextModel;
use vision::MLlamaVisionModel;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::attention::AttentionMask;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    layers::{linear, GetFloatInfo},
    layers_masker::masked_fill,
    ops::RepeatInterleaveOp,
    paged_attention::{
        encoder_cache::{CacheModality, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalLoadingMetadata,
    },
    utils::unvarbuilder::UnVarBuilder,
};

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/modeling_mllama.py#L99
fn prepare_cross_attention_mask(
    cross_attention_mask: &Tensor,
    num_vision_tokens: usize,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let bs = cross_attention_mask.dim(0)?;
    let text_total_length = cross_attention_mask.dim(1)?;
    let mut cross_attn_mask = cross_attention_mask
        .to_dtype(DType::F32)?
        .repeat_interleave(num_vision_tokens, 3)?;
    cross_attn_mask = cross_attn_mask.reshape((bs, text_total_length, ()))?;
    cross_attn_mask = cross_attn_mask.unsqueeze(1)?;

    // Invert the mask
    let inverted_cross_attn_mask = (1. - cross_attn_mask)?;
    let neg_inf_value = dtype.finfo()?.min;
    cross_attn_mask = masked_fill(
        &inverted_cross_attn_mask,
        &inverted_cross_attn_mask.ne(0.)?,
        neg_inf_value as f32,
    )?;

    // Apply full-row bias which return 4d tensor of shape (b, h, s1, 1) where
    // value is 0 if a full row in cross attn mask's last dimension contains
    // negative infinity values, otherwise it's 1
    let full_text_row_masked_out_mask = cross_attn_mask
        .ne(neg_inf_value)?
        .sum(D::Minus1)?
        .ne(0.)?
        .unsqueeze(D::Minus1)?;

    cross_attn_mask = cross_attn_mask
        .broadcast_mul(&full_text_row_masked_out_mask.to_dtype(cross_attn_mask.dtype())?)?
        .to_dtype(DType::F32)?
        .to_dtype(dtype)?;

    Ok((cross_attn_mask, full_text_row_masked_out_mask))
}

fn image_cache_keys(
    image_hashes: &[Vec<u64>],
    batch_size: usize,
    max_num_images: usize,
) -> Result<Vec<(usize, usize, u64)>> {
    if image_hashes.len() != batch_size {
        candle_core::bail!(
            "image hash batch size {} does not match pixel batch size {batch_size}",
            image_hashes.len()
        );
    }

    let mut keys = Vec::new();
    for (batch_idx, hashes) in image_hashes.iter().enumerate() {
        if hashes.len() > max_num_images {
            candle_core::bail!(
                "image hash count {} exceeds padded image count {max_num_images}",
                hashes.len()
            );
        }
        keys.extend(
            hashes
                .iter()
                .enumerate()
                .map(|(image_idx, &hash)| (batch_idx, image_idx, hash)),
        );
    }
    Ok(keys)
}

pub(crate) struct MLlamaModel {
    vision_model: MLlamaVisionModel,
    language_model: MLlamaTextModel,
    multi_modal_projector: Linear,
    hidden_size: usize,
    dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl MLlamaModel {
    pub(crate) fn new(
        cfg: &MLlamaConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let real_dev = normal_loading_metadata.real_device.clone();
        Ok(Self {
            vision_model: MLlamaVisionModel::new(
                &cfg.vision_config,
                vb.pp("vision_model"),
                &real_dev,
                &normal_loading_metadata.mapper.get_comm_for(0)?,
            )?,
            language_model: MLlamaTextModel::new(
                &cfg.text_config,
                vb.pp("language_model"),
                is_gptx,
                normal_loading_metadata,
                attention_mechanism,
            )?,
            multi_modal_projector: linear(
                cfg.vision_config.vision_output_dim,
                cfg.text_config.hidden_size,
                vb.pp("multi_modal_projector").set_device(real_dev.clone()),
            )?,
            hidden_size: cfg.text_config.hidden_size,
            dtype: vb.dtype(),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn project_vision_outputs(&self, vision_outputs: &Tensor) -> Result<Tensor> {
        let batch_size = vision_outputs.dim(0)?;
        let num_images = vision_outputs.dim(1)?;
        let num_tiles = vision_outputs.dim(2)?;
        let num_patches = vision_outputs.dim(3)?;
        self.multi_modal_projector
            .forward(&vision_outputs.flatten(0, 1)?)?
            .reshape((
                batch_size,
                num_images * num_tiles * num_patches,
                self.hidden_size,
            ))?
            .to_dtype(self.dtype)
    }

    fn encode_image(
        &self,
        pixel_values: &Tensor,
        aspect_ratio_ids: &Tensor,
        aspect_ratio_mask: &Tensor,
        batch_idx: usize,
        image_idx: usize,
    ) -> Result<Tensor> {
        let pixel_values = pixel_values
            .get(batch_idx)?
            .get(image_idx)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let aspect_ratio_ids = aspect_ratio_ids
            .get(batch_idx)?
            .get(image_idx)?
            .reshape((1, 1))?;
        let aspect_ratio_mask =
            aspect_ratio_mask
                .get(batch_idx)?
                .get(image_idx)?
                .reshape((1, 1, ()))?;
        let vision_outputs =
            self.vision_model
                .forward(&pixel_values, &aspect_ratio_ids, &aspect_ratio_mask)?;
        self.project_vision_outputs(&vision_outputs)?.squeeze(0)
    }

    fn cached_cross_attention_states(
        &self,
        pixel_values: &Tensor,
        aspect_ratio_ids: &Tensor,
        aspect_ratio_mask: &Tensor,
        image_hashes: &[Vec<u64>],
    ) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;
        let max_num_images = pixel_values.dim(1)?;
        let max_num_tiles = pixel_values.dim(2)?;
        let keys = image_cache_keys(image_hashes, batch_size, max_num_images)?;
        let mut states = image_hashes
            .iter()
            .map(|hashes| vec![None; hashes.len()])
            .collect::<Vec<Vec<Option<Tensor>>>>();
        let mut misses = Vec::new();

        {
            let mut cache = self
                .encoder_cache
                .lock()
                .expect("encoder cache lock poisoned");
            for &(batch_idx, image_idx, hash) in &keys {
                if let Some(cached) = cache.get(CacheModality::Image, hash) {
                    states[batch_idx][image_idx] = Some(cached[0].clone());
                } else {
                    misses.push((batch_idx, image_idx, hash));
                }
            }
        }

        for (batch_idx, image_idx, hash) in misses {
            let state = self.encode_image(
                pixel_values,
                aspect_ratio_ids,
                aspect_ratio_mask,
                batch_idx,
                image_idx,
            )?;
            self.encoder_cache
                .lock()
                .expect("encoder cache lock poisoned")
                .insert(CacheModality::Image, hash, vec![state.clone()]);
            states[batch_idx][image_idx] = Some(state);
        }

        let tokens_per_image = max_num_tiles * self.vision_model.num_patches;
        let mut batch_states = Vec::with_capacity(batch_size);
        for sample_states in states {
            let num_images = sample_states.len();
            let mut sample_states = sample_states
                .into_iter()
                .map(|state| state.expect("all image states should be resolved"))
                .collect::<Vec<_>>();
            for _ in num_images..max_num_images {
                sample_states.push(Tensor::zeros(
                    (tokens_per_image, self.hidden_size),
                    self.dtype,
                    pixel_values.device(),
                )?);
            }
            batch_states.push(Tensor::cat(&sample_states, 0)?);
        }
        Tensor::stack(&batch_states, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        aspect_ratio_mask: Option<&Tensor>,
        aspect_ratio_ids: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        image_hashes: &[Vec<u64>],
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let cross_attn_states = if let Some(pixel_values) = pixel_values {
            let Some(aspect_ratio_mask) = aspect_ratio_mask else {
                candle_core::bail!("`aspect_ratio_mask` must be specified if `pixel_values` is.");
            };
            let Some(aspect_ratio_ids) = aspect_ratio_ids else {
                candle_core::bail!("`aspect_ratio_ids` must be specified if `pixel_values` is.");
            };

            if image_hashes.iter().any(|hashes| !hashes.is_empty()) {
                Some(self.cached_cross_attention_states(
                    pixel_values,
                    aspect_ratio_ids,
                    aspect_ratio_mask,
                    image_hashes,
                )?)
            } else {
                let vision_outputs =
                    self.vision_model
                        .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask)?;
                Some(self.project_vision_outputs(&vision_outputs)?)
            }
        } else {
            None
        };

        let (cross_attn_mask, full_text_row_masked_out_mask) =
            if let Some(cross_attn_mask) = cross_attn_mask {
                let (mut cmask, fmask) = prepare_cross_attention_mask(
                    cross_attn_mask,
                    self.vision_model.num_patches,
                    self.dtype,
                )?;
                cmask = cmask.squeeze(1)?;
                (Some(cmask), Some(fmask))
            } else {
                (None, None)
            };

        let cross_attn_mask_enum = match &cross_attn_mask {
            Some(t) => AttentionMask::Custom(t.clone()),
            None => AttentionMask::None,
        };
        self.language_model.forward(
            input_ids,
            cross_attn_states.as_ref(),
            &cross_attn_mask_enum,
            full_text_row_masked_out_mask.as_ref(),
            ctx,
        )
    }
}

#[derive(Default)]
pub(crate) struct MLlamaSpecificArgs {
    pub aspect_ratio_ids: Option<Tensor>,
    pub aspect_ratio_mask: Option<Tensor>,
    pub cross_attn_mask: Option<Tensor>,
    pub image_hashes: Vec<Vec<u64>>,
}

impl crate::speculative::SpeculativeTargetMixin for MLlamaModel {}

impl crate::block_diffusion::BlockDiffusionMixin for MLlamaModel {}

impl MultimodalModel for MLlamaModel {
    fn cache(&self) -> &EitherCache {
        &self.language_model.cache
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.language_model.cfg
    }
    fn device(&self) -> &Device {
        &self.language_model.device
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_position_embeddings
    }
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        ctx: &mut crate::pipeline::ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        let MLlamaSpecificArgs {
            aspect_ratio_ids,
            aspect_ratio_mask,
            cross_attn_mask,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `MLlamaSpecificArgs`");
        self.forward_inner(
            input_ids,
            pixel_values.as_ref(),
            aspect_ratio_mask.as_ref(),
            aspect_ratio_ids.as_ref(),
            cross_attn_mask.as_ref(),
            &image_hashes,
            ctx,
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(MLlamaSpecificArgs::default())
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

impl IsqModel for MLlamaModel {
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        uvb.pp("multi_modal_projector")
            .add(&self.multi_modal_projector);
        uvb.pp("language_model")
            .extend(self.language_model.residual_tensors());
        uvb.pp("vision_model")
            .extend(self.vision_model.residual_tensors());

        uvb.to_safetensors()
    }
}

impl AnyMoeBaseModelMixin for MLlamaModel {}

#[cfg(test)]
mod tests {
    use super::image_cache_keys;

    #[test]
    fn image_cache_keys_keep_batch_and_image_coordinates() {
        let keys = image_cache_keys(&[vec![11, 12], vec![21]], 2, 2).unwrap();
        assert_eq!(keys, vec![(0, 0, 11), (0, 1, 12), (1, 0, 21)]);
    }

    #[test]
    fn image_cache_keys_reject_invalid_padding_layout() {
        assert!(image_cache_keys(&[vec![11, 12]], 1, 1).is_err());
        assert!(image_cache_keys(&[vec![11]], 2, 1).is_err());
    }
}
