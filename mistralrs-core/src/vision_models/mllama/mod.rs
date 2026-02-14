#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod inputs_processor;
mod text;
mod vision;

use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, Mutex},
};

pub(crate) use config::{MLlamaConfig, MLlamaRopeScaling, MLlamaRopeType, MLlamaTextConfig};
use config::{MLlamaVisionConfig, VisionActivation};
pub(crate) use inputs_processor::MLlamaProcessor;
use text::MLlamaTextModel;
use vision::MLlamaVisionModel;

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::{CollectedImatrixData, QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{linear, GetFloatInfo},
    layers_masker::masked_fill,
    ops::RepeatInterleaveOp,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
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

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        aspect_ratio_mask: Option<&Tensor>,
        aspect_ratio_ids: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        image_hashes: &[u64],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        let cross_attn_states = if let Some(pixel_values) = pixel_values {
            let Some(aspect_ratio_mask) = aspect_ratio_mask else {
                candle_core::bail!("`aspect_ratio_mask` must be specified if `pixel_values` is.");
            };
            let Some(aspect_ratio_ids) = aspect_ratio_ids else {
                candle_core::bail!("`aspect_ratio_ids` must be specified if `pixel_values` is.");
            };

            let n_images = image_hashes.len();
            if n_images > 0 {
                let mut per_image: Vec<Tensor> = Vec::with_capacity(n_images);
                let mut miss_indices = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image.push(cached[0].clone());
                        } else {
                            per_image.push(Tensor::zeros(
                                1,
                                candle_core::DType::F32,
                                pixel_values.device(),
                            )?);
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &idx in &miss_indices {
                        let single_pv = pixel_values.get(idx)?.unsqueeze(0)?;
                        let single_ar_mask = aspect_ratio_mask.get(idx)?.unsqueeze(0)?;
                        let single_ar_ids = aspect_ratio_ids.get(idx)?.unsqueeze(0)?;
                        let vision_outputs = self.vision_model.forward(
                            &single_pv,
                            &single_ar_ids,
                            &single_ar_mask,
                        )?;
                        let feats = self
                            .multi_modal_projector
                            .forward(&vision_outputs.flatten(0, 1)?)?
                            .reshape(((), vision_outputs.dim(D::Minus2)?, self.hidden_size))?
                            .to_dtype(self.dtype)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[idx], vec![feats.clone()]);
                        }
                        per_image[idx] = feats;
                    }
                }
                Some(Tensor::cat(&per_image, 0)?)
            } else {
                let vision_outputs =
                    self.vision_model
                        .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask)?;
                let cross_attention_states = self
                    .multi_modal_projector
                    .forward(&vision_outputs.flatten(0, 1)?)?
                    .reshape(((), vision_outputs.dim(D::Minus2)?, self.hidden_size))?
                    .to_dtype(self.dtype)?;
                Some(cross_attention_states)
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

        self.language_model.forward(
            input_ids,
            cross_attn_states.as_ref(),
            cross_attn_mask.as_ref(),
            full_text_row_masked_out_mask.as_ref(),
            seqlen_offsets,
            context_lens,
        )
    }
}

#[derive(Default)]
pub(crate) struct MLlamaSpecificArgs {
    pub aspect_ratio_ids: Option<Tensor>,
    pub aspect_ratio_mask: Option<Tensor>,
    pub cross_attn_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
}

impl VisionModel for MLlamaModel {
    fn cache(&self) -> &EitherCache {
        &self.language_model.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.language_model.cache
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
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        _metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        _flash_params: &FlashParams,
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
            seqlen_offsets,
            context_lens,
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
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let (mut layers, mapper) = self.language_model.get_layers();
        layers.extend(
            self.vision_model
                .get_isq_layers()
                .into_iter()
                .map(|layer| (layer, None)),
        );
        (layers, mapper)
    }

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

    // NOTE: We ONLY calibrate the text bits of these models, so we should only track/return those parts!!

    /// This is used for imatrix generation internally. Begin stats tracking.
    fn begin_track_stats(&mut self) -> anyhow::Result<()> {
        let layers = self
            .language_model
            .get_layers()
            .0
            .into_iter()
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        for layer in layers {
            Arc::get_mut(layer).unwrap().begin_track_stats()?;
        }
        Ok(())
    }

    /// End stats tracking and return the imatrix data
    fn extract_imatrix_data(&mut self) -> candle_core::Result<CollectedImatrixData> {
        let layers = self
            .language_model
            .get_layers()
            .0
            .into_iter()
            .enumerate()
            .map(|(i, (layer, _))| (i, layer))
            .collect::<Vec<_>>();
        let mut data = HashMap::new();
        for (i, layer) in layers {
            data.insert(i, Some(layer.end_track_stats()?.to_vec1::<f32>()?));
        }
        Ok(CollectedImatrixData(data))
    }
}

impl AnyMoeBaseModelMixin for MLlamaModel {}
