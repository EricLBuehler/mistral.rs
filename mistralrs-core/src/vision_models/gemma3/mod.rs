#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{Context, DType, Device, Result, Tensor, D};
use config::Gemma3Config;
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use mmproj::Gemma3MultiModalProjector;
use text::TextModel;

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    paged_attention::{
        encoder_cache::{cached_encode_images, EncoderCacheManager},
        AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    AnyMoeConfig, AnyMoeExpertType,
};

pub mod config;
mod inputs_processor;
mod mmproj;
mod text;
pub(crate) use inputs_processor::Gemma3Processor;

use super::siglip::SiglipVisionTransformer;

pub struct Gemma3Model {
    language_model: TextModel,
    multi_modal_projector: Option<Gemma3MultiModalProjector>,
    vision_tower: Option<SiglipVisionTransformer>,
    cfg: Gemma3Config,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Gemma3Model {
    pub fn new(
        cfg: &Gemma3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        match cfg {
            Gemma3Config::Text(text_cfg) => Ok(Self {
                language_model: TextModel::new(
                    text_cfg,
                    vb,
                    is_gptx,
                    normal_loading_metadata,
                    attention_mechanism,
                    None,
                )?,
                multi_modal_projector: None,
                vision_tower: None,
                cfg: cfg.clone(),
                encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
            }),
            Gemma3Config::WithVision {
                text_config,
                vision_config,
                image_token_index,
                mm_tokens_per_image: _,
            } => {
                assert!(*image_token_index < text_config.vocab_size);
                Ok(Self {
                    multi_modal_projector: Some(Gemma3MultiModalProjector::new(
                        cfg,
                        vb.pp("multi_modal_projector")
                            .set_device(normal_loading_metadata.real_device.clone()),
                    )?),
                    vision_tower: Some(SiglipVisionTransformer::new(
                        vision_config,
                        vb.pp("vision_tower")
                            .pp("vision_model")
                            .set_device(normal_loading_metadata.real_device.clone()),
                    )?),
                    language_model: TextModel::new(
                        text_config,
                        vb.pp("language_model"),
                        is_gptx,
                        normal_loading_metadata,
                        attention_mechanism,
                        Some(*image_token_index),
                    )?,
                    cfg: cfg.clone(),
                    encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
                })
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        image_hashes: &[u64],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens(input_ids)?;
        let has_images = pixel_values.is_some();
        if let Some(pixel_values) = pixel_values {
            let Gemma3Config::WithVision {
                image_token_index, ..
            } = &self.cfg
            else {
                unreachable!()
            };
            let special_image_mask = input_ids
                .eq(*image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            let vision_tower = self
                .vision_tower
                .as_ref()
                .context("This model does not support vision.")?;
            let multi_modal_projector = self.multi_modal_projector.as_ref().unwrap();
            let dtype = vision_tower.dtype();

            let image_features = cached_encode_images(
                image_hashes,
                &pixel_values.to_dtype(dtype)?,
                &self.encoder_cache,
                |pv| {
                    let vision_outputs = vision_tower.forward(pv, None, None)?;
                    Ok(vec![multi_modal_projector.forward(&vision_outputs)?])
                },
            )?[0]
                .clone();

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;
        };
        let res = self.language_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
            has_images,
        )?;
        Ok(res)
    }
}

impl IsqModel for Gemma3Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        self.language_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        match &self.cfg {
            Gemma3Config::Text(_) => self.language_model.residual_tensors(),
            Gemma3Config::WithVision { .. } => {
                let vision_tower = self.vision_tower.as_ref().unwrap();
                let multi_modal_projector = self.multi_modal_projector.as_ref().unwrap();

                let uvb = UnVarBuilder::new();
                uvb.pp("multi_modal_projector")
                    .extend(multi_modal_projector.residual_tensors());
                uvb.pp("language_model")
                    .extend(self.language_model.residual_tensors());
                uvb.pp("vision_tower")
                    .pp("vision_model")
                    .extend(vision_tower.residual_tensors());

                uvb.to_safetensors()
            }
        }
    }

    fn imatrix_names(&self) -> candle_core::Result<Vec<Option<String>>> {
        self.language_model.imatrix_names()
    }
}

pub struct Gemma3SpecificArgs {
    pub image_hashes: Vec<u64>,
}

impl VisionModel for Gemma3Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let Gemma3SpecificArgs { image_hashes } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Gemma3SpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            &image_hashes,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn std::any::Any> {
        Box::new(Gemma3SpecificArgs {
            image_hashes: vec![],
        })
    }
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
    }
    fn device(&self) -> &Device {
        self.language_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.language_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.language_model.config()
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

impl AnyMoeBaseModelMixin for Gemma3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.language_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.language_model.get_mlps_mut()
    }
    fn create_anymoe_layers(
        &mut self,
        additional_vbs: Vec<ShardedVarBuilder>,
        config: AnyMoeConfig,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        gate_vb: Option<ShardedVarBuilder>,
    ) -> Result<()> {
        self.language_model.create_anymoe_layers(
            additional_vbs,
            config,
            (prefix, mlp),
            layers,
            expert_type,
            gate_vb,
        )
    }
    fn amoe_supported(&self) -> bool {
        true
    }
}
