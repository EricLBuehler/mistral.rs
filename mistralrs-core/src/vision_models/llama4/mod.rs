#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod text;

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use mistralrs_quant::{NonZeroOp, QuantMethod, ShardedVarBuilder};
use text::TextModel;
use vision::Llama4VisionModel;

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::linear_no_bias,
    paged_attention::encoder_cache::{cached_encode_images, EncoderCacheManager},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
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
        let mut input_embeds = self.language_model.get_input_embeddings(input_ids)?;

        if let Some(pixel_values) = pixel_values {
            let special_image_mask = input_ids
                .eq(self.image_token_index as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape().clone())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            let image_features =
                cached_encode_images(image_hashes, &pixel_values, &self.encoder_cache, |pv| {
                    let feats = self.vision_model.forward(pv)?;
                    let flat = feats.reshape(((), feats.dim(D::Minus1)?))?;
                    Ok(vec![self.multi_modal_projector.forward(&flat)?])
                })?[0]
                    .clone();

            let mut x_flat = input_embeds.flatten_all()?;
            let src_flat = image_features.flatten_all()?;

            let current_vals = x_flat.gather(&indices, 0)?;
            let diff = (src_flat - current_vals)?;
            x_flat = x_flat.scatter_add(&indices, &diff, 0)?;

            input_embeds = x_flat.reshape(input_embeds.shape())?;
        }

        self.language_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Llama4Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let (mut layers, device_map) = self.language_model.get_layers();
        layers.extend(
            self.vision_model
                .get_isq_layers()
                .into_iter()
                .map(|x| (x, None)),
        );
        (layers, device_map)
    }

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
}

impl NormalModel for Llama4Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
            None,
            &[],
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
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
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
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

impl VisionModel for Llama4Model {
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
        let Llama4ModelSpecificArgs { image_hashes } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Llama4ModelSpecificArgs`");
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
    fn cache(&self) -> &EitherCache {
        self.language_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.language_model.cache_mut()
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
