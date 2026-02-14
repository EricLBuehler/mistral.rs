#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod config;
mod inputs_processor;
mod vision;

use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor, D};
pub use config::Idefics3Config;
pub use inputs_processor::Idefics3Processor;
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};
use vision::{Idefics3Connector, Idefics3VisionTransformer};

use crate::{
    amoe::{AnyMoeBaseModelMixin, MlpLayer},
    device_map::DeviceMapper,
    models::llama::Llama,
    paged_attention::{
        encoder_cache::EncoderCacheManager, AttentionImplementation, ModelConfigMetadata,
    },
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, NormalLoadingMetadata, NormalModel, VisionModel,
    },
    utils::unvarbuilder::UnVarBuilder,
    AnyMoeConfig, AnyMoeExpertType,
};

pub(crate) struct Idefics3SpecificArgs {
    pub pixel_attention_mask: Option<Tensor>,
    pub image_hashes: Vec<u64>,
}

pub struct Idefics3Model {
    text_model: Llama,
    connector: Idefics3Connector,
    vision: Idefics3VisionTransformer,
    config: Idefics3Config,
    dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Idefics3Model {
    pub fn new(
        cfg: &Idefics3Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let connector = Idefics3Connector::new(
            cfg,
            vb_m.pp("connector")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let vision = Idefics3VisionTransformer::new(
            &cfg.vision_config,
            vb_m.pp("vision_model")
                .set_device(normal_loading_metadata.real_device.clone()),
        )?;
        let text_model = Llama::new_inner(
            &cfg.text_config,
            vb_m.pp("text_model"),
            vb.pp("lm_head"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;
        Ok(Self {
            text_model,
            connector,
            vision,
            config: cfg.clone(),
            dtype: vb.dtype(),
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    fn inputs_merger(
        &self,
        indices: &Tensor,
        input_embeds: &Tensor,
        image_hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let mut x_flat = input_embeds.flatten_all()?;
        let src_flat = image_hidden_states.flatten_all()?;

        let current_vals = x_flat.gather(indices, 0)?;
        let diff = (src_flat - current_vals)?;
        x_flat = x_flat.scatter_add(indices, &diff, 0)?;

        x_flat.reshape(input_embeds.shape())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        pixel_attention_mask: Option<Tensor>,
        image_hashes: &[u64],
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let input_embeds = if let Some(pixel_values) = pixel_values {
            let input_embeds = self.text_model.get_input_embeddings(input_ids)?;
            let special_image_mask = input_ids
                .eq(self.config.image_token_id as f64)?
                .unsqueeze(D::Minus1)?
                .broadcast_as(input_embeds.shape())?
                .to_dtype(DType::U32)?;

            let mask_flat = special_image_mask.flatten_all()?;
            // Nonzero before vision model to allow async processing all the way through logits.
            let indices = mask_flat.nonzero()?.squeeze(1)?;

            // == START VISUAL INPUTS INTEGRATION ==
            let (batch_size, num_images, _, _, _) = pixel_values.dims5()?;
            let mut s = vec![batch_size * num_images];
            s.extend(pixel_values.dims()[2..].to_vec());
            let pixel_values = pixel_values.reshape(s)?;

            // Remove padding images which are full of 0s
            let nb_values_per_image = pixel_values.dims()[1..].iter().product::<usize>();
            let real_images_inds = pixel_values
                .eq(0.0f64)?
                .sum(vec![
                    pixel_values.dims().len() - 1,
                    pixel_values.dims().len() - 2,
                    pixel_values.dims().len() - 3,
                ])?
                .ne(nb_values_per_image as f64)?;
            let mut batches = Vec::new();
            for (batch, use_it) in pixel_values
                .chunk(pixel_values.dim(0)?, 0)?
                .iter()
                .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
            {
                let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                if use_it {
                    batches.push(batch.clone());
                }
            }
            let pixel_values = Tensor::cat(&batches, 0)?;

            // Vision attention mask
            let pixel_attention_mask = if let Some(pixel_attention_mask) = pixel_attention_mask {
                let pixel_attention_mask = pixel_attention_mask.reshape((
                    batch_size * num_images,
                    pixel_attention_mask.dims()[2],
                    pixel_attention_mask.dims()[3],
                ))?;
                let mut batches = Vec::new();
                for (batch, use_it) in pixel_attention_mask
                    .chunk(pixel_attention_mask.dim(0)?, 0)?
                    .iter()
                    .zip(real_images_inds.chunk(real_images_inds.dim(0)?, 0)?)
                {
                    let use_it = use_it.squeeze(0)?.to_scalar::<u8>()? != 0;
                    if use_it {
                        batches.push(batch.clone());
                    }
                }
                Tensor::cat(&batches, 0)?
            } else {
                Tensor::ones(
                    (
                        pixel_values.dims()[0],
                        pixel_values.dims()[2],
                        pixel_values.dims()[3],
                    ),
                    DType::U8,
                    pixel_values.device(),
                )?
            };

            let patch_size = self.config.vision_config.patch_size;
            let patches_subgrid = pixel_attention_mask.unfold(1, patch_size, patch_size)?;
            let patches_subgrid = patches_subgrid.unfold(2, patch_size, patch_size)?;

            let patch_attention_mask = patches_subgrid
                .sum((D::Minus1, D::Minus2))?
                .gt(0.0)?
                .to_dtype(DType::U8)?;

            let pixel_values = pixel_values.to_dtype(self.dtype)?;

            // Get seq from vision encoder + connector, with per-image caching
            let image_hidden_states = if !image_hashes.is_empty() {
                let n = pixel_values.dim(0)?;
                let mut per_image: Vec<Option<Tensor>> = vec![None; n];
                let mut miss_indices: Vec<usize> = Vec::new();
                {
                    let mut guard = self
                        .encoder_cache
                        .lock()
                        .expect("encoder cache lock poisoned");
                    for (i, &hash) in image_hashes.iter().enumerate() {
                        if let Some(cached) = guard.get(hash) {
                            per_image[i] = Some(cached[0].clone());
                        } else {
                            miss_indices.push(i);
                        }
                    }
                }
                if !miss_indices.is_empty() {
                    for &i in &miss_indices {
                        let pv = pixel_values.get(i)?.unsqueeze(0)?;
                        let mask = patch_attention_mask.get(i)?.unsqueeze(0)?;
                        let hidden = self.vision.forward(&pv, Some(&mask))?;
                        let hidden = self.connector.forward(&hidden)?;
                        let result = hidden.squeeze(0)?;
                        {
                            let mut guard = self
                                .encoder_cache
                                .lock()
                                .expect("encoder cache lock poisoned");
                            guard.insert(image_hashes[i], vec![result.clone()]);
                        }
                        per_image[i] = Some(result);
                    }
                }
                let slices: Vec<Tensor> = per_image.into_iter().map(|t| t.unwrap()).collect();
                Tensor::stack(&slices, 0)?
            } else {
                // No caching: original path
                let image_hidden_states = self
                    .vision
                    .forward(&pixel_values, Some(&patch_attention_mask))?;
                self.connector.forward(&image_hidden_states)?
            };

            self.inputs_merger(&indices, &input_embeds, &image_hidden_states)?
                .to_dtype(self.dtype)?
        } else {
            self.text_model.get_input_embeddings(input_ids)?
        };

        self.text_model.forward_embeds(
            input_ids,
            input_embeds,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
}

impl IsqModel for Idefics3Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn DeviceMapper,
    ) {
        self.text_model.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m
            .pp("connector")
            .pp("modality_projection")
            .pp("proj")
            .add(&self.connector.modality_projection.proj);
        uvb.extend(self.text_model.residual_tensors_m(uvb_m.pp("text_model")));
        uvb_m
            .pp("vision_model")
            .extend(self.vision.residual_tensors());

        uvb.to_safetensors()
    }
}

// AnyMoE is forwarded to the base model
impl AnyMoeBaseModelMixin for Idefics3Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.text_model.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.text_model.get_mlps_mut()
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
        self.text_model.create_anymoe_layers(
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

impl VisionModel for Idefics3Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _: Vec<usize>, // Ignore, it is for phi3
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let Idefics3SpecificArgs {
            pixel_attention_mask,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Idefics3SpecificArgs`");
        self.forward_inner(
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            pixel_attention_mask,
            &image_hashes,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.text_model.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.text_model.cache_mut()
    }
    fn device(&self) -> &Device {
        self.text_model.device()
    }
    fn max_seq_len(&self) -> usize {
        self.text_model.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.text_model.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(Idefics3SpecificArgs {
            pixel_attention_mask: None,
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
