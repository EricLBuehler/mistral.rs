#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
use std::any::Any;
use std::sync::{Arc, Mutex};

use candle_core::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Linear};
use mistralrs_quant::{NonZeroOp, ShardedVarBuilder};

use crate::amoe::{AnyMoeBaseModelMixin, MlpLayer};
use crate::device_map::DeviceMapper;
use crate::paged_attention::encoder_cache::EncoderCacheManager;
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::IsqModel;
use crate::pipeline::NormalLoadingMetadata;
use crate::pipeline::VisionModel;

use crate::utils::unvarbuilder::UnVarBuilder;
use crate::vision_models::clip::{ClipConfig, ClipVisionTransformer};
use crate::vision_models::llava::config::Config;
use crate::vision_models::llava::utils::get_anyres_image_grid_shape;
use crate::{layers, AnyMoeConfig, AnyMoeExpertType};

use super::llava_llm::{LLaVALLM, Llama, Mistral};

#[derive(Default)]
pub(crate) struct LLaVANextVisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>, // width, height
    pub num_image_tokens: Option<Vec<usize>>,     // number of image tokens for each image
    pub num_image_samples: Option<Vec<usize>>,    // number of image samples for each image
    pub image_hashes: Vec<u64>,
}

pub struct MMProjector {
    linear_1: Linear,
    activation: Activation,
    linear_2: Linear,
}

impl MMProjector {
    pub fn new(vb: &ShardedVarBuilder, config: &Config, device: &Device) -> Result<Self> {
        let linear_1 = layers::linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_1")
                .set_device(device.clone()),
        )?;
        let activation = match config.projector_hidden_act.as_str() {
            "gelu" => Activation::Gelu,
            _ => {
                bail!(
                    "Unsupporg projector hidden act: {}",
                    config.projector_hidden_act
                );
            }
        };
        let linear_2 = layers::linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_2")
                .set_device(device.clone()),
        )?;
        Ok(Self {
            linear_1,
            activation,
            linear_2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.apply(&self.linear_1)?
            .apply(&self.activation)?
            .apply(&self.linear_2)
    }
}

pub struct ClipVisionTower {
    model: ClipVisionTransformer,
    select_layer: isize,
    select_feature_method: String,
    config: ClipConfig,
}

impl ClipVisionTower {
    pub fn new(
        vb: ShardedVarBuilder,
        select_layer: isize,
        select_feature_method: &str,
        config: &ClipConfig,
    ) -> Result<Self> {
        let model = ClipVisionTransformer::new(vb, config)?;
        Ok(Self {
            model,
            select_layer,
            select_feature_method: select_feature_method.to_string(),
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let result = self.model.forward_get_hidden_states(x)?;
        let index = result.len() as isize + self.select_layer;
        let result = result[index as usize].clone();
        if self.select_feature_method == "cls_patch" || self.select_feature_method == "full" {
            Ok(result)
        } else {
            result.i((.., 1..))
        }
    }

    pub fn num_patches_per_side(&self) -> usize {
        self.config.image_size / self.config.patch_size
    }
}

pub struct Model {
    clip_vision_tower: ClipVisionTower,
    image_newline: Tensor,
    mm_projector: MMProjector,
    llm: Box<dyn LLaVALLM>,
    config: Config,
    device: Device,
    dtype: DType,
    encoder_cache: Arc<Mutex<EncoderCacheManager>>,
}

impl Model {
    pub fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let dtype = vb.dtype();
        let clip_config = config.to_clip_config();
        let mm_projector = MMProjector::new(&vb, config, &device)?;
        let clip_vision_tower = ClipVisionTower::new(
            vb.pp("vision_tower.vision_model")
                .set_device(device.clone()),
            config.vision_feature_layer,
            &config.vision_feature_select_strategy,
            &clip_config,
        )?;
        let image_newline = vb
            .get(&[config.text_config.hidden_size], "image_newline")?
            .to_device(&device)?;

        let llm: Box<dyn LLaVALLM> = match config.text_config.model_type.as_str() {
            "llama" => {
                let llama_config = config.to_llama_config();
                let llama = Llama::new(
                    &llama_config,
                    vb.pp("language_model"),
                    is_gptx,
                    normal_loading_metadata,
                    attention_mechanism,
                )?;
                Box::new(llama)
            }
            "mistral" => {
                let mistral_config = config.to_mistral_config();
                let mistral = Mistral::new(
                    &mistral_config,
                    vb.pp("language_model"),
                    is_gptx,
                    normal_loading_metadata,
                    attention_mechanism,
                )?;
                Box::new(mistral)
            }
            _ => {
                bail!("Unsupported model type: {}", config.text_config.model_type);
            }
        };
        Ok(Self {
            clip_vision_tower,
            image_newline,
            mm_projector,
            llm,
            config: config.clone(),
            device,
            dtype,
            encoder_cache: Arc::new(Mutex::new(EncoderCacheManager::new(32))),
        })
    }

    pub fn encode_images(&self, x: &Tensor) -> Result<Tensor> {
        let mut image_features = self.clip_vision_tower.forward(x)?;
        image_features = self.mm_projector.forward(&image_features)?;
        Ok(image_features)
    }

    fn unpad_image(&self, tensor: &Tensor, original_size: (u32, u32)) -> Result<Tensor> {
        assert_eq!(tensor.dims().len(), 3);
        let (original_width, original_height) = original_size;
        let tensor_dims = tensor.dims();
        let current_height = tensor_dims[1];
        let current_width = tensor_dims[2];
        let original_aspect_ratio = (original_width as f32) / (original_height as f32);
        let current_aspect_ratio = (current_width as f32) / (current_height as f32);
        if original_aspect_ratio > current_aspect_ratio {
            let scale_factor = (current_width as f32) / (original_width as f32);
            let new_height = (original_height as f32 * scale_factor).floor() as usize;
            let padding = (current_height - new_height) / 2;
            tensor.i((.., padding..current_height - padding, ..))
        } else {
            let scale_factor = (current_height as f32) / (original_height as f32);
            let new_width = (original_width as f32 * scale_factor).floor() as usize;
            let padding = (current_width - new_width) / 2;
            tensor.i((.., .., padding..current_width - padding))
        }
    }

    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        input_ids: &Tensor, //[1,seq_len]
        images: &Tensor,    //[sum of samples of all images,channel,width,height]
        num_image_tokens: Vec<usize>,
        num_image_samples: Vec<usize>,
        image_sizes: &[(u32, u32)],
        image_hashes: &[u64],
    ) -> Result<Tensor> {
        let image_indexes = input_ids
            .squeeze(0)?
            .lt(0i64)?
            .nonzero()?
            .squeeze(1)?
            .to_vec1::<u32>()?;
        let mut result = input_ids.clamp(0i64, i64::MAX)?.to_dtype(DType::U32)?;
        result = self.llm.embed(&result)?; //[seq_len,hidden_size]

        let images_typed = images.to_dtype(self.dtype)?;
        let n_images = num_image_samples.len();

        // Per-image caching: each image may have multiple samples (base + tiles).
        let image_features_vec: Vec<Tensor>;
        if image_hashes.len() == n_images && n_images > 0 {
            // Compute sample offset ranges per image.
            let mut sample_offsets = vec![0usize; n_images + 1];
            for (i, &ns) in num_image_samples.iter().enumerate() {
                sample_offsets[i + 1] = sample_offsets[i] + ns;
            }

            let mut per_image: Vec<Option<Tensor>> = vec![None; n_images];
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
                // Collect miss samples and encode them as a batch.
                let miss_samples: Vec<Tensor> = miss_indices
                    .iter()
                    .flat_map(|&idx| {
                        let (start, end) = (sample_offsets[idx], sample_offsets[idx + 1]);
                        (start..end).map(|j| images_typed.get(j).unwrap())
                    })
                    .collect();
                let miss_pixels = Tensor::stack(&miss_samples, 0)?;
                let miss_encoded = self.encode_images(&miss_pixels)?;

                let mut offset = 0;
                let mut guard = self
                    .encoder_cache
                    .lock()
                    .expect("encoder cache lock poisoned");
                for &idx in &miss_indices {
                    let ns = num_image_samples[idx];
                    let encoded = miss_encoded.i(offset..offset + ns)?;
                    guard.insert(image_hashes[idx], vec![encoded.clone()]);
                    per_image[idx] = Some(encoded);
                    offset += ns;
                }
            }

            image_features_vec = per_image
                .into_iter()
                .map(|o| o.expect("all images should be resolved"))
                .collect();
        } else {
            // Fallback: no hashes, encode all at once.
            let image_features = self.encode_images(&images_typed)?;
            let mut feats = Vec::new();
            let mut index = 0;
            for num_image_sample in &num_image_samples {
                feats.push(image_features.i(index..index + num_image_sample)?);
                index += num_image_sample;
            }
            image_features_vec = feats;
        }
        let image_features_vec = image_features_vec
            .iter()
            .enumerate()
            .map(|(image_idx, image_feature)| {
                let base_image_feature = image_feature.get(0).unwrap();
                let patch_image_feature = image_feature.i(1..).unwrap();
                let height = self.clip_vision_tower.num_patches_per_side();
                let width = height;
                assert_eq!(height * width, base_image_feature.dims()[0]);
                let image_size = image_sizes[image_idx];
                let image_grid_pinpoints = self.config.image_grid_pinpoints.clone().unwrap();
                let (num_patch_width, num_patch_height) = get_anyres_image_grid_shape(
                    image_size,
                    &image_grid_pinpoints,
                    self.clip_vision_tower.config.image_size as u32,
                );
                let mut new_image_feature = patch_image_feature.reshape((
                    num_patch_height as usize,
                    num_patch_width as usize,
                    height,
                    width,
                    (),
                ))?;
                new_image_feature = new_image_feature
                    .permute((4, 0, 2, 1, 3))?
                    .flatten(1, 2)?
                    .flatten(2, 3)?;
                new_image_feature = self.unpad_image(&new_image_feature, image_size)?;
                let new_image_feature_dims = new_image_feature.dims();
                let image_new_line = self
                    .image_newline
                    .reshape((self.config.text_config.hidden_size, 1, 1))?
                    .broadcast_as((new_image_feature_dims[0], new_image_feature_dims[1], 1))?;
                new_image_feature = Tensor::cat(&[new_image_feature, image_new_line], 2)?
                    .flatten(1, 2)?
                    .transpose(0, 1)?;
                new_image_feature =
                    Tensor::cat(&[base_image_feature, new_image_feature], 0)?.unsqueeze(0)?;
                Ok(new_image_feature)
            })
            .collect::<Result<Vec<Tensor>>>()?;
        for (i, image_index) in image_indexes.iter().enumerate() {
            result = result.slice_assign(
                &[
                    0usize..1usize,
                    *image_index as usize..*image_index as usize + num_image_tokens[i],
                    0..result.dim(2)?,
                ],
                &image_features_vec[i],
            )?;
        }
        //truncate
        let (_, seq_len) = input_ids.shape().dims2()?;
        if seq_len > self.config.text_config.max_length {
            result = result.i((.., ..self.config.text_config.max_length, ..))?
        }
        Ok(result)
    }

    pub fn forward_inputs(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        num_image_tokens: Option<Vec<usize>>,
        num_image_samples: Option<Vec<usize>>,
        image_hashes: &[u64],
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        if let Some(ref pixel_values) = pixel_values {
            // we assume(as it should be) only prompt request contains image
            let input_embeds = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                pixel_values,
                num_image_tokens.unwrap(),
                num_image_samples.unwrap(),
                &image_sizes.unwrap(),
                image_hashes,
            )?;
            self.llm.forward_input_embed(
                input_ids,
                input_embeds,
                seqlen_offsets,
                context_lens,
                metadata,
                flash_params,
            )
        } else {
            self.llm.forward(
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                metadata,
                flash_params,
            )
        }
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(
            &mut std::sync::Arc<dyn mistralrs_quant::QuantMethod>,
            Option<usize>,
        )>,
        &dyn DeviceMapper,
    ) {
        self.llm.get_layers()
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        // MM projectors
        uvb.pp("multi_modal_projector.linear_1")
            .add(&self.mm_projector.linear_1);
        uvb.pp("multi_modal_projector.linear_2")
            .add(&self.mm_projector.linear_2);

        // Vision tower
        {
            let uvb_vt = uvb.pp("vision_tower.vision_model");
            uvb_vt.extend(self.clip_vision_tower.model.residual_tensors());
        }

        uvb.add_tensor("image_newline", self.image_newline.clone());

        uvb.to_safetensors()
    }
}

impl VisionModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let LLaVANextVisionSpecificArgs {
            image_sizes,
            num_image_tokens,
            num_image_samples,
            image_hashes,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `LLaVANextVisionSpecificArgs`");
        let image_sizes = image_sizes.map(|image_sizes| {
            image_sizes
                .iter()
                .map(|(w, h)| (*w as u32, *h as u32))
                .collect::<Vec<_>>()
        });
        self.forward_inputs(
            input_ids,
            pixel_values,
            image_sizes,
            num_image_tokens,
            num_image_samples,
            &image_hashes,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            flash_params,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &crate::pipeline::EitherCache {
        self.llm.cache()
    }
    fn cache_mut(&mut self) -> &mut crate::pipeline::EitherCache {
        self.llm.cache_mut()
    }

    fn max_seq_len(&self) -> usize {
        self.config.text_config.max_length
    }

    fn config(&self) -> &ModelConfigMetadata {
        self.llm.config()
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        Box::new(LLaVANextVisionSpecificArgs::default())
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

impl AnyMoeBaseModelMixin for Model {
    fn get_mlps(&self) -> Vec<&dyn MlpLayer> {
        self.llm.get_mlps()
    }
    fn get_mlps_mut(&mut self) -> Vec<&mut Box<dyn MlpLayer>> {
        self.llm.get_mlps_mut()
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
        self.llm.create_anymoe_layers(
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
