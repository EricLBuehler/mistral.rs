#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
use std::any::Any;
use std::sync::{Arc, Mutex};

use super::llava_llm::{LLaVALLM, Llama, Mistral};
use crate::amoe::AnyMoeBaseModelMixin;
use crate::amoe::MlpLayer;
use crate::device_map::DeviceMapper;
use crate::layers;
use crate::paged_attention::encoder_cache::{cached_encode_images, EncoderCacheManager};
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::IsqModel;
use crate::pipeline::NormalLoadingMetadata;
use crate::pipeline::VisionModel;
use crate::utils::unvarbuilder::UnVarBuilder;
use crate::vision_models::clip::{ClipConfig, ClipVisionTransformer};
use crate::vision_models::llava::config::Config;
use crate::AnyMoeConfig;
use crate::AnyMoeExpertType;
use candle_core::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Activation, Linear};
use mistralrs_quant::NonZeroOp;
use mistralrs_quant::ShardedVarBuilder;

pub(crate) struct LLaVAVisionSpecificArgs {
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

    pub fn prepare_inputs_labels_for_multimodal(
        &self,
        input_ids: &Tensor, //[1,seq_len]
        images: &Tensor,    //[sum of samples of all images,channel,width,height]
        num_image_tokens: usize,
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
        let image_features =
            cached_encode_images(image_hashes, &images_typed, &self.encoder_cache, |pv| {
                Ok(vec![self.encode_images(pv)?])
            })?[0]
                .clone(); //[num of images,patch_size*patch_size,hidden_size]
        let num_of_images = image_features.shape().dims()[0];
        let mut image_features_vec = Vec::new();
        for i in 0..num_of_images {
            image_features_vec.push(image_features.get(i)?.unsqueeze(0)?);
        }
        for (i, image_index) in image_indexes.iter().enumerate() {
            result = result.slice_assign(
                &[
                    0usize..1usize,
                    *image_index as usize..*image_index as usize + num_image_tokens,
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
        num_image_tokens: Option<usize>,
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
        model_specific_args: Box<dyn std::any::Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor> {
        let LLaVAVisionSpecificArgs { image_hashes } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `LLaVAVisionSpecificArgs`");
        self.forward_inputs(
            input_ids,
            pixel_values,
            Some(
                self.clip_vision_tower.num_patches_per_side()
                    * self.clip_vision_tower.num_patches_per_side(),
            ),
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
        Box::new(LLaVAVisionSpecificArgs {
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
