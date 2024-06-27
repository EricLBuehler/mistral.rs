#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use candle_core::quantized::QMatMul;
use candle_core::{bail, DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, seq, Activation, Linear, Module, Sequential, VarBuilder};
use itertools::Itertools;
use serde::Deserialize;

use crate::device_map::DeviceMapper;
use crate::ops::NonZeroOp;
use crate::pipeline::IsqModel;
use crate::pipeline::NormalLoadingMetadata;
use crate::pipeline::NormalModel;
use crate::pipeline::VisionModel;
use crate::serde_default_fn;

use crate::models::llama::Config as LLaMAConfig;
use crate::models::llama::Llama;
use crate::vision_models::clip::ClipConfig;
use crate::vision_models::llava_next_inputs_processor::get_anyres_image_grid_shape;

use super::clip::{Activation as ClipActivation, ClipVisionTransformer};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub projector_hidden_act: String,
    pub text_config: LLaVATextConfig,
    pub torch_dtype: String,
    pub vision_config: LLaVAVisionConfig,
    pub vision_feature_layer: isize,
    pub vision_feature_select_strategy: String,
    pub vocab_size: usize,
    #[serde(default = "default_use_flash_attn")]
    pub use_flash_attn: bool,
}

serde_default_fn!(bool, default_use_flash_attn, false);

#[derive(Deserialize, Debug, Clone)]
pub struct LLaVATextConfig {
    pub architectures: Vec<String>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub vocab_size: usize,
}

serde_default_fn!(usize, default_num_hidden_layers, 32);
serde_default_fn!(bool, default_use_cache, true);
serde_default_fn!(usize, default_hidden_size, 4096);
serde_default_fn!(usize, default_intermediate_size, 11008);
serde_default_fn!(usize, default_max_length, 4096);
serde_default_fn!(usize, default_num_attention_heads, 32);
serde_default_fn!(usize, default_num_key_value_heads, 32);
serde_default_fn!(f32, default_rope_theta, 10000.0);

#[derive(Deserialize, Debug, Clone)]
pub struct LLaVAVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub projection_dim: usize,
    pub vocab_size: usize,
}

impl Config {
    fn to_llama_config(&self) -> LLaMAConfig {
        LLaMAConfig {
            hidden_size: self.text_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            num_key_value_heads: self.text_config.num_key_value_heads,
            use_flash_attn: self.use_flash_attn,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            max_position_embeddings: self.text_config.max_position_embeddings,
        }
    }

    fn to_clip_config(&self) -> ClipConfig {
        ClipConfig {
            hidden_size: self.vision_config.hidden_size,
            intermediate_size: self.vision_config.intermediate_size,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            num_channels: 3,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
            hidden_act: ClipActivation::QuickGelu,
        }
    }
}

pub struct MMProjector {
    linear_1: Linear,
    activation: Activation,
    linear_2: Linear,
}

impl MMProjector {
    pub fn new(vb: &VarBuilder, config: &Config) -> Result<Self> {
        let linear_1 = linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_1"),
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
        let linear_2 = linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_2"),
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
        vb: VarBuilder,
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
    llama: Llama,
    config: Config,
    device: Device,
}

impl Model {
    pub fn new(
        config: &Config,
        vb: VarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let llama_config = config.to_llama_config();
        let clip_config = config.to_clip_config();
        let mm_projector = MMProjector::new(&vb, config)?;
        let clip_vision_tower = ClipVisionTower::new(
            vb.pp("vision_tower.vision_model"),
            config.vision_feature_layer,
            &config.vision_feature_select_strategy,
            &clip_config,
        )?;
        let image_newline = vb
            .get(&[config.text_config.hidden_size], "image_newline")?
            .to_device(&device)?;
        let llama = Llama::new(
            &llama_config,
            vb.pp("language_model"),
            is_gptx,
            normal_loading_metadata,
        )?;
        Ok(Self {
            clip_vision_tower,
            image_newline,
            mm_projector,
            llama,
            config: config.clone(),
            device,
        })
    }

    pub fn encode_images(&self, x: &Tensor) -> Result<Tensor> {
        let mut image_features = self.clip_vision_tower.forward(x)?;
        image_features = self.mm_projector.forward(&image_features)?;
        Ok(image_features)
    }

    fn unpad_image(&self, tensor: &Tensor, original_size: &(u32, u32)) -> Result<Tensor> {
        assert_eq!(tensor.dims().len(), 3);
        let (original_width, original_height) = *original_size;
        let tensor_dims = tensor.dims();
        let current_height = tensor_dims[1];
        let current_width = tensor_dims[2];
        let original_aspect_ratio = (original_width as f32) / (original_height as f32);
        let current_aspect_ratio = (current_width as f32) / (current_height as f32);
        if original_aspect_ratio > current_aspect_ratio {
            let scale_factor = (current_width as f32) / (original_width as f32);
            let new_height = (original_height as f32 * scale_factor).floor() as usize;
            let padding = (current_height - new_height) / 2;
            tensor.i((.., padding..current_width - padding, ..))
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
        images: &Tensor, //[sample_per_image*image_num,chanel,width,height] for LLaVANext, we fix aspect_ratio_setting to 'anyres', (this is akin to python Transformer). Hence sampler_per_image is 5
        num_image_token: usize,
        image_sizes: &[(u32, u32)],
    ) -> Result<Tensor> {
        let image_indexes = input_ids
            .squeeze(0)?
            .lt(0i64)?
            .nonzero()?
            .squeeze(1)?
            .to_vec1::<u32>()?;
        let image_ids = image_indexes
            .iter()
            .map(|x| Ok(-(input_ids.i((0, *x as usize))?.to_scalar::<i64>()?)))
            .collect::<Result<Vec<i64>>>()?;
        let mut result = input_ids.clamp(0i64, i64::MAX)?.to_dtype(DType::U32)?;
        result = self.llama.embed(&result)?; //[seq_len,hidden_size]
        let image_features = self.encode_images(images)?; //[sample_per_image*image_num,hidden_size]
        let image_nums = images.shape().dims()[0] / 5;
        let mut image_features_vec = Vec::new();
        for i in 0..image_nums {
            image_features_vec.push(image_features.i(i * 5..(i + 1) * 5)?);
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
                let (num_patch_width, num_patch_height) = get_anyres_image_grid_shape(
                    image_size,
                    &self.config.image_grid_pinpoints,
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
                new_image_feature = self.unpad_image(&new_image_feature, &image_size)?;
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
            let image_id = image_ids[i];
            result = result.slice_assign(
                &[
                    &(0usize..1usize),
                    &(*image_index as usize..*image_index as usize + num_image_token),
                    &(..),
                ],
                &image_features_vec[(image_id - 1) as usize],
            )?;
        }
        Ok(result)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        image_sizes: Option<Vec<(u32, u32)>>,
        num_image_token: Option<usize>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
    ) -> Result<Tensor> {
        if let Some(ref pixel_values) = pixel_values {
            // we assume(as it should be) only prompt request contains image
            let input_embeds = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                pixel_values,
                num_image_token.unwrap(),
                &image_sizes.unwrap(),
            )?;
            self.llama.forward_input_embed(
                &input_embeds,
                &seqlen_offsets,
                start_offsets_kernel,
                context_lens,
            )
        } else {
            self.llama.forward(
                input_ids,
                &seqlen_offsets,
                start_offsets_kernel,
                context_lens,
            )
        }
    }
}

pub(crate) struct LLaVANextVisionSpecificArgs {
    pub image_sizes: Option<Vec<(usize, usize)>>,
    pub num_image_token: Option<usize>,
}

impl IsqModel for Model {
    fn get_tensors(&mut self) -> (Vec<(&mut QMatMul, Option<usize>)>, &dyn DeviceMapper) {
        // I don't really get this part
        self.llama.get_tensors()
    }
}

impl VisionModel for Model {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        model_specific_args: Box<dyn std::any::Any>, // pixel attention mask, or image sizes, or anything else
    ) -> candle_core::Result<Tensor> {
        let LLaVANextVisionSpecificArgs {
            image_sizes,
            num_image_token,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `LLaVANextVisionSpecificArgs`");
        let image_sizes = if image_sizes.is_some() {
            Some(
                image_sizes
                    .unwrap()
                    .iter()
                    .map(|(w, h)| (*w as u32, *h as u32))
                    .collect(),
            )
        } else {
            None
        };
        self.forward(
            input_ids,
            pixel_values,
            image_sizes,
            num_image_token,
            seqlen_offsets,
            start_offsets_kernel,
            context_lens,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn cache(&self) -> &crate::pipeline::Cache {
        self.llama.cache()
    }

    fn max_seq_len(&self) -> usize {
        self.llama.max_seq_len()
    }

    fn has_conv2d(&self) -> bool {
        true
    }
}
