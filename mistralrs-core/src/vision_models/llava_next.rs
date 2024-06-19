use candle_core::{bail, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, seq, Activation, Module, Sequential, VarBuilder};
use serde::Deserialize;

use crate::pipeline::NormalLoadingMetadata;
use crate::serde_default_fn;

use crate::models::llama::Config as LLaMAConfig;
use crate::models::llama::Llama;
use crate::vision_models::clip::ClipConfig;
use crate::vision_models::llava_inputs_processor::get_anyres_image_grid_shape;

use super::clip::{Activation as ClipActivation, ClipVisionTransformer};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub ignore_index: isize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_token_index: isize,
    pub model_type: String,
    pub projector_hidden_act: String,
    pub text_config: LLaVATextConfig,
    pub torch_dtype: String,
    pub use_image_newline_parameter: bool,
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
            projection_dim: self.vision_config.projection_dim,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            num_channels: 3,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
            hidden_act: ClipActivation::QuickGelu,
            layer_norm_eps: 1e-5, // default value, seem useless
        }
    }
}

pub struct MMProjector {
    pub modules: Sequential,
}

impl MMProjector {
    pub fn new(vb: &VarBuilder, config: &Config) -> Result<Self> {
        let mut modules = seq().add(linear(
            1024,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_1"),
        )?);
        match config.projector_hidden_act.as_str() {
            "gelu" => {
                modules = modules.add(Activation::Gelu);
            }
            _ => {
                bail!(
                    "Unsupporg projector hidden act: {}",
                    config.projector_hidden_act
                );
            }
        }
        modules = modules.add(linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            vb.pp("multi_modal_projector.linear_2"),
        )?);
        Ok(Self { modules })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.modules.forward(x)
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
    pub clip_vision_tower: ClipVisionTower,
    pub image_newline: Tensor,
    pub mm_projector: MMProjector,
    pub llama: Llama,
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
        input_ids: &Tensor,
        images: &[Tensor],
        image_sizes: &[(u32, u32)],
    ) -> Result<Tensor> {
        let mut x = Tensor::cat(images, 0)?;
        x = self.encode_images(&x)?;
        let split_sizes = images
            .iter()
            .map(|x| x.shape().dims()[0])
            .collect::<Vec<usize>>();
        let mut index_pos = 0;
        let mut image_features = Vec::new();
        // can be replaced by split
        for split_size in split_sizes.iter() {
            image_features.push(x.i(index_pos..index_pos + (*split_size))?);
            index_pos += *split_size;
        }
        drop(x);
        let mut new_image_features = Vec::new();
        for (image_idx, image_feature) in image_features.iter().enumerate() {
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
            new_image_feature = Tensor::cat(&[base_image_feature, new_image_feature], 0)?;
            new_image_features.push(new_image_feature);
        }
        image_features = new_image_features; //moved
        let image_indices = input_ids.squeeze(0)?.eq(self.config.image_token_index as i64)?.nonzero()?;
        //TODO: replace using nonzero
        /*
        let input_ids_vec = input_ids.squeeze(0)?.to_vec1::<i64>()?;
        let mut image_indices = {
            let mut image_indices = vec![0_i64];
            image_indices.extend(
                input_ids_vec
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| {
                        if *x == self.config.image_token_index as i64 {
                            Some(i as i64)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<i64>>(),
            );
            image_indices
        };
        */

        todo!()
    }
}
