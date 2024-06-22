#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::any::Any;

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use regex_automata::meta::Regex;

use crate::pipeline::InputsProcessor;

use super::image_processor::ImagePreProcessor;
use crate::vision_models::preprocessor_config::{PreProcessorConfig, ToFilter};
use mistralrs_vision::Normalize;

pub fn get_anyres_image_grid_shape(
    image_size: (u32, u32),
    grid_pinpoints: &[(u32, u32)],
    patch_size: u32,
) -> (u32, u32) {
    let (width, height) = select_best_resolution(image_size, grid_pinpoints);
    (width / patch_size, height / patch_size)
}

pub fn select_best_resolution(
    original_size: (u32, u32),
    possible_resolutions: &[(u32, u32)],
) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let mut best_fit = (0, 0);
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let mut max_effective_resolution = 0_u32;
    let mut min_wasted_resolution = u32::MAX;
    for (width, height) in possible_resolutions {
        let width_f = *width as f32;
        let height_f = *height as f32;
        let scale = (width_f / original_width_f).min(height_f / original_height_f);
        let (downscaled_width, downscaled_height) = (
            (original_width_f * scale) as u32,
            (original_height_f * scale) as u32,
        );
        let effective_resolution =
            std::cmp::min((*width) * (*height), downscaled_width * downscaled_height);
        let wasted_resolution = (*width) * (*height) - effective_resolution;
        if effective_resolution > max_effective_resolution
            || (effective_resolution == max_effective_resolution
                && wasted_resolution < min_wasted_resolution)
        {
            best_fit = (*width, *height);
            max_effective_resolution = effective_resolution;
            min_wasted_resolution = wasted_resolution;
        }
    }
    best_fit
}

pub fn calculate_middle(image_size: (u32, u32), center_size: (u32, u32)) -> (u32, u32) {
    let (width, height) = image_size;
    let (center_width, center_height) = center_size;
    let left = if width <= center_width {
        0
    } else {
        ((width as f32 - center_width as f32) / 2.0).ceil() as u32
    };
    let top = if height <= center_height {
        0
    } else {
        ((height as f32 - center_height as f32) / 2.0).ceil() as u32
    };
    (left, top)
}

pub struct LLaVANextInputProcessor {
    image_tag_splitter: Regex,
}

impl InputsProcessor for LLaVANextInputProcessor {
    fn process_inputs(
        &self,
        tokenizer: std::sync::Arc<tokenizers::Tokenizer>,
        input_seqs: &mut [&mut crate::sequence::Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &candle_core::Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<std::sync::Arc<dyn std::any::Any>>,
    ) -> anyhow::Result<Box<dyn std::any::Any>> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let config: std::sync::Arc<dyn Any> = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        todo!()
    }

    fn get_type(&self) -> crate::pipeline::InputsProcessorType {
        todo!()
    }
}

impl LLaVANextInputProcessor {
    fn resize(&self, image: &DynamicImage, size: u32, filter: FilterType) -> DynamicImage {
        let (width, height) = image.dimensions();
        if width == size && height == size {
            image.clone()
        } else {
            let (new_width, new_height) = if width < height {
                (
                    size,
                    (((size * height) as f32) / width as f32).ceil() as u32,
                )
            } else {
                (
                    (((size * width) as f32) / height as f32).ceil() as u32,
                    size,
                )
            };
            image.resize(new_width, new_height, filter)
        }
    }

    fn center_crop(&self, image: &DynamicImage, crop_size: (u32, u32)) -> DynamicImage {
        let (width, height) = image.dimensions();
        let (left, top) = calculate_middle((width, height), crop_size);
        image.crop_imm(left, top, crop_size.0, crop_size.1)
    }

    fn rescale(&self, tensor: &Tensor, rescale_factor: f64) -> Result<Tensor> {
        tensor.affine(rescale_factor, 0.0)
    }

    fn to_tensor(&self, image: &DynamicImage, device: &Device) -> Result<Tensor> {
        let img = image.to_rgb8().into_raw();
        let (width, height) = image.dimensions();
        Tensor::from_vec(img, (height as usize, width as usize, 3), device)?.to_dtype(DType::F32)
    }

    fn normalize(&self, tensor: &Tensor, image_mean: &[f32], image_std: &[f32]) -> Result<Tensor> {
        let mean = Tensor::from_slice(image_mean, (3,), &Device::Cpu)?;
        let std = Tensor::from_slice(image_std, (3,), &Device::Cpu)?;
        tensor.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    fn to_channel_dimension_format(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.permute((2, 0, 1))
    }
}

impl ImagePreProcessor for LLaVANextInputProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        images: Vec<image::DynamicImage>,
        config: &super::preprocessor_config::PreProcessorConfig,
        device: &candle_core::Device,
    ) -> candle_core::Result<super::image_processor::PreprocessedImages> {
        if images.len() > 1 {
            candle_core::bail!("Can only process one image per batch"); // This is no different from phi3_input_processor
        };
        let mut image = images[0].clone();
        let image_size = (image.width() as usize, image.height() as usize);
        if config.do_resize.unwrap_or(true) {
            let size = config.size.as_ref().unwrap().get("shortest_edge").unwrap();
            let filter = config.resampling.to_filter()?;
            image = self.resize(&image, *size, filter);
        }
        if config.do_center_crop.unwrap_or(true) {
            let crop_size = (
                *config.crop_size.as_ref().unwrap().get("width").unwrap(),
                *config.crop_size.as_ref().unwrap().get("height").unwrap(),
            );
            image = self.center_crop(&image, crop_size);
        }
        let mut pixel_values = self.to_tensor(&image, &device)?;
        if config.do_rescale.unwrap_or(true) {
            let rescale_factor = config.rescale_factor.unwrap();
            pixel_values = self.rescale(&pixel_values, rescale_factor)?;
        }
        if config.do_normalize.unwrap_or(true) {
            let image_mean = config
                .image_mean
                .unwrap_or(Self::DEFAULT_MEAN)
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>();
            let image_std = config
                .image_std
                .unwrap_or(Self::DEFAULT_STD)
                .iter()
                .map(|x| *x as f32)
                .collect::<Vec<f32>>();
            pixel_values = self.normalize(&pixel_values, &image_mean, &image_std)?;
        }  
        pixel_values = self.to_channel_dimension_format(&pixel_values)?;
        Ok(super::image_processor::PreprocessedImages {
            pixel_values: pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some(image_size),
            num_img_tokens: None,
        })
    }
}
