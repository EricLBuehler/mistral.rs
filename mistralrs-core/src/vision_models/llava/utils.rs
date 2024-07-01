#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_arguments
)]
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use candle_core::{DType, Device, Result, Tensor};
use image::{
    imageops::{overlay, FilterType},
    DynamicImage, GenericImageView, Rgb, RgbImage,
};
use std::cmp::min;

pub(crate) fn get_anyres_image_grid_shape(
    image_size: (u32, u32),
    grid_pinpoints: &[(u32, u32)],
    patch_size: u32,
) -> (u32, u32) {
    let (width, height) = select_best_resolution(image_size, grid_pinpoints);
    (width / patch_size, height / patch_size)
}

pub(crate) fn get_num_samples(
    image_size: (u32, u32),
    grid_pinpoints: &[(u32, u32)],
    crop_size: (u32, u32),
) -> u32 {
    let (width, height) = select_best_resolution(image_size, grid_pinpoints);
    width / crop_size.0 * height / crop_size.1 + 1
}

pub(crate) fn select_best_resolution(
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

pub(crate) fn calculate_unpad(size: (u32, u32), original_size: (u32, u32)) -> (u32, u32) {
    let (original_width, original_height) = original_size;
    let (current_width, current_height) = size;
    let original_aspect_ratio = (original_width as f32) / (original_height as f32);
    let current_aspect_ratio = (current_width as f32) / (current_height as f32);
    if original_aspect_ratio > current_aspect_ratio {
        let scale_factor = (current_width as f32) / (original_width as f32);
        let new_height = (original_height as f32 * scale_factor).floor() as u32;
        let padding = (current_height - new_height) / 2;
        (current_width, current_height - 2 * padding) // as it is in unpad_image
    } else {
        let scale_factor = (current_height as f32) / (original_height as f32);
        let new_width = (original_width as f32 * scale_factor).floor() as u32;
        let padding = (current_width - new_width) / 2;
        (current_width - 2 * padding, current_height)
    }
}

pub(crate) fn resize_and_pad_image(
    image: &DynamicImage,
    target_resolution: (u32, u32),
) -> DynamicImage {
    let (original_width, original_height) = image.dimensions();
    let original_width_f = original_width as f32;
    let original_height_f = original_height as f32;
    let (target_width, target_height) = target_resolution;
    let target_width_f = target_width as f32;
    let target_height_f = target_height as f32;
    let scale_w = target_width_f / original_width_f;
    let scale_h = target_height_f / original_height_f;
    let (new_width, new_height) = if scale_w < scale_h {
        (
            target_width,
            min((original_height_f * scale_w).ceil() as u32, target_height),
        )
    } else {
        (
            min((original_width_f * scale_h).ceil() as u32, target_width),
            target_height,
        )
    };
    let resized_image = image.resize_exact(
        new_width,
        new_height,
        image::imageops::FilterType::CatmullRom,
    );
    let mut new_image = DynamicImage::new_rgb8(target_width, target_height);
    let (paste_x, paste_y) =
        calculate_middle((target_width, target_height), (new_width, new_height));
    overlay(
        &mut new_image,
        &resized_image,
        paste_x.into(),
        paste_y.into(),
    );
    new_image
}

pub(crate) fn divide_to_samples(image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut samples = Vec::new();
    for y in (0..height).step_by(crop_size.1 as usize) {
        for x in (0..width).step_by(crop_size.0 as usize) {
            let patch = image.crop_imm(x, y, crop_size.0, crop_size.1);
            samples.push(patch);
        }
    }
    samples
}

pub(crate) fn calculate_middle(image_size: (u32, u32), center_size: (u32, u32)) -> (u32, u32) {
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

pub(crate) fn expand2square(image: &DynamicImage, background_color: Rgb<u8>) -> DynamicImage {
    let (width, height) = image.dimensions();
    match width.cmp(&height) {
        std::cmp::Ordering::Less => {
            let mut new_image =
                DynamicImage::from(RgbImage::from_pixel(height, height, background_color));
            overlay(&mut new_image, image, ((height - width) / 2) as i64, 0);
            new_image
        }
        std::cmp::Ordering::Equal => image.clone(),
        std::cmp::Ordering::Greater => {
            let mut new_image =
                DynamicImage::from(RgbImage::from_pixel(width, width, background_color));
            overlay(&mut new_image, image, 0, ((width - height) / 2) as i64);
            new_image
        }
    }
}

pub struct LLaVAImageProcessor;

impl LLaVAImageProcessor {
    fn resize(image: &DynamicImage, size: u32, filter: FilterType) -> DynamicImage {
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

    fn center_crop(image: &DynamicImage, crop_size: (u32, u32)) -> DynamicImage {
        let (width, height) = image.dimensions();
        let (left, top) = calculate_middle((width, height), crop_size);
        image.crop_imm(left, top, crop_size.0, crop_size.1)
    }

    fn rescale(tensor: &Tensor, rescale_factor: f64) -> Result<Tensor> {
        tensor.affine(rescale_factor, 0.0)
    }

    fn to_tensor(image: &DynamicImage, device: &Device) -> Result<Tensor> {
        let img = image.to_rgb8().into_raw();
        let (width, height) = image.dimensions();
        Tensor::from_vec(img, (height as usize, width as usize, 3), device)?.to_dtype(DType::F32)
    }

    fn normalize(tensor: &Tensor, image_mean: &[f32], image_std: &[f32]) -> Result<Tensor> {
        let mean = Tensor::from_slice(image_mean, (3,), &Device::Cpu)?;
        let std = Tensor::from_slice(image_std, (3,), &Device::Cpu)?;
        tensor.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    fn to_channel_dimension_format(tensor: &Tensor) -> Result<Tensor> {
        tensor.permute((2, 0, 1))
    }
    pub fn process_one_image(
        image: &DynamicImage,
        preprocessor_config: &PreProcessorConfig,
        resize_size: u32,
        filter: FilterType,
        dtype: DType,
        device: &Device,
        image_mean: &[f32],
        image_std: &[f32],
    ) -> Result<Tensor> {
        let mut image = if preprocessor_config.do_resize.unwrap_or(true) {
            Self::resize(image, resize_size, filter)
        } else {
            image.clone()
        };
        image = if preprocessor_config.do_center_crop.unwrap_or(true) {
            let crop_width = *preprocessor_config
                .crop_size
                .as_ref()
                .unwrap()
                .get("width")
                .unwrap();
            let crop_height = *preprocessor_config
                .crop_size
                .as_ref()
                .unwrap()
                .get("height")
                .unwrap();
            Self::center_crop(&image, (crop_width, crop_height))
        } else {
            image
        };
        let mut pixel_value = Self::to_tensor(&image, &Device::Cpu)?;
        if preprocessor_config.do_rescale.unwrap_or(true) {
            let rescale_factor = preprocessor_config.rescale_factor.unwrap();
            pixel_value = Self::rescale(&pixel_value, rescale_factor)?;
        }
        if preprocessor_config.do_normalize.unwrap_or(true) {
            pixel_value = Self::normalize(&pixel_value, image_mean, image_std)?;
        }
        Self::to_channel_dimension_format(&pixel_value)?
            .to_dtype(dtype)?
            .to_device(device)
    }
}
