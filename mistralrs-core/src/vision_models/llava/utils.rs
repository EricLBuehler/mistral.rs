use std::cmp::min;

use image::{imageops::overlay, DynamicImage, GenericImageView};

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

pub(crate) fn resize_and_pad_image(image: &DynamicImage, target_resolution: (u32, u32)) -> DynamicImage {
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
