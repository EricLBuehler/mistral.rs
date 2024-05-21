use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

use crate::vision_models::image_processor::{from_pixel_data, get_pixel_data, resize};

use super::image_processor::{ImagePreProcessor, NormalizationMetadata, PreprocessedImages};

pub struct Idefics2ImageProcessor;

const IDEFICS_STANDARD_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IDEFICS_STANDARD_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Generate pixel mask. 1 indicates valid pixel, 0 indicates padding
fn make_pixel_mask(
    image: &DynamicImage,
    max_h: usize,
    max_w: usize,
    device: &Device,
) -> Result<Tensor> {
    let (img_w, img_h) = image.dimensions();
    let data = (0..max_h as u32)
        .map(|h| {
            (0..max_w as u32)
                .map(|w| if w >= img_w || h >= img_h { 0u8 } else { 1u8 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut accum = Vec::new();
    for row in data {
        accum.push(Tensor::from_vec(row, (1, max_w), device)?)
    }
    Tensor::cat(&accum, 0)
}

/// Pad image to bottom and the right to the largest height and width.
/// Returns the new image and the pixel mask
fn pad(
    image: &DynamicImage,
    max_h: usize,
    max_w: usize,
    device: &Device,
) -> Result<(DynamicImage, Tensor)> {
    let new_image = from_pixel_data(get_pixel_data(image, max_h, max_w), max_h, max_w);
    Ok((new_image, make_pixel_mask(&image, max_h, max_w, device)?))
}

impl ImagePreProcessor for Idefics2ImageProcessor {
    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        do_resize: bool,
        rescale: Option<f32>,
        normalize: Option<NormalizationMetadata>,
        do_pad: bool,
        filter: FilterType,
        device: &Device,
    ) -> Result<PreprocessedImages> {
        let mut max_h = 0;
        let mut max_w = 0;
        for image in &images {
            let (w, h) = image.dimensions();
            if w > max_w {
                max_w = w;
            }
            if h > max_h {
                max_h = h;
            }
        }
        let mut patch_masks = Vec::new();
        for image in images.iter_mut() {
            // Convert image to rgb8 always
            // TODO configurable? Will need to update the image_processor.rs functions
            *image = DynamicImage::ImageRgb8(image.to_rgb8());

            // TODO: implement image splitting?

            // Resize
            if do_resize {
                *image = resize(&image, image.dimensions().0, image.dimensions().1, filter);
            }

            // Rescale
            if let Some(factor) = rescale {
                *image = self.rescale(image, factor);
            }

            // Normalize
            if let Some(NormalizationMetadata {
                image_mean,
                image_std,
            }) = normalize
            {
                *image = self.normalize(image, image_mean, image_std);
            }

            // Pad images, calculating attention mask.
            if do_pad {
                let (new_image, mask) = pad(image, max_h as usize, max_w as usize, device)?;
                *image = new_image;
                patch_masks.push(mask);
            }
        }
        todo!()
    }
}
