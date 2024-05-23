#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};

use crate::{
    pipeline::{
        text_models_inputs_processor::{self, get_completion_input, get_prompt_input},
        InputsProcessor, InputsProcessorType,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{from_pixel_data, get_pixel_data, make_pixel_values},
        ModelInputs,
    },
};

use super::image_processor::{ImagePreProcessor, NormalizationMetadata, PreprocessedImages};

pub struct Idefics2ImageProcessor;

/// Generate pixel mask. 1 indicates valid pixel, 0 indicates padding
/// Shape is (h,w)
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
    Ok((new_image, make_pixel_mask(image, max_h, max_w, device)?))
}

impl InputsProcessor for Idefics2ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
    ) -> anyhow::Result<Box<dyn std::any::Any>> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let text_models_inputs_processor::InputMetadata {
            input,
            positions,
            positions_kernel,
            context_lens,
            position_ids,
        } = if is_prompt {
            get_prompt_input(input_seqs, device, last_n_context_len)?
        } else {
            get_completion_input(input_seqs, device, no_kv_cache, last_n_context_len)?
        };

        let (pixel_values, pixel_attention_mask) = if is_prompt {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_mask_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                // TODO: Properly pass params here?
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask,
                } = self.preprocess(
                    seq.take_images()
                        .expect("Need to have images by this point."),
                    None,
                    None,
                    None,
                    None,
                    true,
                    device,
                )?;
                pixel_values_accum.push(pixel_values.unsqueeze(0)?);
                pixel_attention_mask_accum.push(pixel_attention_mask.unsqueeze(0)?);
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0)?),
                Some(Tensor::cat(&pixel_attention_mask_accum, 0)?),
            )
        } else {
            (None, None)
        };

        Ok(Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            seqlen_offsets_kernel: positions_kernel,
            context_lens,
            position_ids,
            pixel_values,
            pixel_attention_mask,
        }))
    }
}

impl ImagePreProcessor for Idefics2ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        filter: Option<FilterType>,
        resize: Option<(usize, usize)>,
        rescale: Option<f32>,
        normalize: Option<NormalizationMetadata>,
        do_pad: bool,
        pad_to: Option<(u32,u32)>,
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
        if let Some((h,w)) = pad_to {
            if h < max_h {
                candle_core::bail!("`pad_to` height ({h}) less than maximum height of specified images ({max_h}).");
            }
            if w < max_w {
                candle_core::bail!("`pad_to` width ({w}) less than maximum width of specified images ({max_w}).");
            }
        }
        let mut patch_masks = Vec::new();
        let mut pixel_values = Vec::new();
        for image in images.iter_mut() {
            // Convert image to rgb8 always
            // TODO configurable? Will need to update the image_processor.rs functions
            *image = DynamicImage::ImageRgb8(image.to_rgb8());

            // TODO: implement image splitting?

            // Resize
            if let Some((w, h)) = resize {
                *image = super::image_processor::resize(
                    image,
                    w as u32,
                    h as u32,
                    filter.expect("Need filter if resizing."),
                );
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
                *image = self.normalize(image, image_mean.unwrap_or(Self::DEFAULT_MEAN), image_std.unwrap_or(Self::DEFAULT_STD));
            }

            // Pad images, calculating attention mask.
            if do_pad {
                let (new_image, mask) = pad(image, max_h as usize, max_w as usize, device)?;
                *image = new_image;
                patch_masks.push(mask.unsqueeze(0)?);
            }

            // Get pixel values
            pixel_values.push(make_pixel_values(image, device)?.unsqueeze(0)?)
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
            pixel_attention_mask: Tensor::cat(&patch_masks, 0)?,
        })
    }
}
