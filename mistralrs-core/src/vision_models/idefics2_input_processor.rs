#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        apply_chat_template,
        text_models_inputs_processor::{self, get_completion_input, get_prompt_input},
        InputsProcessor, InputsProcessorType, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{from_pixel_data, get_pixel_data, make_pixel_values},
        ModelInputs,
    },
    Content, Pipeline,
};

use super::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
};

// Input processor
pub struct Idefics2ImageProcessor;
// Processor
pub struct Idefics2Processor {
    config: ProcessorConfig,
    preprocessor_config: PreProcessorConfig,
    fake_image_token: &'static str,
    image_token: &'static str,
}

impl Idefics2Processor {
    pub fn new(config: ProcessorConfig, preprocessor_config: PreProcessorConfig) -> Self {
        Self {
            config,
            preprocessor_config,
            fake_image_token: "<fake_token_around_image>",
            image_token: "<image>",
        }
    }
}

impl Processor for Idefics2Processor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, Content>>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<Vec<u32>> {
        let mut prompt = apply_chat_template(pipeline, messages, add_generation_prompt)?;

        let mut image_str = format!(
            "{}{}{}",
            self.fake_image_token,
            self.image_token.repeat(self.config.image_seq_len),
            self.fake_image_token
        );
        if self.preprocessor_config.do_image_splitting {
            // 4 patches + 1 original
            image_str = image_str.repeat(5);
        }

        prompt = prompt.replace(self.image_token, &image_str);
        // Deal with any adjacent images.
        prompt = prompt.replace(
            &format!("{}{}", self.fake_image_token, self.fake_image_token),
            self.fake_image_token,
        );

        let encoding = pipeline
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Idefics2ImageProcessor)
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }
}

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
    let new_image = from_pixel_data(
        get_pixel_data(
            image,
            image.dimensions().1 as usize,
            image.dimensions().0 as usize,
        ),
        max_h,
        max_w,
    );
    Ok((new_image, make_pixel_mask(image, max_h, max_w, device)?))
}

impl InputsProcessor for Idefics2ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        _: Arc<Tokenizer>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
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
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
            )?
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
            )?
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let (pixel_values, pixel_attention_mask) = if is_prompt {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_mask_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask,
                } = self.preprocess(
                    seq.take_images()
                        .expect("Need to have images by this point."),
                    config,
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
            model_specific_args: Box::new(pixel_attention_mask),
        }))
    }
}

impl ImagePreProcessor for Idefics2ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<PreprocessedImages> {
        let mut patch_masks = Vec::new();
        let mut pixel_values = Vec::new();

        // Image splitting
        if config.do_image_splitting {
            let mut new_images = Vec::new();
            for image in images {
                let (w, h) = image.dimensions();
                let mid_w = w / 2;
                let mid_h = h / 2;
                new_images.push(image.crop_imm(0, 0, mid_w, mid_h));
                new_images.push(image.crop_imm(mid_w, 0, w, mid_h));
                new_images.push(image.crop_imm(0, mid_h, mid_w, h));
                new_images.push(image.crop_imm(mid_w, mid_h, w, h));
                new_images.push(image);
            }
            images = new_images;
        }

        for image in images.iter_mut() {
            // Resize
            if config.do_resize {
                *image = super::image_processor::resize(image, &config.size, config.resampling)?;
            }
        }

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

        for image in images.iter_mut() {
            // Convert to rgb
            if config.do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            // Rescale
            if config.do_rescale {
                *image = self.rescale(image, config.rescale_factor);
            }

            // Normalize
            if config.do_normalize {
                *image = self.normalize(
                    image,
                    config.image_mean.unwrap_or(Self::DEFAULT_MEAN),
                    config.image_std.unwrap_or(Self::DEFAULT_STD),
                );
            }

            // Pad images, calculating attention mask.
            if config.do_pad {
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
