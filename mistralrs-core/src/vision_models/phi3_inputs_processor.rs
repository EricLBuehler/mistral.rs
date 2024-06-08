#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use itertools::Itertools;
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        text_models_inputs_processor::{self, get_completion_input, get_prompt_input},
        InputsProcessor, InputsProcessorType, MessagesAction, Processor, ProcessorCreator,
    },
    sequence::Sequence,
};

use super::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    phi3::Phi3VisionSpecificArgs,
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
    ModelInputs,
};

// Input processor
pub struct Phi3InputsProcessor {
    image_tag_splitter: Regex,
}
// Processor
pub struct Phi3Processor {
    inputs_processor: Arc<Phi3InputsProcessor>,
}

impl ProcessorCreator for Phi3Processor {
    fn new_processor(
        _: Option<ProcessorConfig>,
        _: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi3InputsProcessor {
                image_tag_splitter: Regex::new(r"<\|image_\d+\|>")
                    .expect("Failed to compile split regex."),
            }),
        })
    }
}

impl Processor for Phi3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.inputs_processor.clone()
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl InputsProcessor for Phi3InputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        tokenizer: Arc<Tokenizer>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
    ) -> anyhow::Result<Box<dyn Any>> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        let (pixel_values, image_sizes, num_img_tokens, n_images) = if is_prompt
            && input_seqs
                .iter()
                .map(|seq| seq.images().is_some())
                .all(|x| x)
        {
            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            let mut n_images = Vec::new();
            for seq in input_seqs.iter_mut() {
                let imgs = seq
                    .take_images()
                    .expect("Need to have images by this point.");
                let imgs_len = imgs.len();
                n_images.push(imgs_len);
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes,
                    num_img_tokens,
                } = self.preprocess(imgs, config, device)?;
                let image_sizes = image_sizes.unwrap();
                pixel_values_accum.push(pixel_values);
                image_sizes_accum.push(image_sizes);
                num_img_tokens_accum.push(num_img_tokens.unwrap());
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0)?),
                Some(image_sizes_accum),
                Some(num_img_tokens_accum),
                n_images,
            )
        } else {
            let text_models_inputs_processor::ModelInputs {
                input_ids,
                input_ids_full: _,
                seqlen_offsets,
                seqlen_offsets_full: _,
                seqlen_offsets_kernel,
                seqlen_offsets_kernel_full: _,
                context_lens,
                position_ids,
            } = *text_models_inputs_processor::TextInputsProcessor
                .process_inputs(
                    tokenizer,
                    input_seqs,
                    is_prompt,
                    is_xlora,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    other_config,
                )?
                .downcast::<text_models_inputs_processor::ModelInputs>()
                .expect("Downcast failed.");

            return Ok(Box::new(ModelInputs {
                input_ids,
                seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
                position_ids,
                pixel_values: None,
                model_specific_args: Box::new(Phi3VisionSpecificArgs { image_sizes: None }),
            }));
        };

        let mut toks = Vec::new();
        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .map_err(anyhow::Error::msg)?;

        for (detokenized, (seq, (num_img_tokens, n_images))) in detokenized.into_iter().zip(
            input_seqs
                .iter_mut()
                .zip(num_img_tokens.unwrap().into_iter().zip(n_images)),
        ) {
            let splits = self
                .image_tag_splitter
                .split(&detokenized)
                .map(|span| &detokenized[span.range()])
                .collect::<Vec<_>>();
            let prompt_chunks = tokenizer
                .encode_batch(splits, true)
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .map(|enc| enc.get_ids().to_vec())
                .collect::<Vec<_>>();

            let image_tags = self.image_tag_splitter.find_iter(&detokenized);
            let image_ids = image_tags
                .into_iter()
                .map(|s| {
                    let s = &detokenized[s.range()];
                    s.split('|')
                        .nth(1)
                        .unwrap()
                        .split('_')
                        .nth(1)
                        .unwrap()
                        .parse::<u32>()
                        .expect("Failed to parse image id to u32")
                })
                .collect::<Vec<_>>();
            let unique_image_ids = image_ids
                .iter()
                .copied()
                .unique()
                .sorted()
                .collect::<Vec<_>>();
            // `image_ids` must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be [1, 4, 5]
            if unique_image_ids != (1u32..unique_image_ids.len() as u32 + 1).collect::<Vec<_>>() {
                anyhow::bail!("`image_ids` must start from 1, and must be continuous, e.g. [1, 2, 3], cannot be [1, 4, 5].");
            }
            // Total images must be the same as the number of image tags
            if unique_image_ids.len() != n_images {
                anyhow::bail!("Total images must be the same as the number of image tags.");
            }

            // Use the TryInto + unwrap_or to handle case when id==0
            let image_ids_pad = image_ids
                .iter()
                .map(|id| {
                    [-(*id as i64)].repeat(
                        num_img_tokens[TryInto::<usize>::try_into(*id as isize - 1)
                            .unwrap_or(num_img_tokens.len() - 1)],
                    )
                })
                .collect::<Vec<_>>();

            let mut input_ids: Vec<i64> = Vec::new();
            for item in prompt_chunks
                .iter()
                .map(|x| x.iter().map(|x| *x as i64).collect::<Vec<_>>())
                .interleave(image_ids_pad)
            {
                input_ids.extend(item);
            }

            // NOTE(EricLBuehler): Casting to u32 is fine, we don't care about the other toks
            seq.set_toks(
                input_ids
                    .iter()
                    .map(|x| if *x < 0 { 0u32 } else { *x as u32 })
                    .collect::<Vec<_>>(),
            );

            toks.push(input_ids);
        }

        let text_models_inputs_processor::InputMetadata {
            input,
            positions,
            positions_kernel,
            context_lens,
            position_ids,
        } = if is_prompt {
            get_prompt_input(toks, input_seqs, device, last_n_context_len)?
        } else {
            get_completion_input(toks, input_seqs, device, no_kv_cache, last_n_context_len)?
        };

        Ok(Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            seqlen_offsets_kernel: positions_kernel,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Phi3VisionSpecificArgs { image_sizes }),
        }))
    }
}

impl Phi3InputsProcessor {
    fn pad_image(
        image: &DynamicImage,
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        pad_color: Rgba<u8>,
    ) -> DynamicImage {
        // Calculate the new dimensions
        let new_width = image.width() + left + right;
        let new_height = image.height() + top + bottom;

        // Create a new image with the new dimensions and fill it with the pad color
        let mut new_image = DynamicImage::new_rgb8(new_width, new_height);
        for x in 0..new_width {
            for y in 0..new_height {
                new_image.put_pixel(x, y, pad_color);
            }
        }

        // Paste the original image into the center of the new image
        new_image
            .copy_from(image, left, top)
            .expect("Failed to copy image");

        new_image
    }

    fn padding_336(img: &DynamicImage) -> DynamicImage {
        let (_width, height) = img.dimensions();
        let tar = ((height as f64 / 336.0).ceil() * 336.0) as u32;
        let top_padding = ((tar as f64 - height as f64 + 1.) / 2.) as u32;
        let bottom_padding = tar - height - top_padding;
        let left_padding = 0u32;
        let right_padding = 0u32;
        Self::pad_image(
            img,
            top_padding,
            bottom_padding,
            left_padding,
            right_padding,
            Rgba([255u8, 255, 255, 255]),
        )
    }

    fn hd_transform(img: &DynamicImage, hd_num: usize) -> DynamicImage {
        let (mut width, mut height) = img.dimensions();
        let mut transposed = false;

        let img = if width < height {
            let img = img.rotate90();
            transposed = true;
            width = img.width();
            height = img.height();
            img
        } else {
            // NOTE: Don't love the clone.
            img.clone()
        };

        let ratio = width as f64 / height as f64;
        let mut scale = 1.0;
        while (scale * (scale / ratio).ceil()) <= hd_num as f64 {
            scale += 1.0;
        }
        scale -= 1.0;

        let new_width = (scale * 336.0) as u32;
        let new_height = (new_width as f64 / ratio) as u32;

        let resized_img = img.resize_exact(new_width, new_height, FilterType::Nearest);
        let padded_img = Self::padding_336(&resized_img);

        if transposed {
            return padded_img.rotate270();
        }

        padded_img
    }
}

fn pad_to_max_num_crops_tensor(image: &Tensor, max_crops: usize) -> Result<Tensor> {
    let (b, _, h, w) = image.dims4()?;
    if b < max_crops {
        let pad = Tensor::zeros((max_crops - b, 3, h, w), image.dtype(), image.device())?;
        Tensor::cat(&[image, &pad], 0)
    } else {
        Ok(image.clone())
    }
}

impl ImagePreProcessor for Phi3InputsProcessor {
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
        // If no images, will not call this.
        assert!(!images.is_empty());

        let mut image_sizes = Vec::new();
        let mut padded_images = Vec::new();
        let mut num_img_tokens = Vec::new();
        // If >1 images, resize them all to the largest, potentially destroying aspect ratio
        let mut max_size = None;
        for image in images.iter() {
            if max_size.is_none() {
                max_size = Some((image.dimensions().0 as usize, image.dimensions().1 as usize))
            } else if max_size.is_some_and(|(x, _)| image.dimensions().0 as usize > x) {
                max_size = Some((image.dimensions().0 as usize, max_size.unwrap().1));
            } else if max_size.is_some_and(|(_, y)| image.dimensions().1 as usize > y) {
                max_size = Some((max_size.unwrap().0, image.dimensions().1 as usize));
            }
        }
        let (max_h, max_w) = max_size.unwrap();
        for image in images.iter_mut() {
            *image = image.resize_exact(max_w as u32, max_h as u32, FilterType::Nearest);
        }

        for image in images.iter_mut() {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let hd_image = Self::hd_transform(image, config.num_crops.expect("Need `num_crops`"));

            // Both hd and global have a normalization
            // Transforms for the HD image
            let transforms_hd = Transforms {
                input: &ToTensor,
                inner_transforms: &[&Normalize {
                    mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                    std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                }],
            };

            // (3,h,w)
            let hd_image = hd_image.apply(transforms_hd, device)?;

            // Resize with bicubic interpolation
            // (3,336,336)
            let global_image = hd_image.unsqueeze(0)?.interpolate2d(336, 336)?;

            let (_, h, w) = hd_image.dims3()?;
            let num_image_tokens = ((h as f32 / 336. * w as f32 / 336. + 1.) * 144.
                + ((h as f32 / 336.) + 1.) * 12.
                + 1.) as usize;

            let hd_image_reshape = hd_image
                .reshape((
                    1,
                    3,
                    (h as f32 / 336.) as usize,
                    336,
                    (w as f32 / 336.) as usize,
                    336,
                ))?
                .permute((0, 2, 4, 1, 3, 5))?
                .reshape(((), 3, 336, 336))?;
            let hd_image_reshape = Tensor::cat(&[global_image, hd_image_reshape], 0)?;
            let image_transformed = pad_to_max_num_crops_tensor(
                &hd_image_reshape,
                config.num_crops.expect("Need `num_crops`") + 1,
            )?;
            image_sizes.push((h, w));
            padded_images.push(image_transformed);
            num_img_tokens.push(num_image_tokens);
        }
        if padded_images.len() > 1 {
            candle_core::bail!("Can only process one image per batch");
        }
        let image_sizes = image_sizes[0];

        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&padded_images, 0)?,
            image_sizes: Some((image_sizes.0, image_sizes.1)),
            pixel_attention_mask: None,
            num_img_tokens: Some(num_img_tokens),
        })
    }
}
