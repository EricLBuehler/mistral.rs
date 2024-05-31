#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView, RgbImage, RgbaImage};
use itertools::Itertools;
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        text_models_inputs_processor::{self, get_completion_input, get_prompt_input},
        InputsProcessor, InputsProcessorType, Processor, ProcessorCreator,
    },
    sequence::Sequence,
    vision_models::image_processor::make_pixel_values,
};

use super::{
    image_processor::{get_pixel_data, ImagePreProcessor, PreprocessedImages},
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
        _: ProcessorConfig,
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

        let (pixel_values, image_sizes, num_img_tokens) = if is_prompt {
            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes,
                    num_img_tokens,
                } = self.preprocess(
                    seq.take_images()
                        .expect("Need to have images by this point."),
                    config,
                    device,
                )?;
                let image_sizes = image_sizes.unwrap();
                pixel_values_accum.push(pixel_values.unsqueeze(0)?);
                image_sizes_accum.push(image_sizes);
                num_img_tokens_accum.push(num_img_tokens.unwrap());
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0)?),
                Some(image_sizes_accum),
                Some(num_img_tokens_accum),
            )
        } else {
            return text_models_inputs_processor::TextInputsProcessor.process_inputs(
                tokenizer,
                input_seqs,
                is_prompt,
                is_xlora,
                device,
                no_kv_cache,
                last_n_context_len,
                other_config,
            );
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

        let num_img_tokens = num_img_tokens.unwrap();
        for (detokenized, seq) in detokenized.into_iter().zip(input_seqs.iter()) {
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
            assert_eq!(
                unique_image_ids,
                (1u32..unique_image_ids.len() as u32 + 1).collect::<Vec<_>>()
            );
            // Total images must be the same as the number of image tags
            assert_eq!(unique_image_ids.len(), seq.images().unwrap().len());

            // Use the TryInto + unwrap_or to handle case when id==0
            let image_ids_pad = image_ids
                .iter()
                .map(|id| {
                    [*id].repeat(
                        num_img_tokens[TryInto::<usize>::try_into(*id as isize - 1)
                            .unwrap_or(num_img_tokens.len() - 1)],
                    )
                })
                .collect::<Vec<_>>();

            let mut input_ids: Vec<u32> = Vec::new();
            for item in prompt_chunks.iter().interleave(&image_ids_pad) {
                input_ids.extend(item);
            }
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

pub(crate) fn from_pixel_data_top_bottom(
    mut data: Vec<Vec<Vec<u8>>>,
    h: usize,
    w: usize,
    fill: u8,
    pad_top: usize,
    pad_bottom: usize,
) -> DynamicImage {
    let channels = data[0][0].len();
    let cur_w = data[0].len();

    let mut flat_data: Vec<u8> = Vec::with_capacity(w * h * channels);
    // Add top padding
    for _ in 0..pad_top {
        data.insert(0, vec![vec![fill; channels]; cur_w]);
    }
    // Add the bottom padding
    data.extend(vec![vec![vec![fill; channels]; cur_w]; pad_bottom]);
    for row in data {
        for pixel in row {
            flat_data.extend_from_slice(&pixel);
        }
    }

    if channels == 3 {
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    } else {
        DynamicImage::ImageRgba8(
            RgbaImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    }
}

pub(crate) fn from_pixel_data_left_right(
    data: Vec<Vec<Vec<u8>>>,
    h: usize,
    w: usize,
    fill: u8,
    pad_left: usize,
    pad_right: usize,
) -> DynamicImage {
    let channels = data[0][0].len();

    let mut flat_data: Vec<u8> = Vec::with_capacity(w * h * channels);
    for mut row in data {
        // Add left padding
        for _ in 0..pad_left {
            row.insert(0, vec![fill; channels]);
        }
        // Add the right padding
        row.extend(vec![vec![fill; channels]; pad_right]);
        for pixel in row {
            flat_data.extend_from_slice(&pixel);
        }
    }

    if channels == 3 {
        DynamicImage::ImageRgb8(
            RgbImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    } else {
        DynamicImage::ImageRgba8(
            RgbaImage::from_raw(w as u32, h as u32, flat_data).expect("Unable to create RgbaImage"),
        )
    }
}

/// Pad image, left and right if transposed, and top to bottom if not
fn pad_336(image: &DynamicImage, trans: bool) -> DynamicImage {
    let (mut w, mut h) = image.dimensions();
    if w < h {
        std::mem::swap(&mut w, &mut h);
    }

    let tar = ((h as f32 / 336.).ceil() * 336.) as u32;
    let top_pad = ((tar - h) as f32 / 2.) as u32; // also right if transposed
    let bottom_pad = tar - h - top_pad; // also left if transposed

    let data = get_pixel_data(
        image,
        image.dimensions().1 as usize,
        image.dimensions().0 as usize,
    );
    if trans {
        from_pixel_data_top_bottom(
            data,
            (h + top_pad + bottom_pad) as usize,
            w as usize,
            255,
            top_pad as usize,
            bottom_pad as usize,
        )
    } else {
        from_pixel_data_left_right(
            data,
            h as usize,
            (w + top_pad + bottom_pad) as usize,
            255,
            top_pad as usize,
            bottom_pad as usize,
        )
    }
}

impl Phi3InputsProcessor {
    fn hd_transform(image: &DynamicImage, num_crops: usize) -> Result<DynamicImage> {
        let (mut w, mut h) = image.dimensions();
        let trans = if w < h {
            std::mem::swap(&mut w, &mut h);
            true
        } else {
            false
        };
        let ratio = w as f32 / h as f32;
        let mut scale = 1.0;
        while (scale * (scale / ratio).ceil()) as usize <= num_crops {
            scale += 1.0;
        }
        let new_w = (scale * 336.) as u32;
        let new_h = (new_w as f32 / ratio) as u32;

        // torchvision.transforms.functional.resize's default interpolation mode is bilinear
        let img = image.resize(
            if trans { new_h } else { new_w },
            if trans { new_w } else { new_h },
            FilterType::Triangle,
        );
        Ok(pad_336(&img, trans))
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
        let mut image_sizes = Vec::new();
        let mut padded_images = Vec::new();
        let mut num_img_tokens = Vec::new();
        for image in images.iter_mut() {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            *image = Self::hd_transform(image, config.num_crops.expect("Need `num_crops`"))?;

            // Normalize
            *image = self.normalize(
                image,
                config.image_mean.unwrap_or(Self::DEFAULT_MEAN),
                config.image_std.unwrap_or(Self::DEFAULT_STD),
            );

            // Resize with bicubic interpolation
            let global_image = image.resize(336, 336, FilterType::Triangle);
            let global_image = make_pixel_values(&global_image, device)?.unsqueeze(0)?;

            let (w, h) = image.dimensions();
            let num_image_tokens =
                (h as f32 / 336. * w as f32 / 336. + 1. + ((h as f32 / 336.) + 1.) * 12.) as usize;

            // (3,336,336)
            let image = make_pixel_values(image, device)?;
            let hd_image_reshape = image
                .reshape((
                    1,
                    3,
                    (h as f32 / 336.) as usize,
                    336,
                    (w as f32 / 336.) as usize,
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
        let num_img_tokens = num_img_tokens[0];

        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&padded_images, 0)?,
            image_sizes: Some((image_sizes.0 as usize, image_sizes.1 as usize)),
            pixel_attention_mask: None,
            num_img_tokens: Some(num_img_tokens),
        })
    }
}
