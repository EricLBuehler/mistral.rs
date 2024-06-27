#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::any::Any;
use std::cmp::min;
use std::sync::Arc;

use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use image::imageops::overlay;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use itertools::Itertools;
use regex_automata::meta::Regex;
use tokenizers::Tokenizer;

use crate::pipeline::text_models_inputs_processor::{get_completion_input, get_prompt_input};
use crate::pipeline::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
};
use crate::sequence::Sequence;
use crate::vision_models::image_processor::PreprocessedImages;
use crate::vision_models::llava_next::Config;
use crate::vision_models::llava_next::LLaVANextVisionSpecificArgs;
use crate::vision_models::ModelInputs;

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

fn resize_and_pad_image(image: &DynamicImage, target_resolution: (u32, u32)) -> DynamicImage {
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

fn divide_to_patches(image: &DynamicImage, crop_size: (u32, u32)) -> Vec<DynamicImage> {
    let (width, height) = image.dimensions();
    let mut patches = Vec::new();
    for y in (0..height).step_by(crop_size.1 as usize) {
        for x in (0..width).step_by(crop_size.0 as usize) {
            let patch = image.crop_imm(x, y, crop_size.0, crop_size.1);
            patches.push(patch);
        }
    }
    patches
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

pub struct LLaVANextProcessor {
    inputs_processor: Arc<LLaVANextInputProcessor>,
}

impl Processor for LLaVANextProcessor {
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

impl LLaVANextProcessor {
    pub fn new(config: &str) -> Self {
        let model_config = serde_json::from_str::<crate::vision_models::llava_next::Config>(config)
            .expect("Failed to parse model config.");
        let image_tag_splitter =
            Regex::new(r"<\|image_\d+\|>").expect("Failed to compile split regex.");
        let inputs_processor = Arc::new(LLaVANextInputProcessor {
            image_tag_splitter,
            model_config,
        });
        Self { inputs_processor }
    }
}

pub struct LLaVANextInputProcessor {
    image_tag_splitter: Regex,
    model_config: crate::vision_models::llava_next::Config,
}

// Copy from phi3_inputs_processor. different is (1) calculate of num_image_token (2) process_anyres_image (3)image_ids_pad
impl InputsProcessor for LLaVANextInputProcessor {
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
                model_specific_args: Box::new(LLaVANextVisionSpecificArgs {
                    image_sizes: None,
                    num_image_token: None,
                }),
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
            let prompt_chunks = splits
                .iter()
                .map(|s| { // we don't use encode_batch here, because encode_batch will pad 0 to the end of the shor sequences, which will cause the image_ids_pad to be wrong.
                    tokenizer
                        .encode(*s, true)
                        .unwrap()
                        .get_ids()
                        .to_vec()
                        .iter()
                        .map(|x| *x as i64)
                        .collect()
                })
                .collect::<Vec<Vec<_>>>();
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
            //only start position is -id, other positions are 0. This is for targeting image positions.
            let mut image_ids_pad = Vec::new();
            for image_id in image_ids.iter() {
                let mut image_id_pad = vec![0; num_img_tokens[*image_id as usize - 1]];
                image_id_pad[0] = -(*image_id as i64);
                image_ids_pad.push(image_id_pad);
            }
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
            model_specific_args: Box::new(LLaVANextVisionSpecificArgs {
                image_sizes,
                num_image_token: Some(self.get_num_image_tokens()),
            }),
        }))
    }
}

impl LLaVANextInputProcessor {
    fn get_num_image_tokens(&self) -> usize {
        let patch_size = self.model_config.vision_config.patch_size;
        let image_size = self.model_config.vision_config.image_size;
        let patch_per_side = image_size / patch_size;
        patch_per_side * patch_per_side + (patch_per_side * 2) * (patch_per_side * 2 + 1)
    }
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
        let image_size = *config.size.as_ref().unwrap().get("shortest_edge").unwrap() as usize;
        let image = images[0].clone();
        let original_size = image.dimensions();
        let best_resolution =
            select_best_resolution(original_size, &self.model_config.image_grid_pinpoints);
        // Here I didn't use mistral_vision::Transform, because a lot transformations are before turning the image into a tensor
        let image_padded = resize_and_pad_image(&image, best_resolution);
        let filter = config.resampling.to_filter()?;
        let image_original_resize =
            image.resize_exact(image_size as u32, image_size as u32, filter);
        let mut patches = vec![image_original_resize];
        for patch in divide_to_patches(
            &image_padded,
            (
                *config.crop_size.as_ref().unwrap().get("width").unwrap(),
                *config.crop_size.as_ref().unwrap().get("height").unwrap(),
            ),
        ) {
            patches.push(patch);
        }
        let dtype = match self.model_config.torch_dtype.as_str() {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => candle_core::bail!("unsupported dtype"),
        };
        let process_one_image = |image: &DynamicImage| -> Result<Tensor> {
            let mut image = if config.do_resize.unwrap_or(true) {
                self.resize(image, image_size as u32, filter)
            } else {
                image.clone()
            };
            image = if config.do_center_crop.unwrap_or(true) {
                let crop_width = *config.crop_size.as_ref().unwrap().get("width").unwrap() as u32;
                let crop_height = *config.crop_size.as_ref().unwrap().get("height").unwrap() as u32;
                self.center_crop(&image, (crop_width, crop_height))
            } else {
                image
            };
            let mut pixel_value = self.to_tensor(&image, &Device::Cpu)?;
            if config.do_rescale.unwrap_or(true) {
                let rescale_factor = config.rescale_factor.unwrap();
                pixel_value = self.rescale(&pixel_value, rescale_factor)?;
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
                pixel_value = self.normalize(&pixel_value, &image_mean, &image_std)?;
                pixel_value = self
                    .to_channel_dimension_format(&pixel_value)?
                    .to_dtype(dtype)?
                    .to_device(device)?;
            }
            Ok(pixel_value)
        };
        let pixel_values = patches
            .iter()
            .map(process_one_image)
            .collect::<Result<Vec<Tensor>>>()?;
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        Ok(super::image_processor::PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some((image_size, image_size)),
            num_img_tokens: Some(vec![self.get_num_image_tokens()]),
        })
    }
}
