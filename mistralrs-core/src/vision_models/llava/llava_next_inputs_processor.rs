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
use crate::vision_models::image_processor::{self, ImagePreProcessor, PreprocessedImages};
use crate::vision_models::llava_next::Config as LLaVANextConfig;
use crate::vision_models::llava_next::LLaVANextVisionSpecificArgs;
use crate::vision_models::preprocessor_config::{PreProcessorConfig, ToFilter};
use crate::vision_models::{preprocessor_config, ModelInputs};

use super::utils::{
    calculate_middle, calculate_unpad, divide_to_samples, get_anyres_image_grid_shape,
    get_num_samples, resize_and_pad_image, select_best_resolution,
};

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
        let model_config =
            serde_json::from_str::<LLaVANextConfig>(config).expect("Failed to parse model config.");
        let image_tag_splitter =
            Regex::new(r"<\|image_\d+\|>").expect("Failed to compile split regex.");
        let inputs_processor = Arc::new(LLaVANextInputProcessor {
            image_tag_splitter,
            model_config: model_config.clone(),
        });
        Self { inputs_processor }
    }
}

pub struct LLaVANextInputProcessor {
    image_tag_splitter: Regex,
    model_config: crate::vision_models::llava_next::Config,
}

impl LLaVANextInputProcessor {
    fn get_num_image_tokens(
        &self,
        image_size: (u32, u32),
        grid_pinpoints: &[(u32, u32)],
        patch_size: u32,
    ) -> usize {
        let anyres_grid_shape = get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size);
        let patch_per_side = self.model_config.vision_config.image_size / patch_size as usize;
        let unpad_shape = calculate_unpad(anyres_grid_shape, image_size);
        patch_per_side * patch_per_side + (unpad_shape.0 as usize + 1) * (unpad_shape.1 as usize)
    }
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
        let crop_size = (
            *config.crop_size.as_ref().unwrap().get("width").unwrap(),
            *config.crop_size.as_ref().unwrap().get("height").unwrap(),
        );
        let (pixel_values, image_sizes, num_img_tokens, num_image_samples, n_images) = if is_prompt
            && input_seqs
                .iter()
                .map(|seq| seq.images().is_some())
                .all(|x| x)
        {
            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            let mut num_image_samples_accum = Vec::new();
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
                } = self.preprocess(imgs.clone(), config, device)?;
                let image_sizes = image_sizes.unwrap();
                pixel_values_accum.push(pixel_values);
                image_sizes_accum.push(image_sizes);
                num_img_tokens_accum.push(num_img_tokens.unwrap());
                let num_img_samples = imgs
                    .iter()
                    .map(|img| {
                        let original_size = img.dimensions();
                        get_num_samples(
                            original_size,
                            &self.model_config.image_grid_pinpoints,
                            crop_size,
                        ) as usize
                    })
                    .collect::<Vec<_>>();
                num_image_samples_accum.push(num_img_samples);
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0)?),
                Some(image_sizes_accum),
                Some(num_img_tokens_accum),
                Some(num_image_samples_accum),
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
                    num_image_tokens: None,
                    num_image_samples: None,
                }),
            }));
        };

        let num_image_tokens_flat = num_img_tokens
            .clone()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let num_image_samples = num_image_samples
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

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
                .map(|s| {
                    // we don't use encode_batch here, because encode_batch will pad 0 to the end of the shor sequences, which will cause the image_ids_pad to be wrong.
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
                .map(|x| x.to_vec())
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
                num_image_tokens: Some(num_image_tokens_flat),
                num_image_samples: Some(num_image_samples),
            }),
        }))
    }
}

struct LLaVANextImageProcessor;

impl LLaVANextImageProcessor {
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
            Self::resize(image, resize_size as u32, filter)
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
            pixel_value = Self::normalize(&pixel_value, &image_mean, &image_std)?;
            pixel_value = Self::to_channel_dimension_format(&pixel_value)?
                .to_dtype(dtype)?
                .to_device(device)?;
        }
        Ok(pixel_value)
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
        config: &preprocessor_config::PreProcessorConfig,
        device: &candle_core::Device,
    ) -> candle_core::Result<image_processor::PreprocessedImages> {
        if images.len() > 1 {
            candle_core::bail!("Can only process one image per batch"); // This is no different from phi3_input_processor
        };
        let resized_size = *config.size.as_ref().unwrap().get("shortest_edge").unwrap() as usize;
        let image = images[0].clone();
        let original_size = image.dimensions();
        let best_resolution =
            select_best_resolution(original_size, &self.model_config.image_grid_pinpoints);
        // Here I didn't use mistral_vision::Transform, because a lot transformations are before turning the image into a tensor
        let image_padded = resize_and_pad_image(&image, best_resolution);
        let filter = config.resampling.to_filter()?;
        let image_original_resize =
            image.resize_exact(resized_size as u32, resized_size as u32, filter);
        let mut samples = vec![image_original_resize];
        for patch in divide_to_samples(
            &image_padded,
            (
                *config.crop_size.as_ref().unwrap().get("width").unwrap(),
                *config.crop_size.as_ref().unwrap().get("height").unwrap(),
            ),
        ) {
            samples.push(patch);
        }
        let dtype = match self.model_config.torch_dtype.as_str() {
            "float16" => DType::F16,
            "bfloat16" => DType::BF16,
            _ => candle_core::bail!("unsupported dtype"),
        };
        let image_mean = config
            .image_mean
            .unwrap_or(Self::DEFAULT_MEAN)
            .map(|x| x as f32);
        let image_std = config
            .image_std
            .unwrap_or(Self::DEFAULT_STD)
            .map(|x| x as f32);
        let pixel_values = samples
            .iter()
            .map(|x| {
                LLaVANextImageProcessor::process_one_image(
                    x,
                    config,
                    resized_size as u32,
                    filter,
                    dtype,
                    device,
                    &image_mean,
                    &image_std,
                )
            })
            .collect::<Result<Vec<Tensor>>>()?;
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        Ok(image_processor::PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: Some((original_size.0 as usize, original_size.1 as usize)),
            num_img_tokens: Some(vec![self.get_num_image_tokens(
                original_size,
                &self.model_config.image_grid_pinpoints,
                self.model_config.vision_config.patch_size as u32,
            )]),
        })
    }
}
