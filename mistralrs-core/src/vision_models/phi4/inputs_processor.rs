#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{any::Any, collections::HashSet, num::NonZeroUsize, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex::Regex;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
        ProcessorCreator,
    },
    sequence::Sequence,
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    phi4::Phi4MMVisionSpecificArgs,
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
    ModelInputs,
};

use super::image_embedding::IMAGE_SPECIAL_TOKEN_ID;

const COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN: &str = r"<\|image_\d+\|>";
const IMAGE_SPECIAL_TOKEN: &str = "<|endoftext10|>";
pub(crate) const DYHD_BASE_RESOLUTION: usize = 448;

// Input processor
pub struct Phi4MMInputsProcessor;
// Processor
pub struct Phi4MMProcessor {
    inputs_processor: Arc<Phi4MMInputsProcessor>,
}

impl ProcessorCreator for Phi4MMProcessor {
    fn new_processor(
        _: Option<ProcessorConfig>,
        _: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi4MMInputsProcessor),
        })
    }
}

impl Processor for Phi4MMProcessor {
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

impl InputsProcessor for Phi4MMInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_chunksize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = anyhow::Result<InputProcessorOutput>>> {
        if is_xlora {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ))));
        }
        if no_kv_cache {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Vision model must have kv cache.",
            ))));
        }
        // TODO(EricLBuehler): support this? Would require some handling of image tokens.
        if prompt_chunksize.is_some() {
            warn!("`prompt_chunksize` is set. Idefics 2 does not support prompt batching.");
        }
        let Some(tokenizer) = tokenizer else {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Phi4MMInputProcessor requires a specified tokenizer.",
            ))));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, pixel_attention_mask, image_sizes, num_img_tokens) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_masks_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let imgs = seq
                    .take_images()
                    .expect("Need to have images by this point.");
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask,
                    image_sizes: _,
                    num_img_tokens,
                    aspect_ratio_ids: _,
                    aspect_ratio_mask: _,
                    num_tiles: _,
                    image_grid_thw: _,
                    video_grid_thw: _,
                    rows: _,
                    cols: _,
                    pixel_values_list: _,
                    tgt_sizes: _,
                    image_sizes_all,
                    num_crops: _,
                } = self
                    .preprocess(
                        imgs,
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX), // Don't use it here...
                    )
                    .expect("Preprocessor failed");
                let image_sizes = image_sizes_all.unwrap();
                let pixel_attention_mask = pixel_attention_mask.unwrap();
                pixel_values_accum.push(pixel_values);
                pixel_attention_masks_accum.push(pixel_attention_mask);
                // Using extend on purpose
                image_sizes_accum.extend(image_sizes);
                num_img_tokens_accum.push(num_img_tokens.unwrap());
            }
            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                Some(Tensor::cat(&pixel_attention_masks_accum, 0).unwrap()),
                Some(image_sizes_accum),
                Some(num_img_tokens_accum),
            )
        } else {
            return Box::new(
                text_models_inputs_processor::TextInputsProcessor
                    .process_inputs(
                        Some(tokenizer),
                        input_seqs,
                        is_prompt,
                        is_xlora,
                        device,
                        no_kv_cache,
                        last_n_context_len,
                        return_raw_logits,
                        other_config,
                        paged_attn_metadata,
                        None, // TODO
                        mapper,
                    )
                    .map(|metadata| {
                        let InputProcessorOutput {
                            inputs,
                            seq_indices,
                        } = metadata?;

                        let text_models_inputs_processor::ModelInputs {
                            input_ids,
                            input_ids_full: _,
                            seqlen_offsets,
                            seqlen_offsets_full: _,
                            context_lens,
                            position_ids,
                            paged_attn_meta,
                            flash_meta,
                            flash_meta_full: _,
                        } = *inputs
                            .downcast::<text_models_inputs_processor::ModelInputs>()
                            .expect("Downcast failed.");

                        let inputs: Box<dyn Any> = Box::new(ModelInputs {
                            input_ids,
                            seqlen_offsets,
                            context_lens,
                            position_ids,
                            pixel_values: None,
                            model_specific_args: Box::new(Phi4MMVisionSpecificArgs {
                                image_sizes: None,
                                image_attention_mask: None,
                                input_image_embeds: None,
                            }),
                            paged_attn_meta,
                            flash_meta,
                        });
                        Ok(InputProcessorOutput {
                            inputs,
                            seq_indices,
                        })
                    }),
            );
        };

        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .expect("Decode failed");

        let img_token_pattern = Regex::new(COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN).unwrap();

        let mut toks = Vec::new();

        for (mut detokenized, (seq, num_img_tokens)) in detokenized
            .into_iter()
            .zip(input_seqs.iter_mut().zip(num_img_tokens.unwrap()))
        {
            detokenized = img_token_pattern
                .replace_all(&detokenized, IMAGE_SPECIAL_TOKEN)
                .to_string();

            seq.set_toks_and_reallocate(
                tokenizer
                    .encode_fast(detokenized.clone(), false)
                    .expect("Encode failed")
                    .get_ids()
                    .to_vec(),
                paged_attn_metadata.as_mut(),
            );

            seq.set_initial_prompt(detokenized);

            let mut i = 0;
            let mut image_token_count_iter = num_img_tokens.iter();
            while i < seq.get_toks().len() {
                let token_id = seq.get_toks()[i];
                let token_count = if token_id == IMAGE_SPECIAL_TOKEN_ID as u32 {
                    image_token_count_iter.next().unwrap()
                } else {
                    i += 1;
                    continue;
                };

                let mut new_ids = seq.get_toks()[..i].to_vec();
                new_ids.extend(vec![token_id; *token_count]);
                new_ids.extend(seq.get_toks()[i + 1..].to_vec());
                seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_mut());
                i += token_count;
            }
            toks.push(seq.get_toks().to_vec());
        }

        let iter = if is_prompt {
            get_prompt_input(
                toks,
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
        } else {
            get_completion_input(
                toks,
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
        };

        Box::new(iter.into_iter().map(move |metadata| {
            let pixel_values = pixel_values.clone();
            let pixel_attention_mask = pixel_attention_mask.clone();
            let text_models_inputs_processor::InnerInputProcessorOutput {
                inputs:
                    text_models_inputs_processor::InputMetadata {
                        input,
                        positions,
                        context_lens,
                        position_ids,
                        paged_attn_meta,
                        flash_meta,
                    },
                seq_indices,
            } = metadata?;
            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: pixel_values.clone(),
                model_specific_args: Box::new(Phi4MMVisionSpecificArgs {
                    image_sizes: image_sizes.clone(),
                    image_attention_mask: pixel_attention_mask,
                    input_image_embeds: pixel_values,
                }),
                paged_attn_meta,
                flash_meta,
            });
            Ok(InputProcessorOutput {
                inputs,
                seq_indices,
            })
        }))
    }
}

impl Phi4MMInputsProcessor {
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

    fn compute_target_ratios(min_num: u32, max_num: u32) -> Vec<(u32, u32)> {
        let mut ratios: HashSet<(u32, u32)> = HashSet::new();
        for n in min_num..=max_num {
            for i in 1..=n {
                for j in 1..=n {
                    if i * j >= min_num && i * j <= max_num {
                        ratios.insert((i, j));
                    }
                }
            }
        }
        let mut sorted_ratios: Vec<(u32, u32)> = ratios.into_iter().collect();
        sorted_ratios.sort_by_key(|&(i, j)| i * j);
        sorted_ratios
    }

    fn find_closest_aspect_ratio(
        aspect_ratio: f64,
        target_ratios: Vec<(u32, u32)>,
        width: u32,
        height: u32,
        image_size: usize,
    ) -> (u32, u32) {
        let mut best_ratio_diff = f64::INFINITY;
        let mut best_ratio = (1, 1);
        let area = width * height;
        for ratio in target_ratios {
            let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
            if ratio_diff < best_ratio_diff {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if ratio_diff == best_ratio_diff
                && area as f64 > 0.5 * image_size as f64 * ratio.0 as f64 * ratio.1 as f64
            {
                best_ratio = ratio;
            }
        }
        best_ratio
    }

    fn dynamic_preprocess(
        &self,
        mut image: DynamicImage,
        min_num: usize,
        max_num: usize,
        image_size: usize,
        mask_size: usize,
        device: &Device,
    ) -> Result<(DynamicImage, Tensor)> {
        let (orig_w, orig_h) = image.dimensions();

        let w_crop_num = (orig_w as f64 / image_size as f64).ceil();
        let h_crop_num = (orig_h as f64 / image_size as f64).ceil();
        let (target_aspect_ratio, target_width, target_height) =
            if w_crop_num * h_crop_num > max_num as f64 {
                let aspect_ratio = orig_w as f64 / orig_h as f64;
                let target_ratios = Self::compute_target_ratios(min_num as u32, max_num as u32);

                let target_aspect_ratio = Self::find_closest_aspect_ratio(
                    aspect_ratio,
                    target_ratios,
                    orig_w,
                    orig_h,
                    image_size,
                );

                let target_width = image_size * target_aspect_ratio.0 as usize;
                let target_height = image_size * target_aspect_ratio.1 as usize;

                (
                    (target_aspect_ratio.0 as f64, target_aspect_ratio.1 as f64),
                    target_width,
                    target_height,
                )
            } else {
                let target_width = (image_size as f64 * w_crop_num) as usize;
                let target_height = (image_size as f64 * h_crop_num) as usize;
                let target_aspect_ratio = (w_crop_num, h_crop_num);

                (target_aspect_ratio, target_width, target_height)
            };

        let ratio_width = target_width as f64 / orig_w as f64;
        let ratio_height = target_height as f64 / orig_h as f64;
        let (new_size, padding_width, padding_height) = if ratio_width < ratio_height {
            (
                (target_width, (orig_h as f64 * ratio_width) as usize),
                0_usize,
                target_height - (orig_h as f64 * ratio_width) as usize,
            )
        } else {
            (
                ((orig_w as f64 * ratio_height) as usize, target_height),
                target_width - (orig_w as f64 * ratio_height) as usize,
                0_usize,
            )
        };

        let mut attention_mask = Tensor::ones(
            (
                (mask_size as f64 * target_aspect_ratio.1) as usize,
                (mask_size as f64 * target_aspect_ratio.0) as usize,
            ),
            DType::U32,
            device,
        )?;
        if padding_width >= 14 {
            attention_mask = attention_mask.slice_assign(
                &[&.., &(attention_mask.dim(1)? - padding_width / 14..)],
                &Tensor::zeros(
                    (attention_mask.dim(0)?, padding_width / 14),
                    DType::U32,
                    device,
                )?,
            )?;
        }
        if padding_height >= 14 {
            attention_mask = attention_mask.slice_assign(
                &[&(attention_mask.dim(0)? - padding_height / 14..), &..],
                &Tensor::zeros(
                    (padding_height / 14, attention_mask.dim(1)?),
                    DType::U32,
                    device,
                )?,
            )?;
        }

        image = image.resize_exact(new_size.0 as u32, new_size.1 as u32, FilterType::Nearest);
        image = Self::pad_image(
            &image,
            0,
            padding_height as u32,
            padding_width as u32,
            0,
            Rgba([255u8, 255, 255, 255]),
        );

        Ok((image, attention_mask))
    }
}

impl ImagePreProcessor for Phi4MMInputsProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> Result<PreprocessedImages> {
        // If no images, will not call this.
        assert!(!images.is_empty());
        assert!(videos.is_empty());

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

        let mut image_sizes = Vec::new();
        let mut padded_images = Vec::new();
        let mut padded_masks = Vec::new();
        let mut num_img_tokens = Vec::new();
        for mut image in images {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let transforms = Transforms {
                input: &ToTensor,
                inner_transforms: &[&Normalize {
                    mean: vec![0.5, 0.5, 0.5],
                    std: vec![0.5, 0.5, 0.5],
                }],
            };
            // Dynamic HD
            let dyhd_base_resolution = DYHD_BASE_RESOLUTION;
            let base_resolution = dyhd_base_resolution;
            // over 384 and 448 resolution
            let mask_resolution = base_resolution / 14;
            let min_num = 1;

            let (elem, attention_mask) = self.dynamic_preprocess(
                image,
                min_num,
                config.dynamic_hd.unwrap(),
                base_resolution,
                mask_resolution,
                device,
            )?;

            let hd_image = elem.apply(transforms, device)?;
            let (img_h, img_w) = (hd_image.dim(1)?, hd_image.dim(2)?);
            let (mask_h, mask_w) = (attention_mask.dim(0)?, attention_mask.dim(1)?);

            // Resize with bicubic interpolation
            let global_image = hd_image
                .unsqueeze(0)?
                .interpolate2d(base_resolution, base_resolution)?;
            let global_attention_mask =
                Tensor::ones((1, mask_resolution, mask_resolution), DType::U32, device)?;

            let hd_image_reshape = hd_image
                .reshape((
                    1,
                    3,
                    (img_h as f32 / base_resolution as f32) as usize,
                    base_resolution,
                    (img_w as f32 / base_resolution as f32) as usize,
                    base_resolution,
                ))?
                .permute((0, 2, 4, 1, 3, 5))?
                .reshape(((), 3, base_resolution, base_resolution))?;

            let attention_mask_reshape = attention_mask
                .reshape((
                    1,
                    (mask_h as f32 / mask_resolution as f32) as usize,
                    mask_resolution,
                    (mask_w as f32 / mask_resolution as f32) as usize,
                    mask_resolution,
                ))?
                .permute((0, 1, 3, 2, 4))?
                .reshape(((), mask_resolution, mask_resolution))?;

            let downsample_attention_mask = {
                let h_indices =
                    Tensor::arange_step(0, attention_mask_reshape.dim(1)? as u32, 2, device)?;
                let w_indices =
                    Tensor::arange_step(0, attention_mask_reshape.dim(2)? as u32, 2, device)?;
                let selected = attention_mask_reshape
                    .index_select(&h_indices, 1)?
                    .index_select(&w_indices, 2)?;

                let mask = selected
                    .reshape((
                        1,
                        mask_h / mask_resolution,
                        mask_w / mask_resolution,
                        mask_resolution / 2 + mask_resolution % 2,
                        mask_resolution / 2 + mask_resolution % 2,
                    ))?
                    .permute((0, 1, 3, 2, 4))?;
                mask.reshape((mask.dim(1)? * mask.dim(2)?, mask.dim(3)? * mask.dim(4)?))?
            };

            let img_tokens = 256
                + 1
                + downsample_attention_mask.sum_all()?.to_scalar::<u32>()? as usize
                + downsample_attention_mask
                    .i((.., 0))?
                    .sum_all()?
                    .to_scalar::<u32>()? as usize
                + 16;

            let hd_image_reshape = Tensor::cat(&[global_image, hd_image_reshape], 0)?;
            let hd_mask_reshape = Tensor::cat(&[global_attention_mask, attention_mask_reshape], 0)?;

            image_sizes.push((img_h as u32, img_w as u32));
            padded_images.push(hd_image_reshape);
            padded_masks.push(hd_mask_reshape);
            num_img_tokens.push(img_tokens);
        }
        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&padded_images, 0)?,
            pixel_attention_mask: Some(Tensor::stack(&padded_masks, 0)?),
            image_sizes: None,
            num_img_tokens: Some(num_img_tokens),
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: Some(image_sizes),
            num_crops: None,
        })
    }
}
