#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    num::NonZeroUsize,
    sync::{Arc, RwLock},
};

use candle_core::{Context, DType, Device, Result, Tensor, D};
use image::{imageops::FilterType, DynamicImage};
use itertools::Itertools;
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensorNoNorm,
    Transforms,
};
use ordered_float::NotNan;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};

use super::Llama4ModelSpecificArgs;

const IMAGE_TOKEN: &str = "<|image|>";

// Input processor
struct Llama4ImageProcessor {
    // To represent uninitialized, we do this. Should always be init by the time this is read.
    max_image_tiles: RwLock<Option<usize>>,
}
// Processor
pub struct Llama4Processor;

impl Llama4Processor {
    pub fn new() -> Self {
        Self
    }
}

impl Processor for Llama4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Llama4ImageProcessor {
            max_image_tiles: RwLock::new(None),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_TOKEN, "<|python_tag|>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/Llama4/processing_Llama4.py#L61
/// Generate a cross-attention token mask for image tokens in the input sequence.
fn get_cross_attention_token_mask(input_ids: Vec<u32>, image_token_id: u32) -> Vec<(i64, i64)> {
    let image_token_locations = input_ids
        .iter()
        .positions(|token| *token == image_token_id)
        .collect::<Vec<_>>();

    if image_token_locations.is_empty() {
        return vec![];
    }

    // If only one image present, unmask until end of sequence
    if image_token_locations.len() == 1 {
        return vec![(image_token_locations[0] as i64, -1)];
    }

    let mut vision_masks = image_token_locations[..image_token_locations.len() - 1]
        .iter()
        .zip(&image_token_locations[1..])
        .map(|(a, b)| (*a as i64, *b as i64))
        .collect::<Vec<_>>();

    // Last image will attent to all subsequent text
    vision_masks.push((
        *image_token_locations.last().unwrap() as i64,
        input_ids.len() as i64,
    ));

    // If there are 2 or more consecutive vision tokens, they should all attend
    // to all subsequent text present
    let mut last_mask_end = vision_masks.last().unwrap().1;
    for vision_mask in vision_masks.iter_mut().rev() {
        if vision_mask.0 == vision_mask.1 - 1 {
            vision_mask.1 = last_mask_end;
        }
        last_mask_end = vision_mask.1;
    }

    vision_masks
}

// Convert the cross attention mask indices to a cross attention mask 4D array.
/// `cross_attention_token_mask` structure:
/// - The outer list represents the batch dimension.
/// - The middle list represents different images within each batch item.
/// - The inner list contains pairs of integers [start, end] representing token ranges for each image.
///
/// `num_tiles`: the number of tiles for each image in each batch item.
///
/// NOTE: Special handling is done for cases where the end token is -1, which is interpreted as attending to the end of the sequence.
///
/// Out shape is (batch_size, length, max_num_images, max_num_tiles). 1 means attn is allowed, 0 means it is not
fn convert_sparse_cross_attention_mask_to_dense(
    cross_attn_token_mask: Vec<Vec<(i64, i64)>>,
    num_tiles: Vec<Vec<usize>>,
    max_num_tiles: usize,
    length: usize,
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let bs = cross_attn_token_mask.len();
    let max_num_images = cross_attn_token_mask.iter().map(|x| x.len()).max().unwrap();

    let mut cross_attention_mask = Tensor::zeros(
        (bs, length, max_num_images, max_num_tiles),
        DType::I64,
        &Device::Cpu,
    )?;

    for (sample_idx, (sample_masks, sample_num_tiles)) in
        cross_attn_token_mask.into_iter().zip(num_tiles).enumerate()
    {
        for (mask_idx, ((start, end), mask_num_tiles)) in
            sample_masks.into_iter().zip(sample_num_tiles).enumerate()
        {
            let mut end = end.min(length as i64);
            if end == -1 {
                end = length as i64;
            }
            cross_attention_mask = cross_attention_mask.slice_assign(
                &[
                    &sample_idx,
                    &(start as usize..end as usize),
                    &mask_idx,
                    &(..mask_num_tiles),
                ],
                &Tensor::ones(
                    (1, end as usize - start as usize, 1, mask_num_tiles),
                    DType::I64,
                    &Device::Cpu,
                )?,
            )?;
        }
    }

    cross_attention_mask.to_device(dev)
}

impl InputsProcessor for Llama4ImageProcessor {
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
            warn!("`prompt_chunksize` is set. Llama4 does not support prompt batching.");
        }
        let Some(tokenizer) = tokenizer else {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Llama4InputProcessor requires a specified tokenizer.",
            ))));
        };

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
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
            .nth(0)
            .unwrap()
            .unwrap()
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
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
            .nth(0)
            .unwrap()
            .unwrap()
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, aspect_ratio_ids, aspect_ratio_mask, cross_attn_mask) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut aspect_ratio_ids_accum = Vec::new();
            let mut aspect_ratio_mask_accum = Vec::new();
            let mut num_tiles_accum = Vec::new();

            let bs = input_seqs.len();
            let detokenized = tokenizer
                .decode_batch(
                    &input_seqs
                        .iter()
                        .map(|seq| seq.get_toks())
                        .collect::<Vec<_>>(),
                    false,
                )
                .expect("Detokenization failed!");
            let n_images_in_text = detokenized
                .iter()
                .map(|text| text.matches(IMAGE_TOKEN).count())
                .collect::<Vec<_>>();
            let n_images_in_images = input_seqs
                .iter()
                .map(|seq| seq.images().map(|imgs| imgs.len()).unwrap_or(0))
                .collect::<Vec<_>>();

            if n_images_in_text != n_images_in_images {
                return Box::new(std::iter::once(Err(anyhow::Error::msg(format!(
                    "The number of images in each batch {n_images_in_text:?} should be the same as the number of images {n_images_in_images:?}. The model cannot support a different number of images per patch. Perhaps you forgot a `<|image|>` tag?"
                )))));
            }

            let max_num_images = *n_images_in_images
                .iter()
                .max()
                .expect("No max images per batch!");

            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask: _,
                    image_sizes: _,
                    num_img_tokens: _,
                    aspect_ratio_ids,
                    aspect_ratio_mask,
                    num_tiles,
                    image_grid_thw: _,
                    video_grid_thw: _,
                    rows: _,
                    cols: _,
                    pixel_values_list: _,
                    tgt_sizes: _,
                    image_sizes_all: _,
                    num_crops: _,
                } = self
                    .preprocess(
                        seq.take_images()
                            .expect("Need to have images by this point."),
                        vec![],
                        config,
                        device,
                        (bs, max_num_images), // Don't use it here...
                    )
                    .expect("Preprocessing failed");
                pixel_values_accum.push(pixel_values.unsqueeze(0).unwrap());
                aspect_ratio_ids_accum.push(aspect_ratio_ids.unwrap().unsqueeze(0).unwrap());
                aspect_ratio_mask_accum.push(aspect_ratio_mask.unwrap().unsqueeze(0).unwrap());
                num_tiles_accum.push(num_tiles.unwrap());
            }

            // Create cross attn mask
            let image_token_id = tokenizer
                .encode_fast(IMAGE_TOKEN, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let image_token_id = if image_token_id.len() == 1 {
                image_token_id[0]
            } else {
                panic!("{IMAGE_TOKEN} encoding should be one token, got {image_token_id:?}");
            };
            let chunks = input.chunk(input.dim(0).unwrap(), 0).unwrap();
            let cross_attention_token_mask = chunks
                .iter()
                .map(|token_ids| {
                    get_cross_attention_token_mask(
                        token_ids.squeeze(0).unwrap().to_vec1::<u32>().unwrap(),
                        image_token_id,
                    )
                })
                .collect::<Vec<_>>();

            let cross_attn_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles_accum,
                self.max_image_tiles
                    .read()
                    .unwrap()
                    .expect("`max_image_tiles` must be set!"),
                chunks
                    .iter()
                    .map(|input_ids| *input_ids.dims().last().unwrap())
                    .max()
                    .unwrap(),
                chunks[0].device(),
            );

            let cross_attn_mask = match cross_attn_mask {
                Ok(v) => v,
                Err(e) => return Box::new(std::iter::once(Err(anyhow::Error::msg(e.to_string())))),
            };

            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                Some(Tensor::cat(&aspect_ratio_ids_accum, 0).unwrap()),
                Some(Tensor::cat(&aspect_ratio_mask_accum, 0).unwrap()),
                Some(cross_attn_mask),
            )
        } else {
            (None, None, None, None)
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(Llama4ModelSpecificArgs),
            paged_attn_meta,
            flash_meta,
        });
        Box::new(std::iter::once(Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })))
    }
}

impl Llama4ImageProcessor {
    fn get_factors(dividend: u32) -> HashSet<u32> {
        let mut factors_set = HashSet::new();

        let sqrt = (dividend as f64).sqrt() as u32;
        for i in 1..=sqrt {
            if dividend % i == 0 {
                factors_set.insert(i);
                factors_set.insert(dividend / i);
            }
        }

        factors_set
    }

    fn find_supported_resolutions(
        &self,
        max_num_chunks: usize,
        size: &HashMap<String, u32>,
    ) -> Result<Vec<(u32, u32)>> {
        let height = size["height"];
        let width = size["width"];
        if height != width {
            candle_core::bail!("Expected config size height==width ({height}!={width})");
        }

        let patch_size = height;

        let mut asp_map = HashMap::new();
        for chunk_size in (0..max_num_chunks).rev() {
            let factors = Self::get_factors(chunk_size as u32);
            let asp_ratios = factors
                .into_iter()
                .sorted()
                .map(|factors| (factors, chunk_size as u32 / factors));
            for (h, w) in asp_ratios {
                let ratio_float = h as f32 / w as f32;
                asp_map
                    .entry(NotNan::new(ratio_float).context("f32 is NaN")?)
                    .or_insert_with(Vec::new)
                    .push((h, w));
            }
        }

        // Get the resolutions multiplied by the patch size
        let possible_resolutions = asp_map
            .into_values()
            .flatten()
            .map(|(height, depth)| (height * patch_size, depth * patch_size))
            .collect::<Vec<_>>();

        Ok(possible_resolutions)
    }

    fn group_images_by_shape(
        &self,
        images: &Vec<Tensor>,
    ) -> Result<(
        HashMap<(usize, usize), Tensor>,
        HashMap<usize, ((usize, usize), usize)>,
    )> {
        let mut grouped_images = HashMap::new();
        let mut grouped_images_index = HashMap::new();
        for (i, image) in images.into_iter().enumerate() {
            let (_c, h, w) = image.dims3()?;
            let shape = (h, w);
            grouped_images
                .entry(shape)
                .or_insert_with(Vec::new)
                .push(image.clone());
            grouped_images_index.insert(i, (shape, grouped_images[&shape].len() - 1));
        }
        // Stack images with the same shape
        let mut grouped_images_stack = HashMap::new();
        for (shape, images) in grouped_images {
            grouped_images_stack.insert(shape, Tensor::stack(&images, 0)?);
        }

        Ok((grouped_images_stack, grouped_images_index))
    }

    fn get_best_fit(
        &self,
        (original_height, original_width): (u32, u32),
        possible_resolutions: Vec<(u32, u32)>,
        resize_to_max_canvas: bool,
    ) -> Result<(u32, u32)> {
        // All possible reslns h/w
        let (target_heights, target_widths): (Vec<u32>, Vec<u32>) =
            possible_resolutions.iter().copied().unzip();

        // Scaling factors to resize image without distortion
        let scale_w = target_widths
            .iter()
            .map(|tw| *tw as f32 / original_width as f32);
        let scale_h = target_heights
            .iter()
            .map(|th| *th as f32 / original_height as f32);

        // Min scale beetween w and h (limiting size -> no distortion)
        let scales = scale_w.zip(scale_h).map(|(w, h)| if h > w { w } else { h });

        // Filter only scales that allow upscaling
        let upscaling_options = scales
            .clone()
            .filter(|s| *s >= 1.)
            .map(|x| NotNan::new(x).unwrap())
            .collect::<Vec<_>>();
        let downscaling_options = scales
            .clone()
            .filter(|s| *s < 1.)
            .map(|x| NotNan::new(x).unwrap())
            .collect::<Vec<_>>();
        let selected_scale = if !upscaling_options.is_empty() {
            if resize_to_max_canvas {
                upscaling_options.into_iter().max().unwrap().into_inner()
            } else {
                upscaling_options.into_iter().min().unwrap().into_inner()
            }
        } else {
            // No upscaling; get min downscaling (max scale for scales < 1)
            downscaling_options.into_iter().max().unwrap().into_inner()
        };

        // All reslns that support this scaling factor
        // Ex. can upscale 224x224, 224x448, 224x672 without distortion
        // If there are multiple resolutions, get the one with minimum area to reduce padding.
        // Sort by increasing areas and take 1.
        let chosen_canvas = possible_resolutions
            .into_iter()
            .zip(scales)
            .enumerate()
            .filter_map(|(i, (possible, scale))| {
                if scale == selected_scale {
                    Some(possible)
                } else {
                    None
                }
            })
            .sorted_by_key(|(h, w)| h * w)
            .take(1)
            .collect::<Vec<_>>()[0];

        Ok(chosen_canvas)
    }

    fn get_max_res_without_distortion(
        &self,
        image_size: (u32, u32),
        target_size: (u32, u32),
    ) -> (u32, u32) {
        let (original_height, original_width) = image_size;
        let (target_height, target_width) = target_size;

        let scale_w = target_width as f64 / original_width as f64;
        let scale_h = target_height as f64 / original_height as f64;

        if scale_w < scale_h {
            let new_width = target_width;
            // Calculate new height and ensure it doesn't exceed target_height
            let new_height = std::cmp::min(
                (original_height as f64 * scale_w).floor() as u32,
                target_height,
            );
            (new_height, new_width)
        } else {
            let new_height = target_height;
            // Calculate new width and ensure it doesn't exceed target_width
            let new_width = std::cmp::min(
                (original_width as f64 * scale_h).floor() as u32,
                target_width,
            );
            (new_height, new_width)
        }
    }
}
impl ImagePreProcessor for Llama4ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images_d: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (bs, max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let max_patches = config.max_patches.context("Require `max_patches`")?;
        let size = config.size.context("Require `size`")?;
        let resize_to_max_canvas = config
            .resize_to_max_canvas
            .context("Require `resize_to_max_canvas`")?;

        let mut sample_images = Vec::new();
        let mut sample_aspect_ratios = Vec::new();

        let possible_resolutions = self.find_supported_resolutions(max_patches, &size)?;

        let mut images = Vec::new();
        for mut image in images_d {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let to_tensor_rescale = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[],
            };
            let mut image = image.apply(to_tensor_rescale, device)?;
            images.push(image);
        }

        let (grouped_images, grouped_images_index) = self.group_images_by_shape(&images)?;

        for (shape, stacked_images) in grouped_images {
            let image_size = (
                stacked_images.dim(D::Minus2)? as u32,
                stacked_images.dim(D::Minus1)? as u32,
            );
            let target_size =
                self.get_best_fit(image_size, possible_resolutions, resize_to_max_canvas)?;
            // If target_size requires upscaling, we might want to limit the upscaling to max_upscaling_size
            let max_upscaling_size = if resize_to_max_canvas {
                None
            } else {
                Some(size["height"])
            };
            let target_size_without_distortion =
                if let Some(max_upscaling_size) = max_upscaling_size {
                    let nt_h = image_size.0.max(max_upscaling_size).min(target_size.0);
                    let nt_w = image_size.1.max(max_upscaling_size).min(target_size.1);
                    (nt_h, nt_w)
                } else {
                    candle_core::bail!("Currently resize_to_max_canvas is assumed!");
                };

            // Resize to target_size while preserving aspect ratio
            let new_size_without_distortion =
                self.get_max_res_without_distortion(image_size, target_size_without_distortion);
            let mut processed_images = stacked_images.interpolate2d(
                new_size_without_distortion.0.max(1) as usize,
                new_size_without_distortion.1.max(1) as usize,
            )?;

            // Pad to target_size to be able to split into tiles
            processed_images = {
                let (target_h, target_w) = target_size;
                let (h, w) = (
                    processed_images.dim(D::Minus2)?,
                    processed_images.dim(D::Minus1)?,
                );
                let paste_x_r = target_w as usize - w;
                let paste_y_r = target_h as usize - h;
                processed_images
                    .pad_with_zeros(D::Minus2, 0, paste_y_r)?
                    .pad_with_zeros(D::Minus1, 0, paste_x_r)?
            };
            // https://github.com/huggingface/transformers/blob/9cda4265d61b0ecc276b705bd9b361a452106128/src/transformers/models/llama4/image_processing_llama4_fast.py#L446
        }

        todo!()
    }
}
