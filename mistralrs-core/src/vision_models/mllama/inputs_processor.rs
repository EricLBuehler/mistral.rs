#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    collections::HashMap,
    sync::{Arc, RwLock},
};

use candle_core::{Context, DType, Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage};
use itertools::Itertools;
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensorNoNorm,
    Transforms,
};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};

use super::MLlamaSpecificArgs;

const IMAGE_TOKEN: &str = "<|image|>";

// Input processor
struct MLlamaImageProcessor {
    // To represent uninitialized, we do this. Should always be init by the time this is read.
    max_image_tiles: RwLock<Option<usize>>,
}
// Processor
pub struct MLlamaProcessor;

impl MLlamaProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Processor for MLlamaProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MLlamaImageProcessor {
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

// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/processing_mllama.py#L61
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
                    sample_idx..sample_idx + 1,
                    start as usize..end as usize,
                    mask_idx..mask_idx + 1,
                    0..mask_num_tiles,
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

impl InputsProcessor for MLlamaImageProcessor {
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
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if is_xlora {
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        let Some(tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "MLlamaInputProcessor requires a specified tokenizer.",
            ));
        };

        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions: _,
                    context_lens: _,
                    position_ids: _,
                    paged_attn_meta: _,
                    flash_meta: _,
                },
            seq_indices: _,
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
            .unwrap()
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
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
                return Err(anyhow::Error::msg(format!(
                    "The number of images in each batch {n_images_in_text:?} should be the same as the number of images {n_images_in_images:?}. The model cannot support a different number of images per patch. Perhaps you forgot a `<|image|>` tag?"
                )));
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

                // Build mm_features for position-aware prefix cache hashing
                if seq.mm_features().is_empty() {
                    if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                        let img_tok_id =
                            tokenizer.encode_fast(IMAGE_TOKEN, false).unwrap().get_ids()[0];
                        let ranges = find_image_placeholder_ranges(seq.get_toks(), img_tok_id);
                        seq.set_mm_features(build_mm_features_from_ranges(&ranges, &hashes, "img"));
                    }
                }

                seq.multimodal.has_changed_prompt = true;
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
                Err(e) => return Err(anyhow::Error::msg(e.to_string())),
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
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
            .unwrap()
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
            .unwrap()
        };

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .flat_map(|seq| {
                    seq.image_hashes()
                        .map(|h| {
                            let cached = seq.count_prefix_cached_mm_items();
                            if cached < h.len() {
                                h[cached..].to_vec()
                            } else {
                                vec![]
                            }
                        })
                        .unwrap_or_default()
                })
                .collect()
        } else {
            vec![]
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(MLlamaSpecificArgs {
                aspect_ratio_ids,
                aspect_ratio_mask,
                cross_attn_mask,
                image_hashes,
            }),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

fn argmin<T, I>(iter: I) -> Option<usize>
where
    T: PartialOrd,
    I: Iterator<Item = T>,
{
    iter.enumerate()
        .fold(None, |min, (idx, item)| match min {
            None => Some((idx, item)),
            Some((min_idx, min_item)) => {
                if item < min_item {
                    Some((idx, item))
                } else {
                    Some((min_idx, min_item))
                }
            }
        })
        .map(|(min_idx, _)| min_idx)
}

impl MLlamaImageProcessor {
    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L53
    fn get_all_supported_aspect_ratios(max_image_tiles: usize) -> Vec<(usize, usize)> {
        (1..max_image_tiles + 1)
            .flat_map(|width| {
                (1..max_image_tiles + 1).filter_map(move |height| {
                    if width * height <= max_image_tiles {
                        Some((width, height))
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>()
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L132
    fn get_optimal_tiled_canvas(
        image_height: u32,
        image_width: u32,
        max_image_tiles: usize,
        tile_size: usize,
    ) -> Result<(usize, usize)> {
        let possible_tile_arrangements = Self::get_all_supported_aspect_ratios(max_image_tiles);
        let possible_canvas_sizes: (Vec<_>, Vec<_>) = possible_tile_arrangements
            .into_iter()
            .map(|(h, w)| (h * tile_size, w * tile_size))
            .unzip();
        // Get all possible resolution heights/widths
        let (target_heights, target_widths) = possible_canvas_sizes;

        // Get scaling factors to resize the image without distortion
        let scale_h = target_heights
            .iter()
            .map(|h| *h as f32 / image_height as f32)
            .collect::<Vec<_>>();
        let scale_w = target_widths
            .iter()
            .map(|w| *w as f32 / image_width as f32)
            .collect::<Vec<_>>();

        // Get the min scale between width and height
        let scales = scale_h
            .into_iter()
            .zip(scale_w)
            .map(|(scale_h, scale_w)| if scale_w > scale_h { scale_h } else { scale_w })
            .collect::<Vec<_>>();

        // Filter only scales that allow upscaling
        let upscaling_options = scales
            .iter()
            .copied()
            .filter(|scale| *scale >= 1.)
            .collect::<Vec<_>>();
        let selected_scale = if !upscaling_options.is_empty() {
            upscaling_options
                .into_iter()
                .min_by(|x, y| x.partial_cmp(y).expect("No ordering!"))
                .context("No min, upscale")?
        } else {
            // No upscaling possible, get min downscaling (max scale for scales<1)
            let downscaling_options = scales
                .iter()
                .copied()
                .filter(|scale| *scale < 1.)
                .collect::<Vec<_>>();
            downscaling_options
                .into_iter()
                .max_by(|x, y| x.partial_cmp(y).expect("No ordering!"))
                .context("No max, downscale")?
        };

        // Get all resolutions that support this scaling factor
        let chosen_canvas_h = target_heights
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, h)| {
                if scales[i] == selected_scale {
                    Some(h)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let chosen_canvas_w = target_widths
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(i, w)| {
                if scales[i] == selected_scale {
                    Some(w)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        assert_eq!(chosen_canvas_h.len(), chosen_canvas_w.len());
        if chosen_canvas_h.len() > 1 {
            let optimal_idx = argmin(
                chosen_canvas_h
                    .iter()
                    .zip(&chosen_canvas_w)
                    .map(|(h, w)| *h * *w),
            )
            .context("No argmin")?;
            Ok((chosen_canvas_h[optimal_idx], chosen_canvas_w[optimal_idx]))
        } else {
            Ok((chosen_canvas_h[0], chosen_canvas_w[0]))
        }
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L82
    fn get_image_size_fit_to_canvas(
        image_height: u32,
        image_width: u32,
        canvas_height: usize,
        canvas_width: usize,
        tile_size: usize,
    ) -> (usize, usize) {
        let target_width = (image_width as usize).clamp(tile_size, canvas_width);
        let target_height = (image_height as usize).clamp(tile_size, canvas_height);

        let scale_h = (target_height as f32) / (image_height as f32);
        let scale_w = (target_width as f32) / (image_width as f32);

        if scale_w < scale_h {
            (
                target_height.min((image_height as f32 * scale_w).floor() as usize),
                target_width,
            )
        } else {
            (
                target_height,
                target_width.min((image_width as f32 * scale_h).floor() as usize),
            )
        }
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L796
    /// Resizes an image to fit within a tiled canvas while maintaining its aspect ratio.
    /// The optimal canvas size is calculated based on the maximum number of tiles and the tile size.
    fn resize(
        &self,
        image: DynamicImage,
        size: &HashMap<String, u32>,
        max_image_tiles: usize,
        filter: FilterType,
    ) -> Result<(DynamicImage, (usize, usize))> {
        let image_height = image.height();
        let image_width = image.width();
        let tile_size = size["height"] as usize;

        let (canvas_height, canvas_width) =
            Self::get_optimal_tiled_canvas(image_height, image_width, max_image_tiles, tile_size)?;
        let num_tiles_height = canvas_height / tile_size;
        let num_tiles_width = canvas_width / tile_size;

        let (new_height, new_width) = Self::get_image_size_fit_to_canvas(
            image_height,
            image_width,
            canvas_height,
            canvas_width,
            tile_size,
        );

        Ok((
            image.resize_exact(new_width as u32, new_height as u32, filter),
            (num_tiles_height, num_tiles_width),
        ))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L749
    /// Pad an image to the `size` x `aspect_ratio`. For example, if size is {height: 224, width: 224} and aspect ratio is
    /// (1, 2), the image will be padded to 224x448.
    fn pad(
        &self,
        image: &Tensor,
        size: &HashMap<String, u32>,
        aspect_ratio: (usize, usize),
    ) -> Result<Tensor> {
        let (num_tiles_h, num_tiles_w) = aspect_ratio;
        let padded_height = num_tiles_h * size["height"] as usize;
        let padded_width = num_tiles_w * size["width"] as usize;

        // Add padding on bottom and right sides
        mistralrs_vision::pad(image, padded_height, padded_width)
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L213
    /// Split an image into a specified number of tiles along its width and height dimensions.
    fn split_to_tiles(
        &self,
        image: &Tensor,
        num_tiles_height: usize,
        num_tiles_width: usize,
    ) -> Result<Tensor> {
        let (ch, h, w) = image.dims3()?;
        let tile_height = h / num_tiles_height;
        let tile_width = w / num_tiles_width;

        let mut image = image.reshape((
            ch,
            num_tiles_height,
            tile_height,
            num_tiles_width,
            tile_width,
        ))?;

        // Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
        image = image.permute((1, 3, 0, 2, 4))?;

        // Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
        image
            .reshape((
                num_tiles_width * num_tiles_height,
                ch,
                tile_height,
                tile_width,
            ))?
            .contiguous()
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L277
    /// Returns
    /// - stacked and packed images
    /// - a list of lists containing the number of tiles for each image in each batch sample.
    ///   Padding uses 0
    fn pack_images(
        &self,
        images: Vec<Tensor>,
        max_image_tiles: usize,
        (_bs, max_num_images): (usize, usize),
    ) -> Result<(Tensor, Vec<usize>)> {
        let (_, ch, tile_h, tile_w) = images[0].dims4()?;

        let mut stacked_images = Tensor::zeros(
            (max_num_images, max_image_tiles, ch, tile_h, tile_w),
            images[0].dtype(),
            images[0].device(),
        )?;
        let mut num_sample_tiles = Vec::new();
        for (i, image) in images.into_iter().enumerate() {
            let num_tiles = image.dim(0)?;
            stacked_images = stacked_images.slice_assign(
                &[i..i + 1, 0..num_tiles, 0..ch, 0..tile_h, 0..tile_w],
                &image.unsqueeze(0)?,
            )?;
            num_sample_tiles.push(num_tiles)
        }
        Ok((stacked_images, num_sample_tiles))
    }

    // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/mllama/image_processing_mllama.py#L354
    /// Convert aspect ratio tuples to unique ids.
    /// Padding uses 0
    fn convert_aspect_ratios_to_ids(
        &self,
        aspect_ratios: Vec<(usize, usize)>,
        max_image_tiles: usize,
        (_bs, max_num_images): (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let supported_aspect_ratios = Self::get_all_supported_aspect_ratios(max_image_tiles);

        let mut aspect_ratios_ids = vec![0i64; max_num_images];
        for (i, (num_tiles_h, num_tiles_w)) in aspect_ratios.iter().enumerate() {
            aspect_ratios_ids[i] = (supported_aspect_ratios
                .iter()
                .position(|(h, w)| *h == *num_tiles_h && *w == *num_tiles_w)
                .context("Could not find aspect ratio")?
                + 1) as i64;
        }

        Tensor::new(aspect_ratios_ids, device)
    }

    fn build_aspect_ratio_mask(
        &self,
        aspect_ratios: Vec<(usize, usize)>,
        max_image_tiles: usize,
        (_bs, max_num_images): (usize, usize),
        device: &Device,
    ) -> Result<Tensor> {
        let mut aspect_ratio_mask =
            Tensor::zeros((max_num_images, max_image_tiles), DType::I64, device)?;

        // Set the first tile to 1 for all aspect ratios
        // because in the original implementation, aspect ratios are apdded with (1,1)

        aspect_ratio_mask = aspect_ratio_mask.slice_assign(
            &[0..max_num_images, 0..1],
            &Tensor::ones((max_num_images, 1), DType::I64, device)?,
        )?;

        for (i, (num_tiles_h, num_tiles_w)) in aspect_ratios.iter().enumerate() {
            aspect_ratio_mask = aspect_ratio_mask.slice_assign(
                &[i..i + 1, 0..*num_tiles_h * *num_tiles_w],
                &Tensor::ones((1, *num_tiles_h * *num_tiles_w), DType::I64, device)?,
            )?;
        }

        Ok(aspect_ratio_mask)
    }
}

impl ImagePreProcessor for MLlamaImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (bs, max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let mut sample_images = Vec::new();
        let mut sample_aspect_ratios = Vec::new();
        let max_image_tiles = config
            .max_image_tiles
            .context("`do_resize=false` is not supported, need `max_image_tiles`!")?;
        *self.max_image_tiles.write().unwrap() = Some(max_image_tiles);

        for mut image in images {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let size = config
                .size
                .as_ref()
                .context("`do_resize=false` is not supported, need `size`!")?;

            let (image, aspect_ratio) =
                self.resize(image, size, max_image_tiles, config.resampling.to_filter()?)?;

            // In transformers they rescale from [0, 255] to [0, 1]
            // at the end of resize:
            // https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/image_transforms.py#L340
            let to_tensor_rescale = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[],
            };
            let mut image = image.apply(to_tensor_rescale, device)?;

            image = self.pad(&image, size, aspect_ratio)?;

            let transforms = TensorTransforms {
                inner_transforms: &[
                    &config
                        .do_rescale
                        .is_some_and(|x| x)
                        .then_some(())
                        .map(|_| Rescale {
                            factor: config.rescale_factor,
                        }),
                    &config
                        .do_normalize
                        .is_some_and(|x| x)
                        .then_some(())
                        .map(|_| Normalize {
                            mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                            std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                        }),
                ],
            };
            image = <Tensor as ApplyTensorTransforms>::apply(&image, transforms, device)?;

            let (num_tiles_height, num_tiles_width) = aspect_ratio;
            image = self.split_to_tiles(&image, num_tiles_height, num_tiles_width)?;

            sample_images.push(image);
            sample_aspect_ratios.push((num_tiles_height, num_tiles_width));
        }

        let (images, num_tiles) =
            self.pack_images(sample_images, max_image_tiles, (bs, max_num_images))?;

        let aspect_ratio_ids = self.convert_aspect_ratios_to_ids(
            sample_aspect_ratios.clone(),
            max_image_tiles,
            (bs, max_num_images),
            device,
        )?;
        let aspect_ratio_mask = self.build_aspect_ratio_mask(
            sample_aspect_ratios,
            max_image_tiles,
            (bs, max_num_images),
            device,
        )?;

        Ok(PreprocessedImages {
            pixel_values: images,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: Some(aspect_ratio_ids),
            aspect_ratio_mask: Some(aspect_ratio_mask),
            num_tiles: Some(num_tiles),
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }
}
