#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use candle_core::{Context, Device, IndexOp, Result, Tensor, D};
use image::DynamicImage;
use itertools::Itertools;
use mistralrs_vision::{
    ApplyTensorTransforms, ApplyTransforms, Normalize, Rescale, TensorTransforms, ToTensorNoNorm,
    Transforms,
};
use ordered_float::NotNan;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Llama4ModelSpecificArgs;

pub(crate) const IMAGE_TOKEN: &str = "<|image|>";
const IMAGE_START: &str = "<|image_start|>";
const IMAGE_END: &str = "<|image_end|>";
const PATCH: &str = "<|patch|>";
const TILE_X_SEP: &str = "<|tile_x_separator|>";
const TILE_Y_SEP: &str = "<|tile_y_separator|>";

// Input processor
pub struct Llama4ImageProcessor {
    pub patch_size: usize,
    pub downsample_ratio: usize,
}

impl Llama4ImageProcessor {
    pub fn new(patch_size: Option<usize>, pixel_shuffle_ratio: Option<f32>) -> Self {
        Self {
            patch_size: patch_size.unwrap_or(14),
            downsample_ratio: (1. / pixel_shuffle_ratio.unwrap_or(0.5).powi(2)).round() as usize,
        }
    }
}

// Processor
pub struct Llama4Processor {
    patch_size: usize,
    downsample_ratio: usize,
}

impl Llama4Processor {
    pub fn new(cfg: &ProcessorConfig) -> Self {
        Self {
            patch_size: cfg.patch_size.unwrap_or(14),
            downsample_ratio: (1. / cfg.pixel_shuffle_ratio.unwrap_or(0.5).powi(2)).round()
                as usize,
        }
    }
}

impl Processor for Llama4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Llama4ImageProcessor {
            patch_size: self.patch_size,
            downsample_ratio: self.downsample_ratio,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_START,
            IMAGE_END,
            PATCH,
            TILE_X_SEP,
            TILE_Y_SEP,
            IMAGE_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl Llama4ImageProcessor {
    fn prompt_split_image(&self, aspect_ratio: &Tensor, num_patches_per_chunk: usize) -> String {
        let mut img_string = IMAGE_START.to_string();
        let aspect_ratio = aspect_ratio.to_vec1::<u32>().unwrap();
        let (ratio_h, ratio_w) = (aspect_ratio[0] as usize, aspect_ratio[1] as usize);
        if ratio_h * ratio_w > 1 {
            for _yy in 0..ratio_h {
                for xx in 0..ratio_w {
                    img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
                    if xx < ratio_w - 1 {
                        img_string.push_str(TILE_X_SEP);
                    }
                }
                img_string.push_str(TILE_Y_SEP);
            }
        }
        img_string.push_str(IMAGE_TOKEN);
        img_string.push_str(&PATCH.repeat(num_patches_per_chunk));
        img_string.push_str(IMAGE_END);
        img_string
    }
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
                "Llama4InputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let pixel_values = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut aspect_ratios_accum = Vec::new();

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
                    aspect_ratio_mask: _,
                    num_tiles: _,
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
                // Intentionally don't unsqueeze here as the BS is already included. Just stack now.
                // Trim cached images per-sequence before pushing.
                let cached = seq.count_prefix_cached_mm_items();
                let n_images = pixel_values.dim(0).unwrap_or(0);
                if cached < n_images {
                    if cached > 0 {
                        pixel_values_accum
                            .push(pixel_values.narrow(0, cached, n_images - cached).unwrap());
                    } else {
                        pixel_values_accum.push(pixel_values);
                    }
                }
                aspect_ratios_accum.push(aspect_ratio_ids.unwrap());
            }

            let aspect_ratios = Tensor::cat(&aspect_ratios_accum, 0).unwrap();

            let (image_h, image_w) = (
                pixel_values_accum[0].dim(D::Minus2).unwrap(),
                pixel_values_accum[0].dim(D::Minus1).unwrap(),
            );
            let num_patches_per_chunk =
                (image_h / self.patch_size) * (image_w / self.patch_size) / self.downsample_ratio;

            let placeholder_counts = input_seqs
                .iter()
                .map(|seq| seq.get_initial_prompt().match_indices(IMAGE_TOKEN).count())
                .collect::<Vec<_>>();

            let mut image_index = 0;
            for (seq, placeholder_count) in input_seqs.iter_mut().zip(placeholder_counts) {
                if placeholder_count == 0 {
                    continue;
                }
                let prompt_splits: std::str::Split<'_, &str> =
                    seq.get_initial_prompt().split(IMAGE_TOKEN);
                let mut new_prompt = Vec::new();
                for (local_image_index, split_part) in prompt_splits.enumerate() {
                    new_prompt.push(split_part.to_string());
                    if local_image_index < placeholder_count {
                        let tokens_for_this_image = self.prompt_split_image(
                            &aspect_ratios.i(image_index).unwrap(),
                            num_patches_per_chunk,
                        );
                        image_index += 1;
                        new_prompt.push(tokens_for_this_image);
                    }
                }
                let prompt = new_prompt.join("");

                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt, false)
                        .expect("Detokenization failed!");

                    let ids = toks.get_ids().to_vec();

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(start_id), Some(end_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(IMAGE_START),
                            tokenizer.token_to_id(IMAGE_END),
                        ) {
                            let ranges = find_image_delimited_ranges(&ids, start_id, end_id);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }

                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }
            }

            if !pixel_values_accum.is_empty() {
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap())
            } else {
                None
            }
        } else {
            None
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

        let pixel_values = if is_prompt { pixel_values } else { None };

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
            model_specific_args: Box::new(Llama4ModelSpecificArgs { image_hashes }),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl Llama4ImageProcessor {
    fn get_factors(dividend: u32) -> HashSet<u32> {
        let mut factors_set = HashSet::new();

        let sqrt = (dividend as f64).sqrt() as u32;
        for i in 1..=sqrt {
            if dividend.is_multiple_of(i) {
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

    #[allow(clippy::type_complexity)]
    fn group_images_by_shape(
        &self,
        images: &[Tensor],
    ) -> Result<(
        HashMap<(usize, usize), Tensor>,
        HashMap<usize, ((usize, usize), usize)>,
    )> {
        let mut grouped_images = HashMap::new();
        let mut grouped_images_index = HashMap::new();
        for (i, image) in images.iter().enumerate() {
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

        // Min scale between w and h (limiting size -> no distortion)
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
            .filter_map(|(possible, scale)| {
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

    fn split_to_tiles(
        &self,
        images: &Tensor,
        num_tiles_h: usize,
        num_tiles_w: usize,
    ) -> Result<Tensor> {
        let (bs, c, h, w) = images.dims4()?;
        let mut images = images.reshape((
            bs,
            c,
            num_tiles_h,
            h / num_tiles_h,
            num_tiles_w,
            w / num_tiles_w,
        ))?;
        images = images.permute((0, 2, 4, 1, 3, 5))?.contiguous()?;
        images.reshape((
            bs,
            num_tiles_h * num_tiles_w,
            c,
            h / num_tiles_h,
            w / num_tiles_w,
        ))
    }

    fn reorder_images(
        &self,
        processed_images: HashMap<(usize, usize), Tensor>,
        grouped_images_index: HashMap<usize, ((usize, usize), usize)>,
    ) -> Result<Vec<Tensor>> {
        grouped_images_index
            .values()
            .map(|(k, v)| processed_images[k].i(*v))
            .collect::<Result<Vec<Tensor>>>()
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
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let max_patches = config.max_patches.unwrap_or(16);
        let size = config.size.clone().unwrap_or(HashMap::from_iter([
            ("height".to_string(), 336),
            ("width".to_string(), 336),
        ]));
        let resize_to_max_canvas = config.resize_to_max_canvas.unwrap_or(false);
        let do_rescale = config.do_rescale.unwrap_or(true);
        let do_normalize = config.do_normalize.unwrap_or(true);

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
            let image = image.apply(to_tensor_rescale, device)?;
            images.push(image);
        }

        let (grouped_images, grouped_images_index) = self.group_images_by_shape(&images)?;

        let mut grouped_processed_images = HashMap::new();
        let mut grouped_aspect_ratios = HashMap::new();
        for (shape, stacked_images) in grouped_images {
            let image_size = (
                stacked_images.dim(D::Minus2)? as u32,
                stacked_images.dim(D::Minus1)? as u32,
            );
            let target_size = self.get_best_fit(
                image_size,
                possible_resolutions.clone(),
                resize_to_max_canvas,
            )?;
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

            let rescale_and_norm_transforms = TensorTransforms {
                inner_transforms: &[
                    &do_rescale.then_some(Rescale {
                        factor: config.rescale_factor,
                    }),
                    &do_normalize.then_some(Normalize {
                        mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                        std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                    }),
                ],
            };
            processed_images = <Tensor as ApplyTensorTransforms>::apply(
                &processed_images,
                rescale_and_norm_transforms,
                device,
            )?;

            let (ratio_h, ratio_w) = (
                target_size.0 / size["height"],
                target_size.1 / size["width"],
            );
            // Split into tiles
            processed_images =
                self.split_to_tiles(&processed_images, ratio_h as usize, ratio_w as usize)?;
            grouped_processed_images.insert(shape, processed_images.clone());
            grouped_aspect_ratios.insert(
                shape,
                Tensor::new(vec![vec![ratio_h, ratio_w]; stacked_images.dim(0)?], device)?,
            );

            // Add a global tile to the processed tile if there are more than one tiles
            if ratio_h * ratio_w > 1 {
                let mut global_tiles = stacked_images
                    .interpolate2d(size["height"] as usize, size["width"] as usize)?;
                global_tiles = <Tensor as ApplyTensorTransforms>::apply(
                    &global_tiles,
                    rescale_and_norm_transforms,
                    device,
                )?;
                grouped_processed_images.insert(
                    shape,
                    Tensor::cat(&[processed_images, global_tiles.unsqueeze(1)?], 1)?,
                );
            }
        }

        let processed_images =
            self.reorder_images(grouped_processed_images, grouped_images_index.clone())?;
        let aspect_ratios_list =
            self.reorder_images(grouped_aspect_ratios, grouped_images_index.clone())?;

        let processed_images = Tensor::cat(&processed_images, 0)?;
        let aspect_ratios = Tensor::stack(&aspect_ratios_list, 0)?;

        Ok(PreprocessedImages {
            pixel_values: processed_images,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: Some(aspect_ratios),
            aspect_ratio_mask: None,
            num_tiles: None,
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
