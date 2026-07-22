#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops, DynamicImage, RgbImage};
use itertools::Itertools;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::{build_mm_features_from_ranges, find_image_delimited_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        ModelInputs,
    },
};

use super::{config::Config, Lfm2VlSpecificArgs};

pub(crate) const IMAGE_TOKEN: &str = "<image>";
const IMAGE_START: &str = "<|image_start|>";
const IMAGE_END: &str = "<|image_end|>";
const IMAGE_THUMBNAIL: &str = "<|img_thumbnail|>";

#[derive(Clone)]
pub struct Lfm2VlImageProcessor {
    settings: Lfm2VlProcessorSettings,
}

pub struct Lfm2VlProcessor {
    settings: Lfm2VlProcessorSettings,
}

#[derive(Clone)]
struct Lfm2VlProcessorSettings {
    downsample_factor: usize,
    do_image_splitting: bool,
    min_tiles: usize,
    max_tiles: usize,
    use_thumbnail: bool,
    min_image_tokens: usize,
    max_image_tokens: usize,
    encoder_patch_size: usize,
    tile_size: usize,
    max_pixels_tolerance: f64,
}

struct PreprocessedForSeq {
    pixel_values: Tensor,
    pixel_attention_mask: Tensor,
    spatial_shapes: Tensor,
    rows: Vec<usize>,
    cols: Vec<usize>,
    image_sizes: Vec<(u32, u32)>,
    num_crops: Vec<usize>,
}

impl Lfm2VlProcessorSettings {
    fn from_config(config: &Config, preprocessor_config: &PreProcessorConfig) -> Self {
        let do_image_splitting = preprocessor_config
            .do_image_splitting
            .unwrap_or(config.do_image_splitting);
        Self {
            downsample_factor: preprocessor_config
                .downsample_factor
                .unwrap_or(config.downsample_factor),
            do_image_splitting,
            min_tiles: if do_image_splitting {
                preprocessor_config.min_tiles.unwrap_or(config.min_tiles)
            } else {
                1
            },
            max_tiles: if do_image_splitting {
                preprocessor_config.max_tiles.unwrap_or(config.max_tiles)
            } else {
                1
            },
            use_thumbnail: preprocessor_config
                .use_thumbnail
                .unwrap_or(config.use_thumbnail),
            min_image_tokens: preprocessor_config
                .min_image_tokens
                .unwrap_or(config.min_image_tokens),
            max_image_tokens: preprocessor_config
                .max_image_tokens
                .unwrap_or(config.max_image_tokens),
            encoder_patch_size: preprocessor_config
                .encoder_patch_size
                .unwrap_or(config.encoder_patch_size),
            tile_size: preprocessor_config.tile_size.unwrap_or(config.tile_size),
            max_pixels_tolerance: preprocessor_config
                .max_pixels_tolerance
                .unwrap_or(config.max_pixels_tolerance),
        }
    }

    fn max_num_patches(&self, config: &PreProcessorConfig) -> usize {
        config.max_num_patches.unwrap_or_else(|| {
            let max_thumbnail_image_patches = self.max_image_tokens * self.downsample_factor.pow(2);
            let tile_size_patches = if self.do_image_splitting {
                (self.tile_size / self.encoder_patch_size).pow(2)
            } else {
                0
            };
            max_thumbnail_image_patches.max(tile_size_patches)
        })
    }
}

impl Lfm2VlProcessor {
    pub fn new(config: &Config, preprocessor_config: &PreProcessorConfig) -> Self {
        Self {
            settings: Lfm2VlProcessorSettings::from_config(config, preprocessor_config),
        }
    }
}

impl Processor for Lfm2VlProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Lfm2VlImageProcessor {
            settings: self.settings.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[IMAGE_TOKEN, IMAGE_START, IMAGE_END, IMAGE_THUMBNAIL]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl Lfm2VlImageProcessor {
    fn round_by_factor(number: u32, factor: usize) -> usize {
        ((number as f64 / factor as f64).round() as usize) * factor
    }

    fn target_ratios(&self, min_tiles: usize, max_tiles: usize) -> Vec<(usize, usize)> {
        let mut ratios = Vec::new();
        for n in min_tiles..=max_tiles {
            for width in 1..=n {
                for height in 1..=n {
                    let tiles = width * height;
                    if min_tiles <= tiles && tiles <= max_tiles {
                        ratios.push((width, height));
                    }
                }
            }
        }
        ratios
            .into_iter()
            .unique()
            .sorted_by_key(|x| x.0 * x.1)
            .collect()
    }

    fn closest_aspect_ratio(
        &self,
        aspect_ratio: f64,
        target_ratios: &[(usize, usize)],
        width: u32,
        height: u32,
        image_size: usize,
    ) -> (usize, usize) {
        let mut best_ratio_diff = f64::INFINITY;
        let mut best_ratio = (1, 1);
        let area = width as usize * height as usize;
        for ratio in target_ratios {
            let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
            if ratio_diff < best_ratio_diff {
                best_ratio_diff = ratio_diff;
                best_ratio = *ratio;
            } else if ratio_diff == best_ratio_diff {
                let target_area = image_size * image_size * ratio.0 * ratio.1;
                if area > target_area / 2 {
                    best_ratio = *ratio;
                }
            }
        }
        best_ratio
    }

    fn grid_layout(&self, height: u32, width: u32) -> (usize, usize, u32, u32) {
        let target_ratios = self.target_ratios(self.settings.min_tiles, self.settings.max_tiles);
        let (grid_width, grid_height) = self.closest_aspect_ratio(
            width as f64 / height as f64,
            &target_ratios,
            width,
            height,
            self.settings.tile_size,
        );
        (
            grid_width,
            grid_height,
            (self.settings.tile_size * grid_width) as u32,
            (self.settings.tile_size * grid_height) as u32,
        )
    }

    fn smart_resize(&self, height: u32, width: u32) -> (u32, u32) {
        let total_factor = self.settings.encoder_patch_size * self.settings.downsample_factor;
        let min_pixels = self.settings.min_image_tokens
            * self.settings.encoder_patch_size.pow(2)
            * self.settings.downsample_factor.pow(2);
        let max_pixels = self.settings.max_image_tokens
            * self.settings.encoder_patch_size.pow(2)
            * self.settings.downsample_factor.pow(2);

        let mut h_bar = total_factor.max(Self::round_by_factor(height, total_factor));
        let mut w_bar = total_factor.max(Self::round_by_factor(width, total_factor));

        if h_bar * w_bar > max_pixels {
            let beta = ((height as usize * width as usize) as f64 / max_pixels as f64).sqrt();
            h_bar = total_factor
                .max((height as f64 / beta / total_factor as f64).floor() as usize * total_factor);
            w_bar = total_factor
                .max((width as f64 / beta / total_factor as f64).floor() as usize * total_factor);
        } else if h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f64 / (height as usize * width as usize) as f64).sqrt();
            h_bar = (height as f64 * beta / total_factor as f64).ceil() as usize * total_factor;
            w_bar = (width as f64 * beta / total_factor as f64).ceil() as usize * total_factor;
        }

        (w_bar as u32, h_bar as u32)
    }

    fn is_image_too_large(&self, height: u32, width: u32) -> bool {
        let total_factor = self.settings.encoder_patch_size * self.settings.downsample_factor;
        let h_bar = self
            .settings
            .encoder_patch_size
            .max(Self::round_by_factor(height, total_factor));
        let w_bar = self
            .settings
            .encoder_patch_size
            .max(Self::round_by_factor(width, total_factor));
        (h_bar * w_bar) as f64
            > (self.settings.max_image_tokens
                * self.settings.encoder_patch_size.pow(2)
                * self.settings.downsample_factor.pow(2)) as f64
                * self.settings.max_pixels_tolerance
    }

    fn resize_rgb(
        image: &RgbImage,
        width: u32,
        height: u32,
        filter: imageops::FilterType,
    ) -> RgbImage {
        imageops::resize(image, width, height, filter)
    }

    fn resize_and_split(
        &self,
        image: &RgbImage,
        filter: imageops::FilterType,
    ) -> (Vec<RgbImage>, usize, usize, (u32, u32)) {
        let (width, height) = image.dimensions();
        let (new_width, new_height) = self.smart_resize(height, width);
        let do_image_splitting = self.settings.do_image_splitting
            && !(self.settings.min_tiles == 1 && self.settings.max_tiles == 1);
        if self.is_image_too_large(height, width) && do_image_splitting {
            let (grid_width, grid_height, target_width, target_height) =
                self.grid_layout(height, width);
            let resized = Self::resize_rgb(image, target_width, target_height, filter);
            let mut crops = Vec::with_capacity(grid_width * grid_height + 1);
            for row in 0..grid_height {
                for col in 0..grid_width {
                    let crop = imageops::crop_imm(
                        &resized,
                        (col * self.settings.tile_size) as u32,
                        (row * self.settings.tile_size) as u32,
                        self.settings.tile_size as u32,
                        self.settings.tile_size as u32,
                    )
                    .to_image();
                    crops.push(crop);
                }
            }
            if self.settings.use_thumbnail && grid_width * grid_height != 1 {
                crops.push(Self::resize_rgb(image, new_width, new_height, filter));
            }
            (crops, grid_height, grid_width, (new_height, new_width))
        } else {
            (
                vec![Self::resize_rgb(image, new_width, new_height, filter)],
                1,
                1,
                (new_height, new_width),
            )
        }
    }

    fn patchify(
        &self,
        image: &RgbImage,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let patch_size = self.settings.encoder_patch_size;
        let max_num_patches = self.settings.max_num_patches(config);
        let (width, height) = image.dimensions();
        let height = height as usize;
        let width = width as usize;
        if !height.is_multiple_of(patch_size) || !width.is_multiple_of(patch_size) {
            candle_core::bail!("LFM2-VL image crop is not divisible by patch size {patch_size}");
        }

        let patches_h = height / patch_size;
        let patches_w = width / patch_size;
        let num_patches = patches_h * patches_w;
        if num_patches > max_num_patches {
            candle_core::bail!(
                "LFM2-VL crop has {num_patches} patches but max_num_patches is {max_num_patches}"
            );
        }

        let do_rescale = config.do_rescale.unwrap_or(true);
        let do_normalize = config.do_normalize.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0) as f32;
        let mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let patch_dim = patch_size * patch_size * 3;
        let mut patches = vec![0f32; max_num_patches * patch_dim];
        let raw = image.as_raw();
        for patch_y in 0..patches_h {
            for patch_x in 0..patches_w {
                let patch_idx = patch_y * patches_w + patch_x;
                let patch_offset = patch_idx * patch_dim;
                let mut elem = 0;
                for dy in 0..patch_size {
                    for dx in 0..patch_size {
                        let y = patch_y * patch_size + dy;
                        let x = patch_x * patch_size + dx;
                        let pixel_offset = (y * width + x) * 3;
                        for channel in 0..3 {
                            let mut value = raw[pixel_offset + channel] as f32;
                            if do_rescale {
                                value *= rescale_factor;
                            }
                            if do_normalize {
                                value = (value - mean[channel] as f32) / std[channel] as f32;
                            }
                            patches[patch_offset + elem] = value;
                            elem += 1;
                        }
                    }
                }
            }
        }
        let mut mask = vec![0u32; max_num_patches];
        mask[..num_patches].fill(1);
        Ok((
            Tensor::from_vec(patches, (max_num_patches, patch_dim), device)?,
            Tensor::from_vec(mask, (max_num_patches,), device)?,
            Tensor::from_vec(vec![patches_h as u32, patches_w as u32], (2,), device)?,
        ))
    }

    fn tokens_per_tile(&self) -> usize {
        let num_patches = self.settings.tile_size / self.settings.encoder_patch_size;
        let downsampled_patches = num_patches.div_ceil(self.settings.downsample_factor);
        downsampled_patches * downsampled_patches
    }

    fn tokens_for_image(&self, image_size: (u32, u32)) -> usize {
        let (height, width) = image_size;
        let patches_h = (height as usize / self.settings.encoder_patch_size)
            .div_ceil(self.settings.downsample_factor);
        let patches_w = (width as usize / self.settings.encoder_patch_size)
            .div_ceil(self.settings.downsample_factor);
        patches_h * patches_w
    }

    fn build_image_tokens(&self, rows: usize, cols: usize, image_size: (u32, u32)) -> String {
        let tokens_per_tile = self.tokens_per_tile();
        let tokens_for_image = self.tokens_for_image(image_size);
        let mut parts = vec![IMAGE_START.to_string()];
        if rows > 1 || cols > 1 {
            for row in 0..rows {
                for col in 0..cols {
                    parts.push(format!("<|img_row_{}_col_{}|>", row + 1, col + 1));
                    parts.push(IMAGE_TOKEN.repeat(tokens_per_tile));
                }
            }
            if self.settings.use_thumbnail {
                parts.push(IMAGE_THUMBNAIL.to_string());
                parts.push(IMAGE_TOKEN.repeat(tokens_for_image));
            }
        } else {
            parts.push(IMAGE_TOKEN.repeat(tokens_for_image));
        }
        parts.push(IMAGE_END.to_string());
        parts.join("")
    }

    fn expand_prompt(
        &self,
        prompt: &str,
        rows: &[usize],
        cols: &[usize],
        image_sizes: &[(u32, u32)],
    ) -> anyhow::Result<String> {
        let placeholder_count = prompt.matches(IMAGE_TOKEN).count();
        if placeholder_count != rows.len() {
            anyhow::bail!(
                "The number of `<image>` tokens ({placeholder_count}) must match the number of images ({})",
                rows.len()
            );
        }
        let mut result = String::new();
        let mut splits = prompt.split(IMAGE_TOKEN);
        result.push_str(splits.next().unwrap_or_default());
        for image_idx in 0..placeholder_count {
            result.push_str(&self.build_image_tokens(
                rows[image_idx],
                cols[image_idx],
                image_sizes[image_idx],
            ));
            result.push_str(splits.next().unwrap_or_default());
        }
        Ok(result)
    }

    fn store_cached(seq: &mut Sequence, processed: &PreprocessedForSeq) {
        seq.multimodal.cached_pixel_values = Some(processed.pixel_values.clone());
        seq.multimodal.cached_pixel_attention_mask = Some(processed.pixel_attention_mask.clone());
        seq.multimodal.cached_spatial_shapes = Some(processed.spatial_shapes.clone());
        seq.multimodal.cached_num_crops = Some(processed.num_crops.clone());
    }

    fn cached_or_preprocess(
        &self,
        seq: &mut Sequence,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<PreprocessedForSeq> {
        if let (
            Some(pixel_values),
            Some(pixel_attention_mask),
            Some(spatial_shapes),
            Some(num_crops),
        ) = (
            &seq.multimodal.cached_pixel_values,
            &seq.multimodal.cached_pixel_attention_mask,
            &seq.multimodal.cached_spatial_shapes,
            &seq.multimodal.cached_num_crops,
        ) {
            return Ok(PreprocessedForSeq {
                pixel_values: pixel_values.clone(),
                pixel_attention_mask: pixel_attention_mask.clone(),
                spatial_shapes: spatial_shapes.clone(),
                rows: Vec::new(),
                cols: Vec::new(),
                image_sizes: Vec::new(),
                num_crops: num_crops.clone(),
            });
        }

        let PreprocessedImages {
            pixel_values,
            pixel_attention_mask,
            image_sizes: _,
            num_img_tokens: _,
            aspect_ratio_ids: _,
            aspect_ratio_mask: _,
            num_tiles: _,
            image_grid_thw: _,
            video_grid_thw: _,
            rows,
            cols,
            pixel_values_list: _,
            tgt_sizes,
            image_sizes_all,
            num_crops,
        } = self.preprocess(
            seq.take_images()
                .expect("Need to have images by this point."),
            vec![],
            config,
            device,
            (usize::MAX, usize::MAX),
        )?;
        let processed = PreprocessedForSeq {
            pixel_values,
            pixel_attention_mask: pixel_attention_mask.expect("LFM2-VL returns a pixel mask"),
            spatial_shapes: tgt_sizes.expect("LFM2-VL returns spatial shapes"),
            rows: rows.expect("LFM2-VL returns row info"),
            cols: cols.expect("LFM2-VL returns col info"),
            image_sizes: image_sizes_all.expect("LFM2-VL returns image sizes"),
            num_crops: num_crops.expect("LFM2-VL returns crop counts"),
        };
        Self::store_cached(seq, &processed);
        Ok(processed)
    }

    fn trim_cached_crops(
        &self,
        processed: &PreprocessedForSeq,
        cached_images: usize,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        let crop_offset = processed
            .num_crops
            .iter()
            .take(cached_images)
            .sum::<usize>();
        let total_crops = processed.pixel_values.dim(0)?;
        if crop_offset >= total_crops {
            return Ok(None);
        }
        let len = total_crops - crop_offset;
        Ok(Some((
            processed.pixel_values.narrow(0, crop_offset, len)?,
            processed.pixel_attention_mask.narrow(0, crop_offset, len)?,
            processed.spatial_shapes.narrow(0, crop_offset, len)?,
        )))
    }

    fn maybe_expand_prompt(
        &self,
        tokenizer: &Tokenizer,
        seq: &mut Sequence,
        processed: &PreprocessedForSeq,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        if seq.multimodal.has_changed_prompt {
            return Ok(());
        }
        let prompt = self.expand_prompt(
            seq.get_initial_prompt(),
            &processed.rows,
            &processed.cols,
            &processed.image_sizes,
        )?;
        seq.set_initial_prompt(prompt.clone());
        let toks = tokenizer
            .encode_fast(prompt, false)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let ids = toks.get_ids().to_vec();
        if seq.mm_features().is_empty() {
            if let (Some(hashes), Some(start_id), Some(end_id)) = (
                seq.image_hashes().map(|h| h.to_vec()),
                tokenizer.token_to_id(IMAGE_START),
                tokenizer.token_to_id(IMAGE_END),
            ) {
                let ranges = find_image_delimited_ranges(&ids, start_id, end_id);
                seq.set_mm_features(build_mm_features_from_ranges(
                    &ranges,
                    &hashes,
                    MultimodalKind::Image,
                ));
            }
        }
        seq.set_toks_and_reallocate(ids, paged_attn_metadata);
        seq.multimodal.has_changed_prompt = true;
        Ok(())
    }
}

impl InputsProcessor for Lfm2VlImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("LFM2-VL image processor requires a tokenizer");
        };
        if !input_seqs.iter().all(|seq| seq.has_images()) {
            return Ok(());
        }
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        for seq in input_seqs {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            let processed = self.cached_or_preprocess(seq, config, device)?;
            self.maybe_expand_prompt(
                &tokenizer,
                seq,
                &processed,
                paged_attn_metadata.as_deref_mut(),
            )?;
        }
        Ok(())
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
        sliding_window: Option<usize>,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> anyhow::Result<InputProcessorOutput> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let Some(tokenizer) = tokenizer else {
            anyhow::bail!("LFM2-VL image processor requires a tokenizer");
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = is_prompt && input_seqs.iter().all(|seq| seq.has_images());
        let mut pixel_values_accum = Vec::new();
        let mut pixel_attention_mask_accum = Vec::new();
        let mut spatial_shapes_accum = Vec::new();

        if has_images {
            for seq in input_seqs.iter_mut() {
                let processed = self.cached_or_preprocess(seq, config, device)?;
                self.maybe_expand_prompt(
                    &tokenizer,
                    seq,
                    &processed,
                    paged_attn_metadata.as_mut(),
                )?;
                let cached = seq.count_prefix_cached_mm_items();
                if let Some((pixel_values, pixel_attention_mask, spatial_shapes)) =
                    self.trim_cached_crops(&processed, cached)?
                {
                    pixel_values_accum.push(pixel_values);
                    pixel_attention_mask_accum.push(pixel_attention_mask);
                    spatial_shapes_accum.push(spatial_shapes);
                }
            }
        }

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
                sliding_window,
            )?
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
                sliding_window,
            )?
        };

        let pixel_values = if pixel_values_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_values_accum, 0)?)
        };
        let pixel_attention_mask = if pixel_attention_mask_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&pixel_attention_mask_accum, 0)?)
        };
        let spatial_shapes = if spatial_shapes_accum.is_empty() {
            None
        } else {
            Some(Tensor::cat(&spatial_shapes_accum, 0)?)
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values,
                model_specific_args: Box::new(Lfm2VlSpecificArgs {
                    pixel_attention_mask,
                    spatial_shapes,
                }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::pipeline::RecurrentBatchKind::Prefill
                } else {
                    crate::pipeline::RecurrentBatchKind::Decode
                },
                adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
            }),
            seq_indices,
        })
    }
}

impl ImagePreProcessor for Lfm2VlImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        _batch_info: (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());
        let filter = config.resampling.to_filter()?;
        let mut pixel_values = Vec::new();
        let mut masks = Vec::new();
        let mut spatial_shapes = Vec::new();
        let mut rows = Vec::with_capacity(images.len());
        let mut cols = Vec::with_capacity(images.len());
        let mut image_sizes = Vec::with_capacity(images.len());
        let mut num_crops = Vec::with_capacity(images.len());

        for image in images {
            let rgb = if config.do_convert_rgb.unwrap_or(true) {
                image.to_rgb8()
            } else {
                DynamicImage::ImageRgb8(image.to_rgb8()).to_rgb8()
            };
            let (crops, crop_rows, crop_cols, image_size) = self.resize_and_split(&rgb, filter);
            rows.push(crop_rows);
            cols.push(crop_cols);
            image_sizes.push(image_size);
            num_crops.push(crops.len());
            for crop in crops {
                let (patches, mask, spatial_shape) = self.patchify(&crop, config, device)?;
                pixel_values.push(patches);
                masks.push(mask);
                spatial_shapes.push(spatial_shape);
            }
        }

        if pixel_values.is_empty() {
            candle_core::bail!("LFM2-VL preprocessing received no images");
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&pixel_values, 0)?,
            pixel_attention_mask: Some(Tensor::stack(&masks, 0)?),
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: Some(rows),
            cols: Some(cols),
            pixel_values_list: None,
            tgt_sizes: Some(Tensor::stack(&spatial_shapes, 0)?),
            image_sizes_all: Some(image_sizes),
            num_crops: Some(num_crops),
        })
    }
}
