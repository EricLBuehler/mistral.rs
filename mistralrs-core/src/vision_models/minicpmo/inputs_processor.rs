#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex::Regex;
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
    vision_models::ModelInputs,
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
};

use super::MiniCpmOSpecificArgs;

const DEFAULT_MAX_SLICE_NUMS: usize = 9;
const DEFAULT_SCALE_RESOLUTION: usize = 448;
const DEFAULT_PATCH_SIZE: usize = 14;
const DEFAULT_IMAGE_FEATURE_SIZE: usize = 64;
const DEFAULT_IM_START_TOKEN: &str = "<image>";
const DEFAULT_IM_END_TOKEN: &str = "</image>";
const DEFAULT_IM_ID_START: &str = "<image_id>";
const DEFAULT_IM_ID_END: &str = "</image_id>";
const DEFAULT_SLICE_START_TOKEN: &str = "<slice>";
const DEFAULT_SLICE_END_TOKEN: &str = "</slice>";
const DEFAULT_UNK_TOKEN: &str = "<unk>";
const DEFAULT_USE_IMAGE_ID: bool = false;
const DEFAULT_SLICE_MODE: bool = true;

pub struct MiniCpmOImageProcessor {
    config: PreProcessorConfig,
}

pub struct MiniCpmOProcessor {
    preprocessor_config: PreProcessorConfig,
}

impl MiniCpmOProcessor {
    pub fn new(
        _config: ProcessorConfig,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Self {
        Self {
            preprocessor_config,
        }
    }
}

impl Processor for MiniCpmOProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(MiniCpmOImageProcessor {
            config: self.preprocessor_config.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            DEFAULT_IM_START_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_SLICE_START_TOKEN,
            DEFAULT_SLICE_END_TOKEN,
            DEFAULT_UNK_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl InputsProcessor for MiniCpmOImageProcessor {
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
                "MiniCpmOImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values_all, image_bound, tgt_sizes) = if has_images {
            const IMAGE_TAG: &str = "(<image>./</image>)";
            const IMAGE_PATTERN: &str = r"\(<image>./</image>\)";
            const AUDIO_PATTERN: &str = r"\(<audio>./</audio>\)";

            let image_pattern = Regex::new(IMAGE_PATTERN).unwrap();
            let _audio_pattern = Regex::new(AUDIO_PATTERN).unwrap();
            let split_pattern = Regex::new(&format!(r"({IMAGE_PATTERN}|{AUDIO_PATTERN})")).unwrap();

            let mut pixel_values_accum = Vec::new();
            let mut tgt_sizes_accum = Vec::new();
            let mut image_bounds_accum = Vec::new();

            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values: _,
                    pixel_attention_mask: _,
                    image_sizes: _,
                    num_img_tokens: _,
                    aspect_ratio_ids: _,
                    aspect_ratio_mask: _,
                    num_tiles: _,
                    image_grid_thw: _,
                    video_grid_thw: _,
                    rows: _,
                    cols: _,
                    pixel_values_list,
                    tgt_sizes,
                    image_sizes_all,
                    num_crops: _,
                } = self
                    .preprocess(
                        seq.take_images()
                            .expect("Need to have images by this point."),
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX), // Don't use it here...
                    )
                    .expect("Preprocessing failed");
                let pixel_values_list = pixel_values_list.unwrap();
                let tgt_sizes = tgt_sizes.unwrap();
                let image_sizes_all = image_sizes_all.unwrap();

                let text = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");

                let mut text_chunks = {
                    let mut results = Vec::new();
                    let mut last_end = 0;

                    for m in split_pattern.find_iter(&text) {
                        // Anything between last_end and m.start() is unmatched
                        if m.start() > last_end {
                            results.push((false, &text[last_end..m.start()]));
                        }
                        results.push((true, m.as_str()));
                        last_end = m.end();
                    }
                    // Handle the trailing unmatched part (if any)
                    if last_end < text.len() {
                        results.push((false, &text[last_end..]));
                    }

                    results
                        .into_iter()
                        .map(|(_, x)| x.to_string())
                        .collect::<Vec<_>>()
                };

                let image_tags = image_pattern.find_iter(&text).collect::<Vec<_>>();

                if !image_tags.is_empty() {
                    assert_eq!(image_tags.len(), image_sizes_all.len());
                }

                let mut image_id = 0;
                for chunk in &mut text_chunks {
                    if chunk == IMAGE_TAG {
                        *chunk =
                            self.get_slice_image_placeholder(image_sizes_all[image_id], image_id);
                        image_id += 1;
                    }
                }

                let final_text = text_chunks.join("");

                let input_ids = tokenizer
                    .encode_fast(final_text.clone(), false)
                    .unwrap()
                    .get_ids()
                    .to_vec();

                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(final_text.clone());

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                            let im_start = tokenizer
                                .encode_fast(
                                    self.config
                                        .im_start_token
                                        .clone()
                                        .unwrap_or(DEFAULT_IM_START_TOKEN.to_string()),
                                    false,
                                )
                                .unwrap()
                                .get_ids()[0];
                            let im_end = tokenizer
                                .encode_fast(
                                    self.config
                                        .im_end_token
                                        .clone()
                                        .unwrap_or(DEFAULT_IM_END_TOKEN.to_string()),
                                    false,
                                )
                                .unwrap()
                                .get_ids()[0];
                            let ranges = find_image_delimited_ranges(&input_ids, im_start, im_end);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }

                    seq.set_toks_and_reallocate(input_ids.clone(), paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }

                let image_bounds = {
                    let im_start_id = tokenizer
                        .encode_fast(
                            self.config
                                .im_start_token
                                .clone()
                                .unwrap_or(DEFAULT_IM_START_TOKEN.to_string()),
                            false,
                        )
                        .unwrap()
                        .get_ids()[0];
                    let im_end_id = tokenizer
                        .encode_fast(
                            self.config
                                .im_end_token
                                .clone()
                                .unwrap_or(DEFAULT_IM_END_TOKEN.to_string()),
                            false,
                        )
                        .unwrap()
                        .get_ids()[0];
                    let slice_start_id = tokenizer
                        .encode_fast(
                            self.config
                                .slice_start_token
                                .clone()
                                .unwrap_or(DEFAULT_SLICE_START_TOKEN.to_string()),
                            false,
                        )
                        .unwrap()
                        .get_ids()[0];
                    let slice_end_id = tokenizer
                        .encode_fast(
                            self.config
                                .slice_end_token
                                .clone()
                                .unwrap_or(DEFAULT_SLICE_END_TOKEN.to_string()),
                            false,
                        )
                        .unwrap()
                        .get_ids()[0];

                    let image_start_idx = input_ids
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &id)| {
                            if id == im_start_id || id == slice_start_id {
                                Some(i as u32 + 1)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    let image_end_idx = input_ids
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &id)| {
                            if id == im_end_id || id == slice_end_id {
                                Some(i as u32)
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>();

                    let valid_image_nums = image_start_idx.len().max(image_end_idx.len());

                    let image_start_idx = Tensor::from_slice(
                        &image_start_idx[..valid_image_nums],
                        (valid_image_nums, 1),
                        device,
                    )
                    .unwrap();
                    let image_end_idx = Tensor::from_slice(
                        &image_end_idx[..valid_image_nums],
                        (valid_image_nums, 1),
                        device,
                    )
                    .unwrap();

                    Tensor::cat(&[image_start_idx, image_end_idx], 1).unwrap()
                };

                pixel_values_accum.push(pixel_values_list);
                tgt_sizes_accum.push(tgt_sizes);
                image_bounds_accum.push(image_bounds);
            }

            (
                Some(pixel_values_accum),
                Some(image_bounds_accum),
                Some(tgt_sizes_accum),
            )
        } else {
            (None, None, None)
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

        // Trim pixel_values_all, image_bound, and tgt_sizes to exclude slices
        // already covered by the prefix cache.
        let (mut pixel_values_all, mut image_bound, mut tgt_sizes) =
            (pixel_values_all, image_bound, tgt_sizes);
        if is_prompt {
            if let (Some(ref mut pv_all), Some(ref mut ib_all), Some(ref mut ts_all)) =
                (&mut pixel_values_all, &mut image_bound, &mut tgt_sizes)
            {
                let mut any_remaining = false;
                for (seq_idx, seq) in input_seqs.iter().enumerate() {
                    let prefix_len = seq.prefix_cache_len();
                    if prefix_len == 0 {
                        if !pv_all[seq_idx].is_empty() {
                            any_remaining = true;
                        }
                        continue;
                    }

                    let bounds = ib_all[seq_idx].to_vec2::<u32>().unwrap();
                    let cached_slices = bounds
                        .iter()
                        .filter(|row| (row[0] as usize) < prefix_len)
                        .count();

                    if cached_slices == 0 {
                        if !pv_all[seq_idx].is_empty() {
                            any_remaining = true;
                        }
                        continue;
                    }

                    let remaining = bounds.len() - cached_slices;
                    // Trim pixel_values
                    pv_all[seq_idx] = pv_all[seq_idx].split_off(cached_slices);

                    if remaining > 0 {
                        any_remaining = true;
                        // Trim tgt_sizes
                        ts_all[seq_idx] =
                            ts_all[seq_idx].narrow(0, cached_slices, remaining).unwrap();
                        // Adjust image_bound positions: subtract prefix_cache_len
                        let adjusted: Vec<Vec<u32>> = bounds[cached_slices..]
                            .iter()
                            .map(|row| vec![row[0] - prefix_len as u32, row[1] - prefix_len as u32])
                            .collect();
                        ib_all[seq_idx] = Tensor::new(adjusted, device).unwrap();
                    } else {
                        // All slices cached for this sequence
                        ts_all[seq_idx] = Tensor::zeros((0, 2), DType::U32, device).unwrap();
                        ib_all[seq_idx] = Tensor::zeros((0, 2), DType::U32, device).unwrap();
                    }
                }
                if !any_remaining {
                    pixel_values_all = None;
                    image_bound = None;
                    tgt_sizes = None;
                }
            }
        }

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

        let args = MiniCpmOSpecificArgs {
            pixel_values_all,
            tgt_sizes,
            image_bound,
            image_hashes,
        };

        // Dummy pixel values - real ones are in model specific args
        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(args),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl MiniCpmOImageProcessor {
    fn get_sliced_grid(
        &self,
        (w, h): (usize, usize),
        max_slice_nums: usize,
        scale_resolution: usize,
        never_split: bool,
    ) -> Option<(usize, usize)> {
        let log_ratio = ((w / h) as f32).ln();
        let ratio = (w * h) as f32 / (scale_resolution * scale_resolution) as f32;
        let multiple = ratio.ceil().min(max_slice_nums as f32);
        if multiple <= 1. || never_split {
            return None;
        }

        let mut candidate_split_grid_nums = Vec::new();
        for i in [multiple - 1., multiple, multiple + 1.] {
            if i == 1. || i > max_slice_nums as f32 {
                continue;
            }
            candidate_split_grid_nums.push(i);
        }

        let mut candidate_grids = Vec::new();
        for split_grid_nums in candidate_split_grid_nums {
            let mut m = 1.;
            while m <= split_grid_nums {
                if split_grid_nums % m == 0. {
                    candidate_grids.push((m as usize, split_grid_nums as usize / m as usize));
                }
                m += 1.;
            }
        }

        let mut best_grid = (1, 1);
        let mut min_error = f32::INFINITY;
        for grid in candidate_grids {
            let error = (log_ratio - (grid.0 as f32 / grid.1 as f32).ln()).abs();
            if error < min_error {
                best_grid = grid;
                min_error = error;
            }
        }

        Some(best_grid)
    }

    fn ensure_divide(&self, length: usize, patch_size: usize) -> usize {
        ((length as f32 / patch_size as f32).round() * patch_size as f32).max(patch_size as f32)
            as usize
    }

    fn find_best_resize(
        &self,
        (mut w, mut h): (usize, usize),
        scale_resolution: usize,
        patch_size: usize,
        allow_upscale: bool,
    ) -> (usize, usize) {
        if w * h > scale_resolution * scale_resolution || allow_upscale {
            let r = w as f32 / h as f32;
            h = (scale_resolution as f32 / r.sqrt()) as usize;
            w = (scale_resolution as f32 * r) as usize;
        }
        let best_w = self.ensure_divide(w, patch_size);
        let best_h = self.ensure_divide(h, patch_size);
        (best_w, best_h)
    }

    fn get_refine_size(
        &self,
        (w, h): (usize, usize),
        (grid_x, grid_y): (usize, usize),
        scale_resolution: usize,
        patch_size: usize,
        allow_upscale: bool,
    ) -> (usize, usize) {
        let refine_w = self.ensure_divide(w, grid_x);
        let refine_h = self.ensure_divide(h, grid_y);

        let grid_w = refine_h / grid_x;
        let grid_h = refine_w / grid_y;

        let best_grid_size = self.find_best_resize(
            (grid_w, grid_h),
            scale_resolution,
            patch_size,
            allow_upscale,
        );

        (best_grid_size.0 * grid_x, best_grid_size.1 * grid_y)
    }

    fn split_to_patches(
        &self,
        image: &DynamicImage,
        grid: (usize, usize),
    ) -> Vec<Vec<DynamicImage>> {
        let mut patches = Vec::new();
        let (w, h) = image.dimensions();
        let (w, h) = (w as usize, h as usize);
        let grid_x = w / grid.0;
        let grid_y = h / grid.1;
        for i in (0..h).step_by(grid_y) {
            let mut images = Vec::new();
            for j in (0..w).step_by(grid_x) {
                images.push(image.crop_imm(j as u32, i as u32, grid_x as u32, grid_y as u32));
            }
            patches.push(images);
        }
        patches
    }

    fn get_sliced_images(
        &self,
        image: &DynamicImage,
        max_slice_nums: usize,
        scale_resolution: usize,
        patch_size: usize,
    ) -> Vec<DynamicImage> {
        if !self.config.slice_mode.unwrap_or(DEFAULT_SLICE_MODE) {
            return vec![image.clone()];
        }

        let dims = image.dimensions();
        let (w, h) = (dims.0 as usize, dims.1 as usize);

        let best_grid = self.get_sliced_grid((w, h), max_slice_nums, scale_resolution, false);

        let (source_images, patches) = if let Some(best_grid) = best_grid {
            // Source image, down-sampling and ensure divided by patch_size
            let best_resize = self.find_best_resize((w, h), scale_resolution, patch_size, false);
            let source_image = image.resize_exact(
                best_resize.0 as u32,
                best_resize.1 as u32,
                FilterType::CatmullRom,
            );
            let refine_size =
                self.get_refine_size((w, h), best_grid, scale_resolution, patch_size, true);
            let refine_image = image.resize_exact(
                refine_size.0 as u32,
                refine_size.1 as u32,
                FilterType::CatmullRom,
            );
            let patches = self
                .split_to_patches(&refine_image, best_grid)
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            (source_image, patches)
        } else {
            // Don't need to slice, upsample
            let best_size = self.find_best_resize((w, h), scale_resolution, patch_size, true);
            let source_images = image.resize_exact(
                best_size.0 as u32,
                best_size.1 as u32,
                FilterType::CatmullRom,
            );

            (source_images, vec![])
        };

        [vec![source_images], patches].concat()
    }

    /// image: (3, h, w)
    /// output: (3, patch_size, h*w/patch_size)
    fn reshape_by_patch(&self, image: &Tensor, patch_size: usize) -> Result<Tensor> {
        // Equivalent of torch.nn.functional.unfold with kernel_size and stride both 2-tuples
        let (_c, h, w) = image.dims3()?;
        // Kernel size
        let (kh, kw) = (patch_size, patch_size);
        // Stride
        let (sh, sw) = (patch_size, patch_size);

        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;

        let mut patches = Vec::new();
        for i in 0..out_h {
            for j in 0..out_w {
                // [c, kh, kw]
                let patch = image.i((.., i * sh..i * sh + kh, j * sw..j * sw + kw))?;
                // [c*kh*kw]
                patches.push(patch.flatten_all()?);
            }
        }
        // [C*kH*kW, out_h * out_w]
        let mut patches = Tensor::stack(&patches, 1)?;

        patches = patches.reshape((image.dim(0)?, patch_size, patch_size, ()))?;
        patches
            .permute((0, 1, 3, 2))?
            .reshape((image.dim(0)?, patch_size, ()))
    }

    fn get_image_id_placeholder(&self, image_idx: usize) -> String {
        format!(
            "{}{image_idx}{}",
            self.config
                .im_id_start
                .clone()
                .unwrap_or(DEFAULT_IM_ID_START.to_string()),
            self.config
                .im_id_end
                .clone()
                .unwrap_or(DEFAULT_IM_ID_END.to_string())
        )
    }

    fn get_grid_placeholder(&self, grid: Option<(usize, usize)>) -> String {
        if let Some(grid) = grid {
            let slice_image_placeholder = format!(
                "{}{}{}",
                self.config
                    .slice_start_token
                    .clone()
                    .unwrap_or(DEFAULT_SLICE_START_TOKEN.to_string()),
                self.config
                    .unk_token
                    .clone()
                    .unwrap_or(DEFAULT_UNK_TOKEN.to_string())
                    .repeat(
                        self.config
                            .image_feature_size
                            .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE)
                    ),
                self.config
                    .slice_end_token
                    .clone()
                    .unwrap_or(DEFAULT_SLICE_END_TOKEN.to_string())
            );

            let (cols, rows) = grid;
            let mut slices = Vec::new();
            for _ in 0..rows {
                let mut lines = Vec::new();
                for _ in 0..cols {
                    lines.push(slice_image_placeholder.clone());
                }
                slices.push(lines.join(""));
            }

            slices.join("\n")
        } else {
            "".to_string()
        }
    }

    fn get_slice_image_placeholder(&self, image_size: (u32, u32), image_idx: usize) -> String {
        let max_slice_nums = self.config.max_slice_nums.unwrap_or(DEFAULT_MAX_SLICE_NUMS);
        let use_image_id = self.config.use_image_id.unwrap_or(DEFAULT_USE_IMAGE_ID);
        let slice_mode = self.config.slice_mode.unwrap_or(DEFAULT_SLICE_MODE);

        let grid = self.get_sliced_grid(
            (image_size.0 as usize, image_size.1 as usize),
            max_slice_nums,
            DEFAULT_SCALE_RESOLUTION,
            false,
        );

        let image_placeholder = format!(
            "{}{}{}",
            self.config
                .im_start_token
                .clone()
                .unwrap_or(DEFAULT_IM_START_TOKEN.to_string()),
            self.config
                .unk_token
                .clone()
                .unwrap_or(DEFAULT_UNK_TOKEN.to_string())
                .repeat(
                    self.config
                        .image_feature_size
                        .unwrap_or(DEFAULT_IMAGE_FEATURE_SIZE)
                ),
            self.config
                .im_end_token
                .clone()
                .unwrap_or(DEFAULT_IM_END_TOKEN.to_string())
        );

        let final_placeholder = if use_image_id {
            format!(
                "{}{image_placeholder}",
                self.get_image_id_placeholder(image_idx)
            )
        } else {
            image_placeholder
        };

        if slice_mode {
            format!("{final_placeholder}{}", self.get_grid_placeholder(grid))
        } else {
            final_placeholder
        }
    }
}

impl ImagePreProcessor for MiniCpmOImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        _videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        let mut pixel_values = Vec::new();
        let mut tgt_sizes = Vec::new();
        let image_sizes = images
            .iter()
            .map(|img| img.dimensions())
            .collect::<Vec<_>>();
        for image in images {
            let max_slice_nums = config.max_slice_nums.unwrap_or(DEFAULT_MAX_SLICE_NUMS);
            let scale_resolution = config.scale_resolution.unwrap_or(DEFAULT_SCALE_RESOLUTION);
            let patch_size = config.patch_size.unwrap_or(DEFAULT_PATCH_SIZE);

            let image_patches =
                self.get_sliced_images(&image, max_slice_nums, scale_resolution, patch_size);

            for slice_image in image_patches {
                let (w, h) = slice_image.dimensions();
                let to_tensor_rescale = Transforms {
                    input: &ToTensor,
                    inner_transforms: &[&Normalize {
                        mean: config.image_mean.unwrap_or(Self::DEFAULT_MEAN).to_vec(),
                        std: config.image_std.unwrap_or(Self::DEFAULT_STD).to_vec(),
                    }],
                };
                let mut image = slice_image.apply(to_tensor_rescale, device)?;
                image = self.reshape_by_patch(&image, patch_size)?;
                pixel_values.push(image);
                tgt_sizes.push(Tensor::from_vec(
                    vec![h / patch_size as u32, w / patch_size as u32],
                    (1, 2),
                    &Device::Cpu,
                )?);
            }
        }

        let tgt_sizes = Tensor::cat(&tgt_sizes, 0)?.to_device(device)?;
        // Dummy pixel values
        Ok(PreprocessedImages {
            pixel_values: Tensor::new(0u32, &Device::Cpu)?,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: None,
            cols: None,
            pixel_values_list: Some(pixel_values),
            tgt_sizes: Some(tgt_sizes),
            image_sizes_all: Some(image_sizes),
            num_crops: None,
        })
    }
}
