#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, cmp, collections::HashMap, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
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
    preprocessor_config::{PreProcessorConfig, ToFilter},
    processor_config::ProcessorConfig,
};

// 4k resolution as absolute maximum
const MAX_IMAGE_SIZE: usize = 4096;
const FAKE_IMAGE_TOKEN: &str = "<fake_token_around_image>";
const IMAGE_TOKEN: &str = "<image>";
const GLOBAL_IMAGE_TOKEN: &str = "<global-img>";

pub struct Idefics3ImageProcessor {
    max_edge: Option<u32>,
    image_seq_len: usize,
}

pub struct Idefics3Processor {
    config: ProcessorConfig,
    max_edge: Option<u32>,
}

impl Idefics3Processor {
    pub fn new(
        config: ProcessorConfig,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        Self { config, max_edge }
    }
}

impl Processor for Idefics3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        // Default image_seq_len is 169.
        Arc::new(Idefics3ImageProcessor {
            max_edge: self.max_edge,
            image_seq_len: self.config.image_seq_len.unwrap_or(169),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &["<fake_token_around_image>", "<image>", "<end_of_utterance>"]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

fn get_image_prompt_string(n_rows: usize, n_cols: usize, image_seq_len: usize) -> String {
    if n_rows == 0 && n_cols == 0 {
        format!(
            "{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TOKEN}{}{FAKE_IMAGE_TOKEN}",
            IMAGE_TOKEN.repeat(image_seq_len)
        )
    } else {
        let mut text_split_images = String::new();
        for n_h in 0..n_rows {
            for n_w in 0..n_cols {
                text_split_images.push_str(&format!(
                    "{FAKE_IMAGE_TOKEN}<row_{}_col_{}>{}",
                    n_h + 1,
                    n_w + 1,
                    IMAGE_TOKEN.repeat(image_seq_len)
                ));
            }
            text_split_images.push('\n');
        }
        format!(
            "{text_split_images}\n{FAKE_IMAGE_TOKEN}{GLOBAL_IMAGE_TOKEN}{}{FAKE_IMAGE_TOKEN}",
            IMAGE_TOKEN.repeat(image_seq_len)
        )
    }
}

impl InputsProcessor for Idefics3ImageProcessor {
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
                "Idefics3ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, pixel_attention_mask) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_mask_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
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
                        (usize::MAX, usize::MAX), // Don't use it here...
                    )
                    .expect("Preprocessing failed");
                let pixel_attention_mask = pixel_attention_mask.unwrap();

                if !seq.multimodal.has_changed_prompt {
                    let detok = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");

                    let mut image_prompt_strings = Vec::new();
                    for (n_rows, n_cols) in rows.unwrap().into_iter().zip(cols.unwrap().into_iter())
                    {
                        let image_prompt_string =
                            get_image_prompt_string(n_rows, n_cols, self.image_seq_len);
                        image_prompt_strings.push(image_prompt_string);
                    }

                    let split_sample = detok.split(IMAGE_TOKEN).collect::<Vec<_>>();
                    let mut sample = split_sample
                        .first()
                        .expect("The image token <image> should be present in the text.")
                        .to_string();
                    for (i, image_prompt_string) in image_prompt_strings.into_iter().enumerate() {
                        sample.push_str(&format!(
                            "{image_prompt_string}{}",
                            split_sample
                                .get(i + 1)
                                .expect("Incorrect chat template. Use the one provided in `chat_templates` with the `--chat-template`/`chat_template` settings.")
                        ));
                    }

                    seq.set_initial_prompt(sample.clone());
                    let toks = tokenizer
                        .encode_fast(sample, false)
                        .expect("Detokenization failed!");

                    let ids = toks.get_ids().to_vec();

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(fake_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(FAKE_IMAGE_TOKEN),
                        ) {
                            // Each image is wrapped in FAKE_IMAGE_TOKEN pairs.
                            // Find all FAKE_IMAGE_TOKEN...FAKE_IMAGE_TOKEN ranges.
                            let ranges = find_image_delimited_ranges(&ids, fake_id, fake_id);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }

                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }

                // Per-sequence prefix cache trimming of pixel_values and pixel_attention_mask
                let cached = seq.count_prefix_cached_mm_items();
                let n_sub = pixel_values.dim(0).unwrap_or(0);
                if cached < n_sub {
                    if cached > 0 {
                        pixel_values_accum.push(
                            pixel_values
                                .narrow(0, cached, n_sub - cached)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                        );
                        pixel_attention_mask_accum.push(
                            pixel_attention_mask
                                .narrow(0, cached, n_sub - cached)
                                .unwrap()
                                .unsqueeze(0)
                                .unwrap(),
                        );
                    } else {
                        pixel_values_accum.push(pixel_values.unsqueeze(0).unwrap());
                        pixel_attention_mask_accum.push(pixel_attention_mask.unsqueeze(0).unwrap());
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                (None, None)
            } else {
                (
                    Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                    Some(Tensor::cat(&pixel_attention_mask_accum, 0).unwrap()),
                )
            }
        } else {
            (None, None)
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
        let pixel_attention_mask = if is_prompt {
            pixel_attention_mask
        } else {
            None
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
            model_specific_args: Box::new(super::Idefics3SpecificArgs {
                pixel_attention_mask,
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

// Calculate output size after resizing, rescaling to max length
fn resize_output_size_rescale_to_max_len(
    height: usize,
    width: usize,
    min_len: Option<usize>,
    max_len: Option<usize>,
) -> (usize, usize) {
    let min_len = min_len.unwrap_or(1);
    let max_len = max_len.unwrap_or_else(|| cmp::max(height, width));
    let aspect_ratio = width as f32 / height as f32;
    let (mut height, mut width) = (height, width);

    if width >= height {
        width = max_len;
        height = (width as f32 / aspect_ratio).round() as usize;
        if height % 2 != 0 {
            height += 1;
        }
    } else {
        height = max_len;
        width = (height as f32 * aspect_ratio).round() as usize;
        if width % 2 != 0 {
            width += 1;
        }
    }

    height = cmp::max(height, min_len);
    width = cmp::max(width, min_len);

    (height, width)
}

// Calculate output size after resizing, scaling below upper bound
fn resize_output_size_scale_below_upper_bound(
    height: usize,
    width: usize,
    max_len: Option<usize>,
) -> (usize, usize) {
    let max_len = max_len.unwrap_or_else(|| cmp::max(height, width));
    let aspect_ratio = width as f32 / height as f32;
    let (mut height, mut width) = (height, width);

    if width >= height && width > max_len {
        width = max_len;
        height = (width as f32 / aspect_ratio).round() as usize;
    } else if height > width && height > max_len {
        height = max_len;
        width = (height as f32 * aspect_ratio).round() as usize;
    }

    height = cmp::max(height, 1);
    width = cmp::max(width, 1);

    (height, width)
}

/// Given the image sizes (h, w) and the minimum and maximum lengths, calculate the image dimensions
/// which will preserve aspect ration while respecing the minimum and maximum lengths.
fn get_resize_output_image_size(
    (h, w): (usize, usize),
    resolution_max_side: usize,
) -> (usize, usize) {
    let (h, w) = resize_output_size_rescale_to_max_len(h, w, None, Some(resolution_max_side));
    resize_output_size_scale_below_upper_bound(h, w, Some(MAX_IMAGE_SIZE))
}

fn resize_for_vision_encoder(
    (h, w): (usize, usize),
    vision_encoder_max_size: usize,
) -> (usize, usize) {
    let aspect_ratio = w as f32 / h as f32;

    let (new_h, new_w) = if w >= h {
        let new_w = ((w as f32 / vision_encoder_max_size as f32).ceil()
            * vision_encoder_max_size as f32) as usize;
        let mut new_h = (new_w as f32 / aspect_ratio) as usize;
        new_h = ((new_h as f32 / vision_encoder_max_size as f32).ceil()
            * vision_encoder_max_size as f32) as usize;
        (new_h, new_w)
    } else {
        let new_h = ((h as f32 / vision_encoder_max_size as f32).ceil()
            * vision_encoder_max_size as f32) as usize;
        let mut new_w = (new_h as f32 * aspect_ratio) as usize;
        new_w = ((new_w as f32 / vision_encoder_max_size as f32).ceil()
            * vision_encoder_max_size as f32) as usize;
        (new_h, new_w)
    };

    (new_h, new_w)
}

fn resize(
    image: &DynamicImage,
    size: &HashMap<String, u32>,
    resampling: FilterType,
) -> Result<DynamicImage> {
    let (h, w) = if size.contains_key("longest_edge") {
        get_resize_output_image_size(
            (image.dimensions().1 as usize, image.dimensions().0 as usize),
            size["longest_edge"] as usize,
        )
    } else if size.contains_key("height") && size.contains_key("width") {
        (size["height"] as usize, size["width"] as usize)
    } else {
        candle_core::bail!(
            "Size must be a map of `shortest_edge` and `longest_edge` or `height` and `width`."
        );
    };

    Ok(image.resize_exact(w as u32, h as u32, resampling))
    // Ok(image.resize_exact(w as u32, h as u32,  FilterType::Nearest))
}

/// Returns: frames, num_splits_h, num_splits_w
fn split_image(
    image: &DynamicImage,
    longest_edge: usize,
) -> Result<(Vec<DynamicImage>, usize, usize)> {
    let (width, height) = image.dimensions();
    let mut frames = Vec::new();

    if width > longest_edge as u32 || height > longest_edge as u32 {
        let num_splits_h = (height as f64 / (longest_edge as f64)).ceil() as usize;
        let num_splits_w = (width as f64 / (longest_edge as f64)).ceil() as usize;

        let optimal_height = (height as f64 / num_splits_h as f64).ceil() as u32;
        let optimal_width = (width as f64 / num_splits_w as f64).ceil() as u32;

        for r in 0..num_splits_h {
            for c in 0..num_splits_w {
                let start_x = (c as u32) * optimal_width;
                let start_y = (r as u32) * optimal_height;

                let end_x = std::cmp::min(start_x + optimal_width, width);
                let end_y = std::cmp::min(start_y + optimal_height, height);

                // Crop the image
                let cropped_image =
                    image.crop_imm(start_x, start_y, end_x - start_x, end_y - start_y);
                frames.push(cropped_image);
            }
        }

        // Resize the original image to match `longest_edge` for global efficiency
        let resized_image = resize(
            image,
            &HashMap::from([
                ("height".to_string(), longest_edge as u32),
                ("width".to_string(), longest_edge as u32),
            ]),
            FilterType::Lanczos3,
        )?;
        frames.push(resized_image);

        Ok((frames, num_splits_h, num_splits_w))
    } else {
        frames.push(image.clone());
        Ok((frames, 0, 0))
    }
}

impl ImagePreProcessor for Idefics3ImageProcessor {
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
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let mut patch_masks = Vec::new();
        let mut pixel_values = Vec::new();

        if let Some(max_edge) = self.max_edge {
            images = mistralrs_vision::pad_to_max_edge(&images, max_edge);
        }

        for image in images.iter_mut() {
            // Convert to rgb
            if config.do_convert_rgb.is_some_and(|x| x) {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            // Resize
            if config.do_resize.is_some_and(|x| x) {
                *image = resize(
                    image,
                    config.size.as_ref().unwrap(),
                    config.resampling.to_filter()?,
                )?;
            }
        }

        let mut image_rows = Vec::new();
        let mut image_cols = Vec::new();
        let mut new_images = Vec::new();
        let max_image_size = config
            .max_image_size
            .clone()
            .unwrap_or_else(|| HashMap::from([("longest_edge".to_string(), 364)]));
        if config.do_image_splitting.unwrap_or(true) {
            // We first resize both height and width of each image to the nearest max_image_size multiple, disregarding the aspect ratio
            // for size=(10, max_image_size) -> rescaled_size=(max_image_size, max_image_size)
            // for size=(11, max_image_size+1) -> rescaled_size=(max_image_size, max_image_size*2)
            for image in images.iter_mut() {
                let (new_h, new_w) = resize_for_vision_encoder(
                    (image.dimensions().1 as usize, image.dimensions().0 as usize),
                    max_image_size["longest_edge"] as usize,
                );

                *image =
                    image.resize_exact(new_w as u32, new_h as u32, config.resampling.to_filter()?);

                let (split_image_array, rows, cols) =
                    split_image(image, max_image_size["longest_edge"] as usize)?;
                new_images.extend(split_image_array.into_iter());
                image_rows.push(rows);
                image_cols.push(cols);
            }
        } else {
            // We square the images to max_image_size
            for image in images.iter_mut() {
                new_images.push(resize(
                    image,
                    &HashMap::from([
                        ("height".to_string(), max_image_size["longest_edge"]),
                        ("width".to_string(), max_image_size["longest_edge"]),
                    ]),
                    FilterType::Lanczos3,
                )?);
            }
            image_rows = vec![0; images.len()];
            image_cols = vec![0; images.len()];
        }
        images = new_images;

        let mut max_h = 0;
        let mut max_w = 0;
        for image in &images {
            let (w, h) = image.dimensions();
            if w > max_w {
                max_w = w;
            }
            if h > max_h {
                max_h = h;
            }
        }

        for image in images.iter_mut() {
            let transforms = Transforms {
                input: &ToTensorNoNorm,
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

            let mut image = image.apply(transforms, device)?;
            // Pad images, calculating attention mask.
            if config.do_pad.is_some_and(|x| x) {
                let (_c, h, w) = image.dims3()?;
                let padded = mistralrs_vision::pad(&image, max_h as usize, max_w as usize)?;
                let mask = mistralrs_vision::make_pixel_mask(&padded, h, w)?;
                patch_masks.push(mask.unsqueeze(0)?);
                image = padded;
            }

            // Get pixel values
            pixel_values.push(image.unsqueeze(0)?)
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
            pixel_attention_mask: Some(Tensor::cat(&patch_masks, 0)?),
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: None,
            video_grid_thw: None,
            rows: Some(image_rows),
            cols: Some(image_cols),
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: None,
            num_crops: None,
        })
    }
}
