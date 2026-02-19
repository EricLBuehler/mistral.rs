#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

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
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Mistral3SpecificArgs;

const PLACEHOLDER: &str = "<placeholder>";

struct Mistral3ImageProcessor {
    image_break_token: String,
    image_end_token: String,
    image_token: String,
    patch_size: usize,
    spatial_merge_size: usize,
}

pub struct Mistral3Processor {
    image_break_token: String,
    image_end_token: String,
    image_token: String,
    patch_size: usize,
    spatial_merge_size: usize,
}

impl Mistral3Processor {
    pub fn new(processor_config: ProcessorConfig) -> Self {
        Self {
            image_break_token: processor_config.image_break_token.unwrap().clone(),
            image_end_token: processor_config.image_end_token.unwrap().clone(),
            image_token: processor_config.image_token.unwrap().clone(),
            patch_size: processor_config.patch_size.unwrap(),
            spatial_merge_size: processor_config.spatial_merge_size.unwrap(),
        }
    }
}

impl Processor for Mistral3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Mistral3ImageProcessor {
            image_break_token: self.image_break_token.clone(),
            image_end_token: self.image_end_token.clone(),
            image_token: self.image_token.clone(),
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Mistral3ImageProcessor {
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

        let (pixel_values, image_sizes) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();

            for seq in input_seqs.iter_mut() {
                let PreprocessedImages {
                    pixel_values,
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
                    pixel_values_list: _,
                    tgt_sizes: _,
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
                let image_sizes_all = image_sizes_all.unwrap();

                // Deliberately no .unsqueeze here
                let mut prompt = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");

                let mut image_sizes_all_iter = image_sizes_all.iter().copied();
                let mut replace_strings = Vec::new();
                while prompt.contains(&self.image_token) {
                    let (height, width) = image_sizes_all_iter.next().unwrap();
                    let num_height_tokens =
                        (height as usize) / (self.patch_size * self.spatial_merge_size);
                    let num_width_tokens =
                        (width as usize) / (self.patch_size * self.spatial_merge_size);

                    let mut replace_tokens = vec![
                        [
                            vec![self.image_token.clone(); num_width_tokens],
                            vec![self.image_break_token.clone()],
                        ]
                        .concat();
                        num_height_tokens
                    ]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>();

                    *replace_tokens.last_mut().unwrap() = self.image_end_token.clone();

                    replace_strings.push(replace_tokens.join(""));
                    prompt = prompt.replace(&self.image_token, PLACEHOLDER);
                }

                while prompt.contains(PLACEHOLDER) {
                    let replace_str = replace_strings.pop().unwrap();
                    prompt = prompt.replace(PLACEHOLDER, &replace_str);
                }

                if !seq.multimodal.has_changed_prompt {
                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt, false)
                        .expect("Detokenization failed!");

                    let ids = toks.get_ids().to_vec();

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let (Some(hashes), Some(img_tok_id)) = (
                            seq.image_hashes().map(|h| h.to_vec()),
                            tokenizer.token_to_id(&self.image_token),
                        ) {
                            let ranges = find_image_placeholder_ranges(&ids, img_tok_id);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }

                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }

                // Per-sequence prefix cache trimming of pixel_values and image_sizes
                let cached = seq.count_prefix_cached_mm_items();
                let n_images = pixel_values.dim(0).unwrap_or(0);
                if cached < n_images {
                    if cached > 0 {
                        pixel_values_accum
                            .push(pixel_values.narrow(0, cached, n_images - cached).unwrap());
                        image_sizes_accum.extend_from_slice(&image_sizes_all[cached..]);
                    } else {
                        pixel_values_accum.push(pixel_values.clone());
                        image_sizes_accum.extend_from_slice(&image_sizes_all);
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                (None, None)
            } else {
                (
                    Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                    Some(image_sizes_accum),
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
        let image_sizes = if is_prompt { image_sizes } else { None };

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
            model_specific_args: Box::new(Mistral3SpecificArgs {
                image_sizes,
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

impl Mistral3ImageProcessor {
    #[allow(clippy::too_many_arguments)]
    fn resize(
        &self,
        image: &DynamicImage,
        mut height: usize,
        mut width: usize,
        max_height: usize,
        max_width: usize,
        patch_size: usize,
        filter: FilterType,
    ) -> DynamicImage {
        let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
        if ratio > 1. {
            height = (height as f64 / ratio).floor() as usize;
            width = (width as f64 / ratio).floor() as usize;
        }

        let num_height_tokens = (height - 1) / patch_size + 1;
        let num_width_tokens = (width - 1) / patch_size + 1;

        let resize_height = num_height_tokens * patch_size;
        let resize_width = num_width_tokens * patch_size;

        image.resize_exact(resize_width as u32, resize_height as u32, filter)
    }
}

impl ImagePreProcessor for Mistral3ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/image_processing_pixtral.py
    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let do_resize = config.do_resize.unwrap();
        let do_rescale = config.do_rescale.unwrap();
        let rescale_factor = config.rescale_factor.unwrap();
        let do_normalize = config.do_normalize.unwrap();
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let patch_size = config.patch_size.unwrap();
        let size = config.size.as_ref().unwrap();
        let resample = config.resampling.to_filter()?;

        let default_to_square = config.default_to_square.unwrap();
        assert!(default_to_square);

        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        let (max_height, max_width) = if size.contains_key("longest_edge") {
            (size["longest_edge"] as usize, size["longest_edge"] as usize)
        } else if size.contains_key("height") && size.contains_key("width") {
            (size["height"] as usize, size["width"] as usize)
        } else {
            candle_core::bail!("Size must be a map of `longest_edge` or `height` and `width`.");
        };

        for image in images.iter_mut() {
            let (width, height) = image.dimensions();

            // Convert to rgb
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            if do_resize {
                *image = self.resize(
                    image,
                    height as usize,
                    width as usize,
                    max_height,
                    max_width,
                    patch_size,
                    resample,
                );
            }

            let (width, height) = image.dimensions();

            image_sizes.push((height, width));
        }

        images = mistralrs_vision::pad_to_max_image_size(images);

        for image in images.iter_mut() {
            let transforms = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[
                    &do_rescale.then_some(Rescale {
                        factor: Some(rescale_factor),
                    }),
                    &do_normalize.then(|| Normalize {
                        mean: image_mean.to_vec(),
                        std: image_std.to_vec(),
                    }),
                ],
            };

            let image = image.apply(transforms, device)?;
            pixel_values.push(image.unsqueeze(0)?);
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&pixel_values, 0)?,
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
            pixel_values_list: None,
            tgt_sizes: None,
            image_sizes_all: Some(image_sizes),
            num_crops: None,
        })
    }
}
