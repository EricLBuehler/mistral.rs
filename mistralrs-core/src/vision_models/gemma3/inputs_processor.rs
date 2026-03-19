#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use itertools::Itertools;
use mistralrs_vision::{ApplyTransforms, Normalize, Rescale, ToTensorNoNorm, Transforms};
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
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Gemma3SpecificArgs;

struct Gemma3ImageProcessor {
    full_image_sequence: String,
    supports_images: bool,
}

const IMAGE_TOKEN: &str = "<image_soft_token>";
const BOI_TOKEN: &str = "<start_of_image>";
const EOI_TOKEN: &str = "<end_of_image>";

pub struct Gemma3Processor {
    full_image_sequence: String,
    supports_images: bool,
}

impl Gemma3Processor {
    pub fn new(processor_config: ProcessorConfig, supports_images: bool) -> Self {
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); processor_config.image_seq_len.unwrap_or(256)].join("");
        let full_image_sequence = format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n");

        Self {
            full_image_sequence,
            supports_images,
        }
    }
}

impl Processor for Gemma3Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Gemma3ImageProcessor {
            full_image_sequence: self.full_image_sequence.clone(),
            supports_images: self.supports_images,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[BOI_TOKEN, EOI_TOKEN, IMAGE_TOKEN]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Gemma3ImageProcessor {
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

        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();
            let re = Regex::new(BOI_TOKEN).unwrap();
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
                    image_sizes_all: _,
                    num_crops,
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

                let num_crops = num_crops.unwrap();

                let mut prompt = tokenizer
                    .decode(seq.get_toks(), false)
                    .expect("Detokenization failed!");

                let image_indexes: Vec<usize> =
                    re.find_iter(&prompt).map(|mat| mat.start()).collect();

                for (num, idx) in num_crops.into_iter().zip(image_indexes).rev() {
                    if num != 0 {
                        let formatted_image_text = format!(
                            "Here is the original image {BOI_TOKEN} and here are some crops to help you see better {}", vec![BOI_TOKEN.to_string(); num].join(" ")
                        );
                        prompt = format!(
                            "{}{formatted_image_text}{}",
                            &prompt[..idx],
                            &prompt[idx + BOI_TOKEN.len()..]
                        );
                    }
                }

                prompt = prompt.replace(BOI_TOKEN, &self.full_image_sequence);

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
                            tokenizer.token_to_id(IMAGE_TOKEN),
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

                // Per-sequence prefix cache trimming of pixel_values
                let cached = seq.count_prefix_cached_mm_items();
                let n_images = pixel_values.dim(0).unwrap_or(0);
                if cached < n_images {
                    if cached > 0 {
                        pixel_values_accum
                            .push(pixel_values.narrow(0, cached, n_images - cached).unwrap());
                    } else {
                        pixel_values_accum.push(pixel_values.clone());
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                None
            } else {
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap())
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
            model_specific_args: Box::new(Gemma3SpecificArgs { image_hashes }),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}

impl Gemma3ImageProcessor {
    fn pan_and_scan(
        &self,
        image: &DynamicImage,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> Vec<DynamicImage> {
        let (width, height) = image.dimensions();

        let (num_crops_w, num_crops_h) = if width >= height {
            if (width as f64 / height as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_w = (width as f64 / height as f64 + 0.5).floor() as usize;
            num_crops_w = num_crops_w
                .min((width as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_w = num_crops_w.max(2);
            num_crops_w = num_crops_w.min(pan_and_scan_max_num_crops);

            (num_crops_w, 1)
        } else {
            if (height as f64 / width as f64) < pan_and_scan_min_ratio_to_activate {
                return vec![];
            }

            // Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
            let mut num_crops_h = (height as f64 / width as f64 + 0.5).floor() as usize;
            num_crops_h = num_crops_h
                .min((height as f64 / pan_and_scan_min_crop_size as f64).floor() as usize);

            // Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
            num_crops_h = num_crops_h.max(2);
            num_crops_h = num_crops_h.min(pan_and_scan_max_num_crops);

            (1, num_crops_h)
        };

        let crop_size_w = (width as f64 / num_crops_w as f64).ceil() as usize;
        let crop_size_h = (height as f64 / num_crops_h as f64).ceil() as usize;

        if crop_size_w.min(crop_size_h) < pan_and_scan_min_crop_size {
            return vec![];
        }

        let crop_positions_w = (0..num_crops_w)
            .map(|i| i * crop_size_w)
            .collect::<Vec<_>>();
        let crop_positions_h = (0..num_crops_h)
            .map(|i| i * crop_size_h)
            .collect::<Vec<_>>();

        let mut image_crops = Vec::new();
        for (pos_h, pos_w) in crop_positions_h
            .into_iter()
            .cartesian_product(crop_positions_w)
        {
            image_crops.push(image.crop_imm(
                pos_w as u32,
                pos_h as u32,
                crop_size_w as u32,
                crop_size_h as u32,
            ));
        }

        image_crops
    }

    fn process_images_for_pan_and_scan(
        &self,
        images: Vec<DynamicImage>,
        pan_and_scan_min_crop_size: usize,
        pan_and_scan_max_num_crops: usize,
        pan_and_scan_min_ratio_to_activate: f64,
    ) -> (Vec<DynamicImage>, Vec<usize>) {
        let mut pas_images_list = Vec::new();
        let mut num_crops = Vec::new();

        for image in images {
            let pas_images = self.pan_and_scan(
                &image,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            num_crops.push(pas_images.len());
            pas_images_list.extend([vec![image], pas_images].concat());
        }

        (pas_images_list, num_crops)
    }
}

impl ImagePreProcessor for Gemma3ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
    const DEFAULT_STD: [f64; 3] = [0.5, 0.5, 0.5];

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
        let size = config.size.as_ref().unwrap();
        let (height, width) = (size["height"], size["width"]);
        let resample = config.resampling.to_filter()?;
        let do_rescale = config.do_rescale.unwrap();
        let rescale_factor = config.rescale_factor.unwrap();
        let do_normalize = config.do_normalize.unwrap();
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let do_pan_and_scan = config.do_pan_and_scan.unwrap_or(false);
        // https://github.com/huggingface/transformers/blob/ea219ed164bead55a5513e8cfaa17a25d5613b9e/src/transformers/models/gemma3/processing_gemma3.py#L42
        let pan_and_scan_min_crop_size = config.pan_and_scan_min_crop_size.unwrap_or(256);
        let pan_and_scan_max_num_crops = config.pan_and_scan_max_num_crops.unwrap_or(4);
        let pan_and_scan_min_ratio_to_activate =
            config.pan_and_scan_min_ratio_to_activate.unwrap_or(1.2);

        for image in images.iter_mut() {
            // Convert to rgb
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

        let num_crops = if do_pan_and_scan {
            let (new_images, num_crops) = self.process_images_for_pan_and_scan(
                images,
                pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate,
            );
            images = new_images;
            num_crops
        } else {
            vec![0]
        };

        let mut pixel_values = Vec::new();
        for mut image in images {
            if do_resize {
                image = image.resize_exact(width, height, resample);
            }

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
            image_sizes_all: None,
            num_crops: Some(num_crops),
        })
    }
}
