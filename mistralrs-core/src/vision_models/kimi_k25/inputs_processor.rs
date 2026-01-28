#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTensorTransforms, ApplyTransforms, Normalize, TensorTransforms, ToTensor, Transforms};
use tokenizers::Tokenizer;

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
        preprocessor_config::PreProcessorConfig,
        ModelInputs,
    },
};

use super::KimiK25SpecificArgs;

// ── Constants ──

const PATCH_SIZE: usize = 14;
const MERGE_KERNEL_SIZE: usize = 2;
const FACTOR: usize = PATCH_SIZE * MERGE_KERNEL_SIZE; // 28
const IN_PATCH_LIMIT: usize = 16384;
const PATCH_LIMIT_ON_ONE_SIDE: usize = 512;

const DEFAULT_IMAGE_MEAN: [f64; 3] = [0.5, 0.5, 0.5];
const DEFAULT_IMAGE_STD: [f64; 3] = [0.5, 0.5, 0.5];

const PLACEHOLDER: &str = "<placeholder>";

// ── Processor ──

pub struct KimiK25Processor;

impl KimiK25Processor {
    pub const MEDIA_PAD: &str = "<|media_pad|>";

    pub fn new() -> Self {
        Self
    }
}

impl Processor for KimiK25Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(KimiK25ImageProcessor)
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[Self::MEDIA_PAD]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

// ── Image Processor ──

struct KimiK25ImageProcessor;

/// Compute resized dimensions, padding, and token count for a single image.
///
/// Implements the `navit_resize_image` logic:
/// 1. Scale down to fit within patch limits
/// 2. Pad to make dimensions divisible by `factor` (patch_size * merge_kernel_size)
/// 3. Compute token count after spatial merge
fn navit_resize_image(width: u32, height: u32) -> (u32, u32, u32, u32, usize) {
    let w = width as f64;
    let h = height as f64;

    let patches_w = (w / PATCH_SIZE as f64).max(1.0);
    let patches_h = (h / PATCH_SIZE as f64).max(1.0);

    let s1 = (IN_PATCH_LIMIT as f64 / (patches_w * patches_h)).sqrt();
    let s2 = (PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE) as f64 / w;
    let s3 = (PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE) as f64 / h;
    let scale = 1.0f64.min(s1).min(s2).min(s3);

    let max_side = (PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE) as u32;
    let new_w = ((w * scale) as u32).max(1).min(max_side);
    let new_h = ((h * scale) as u32).max(1).min(max_side);

    let factor = FACTOR as u32;
    let pad_w = (factor - new_w % factor) % factor;
    let pad_h = (factor - new_h % factor) % factor;

    let token_w = (new_w + pad_w) as usize / FACTOR;
    let token_h = (new_h + pad_h) as usize / FACTOR;
    let num_tokens = token_h * token_w;

    (new_w, new_h, pad_w, pad_h, num_tokens)
}

fn replace_first_occurrence(text: &str, to_replace: &str, replacement: &str) -> String {
    if let Some(pos) = text.find(to_replace) {
        let mut result = text.to_string();
        result.replace_range(pos..pos + to_replace.len(), replacement);
        result
    } else {
        text.to_string()
    }
}

impl InputsProcessor for KimiK25ImageProcessor {
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
    ) -> Result<InputProcessorOutput> {
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
                "KimiK25ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, all_grid_thws) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut grid_thws_accum: Vec<Vec<(usize, usize, usize)>> = Vec::new();

            for seq in input_seqs.iter_mut() {
                let (pixel_values, image_grid_thw) =
                    if let Some(cached_pixel_values) = &seq.multimodal.cached_pixel_values {
                        (
                            cached_pixel_values.clone(),
                            seq.multimodal
                                .cached_img_thw
                                .as_ref()
                                .map(|t| {
                                    // Convert cached Tensor back to Vec<(usize,usize,usize)>
                                    let thw: Vec<Vec<u32>> = t
                                        .to_vec2()
                                        .expect("cached_img_thw to_vec2 failed");
                                    thw.iter()
                                        .map(|row| {
                                            (
                                                row[0] as usize,
                                                row[1] as usize,
                                                row[2] as usize,
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .unwrap_or_default(),
                        )
                    } else {
                        let PreprocessedImages {
                            pixel_values,
                            image_grid_thw,
                            ..
                        } = self
                            .preprocess(
                                seq.clone_images()
                                    .expect("Need to have images by this point."),
                                vec![],
                                config,
                                device,
                                (usize::MAX, usize::MAX),
                            )
                            .expect("Preprocessing failed");

                        // Convert image_grid_thw Tensor to Vec<(usize,usize,usize)>
                        let grid_thws_vec = image_grid_thw
                            .as_ref()
                            .map(|t| {
                                let thw: Vec<Vec<u32>> =
                                    t.to_vec2().expect("image_grid_thw to_vec2 failed");
                                thw.iter()
                                    .map(|row| {
                                        (row[0] as usize, row[1] as usize, row[2] as usize)
                                    })
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();

                        seq.multimodal.cached_pixel_values = Some(pixel_values.clone());
                        seq.multimodal.cached_img_thw = image_grid_thw;
                        (pixel_values, grid_thws_vec)
                    };

                pixel_values_accum.push(pixel_values.unsqueeze(0).unwrap());

                // Expand media_pad tokens in prompt for this sequence
                if is_prompt && !seq.multimodal.has_changed_prompt {
                    let mut text = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");

                    for thw in &image_grid_thw {
                        let (t, h, w) = *thw;
                        // After merge: num_tokens = t * (h / merge_h) * (w / merge_w)
                        let num_tokens =
                            t * (h / MERGE_KERNEL_SIZE) * (w / MERGE_KERNEL_SIZE);
                        // Replace one <|media_pad|> with num_tokens copies
                        text = replace_first_occurrence(
                            &text,
                            KimiK25Processor::MEDIA_PAD,
                            &PLACEHOLDER.repeat(num_tokens),
                        );
                    }
                    // Replace placeholders back to media_pad tokens
                    text = text.replace(PLACEHOLDER, KimiK25Processor::MEDIA_PAD);

                    seq.set_initial_prompt(text.clone());
                    let toks = tokenizer
                        .encode_fast(text, false)
                        .expect("Tokenization failed!");
                    let ids = toks.get_ids().to_vec();
                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                }

                grid_thws_accum.push(image_grid_thw);
            }

            let all_grid_thws: Vec<(usize, usize, usize)> =
                grid_thws_accum.into_iter().flatten().collect();

            (
                Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                all_grid_thws,
            )
        } else {
            (None, Vec::new())
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

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args: Box::new(KimiK25SpecificArgs {
                grid_thws: all_grid_thws,
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

impl ImagePreProcessor for KimiK25ImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = DEFAULT_IMAGE_MEAN;
    const DEFAULT_STD: [f64; 3] = DEFAULT_IMAGE_STD;

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        assert!(videos.is_empty(), "Kimi-K2.5 video input is not yet supported");

        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);

        let mut all_patches = Vec::new();
        let mut all_grid_thw = Vec::new();

        for image in images {
            let (orig_w, orig_h) = image.dimensions();
            let (new_w, new_h, pad_w, pad_h, _num_tokens) =
                navit_resize_image(orig_w, orig_h);

            let padded_w = (new_w + pad_w) as usize;
            let padded_h = (new_h + pad_h) as usize;

            // Resize image
            let resized =
                image.resize_exact(new_w, new_h, FilterType::CatmullRom);
            let resized = DynamicImage::ImageRgb8(resized.to_rgb8());

            // Convert to tensor and normalize: ToTensor gives [0,1], then normalize
            let to_tensor = Transforms {
                input: &ToTensor,
                inner_transforms: &[],
            };
            let mut img_tensor = resized.apply(to_tensor, device)?;
            // img_tensor shape: (3, new_h, new_w)

            // Pad with zeros to (3, padded_h, padded_w)
            if pad_h > 0 || pad_w > 0 {
                img_tensor = img_tensor.pad_with_zeros(1, 0, pad_h as usize)?;
                img_tensor = img_tensor.pad_with_zeros(2, 0, pad_w as usize)?;
            }

            // Normalize: (x - mean) / std
            let normalize_transforms = TensorTransforms {
                inner_transforms: &[&Normalize {
                    mean: image_mean.to_vec(),
                    std: image_std.to_vec(),
                }],
            };
            let img_tensor =
                <Tensor as ApplyTensorTransforms>::apply(&img_tensor, normalize_transforms, device)?;
            // img_tensor shape: (3, padded_h, padded_w)

            // Patchify: (3, padded_h, padded_w) -> (num_patches, 3, patch_size, patch_size)
            let grid_h = padded_h / PATCH_SIZE;
            let grid_w = padded_w / PATCH_SIZE;
            let c = 3usize;

            let patches = img_tensor
                .reshape((c, grid_h, PATCH_SIZE, grid_w, PATCH_SIZE))?
                .permute((1, 3, 0, 2, 4))? // (grid_h, grid_w, C, ps, ps)
                .reshape((grid_h * grid_w, c, PATCH_SIZE, PATCH_SIZE))?
                .contiguous()?;

            all_patches.push(patches);
            // T=1 for images
            all_grid_thw.push(Tensor::new(&[1u32, grid_h as u32, grid_w as u32], &Device::Cpu)?);
        }

        let pixel_values = Tensor::cat(&all_patches, 0)?;
        let image_grid_thw = Tensor::stack(&all_grid_thw, 0)?;

        Ok(PreprocessedImages {
            pixel_values,
            pixel_attention_mask: None,
            image_sizes: None,
            num_img_tokens: None,
            aspect_ratio_ids: None,
            aspect_ratio_mask: None,
            num_tiles: None,
            image_grid_thw: Some(image_grid_thw),
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
