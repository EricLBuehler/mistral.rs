#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::{DynamicImage, GenericImageView};
use mistralrs_vision::{ApplyTransforms, Rescale, ToTensorNoNorm, Transforms};
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
    vision_models::gemma4::audio_processing::AudioProcessor,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Gemma4SpecificArgs;

// ── Token constants ────────────────────────────────────────────────────────

const IMAGE_TOKEN: &str = "<|image|>";
const BOI_TOKEN: &str = "<|image>";
const EOI_TOKEN: &str = "<image|>";
pub const IMAGE_TOKEN_ID: u32 = 258880;

const AUDIO_TOKEN: &str = "<|audio|>";
const BOA_TOKEN: &str = "<|audio>";
const EOA_TOKEN: &str = "<audio|>";
pub const AUDIO_TOKEN_ID: u32 = 258881;

// ── Processor (public, created by the pipeline loader) ─────────────────────

pub struct Gemma4Processor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    audio_seq_length: usize,
    supports_images: bool,
    supports_audio: bool,
}

impl Gemma4Processor {
    pub fn new(
        processor_config: ProcessorConfig,
        patch_size: usize,
        pooling_kernel_size: usize,
        default_output_length: usize,
        supports_images: bool,
        supports_audio: bool,
    ) -> Self {
        let max_patches = default_output_length * pooling_kernel_size * pooling_kernel_size;
        let audio_seq_length = processor_config.audio_seq_length.unwrap_or(188);

        Self {
            patch_size,
            pooling_kernel_size,
            default_output_length,
            max_patches,
            audio_seq_length,
            supports_images,
            supports_audio,
        }
    }
}

impl Processor for Gemma4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Gemma4ImageProcessor {
            patch_size: self.patch_size,
            pooling_kernel_size: self.pooling_kernel_size,
            default_output_length: self.default_output_length,
            max_patches: self.max_patches,
            audio_seq_length: self.audio_seq_length,
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            BOI_TOKEN,
            EOI_TOKEN,
            AUDIO_TOKEN,
            BOA_TOKEN,
            EOA_TOKEN,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

// ── Image processor (InputsProcessor + ImagePreProcessor) ──────────────────

#[allow(dead_code)]
struct Gemma4ImageProcessor {
    patch_size: usize,
    pooling_kernel_size: usize,
    default_output_length: usize,
    max_patches: usize,
    audio_seq_length: usize,
    supports_images: bool,
    supports_audio: bool,
}

impl Gemma4ImageProcessor {
    /// Compute how many vision soft tokens a single image will produce after
    /// aspect-ratio-preserving resize + patch embedding + pooling.
    fn output_tokens_for_size(&self, new_h: usize, new_w: usize) -> usize {
        let ph = new_h / self.patch_size;
        let pw = new_w / self.patch_size;
        let pool_area = self.pooling_kernel_size * self.pooling_kernel_size;
        (ph * pw) / pool_area
    }

    /// Aspect-ratio-preserving resize: compute (new_h, new_w) for a given
    /// original image size, ensuring that the result is a multiple of
    /// `grid_unit = pooling_kernel_size * patch_size` and does not exceed
    /// `max_patches` patches.
    fn compute_resize_dims(&self, orig_h: usize, orig_w: usize) -> (usize, usize) {
        let target_px = self.max_patches * self.patch_size * self.patch_size;
        let grid_unit = self.pooling_kernel_size * self.patch_size; // 48

        let factor = (target_px as f64 / (orig_h as f64 * orig_w as f64)).sqrt();

        let new_h = ((orig_h as f64 * factor) as usize / grid_unit).max(1) * grid_unit;
        let new_w = ((orig_w as f64 * factor) as usize / grid_unit).max(1) * grid_unit;

        (new_h, new_w)
    }

    /// Build the expanded token sequence for a single image:
    /// `<start_of_image>{N * <image_soft_token>}<end_of_image>`
    fn build_image_sequence(&self, num_tokens: usize) -> String {
        let image_tokens = vec![IMAGE_TOKEN.to_string(); num_tokens].join("");
        format!("{BOI_TOKEN}{image_tokens}{EOI_TOKEN}")
    }

    fn compute_audio_num_tokens(&self, num_mel_frames: usize) -> usize {
        if num_mel_frames == 0 {
            return 0;
        }

        let mut t = num_mel_frames;
        for _ in 0..2 {
            t = (t + 2 - 3) / 2 + 1;
        }
        t.min(self.audio_seq_length)
    }

    /// Build the expanded token sequence for audio:
    /// `<start_of_audio>{N * <audio_soft_token>}<end_of_audio>`
    fn build_audio_sequence(&self, num_tokens: usize) -> String {
        let audio_tokens = vec![AUDIO_TOKEN.to_string(); num_tokens].join("");
        format!("{BOA_TOKEN}{audio_tokens}{EOA_TOKEN}")
    }
}

// ── InputsProcessor ────────────────────────────────────────────────────────

impl InputsProcessor for Gemma4ImageProcessor {
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
        sliding_window: Option<usize>,
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
                "Gemma4ImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let preprocessor_config: &PreProcessorConfig =
            config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let has_audios = input_seqs.iter().any(|seq| seq.has_audios());

        let mut has_changed_prompt = false;

        // ── Audio processing ───────────────────────────────────────────────
        if has_audios && !self.supports_audio {
            return Err(anyhow::Error::msg(
                "This image processor does not support audio.",
            ));
        }

        let (audio_mel, audio_mel_mask) = if has_audios {
            let mut audio_mel_accum = Vec::new();
            let mut audio_mask_accum = Vec::new();
            let audio_processor = AudioProcessor::new(preprocessor_config);

            for seq in input_seqs.iter_mut() {
                if let Some(mut audios) = seq.take_audios() {
                    let max_audio_len = audios
                        .iter()
                        .map(|x| x.samples.len())
                        .max()
                        .expect("No audios");
                    for audio in &mut audios {
                        let pad_len = max_audio_len - audio.samples.len();
                        audio.samples.extend(std::iter::repeat_n(0., pad_len));
                    }

                    let mut seq_audio_mel = Vec::new();
                    let mut seq_audio_mask = Vec::new();
                    let mut seq_audio_num_tokens = Vec::new();
                    for audio in audios {
                        let (mel, mask) = audio_processor
                            .process_audio(&audio, device)
                            .expect("Audio processing failed");

                        seq_audio_num_tokens.push(self.compute_audio_num_tokens(mel.dim(1)?));
                        seq_audio_mel.push(mel);
                        seq_audio_mask.push(mask);
                    }

                    if !seq.multimodal.has_changed_prompt {
                        let mut prompt = tokenizer
                            .decode(seq.get_toks(), false)
                            .expect("Detokenization failed!");

                        let positions: Vec<usize> = prompt
                            .match_indices(AUDIO_TOKEN)
                            .map(|(idx, _)| idx)
                            .collect();

                        for (i, &pos) in positions.iter().enumerate().rev() {
                            let num_tokens = seq_audio_num_tokens
                                .get(i)
                                .copied()
                                .unwrap_or(self.audio_seq_length);
                            let replacement = self.build_audio_sequence(num_tokens);

                            prompt = format!(
                                "{}{}{}",
                                &prompt[..pos],
                                replacement,
                                &prompt[pos + AUDIO_TOKEN.len()..],
                            );
                        }

                        seq.set_initial_prompt(prompt.clone());
                        let toks = tokenizer
                            .encode_fast(prompt.as_str(), false)
                            .expect("Tokenization failed!");

                        let ids = toks.get_ids().to_vec();
                        seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());

                        has_changed_prompt = true;
                    }

                    let cached_audio = seq.count_prefix_cached_mm_items_by_kind("audio");
                    let n_audio = seq_audio_mel.len();
                    if cached_audio < n_audio {
                        audio_mel_accum.extend(seq_audio_mel.into_iter().skip(cached_audio));
                        audio_mask_accum.extend(seq_audio_mask.into_iter().skip(cached_audio));
                    }
                }
            }

            if !audio_mel_accum.is_empty() {
                match (
                    Tensor::cat(&audio_mel_accum, 0),
                    Tensor::cat(&audio_mask_accum, 0),
                ) {
                    (Ok(mel), Ok(mask)) => (Some(mel), Some(mask)),
                    (Err(e), _) | (_, Err(e)) => {
                        return Err(anyhow::Error::from(e));
                    }
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        // ── Image processing ───────────────────────────────────────────────
        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();

            for seq in input_seqs.iter_mut() {
                let images = seq
                    .take_images()
                    .expect("Need to have images by this point.");

                // Compute per-image resize dimensions *before* preprocessing so
                // we can build the correct per-image token sequences.
                let per_image_dims: Vec<(usize, usize)> = images
                    .iter()
                    .map(|img| {
                        let (w, h) = img.dimensions();
                        self.compute_resize_dims(h as usize, w as usize)
                    })
                    .collect();

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
                        images,
                        vec![],
                        preprocessor_config,
                        device,
                        (usize::MAX, usize::MAX),
                    )
                    .expect("Preprocessing failed");

                // Replace each <start_of_image> placeholder with the correct
                // per-image expanded token sequence.
                if !seq.multimodal.has_changed_prompt {
                    let mut prompt = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");

                    // Replace occurrences of the image placeholder token
                    // (<|image|>) in reverse order so string offsets stay valid.
                    // The chat template emits a single <|image|> per image;
                    // we expand it to <|image>{N × <|image|>}<image|>.
                    let positions: Vec<usize> = prompt
                        .match_indices(IMAGE_TOKEN)
                        .map(|(idx, _)| idx)
                        .collect();

                    for (i, &pos) in positions.iter().enumerate().rev() {
                        let (new_h, new_w) = if i < per_image_dims.len() {
                            per_image_dims[i]
                        } else {
                            let grid_unit = self.pooling_kernel_size * self.patch_size;
                            (grid_unit, grid_unit)
                        };
                        let num_tokens = self.output_tokens_for_size(new_h, new_w);
                        let replacement = self.build_image_sequence(num_tokens);

                        prompt = format!(
                            "{}{}{}",
                            &prompt[..pos],
                            replacement,
                            &prompt[pos + IMAGE_TOKEN.len()..],
                        );
                    }

                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt.as_str(), false)
                        .expect("Tokenization failed!");

                    let ids = toks.get_ids().to_vec();
                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());

                    has_changed_prompt = true;
                }

                // Per-sequence prefix cache trimming of pixel_values
                let cached = seq.count_prefix_cached_mm_items_by_kind("img");
                let n_images = pixel_values.dim(0).unwrap_or(0);
                if cached < n_images {
                    let image_sizes = image_sizes_all.unwrap_or_default();
                    if cached > 0 {
                        pixel_values_accum
                            .push(pixel_values.narrow(0, cached, n_images - cached).unwrap());
                        image_sizes_accum.extend(image_sizes[cached..].iter().copied());
                    } else {
                        pixel_values_accum.push(pixel_values.clone());
                        image_sizes_accum.extend(image_sizes);
                    }
                }
            }

            if pixel_values_accum.is_empty() {
                (None, vec![])
            } else {
                (
                    Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                    image_sizes_accum,
                )
            }
        } else {
            (None, vec![])
        };

        for seq in input_seqs.iter_mut() {
            if seq.mm_features().is_empty() {
                let mut features = Vec::new();
                if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                    if !hashes.is_empty() {
                        let ranges = find_image_placeholder_ranges(seq.get_toks(), IMAGE_TOKEN_ID);
                        features.extend(build_mm_features_from_ranges(&ranges, &hashes, "img"));
                    }
                }
                if let Some(audio_hashes) = seq.audio_hashes().map(|h| h.to_vec()) {
                    if !audio_hashes.is_empty() {
                        let audio_ranges =
                            find_image_placeholder_ranges(seq.get_toks(), AUDIO_TOKEN_ID);
                        features.extend(build_mm_features_from_ranges(
                            &audio_ranges,
                            &audio_hashes,
                            "audio",
                        ));
                    }
                }
                if !features.is_empty() {
                    features.sort_by_key(|f| f.offset);
                    seq.set_mm_features(features);
                }
            }
            seq.multimodal.has_changed_prompt = has_changed_prompt;
        }

        // ── Build final model inputs ───────────────────────────────────────
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
                sliding_window,
            )
            .unwrap()
        };

        let (pixel_values, image_sizes) = if is_prompt {
            pixel_values
        } else {
            (None, vec![])
        };

        let image_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .flat_map(|seq| {
                    seq.image_hashes()
                        .map(|h| {
                            let cached = seq.count_prefix_cached_mm_items_by_kind("img");
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

        let audio_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .flat_map(|seq| {
                    seq.audio_hashes()
                        .map(|h| {
                            let cached = seq.count_prefix_cached_mm_items_by_kind("audio");
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
            model_specific_args: Box::new(Gemma4SpecificArgs {
                audio_mel,
                audio_mel_mask,
                image_hashes,
                image_sizes,
                audio_hashes,
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

// ── ImagePreProcessor ──────────────────────────────────────────────────────

impl ImagePreProcessor for Gemma4ImageProcessor {
    // Gemma4 rescales to [0, 1] but does NOT apply ImageNet normalization.
    const DEFAULT_MEAN: [f64; 3] = [0.0, 0.0, 0.0];
    const DEFAULT_STD: [f64; 3] = [1.0, 1.0, 1.0];

    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        assert!(videos.is_empty());

        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);
        let resample = config.resampling.to_filter()?;

        for image in images.iter_mut() {
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

        let mut pixel_values = Vec::new();
        let mut image_sizes = Vec::new();

        for image in images {
            let (w, h) = image.dimensions();
            let (new_h, new_w) = self.compute_resize_dims(h as usize, w as usize);

            // resize_exact takes (width, height, filter)
            let resized = image.resize_exact(new_w as u32, new_h as u32, resample);

            let transforms = Transforms {
                input: &ToTensorNoNorm,
                inner_transforms: &[&do_rescale.then_some(Rescale {
                    factor: Some(rescale_factor),
                })],
            };

            let tensor = resized.apply(transforms, device)?;
            pixel_values.push(tensor.unsqueeze(0)?);
            image_sizes.push((new_h as u32, new_w as u32));
        }

        // All images may have different spatial dimensions.  We still need to
        // return a single `pixel_values` tensor.  When sizes differ we pad each
        // image tensor to the batch-maximum height/width so they can be
        // concatenated along dim-0.
        let max_h = image_sizes.iter().map(|(h, _)| *h).max().unwrap_or(0) as usize;
        let max_w = image_sizes.iter().map(|(_, w)| *w).max().unwrap_or(0) as usize;

        let mut padded = Vec::new();
        for (pv, &(h, w)) in pixel_values.iter().zip(image_sizes.iter()) {
            let h = h as usize;
            let w = w as usize;
            if h < max_h || w < max_w {
                // pv shape: [1, 3, h, w] -> pad height and width
                let pad_h = max_h - h;
                let pad_w = max_w - w;
                let p = pv
                    .pad_with_zeros(2, 0, pad_h)?
                    .pad_with_zeros(3, 0, pad_w)?;
                padded.push(p);
            } else {
                padded.push(pv.clone());
            }
        }

        Ok(PreprocessedImages {
            pixel_values: Tensor::cat(&padded, 0)?,
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
