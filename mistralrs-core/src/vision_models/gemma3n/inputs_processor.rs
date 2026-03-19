#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Result, Tensor};
use image::DynamicImage;
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
    vision_models::gemma3n::audio_processing::AudioProcessor,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::{PreProcessorConfig, ToFilter},
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Gemma3nSpecificArgs;

struct Gemma3nImageProcessor {
    supports_images: bool,
    supports_audio: bool,
    full_image_sequence: String,
    audio_seq_length: usize,
}

impl Gemma3nImageProcessor {
    fn create_full_audio_sequence(&self) -> String {
        let audio_tokens_expanded = vec![AUDIO_TOKEN.to_string(); self.audio_seq_length].join("");
        format!("\n\n{BOA_TOKEN}{audio_tokens_expanded}{EOA_TOKEN}\n\n")
    }
}

const IMAGE_TOKEN: &str = "<image_soft_token>";
const BOI_TOKEN: &str = "<start_of_image>";
const EOI_TOKEN: &str = "<end_of_image>";
pub const IMAGE_TOKEN_ID: u32 = 262145;

const AUDIO_TOKEN: &str = "<audio_soft_token>";
const BOA_TOKEN: &str = "<start_of_audio>";
const EOA_TOKEN: &str = "<end_of_audio>";
pub const AUDIO_TOKEN_ID: u32 = 262273; // audio_vocab_offset + 1

pub struct Gemma3nProcessor {
    vision_soft_tokens_per_image: usize,
    audio_seq_length: usize,
    supports_images: bool,
    supports_audio: bool,
}

impl Gemma3nProcessor {
    pub fn new(processor_config: ProcessorConfig, supports_images: bool) -> Self {
        // Default to 256 soft tokens per image if not specified
        let vision_soft_tokens_per_image = processor_config.image_seq_len.unwrap_or(256);
        // Default to 188 audio tokens as per transformers implementation
        let audio_seq_length = processor_config.audio_seq_length.unwrap_or(188);

        Self {
            vision_soft_tokens_per_image,
            audio_seq_length,
            supports_images,
            supports_audio: true, // Enable audio support
        }
    }

    fn create_full_image_sequence(&self) -> String {
        // Create the full image token sequence: "\n\n<boi>{repeated image tokens}<eoi>\n\n"
        let image_tokens_expanded =
            vec![IMAGE_TOKEN.to_string(); self.vision_soft_tokens_per_image].join("");
        format!("\n\n{BOI_TOKEN}{image_tokens_expanded}{EOI_TOKEN}\n\n")
    }
}

impl Processor for Gemma3nProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Gemma3nImageProcessor {
            supports_images: self.supports_images,
            supports_audio: self.supports_audio,
            full_image_sequence: self.create_full_image_sequence(),
            audio_seq_length: self.audio_seq_length,
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

impl InputsProcessor for Gemma3nImageProcessor {
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
                "Gemma3nImageProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let preprocessor_config: &PreProcessorConfig =
            config.downcast_ref().expect("Downcast failed.");

        let has_images = input_seqs.iter().any(|seq| seq.has_images());
        let has_audios = input_seqs.iter().any(|seq| seq.has_audios());

        let mut has_changed_prompt = false;

        // Process audio if present
        let (audio_mel, audio_mel_mask) = if has_audios && self.supports_audio {
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

                    for audio in audios {
                        let (mel, mask) = audio_processor
                            .process_audio(&audio, device)
                            .expect("Audio processing failed");

                        audio_mel_accum.push(mel);
                        audio_mask_accum.push(mask);
                    }

                    // Update prompt with audio tokens
                    if !seq.multimodal.has_changed_prompt {
                        let mut prompt = tokenizer
                            .decode(seq.get_toks(), false)
                            .expect("Detokenization failed!");
                        let audio_sequence = self.create_full_audio_sequence();

                        // Replace audio placeholder with tokens
                        prompt = prompt.replace(AUDIO_TOKEN, &audio_sequence);

                        // Re-tokenize
                        seq.set_initial_prompt(prompt.clone());
                        let toks = tokenizer
                            .encode_fast(prompt, false)
                            .expect("Tokenization failed!");

                        let ids = toks.get_ids().to_vec();
                        seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());

                        has_changed_prompt = true;
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

        let pixel_values = if has_images {
            if !self.supports_images {
                return Err(anyhow::Error::msg(
                    "This image processor does not support images.",
                ));
            }

            let mut pixel_values_accum = Vec::new();

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
                    num_crops: _,
                } = self
                    .preprocess(
                        seq.take_images()
                            .expect("Need to have images by this point."),
                        vec![],
                        preprocessor_config,
                        device,
                        (usize::MAX, usize::MAX),
                    )
                    .expect("Preprocessing failed");

                // Replace <image> placeholders with full image sequence
                if !seq.multimodal.has_changed_prompt {
                    let mut prompt = tokenizer
                        .decode(seq.get_toks(), false)
                        .expect("Detokenization failed!");

                    // Replace each <image> token with the full image sequence
                    prompt = prompt.replace(IMAGE_TOKEN, &self.full_image_sequence);

                    // Re-tokenize the modified prompt
                    seq.set_initial_prompt(prompt.clone());
                    let toks = tokenizer
                        .encode_fast(prompt, false)
                        .expect("Tokenization failed!");

                    let ids = toks.get_ids().to_vec();

                    // Build mm_features for position-aware prefix cache hashing
                    if seq.mm_features().is_empty() {
                        if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                            let ranges = find_image_placeholder_ranges(&ids, IMAGE_TOKEN_ID);
                            seq.set_mm_features(build_mm_features_from_ranges(
                                &ranges, &hashes, "img",
                            ));
                        }
                    }
                    // Also include audio features in mm_features for prefix cache hashing
                    if let Some(audio_hashes) = seq.audio_hashes().map(|h| h.to_vec()) {
                        if !audio_hashes.is_empty() {
                            let audio_ranges = find_image_placeholder_ranges(&ids, AUDIO_TOKEN_ID);
                            let audio_features = build_mm_features_from_ranges(
                                &audio_ranges,
                                &audio_hashes,
                                "audio",
                            );
                            let mut features = seq.mm_features().to_vec();
                            features.extend(audio_features);
                            seq.set_mm_features(features);
                        }
                    }

                    seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_mut());

                    has_changed_prompt = true;
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

        for seq in input_seqs.iter_mut() {
            seq.multimodal.has_changed_prompt = has_changed_prompt;
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

        let audio_hashes: Vec<u64> = if is_prompt {
            input_seqs
                .iter()
                .flat_map(|seq| seq.audio_hashes().map(|h| h.to_vec()).unwrap_or_default())
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
            model_specific_args: Box::new(Gemma3nSpecificArgs {
                audio_mel,
                audio_mel_mask,
                image_hashes,
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

impl ImagePreProcessor for Gemma3nImageProcessor {
    // Siglip uses these defaults for normalization
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

        // Get preprocessing parameters with defaults
        let do_resize = config.do_resize.unwrap_or(true);
        let size = config.size.as_ref().unwrap();
        let (height, width) = if let (Some(h), Some(w)) = (size.get("height"), size.get("width")) {
            (*h, *w)
        } else {
            // Default to 768x768 for Gemma3n (based on test files)
            (768, 768)
        };
        let resample = config.resampling.to_filter()?;
        let do_rescale = config.do_rescale.unwrap_or(true);
        let rescale_factor = config.rescale_factor.unwrap_or(1.0 / 255.0);
        let do_normalize = config.do_normalize.unwrap_or(true);
        let image_mean = config.image_mean.unwrap_or(Self::DEFAULT_MEAN);
        let image_std = config.image_std.unwrap_or(Self::DEFAULT_STD);
        let do_convert_rgb = config.do_convert_rgb.unwrap_or(true);

        for image in images.iter_mut() {
            // Convert to rgb
            if do_convert_rgb {
                *image = DynamicImage::ImageRgb8(image.to_rgb8());
            }
        }

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
            num_crops: None,
        })
    }
}
