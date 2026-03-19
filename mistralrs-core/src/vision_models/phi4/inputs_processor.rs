#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::{any::Any, collections::HashSet, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex::Regex;
use tokenizers::Tokenizer;

use apodize::hanning_iter;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
        ProcessorCreator,
    },
    sequence::{build_mm_features_from_ranges, find_image_placeholder_ranges, Sequence},
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    phi4::Phi4MMVisionSpecificArgs,
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
    ModelInputs,
};

use super::audio_embedding::AUDIO_SPECIAL_TOKEN_ID;
use super::image_embedding::IMAGE_SPECIAL_TOKEN_ID;

const COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN: &str = r"<\|image_\d+\|>";
const COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN: &str = r"<\|audio_\d+\|>";
const IMAGE_SPECIAL_TOKEN: &str = "<|endoftext10|>";
const AUDIO_SPECIAL_TOKEN: &str = "<|endoftext11|>";
pub(crate) const DYHD_BASE_RESOLUTION: usize = 448;

const AUDIO_FEATURE_SIZE: usize = 80; // mel bins

type AudioProcessingResult = Result<(Option<Tensor>, Option<Vec<usize>>, Option<Tensor>)>;

// Input processor
pub struct Phi4MMInputsProcessor {
    audio_compression_rate: usize,
    audio_downsample_rate: usize,
    audio_feat_stride: usize,
    eightk_method: String, // "fillzero" or "resample"
}

// Processor
pub struct Phi4MMProcessor {
    inputs_processor: Arc<Phi4MMInputsProcessor>,
}

impl ProcessorCreator for Phi4MMProcessor {
    fn new_processor(
        _: Option<ProcessorConfig>,
        pre_processor_config: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Self {
            inputs_processor: Arc::new(Phi4MMInputsProcessor {
                audio_compression_rate: pre_processor_config
                    .audio_compression_rate
                    .expect("audio_compression_rate"),
                audio_downsample_rate: pre_processor_config
                    .audio_downsample_rate
                    .expect("audio_downsample_rate"),
                audio_feat_stride: pre_processor_config
                    .audio_feat_stride
                    .expect("audio_feat_stride"),
                eightk_method: "fillzero".to_string(), // Default to fillzero
            }),
        })
    }
}

impl Processor for Phi4MMProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        self.inputs_processor.clone()
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

impl InputsProcessor for Phi4MMInputsProcessor {
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
                "Phi4MMInputProcessor requires a specified tokenizer.",
            ));
        };

        let config = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");

        let has_audios = input_seqs.iter().all(|seq| seq.has_audios());
        let has_images = input_seqs.iter().all(|seq| seq.has_images());

        let (pixel_values, pixel_attention_mask, image_sizes, num_img_tokens) = if has_images {
            let mut pixel_values_accum = Vec::new();
            let mut pixel_attention_masks_accum = Vec::new();
            let mut image_sizes_accum = Vec::new();
            let mut num_img_tokens_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                let cached = seq.count_prefix_cached_mm_items();
                let imgs = seq
                    .take_images()
                    .expect("Need to have images by this point.");
                let PreprocessedImages {
                    pixel_values,
                    pixel_attention_mask,
                    image_sizes: _,
                    num_img_tokens,
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
                        imgs,
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX), // Don't use it here...
                    )
                    .expect("Preprocessor failed");
                let image_sizes = image_sizes_all.unwrap();
                let pixel_attention_mask = pixel_attention_mask.unwrap();
                // Trim cached images per-sequence before pushing.
                let n_images = pixel_values.dim(0).unwrap_or(0);
                if cached < n_images {
                    if cached > 0 {
                        pixel_values_accum
                            .push(pixel_values.narrow(0, cached, n_images - cached).unwrap());
                        pixel_attention_masks_accum.push(
                            pixel_attention_mask
                                .narrow(0, cached, n_images - cached)
                                .unwrap(),
                        );
                    } else {
                        pixel_values_accum.push(pixel_values);
                        pixel_attention_masks_accum.push(pixel_attention_mask);
                    }
                    // Using extend on purpose
                    image_sizes_accum.extend(image_sizes[cached..].to_vec());
                }
                num_img_tokens_accum.push(num_img_tokens.unwrap());
            }
            if !pixel_values_accum.is_empty() {
                (
                    Some(Tensor::cat(&pixel_values_accum, 0).unwrap()),
                    Some(Tensor::cat(&pixel_attention_masks_accum, 0).unwrap()),
                    Some(image_sizes_accum),
                    Some(num_img_tokens_accum),
                )
            } else {
                (None, None, None, Some(num_img_tokens_accum))
            }
        } else if has_audios {
            (None, None, None, Some(vec![vec![]; input_seqs.len()]))
        } else {
            return text_models_inputs_processor::TextInputsProcessor
                .process_inputs(
                    Some(tokenizer),
                    input_seqs,
                    is_prompt,
                    is_xlora,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    other_config,
                    paged_attn_metadata,
                    mapper,
                )
                .map(|metadata| {
                    let InputProcessorOutput {
                        inputs,
                        seq_indices,
                    } = metadata;

                    let text_models_inputs_processor::ModelInputs {
                        input_ids,
                        input_ids_full: _,
                        seqlen_offsets,
                        seqlen_offsets_full: _,
                        context_lens,
                        position_ids,
                        paged_attn_meta,
                        flash_meta,
                        flash_meta_full: _,
                    } = *inputs
                        .downcast::<text_models_inputs_processor::ModelInputs>()
                        .expect("Downcast failed.");

                    let inputs: Box<dyn Any> = Box::new(ModelInputs {
                        input_ids,
                        seqlen_offsets,
                        context_lens,
                        position_ids,
                        pixel_values: None,
                        model_specific_args: Box::new(Phi4MMVisionSpecificArgs {
                            input_image_embeds: None,
                            image_attention_mask: None,
                            image_sizes: None,
                            input_audio_embeds: None,
                            audio_embed_sizes: None,
                            audio_attention_mask: None,
                            image_hashes: vec![],
                        }),
                        paged_attn_meta,
                        flash_meta,
                    });
                    InputProcessorOutput {
                        inputs,
                        seq_indices,
                    }
                });
        };

        let detokenized = tokenizer
            .decode_batch(
                &input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
                false,
            )
            .expect("Decode failed");

        let img_token_pattern = Regex::new(COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN).unwrap();
        let audio_token_pattern = Regex::new(COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN).unwrap();

        for (mut detokenized, seq) in detokenized.into_iter().zip(input_seqs.iter_mut()) {
            detokenized = img_token_pattern
                .replace_all(&detokenized, IMAGE_SPECIAL_TOKEN)
                .to_string();
            detokenized = audio_token_pattern
                .replace_all(&detokenized, AUDIO_SPECIAL_TOKEN)
                .to_string();

            let has_changed_prompt = seq.multimodal.has_changed_prompt;
            if !has_changed_prompt {
                seq.set_toks_and_reallocate(
                    tokenizer
                        .encode_fast(detokenized.clone(), false)
                        .expect("Encode failed")
                        .get_ids()
                        .to_vec(),
                    paged_attn_metadata.as_mut(),
                );

                seq.set_initial_prompt(detokenized);
            }
        }

        let (input_audio_embeds, audio_embed_sizes, audio_attention_mask) =
            match self.process_audio_for_sequences(input_seqs, device) {
                Ok(result) => result,
                Err(e) => return Err(anyhow::Error::new(e)),
            };

        let mut toks = Vec::new();

        for (seq, num_img_tokens) in input_seqs.iter_mut().zip(num_img_tokens.unwrap()) {
            let has_changed_prompt = seq.multimodal.has_changed_prompt;

            let mut i = 0;
            let mut image_token_count_iter = num_img_tokens.iter();
            let audio_sizes_tmp = audio_embed_sizes.clone().unwrap_or(vec![]);
            let mut audio_embed_sizes = audio_sizes_tmp.iter();
            while i < seq.get_toks().len() {
                let token_id = seq.get_toks()[i];
                let token_count = if token_id == IMAGE_SPECIAL_TOKEN_ID as u32 {
                    image_token_count_iter.next().unwrap()
                } else if token_id == AUDIO_SPECIAL_TOKEN_ID as u32 {
                    audio_embed_sizes.next().unwrap()
                } else {
                    i += 1;
                    continue;
                };

                let mut new_ids = seq.get_toks()[..i].to_vec();
                new_ids.extend(vec![token_id; *token_count]);
                new_ids.extend(seq.get_toks()[i + 1..].to_vec());
                if !has_changed_prompt {
                    seq.set_toks_and_reallocate(new_ids, paged_attn_metadata.as_mut());
                }
                i += token_count;
            }
            if !has_changed_prompt {
                // Build mm_features for position-aware prefix cache hashing
                if seq.mm_features().is_empty() {
                    if let Some(hashes) = seq.image_hashes().map(|h| h.to_vec()) {
                        let ranges = find_image_placeholder_ranges(
                            seq.get_toks(),
                            IMAGE_SPECIAL_TOKEN_ID as u32,
                        );
                        seq.set_mm_features(build_mm_features_from_ranges(&ranges, &hashes, "img"));
                    }
                }
                // Also include audio features in mm_features for prefix cache hashing
                if let Some(audio_hashes) = seq.audio_hashes().map(|h| h.to_vec()) {
                    if !audio_hashes.is_empty() {
                        let audio_ranges = find_image_placeholder_ranges(
                            seq.get_toks(),
                            AUDIO_SPECIAL_TOKEN_ID as u32,
                        );
                        let audio_features =
                            build_mm_features_from_ranges(&audio_ranges, &audio_hashes, "audio");
                        let mut features = seq.mm_features().to_vec();
                        features.extend(audio_features);
                        seq.set_mm_features(features);
                    }
                }
                seq.multimodal.has_changed_prompt = true;
            }
            toks.push(seq.get_toks().to_vec());
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

        let result = if is_prompt {
            get_prompt_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
        } else {
            get_completion_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
            )
        };

        result.map(move |metadata| {
            let pixel_values = pixel_values.clone();
            let pixel_attention_mask = pixel_attention_mask.clone();
            let image_sizes = image_sizes.clone();

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
            } = metadata;
            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: pixel_values.clone(),
                model_specific_args: Box::new(Phi4MMVisionSpecificArgs {
                    input_image_embeds: pixel_values,
                    image_attention_mask: pixel_attention_mask,
                    image_sizes,
                    input_audio_embeds: input_audio_embeds.clone(),
                    audio_embed_sizes: audio_embed_sizes.clone(),
                    audio_attention_mask: audio_attention_mask.clone(),
                    image_hashes,
                }),
                paged_attn_meta,
                flash_meta,
            });
            InputProcessorOutput {
                inputs,
                seq_indices,
            }
        })
    }
}

impl Phi4MMInputsProcessor {
    fn extract_audio_features(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<Vec<f32>>> {
        // Resample audio to supported rates using rubato
        let (resampled_audio, final_sample_rate) =
            self.resample_audio_with_rubato(audio_data, sample_rate)?;

        // Extract mel spectrogram using rustfft and custom mel filterbank
        let mel_features =
            self.extract_mel_spectrogram_rustfft(&resampled_audio, final_sample_rate)?;

        Ok(mel_features)
    }

    fn resample_audio_with_rubato(&self, wav: &[f32], fs: u32) -> Result<(Vec<f32>, u32)> {
        let target_fs = if fs > 16000 {
            16000
        } else if fs > 8000 && fs < 16000 {
            8000
        } else if fs < 8000 {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported sample rate: {fs}"
            )));
        } else {
            return Ok((wav.to_vec(), fs)); // No resampling needed
        };

        if fs == target_fs {
            return Ok((wav.to_vec(), fs));
        }

        // Handle 8kHz upsampling case
        if fs == 8000 && self.eightk_method == "resample" {
            // Upsample to 16kHz using rubato
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };

            let mut resampler = SincFixedIn::<f32>::new(
                2.0, // resample ratio (16000/8000)
                2.0,
                params,
                wav.len(),
                1, // mono
            )
            .map_err(|e| candle_core::Error::Msg(format!("Resampler creation failed: {e}")))?;

            let input = vec![wav.to_vec()];
            let output = resampler
                .process(&input, None)
                .map_err(|e| candle_core::Error::Msg(format!("Resampling failed: {e}")))?;

            return Ok((output[0].clone(), 16000));
        }

        // Regular downsampling
        let resample_ratio = target_fs as f64 / fs as f64;

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let mut resampler = SincFixedIn::<f32>::new(
            resample_ratio,
            2.0,
            params,
            wav.len(),
            1, // mono
        )
        .map_err(|e| candle_core::Error::Msg(format!("Resampler creation failed: {e}")))?;

        let input = vec![wav.to_vec()];
        let output = resampler
            .process(&input, None)
            .map_err(|e| candle_core::Error::Msg(format!("Resampling failed: {e}")))?;

        Ok((output[0].clone(), target_fs))
    }

    fn extract_mel_spectrogram_rustfft(&self, wav: &[f32], fs: u32) -> Result<Vec<Vec<f32>>> {
        // Set parameters based on sample rate
        let (n_fft, win_length, hop_length) = if fs == 8000 {
            (256, 200, 80)
        } else if fs == 16000 {
            (512, 400, 160)
        } else {
            return Err(candle_core::Error::Msg(format!(
                "Unsupported sample rate: {fs}"
            )));
        };

        // Apply preemphasis first
        let preemphasized = self.apply_preemphasis(wav, 0.97);

        // Create FFT planner
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Create Hanning window
        let window: Vec<f64> = hanning_iter(win_length).collect();

        // Create mel filterbank
        let mel_filters = self.create_mel_filterbank(AUDIO_FEATURE_SIZE, n_fft, fs as f32)?;

        // Extract frames and apply STFT
        let n_batch = (preemphasized.len() - win_length) / hop_length + 1;
        let mut mel_features = Vec::new();

        for i in 0..n_batch {
            let start = i * hop_length;
            let end = start + win_length;
            if end > preemphasized.len() {
                break;
            }

            // Apply window and convert to complex
            let mut windowed: Vec<Complex32> = preemphasized[start..end]
                .iter()
                .zip(window.iter())
                .map(|(s, w)| Complex32::new(s * *w as f32, 0.0))
                .collect();

            // Pad to n_fft length
            windowed.resize(n_fft, Complex32::new(0.0, 0.0));

            // Apply FFT
            fft.process(&mut windowed);

            // Take power spectrum of positive frequencies
            let power_spectrum: Vec<f32> = windowed[0..n_fft / 2 + 1]
                .iter()
                .map(|c| c.norm_sqr())
                .collect();

            // Apply mel filterbank
            let mut mel_frame = vec![0.0; AUDIO_FEATURE_SIZE];
            for (mel_idx, filter) in mel_filters.iter().enumerate() {
                let mut sum = 0.0;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    if freq_idx < power_spectrum.len() {
                        sum += power_spectrum[freq_idx] * coeff;
                    }
                }
                mel_frame[mel_idx] = (sum.max(1.0)).ln(); // Apply log with clipping
            }

            mel_features.push(mel_frame);
        }

        // Handle 8kHz case with fillzero method
        if fs == 8000 && self.eightk_method == "fillzero" {
            for frame in &mut mel_features {
                // Extend each frame with zeros to match 16kHz structure
                let original_len = frame.len();
                frame.extend(vec![0.0; original_len]);
            }
        }

        Ok(mel_features)
    }

    fn apply_preemphasis(&self, wav: &[f32], preemphasis: f32) -> Vec<f32> {
        if wav.is_empty() {
            return vec![];
        }

        let mut preemphasized = Vec::with_capacity(wav.len());

        // First sample: y[0] = x[0] * 32768
        preemphasized.push(wav[0] * 32768.0);

        // Remaining samples: y[n] = (x[n] - preemphasis * x[n-1]) * 32768
        for i in 1..wav.len() {
            let filtered = (wav[i] - preemphasis * wav[i - 1]) * 32768.0;
            preemphasized.push(filtered);
        }

        preemphasized
    }

    fn create_mel_filterbank(
        &self,
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let bank_width = n_fft / 2 + 1;
        let fmax = sample_rate / 2.0;
        let fmin = 0.0;

        // Mel scale conversion functions
        let hz_to_mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
        let mel_to_hz = |mel: f32| 700.0 * (mel / 1127.0).exp() - 700.0;

        let mel_low = hz_to_mel(fmin);
        let mel_high = hz_to_mel(fmax);

        // Create mel centers
        let mel_centers: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (n_mels + 1) as f32)
            .collect();

        let hz_centers: Vec<f32> = mel_centers.iter().map(|&mel| mel_to_hz(mel)).collect();

        // Convert to bin numbers
        let bin_centers: Vec<usize> = hz_centers
            .iter()
            .map(|&f| ((f * n_fft as f32 / sample_rate) + 0.5) as usize)
            .collect();

        // Create triangular filters
        let mut filters = Vec::new();
        for m in 0..n_mels {
            let mut filter = vec![0.0; bank_width];

            let left_bin = bin_centers[m];
            let center_bin = bin_centers[m + 1];
            let right_bin = bin_centers[m + 2];

            // Left slope
            for (bin, filter) in filter
                .iter_mut()
                .enumerate()
                .take(center_bin)
                .skip(left_bin)
            {
                if bin < bank_width {
                    *filter = (bin - left_bin) as f32 / (center_bin - left_bin) as f32;
                }
            }

            // Right slope
            for (bin, filter) in filter
                .iter_mut()
                .enumerate()
                .take(right_bin)
                .skip(center_bin)
            {
                if bin < bank_width {
                    *filter = (right_bin - bin) as f32 / (right_bin - center_bin) as f32;
                }
            }

            filters.push(filter);
        }

        Ok(filters)
    }

    fn compute_audio_embed_size(
        &self,
        audio_frames: usize,
        compression_rate: usize,
        downsample_rate: usize,
    ) -> usize {
        // First compression
        let integer = audio_frames / compression_rate;
        let remainder = audio_frames % compression_rate;
        let result = if remainder == 0 { integer } else { integer + 1 };

        // Second compression (qformer)
        let integer = result / downsample_rate;
        let remainder = result % downsample_rate;
        if remainder == 0 {
            integer
        } else {
            integer + 1
        }
    }

    fn create_audio_attention_mask(
        &self,
        audio_frames_list: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let max_frames = *audio_frames_list.iter().max().unwrap_or(&0);
        let batch_size = audio_frames_list.len();

        let mut mask_data = vec![0u8; batch_size * max_frames];
        for (batch_idx, &frames) in audio_frames_list.iter().enumerate() {
            for frame_idx in 0..frames.min(max_frames) {
                mask_data[batch_idx * max_frames + frame_idx] = 1;
            }
        }

        Tensor::from_slice(&mask_data, (batch_size, max_frames), device)?.to_dtype(DType::F32)
    }

    fn process_audio_for_sequences(
        &self,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
    ) -> AudioProcessingResult {
        // Check if any sequence has audio tokens
        let has_audio_tokens = input_seqs
            .iter()
            .any(|seq| seq.get_toks().contains(&(AUDIO_SPECIAL_TOKEN_ID as u32)));

        if !has_audio_tokens {
            return Ok((None, None, None));
        }

        let mut audio_features_list = Vec::new();
        let mut audio_embed_sizes_list = Vec::new();
        let mut audio_frames_list = Vec::new();

        // Process audio for each sequence that needs it
        for seq in input_seqs.iter_mut() {
            let has_audio = seq.get_toks().contains(&(AUDIO_SPECIAL_TOKEN_ID as u32));

            if has_audio {
                if let Some(audios) = seq.take_audios() {
                    for audio in audios.into_iter() {
                        // Convert multi-channel audio to mono by averaging channels
                        let samples = audio.to_mono();

                        // Extract features
                        let features = self.extract_audio_features(&samples, audio.sample_rate)?;
                        let audio_frames = features.len() * self.audio_feat_stride;

                        let embed_size = self.compute_audio_embed_size(
                            audio_frames,
                            self.audio_compression_rate,
                            self.audio_downsample_rate,
                        );

                        // Convert to tensor
                        let features_len = features.len();
                        let features_flat: Vec<f32> = features.into_iter().flatten().collect();
                        let features_tensor = Tensor::from_slice(
                            &features_flat,
                            (features_len, AUDIO_FEATURE_SIZE),
                            device,
                        )?;

                        audio_features_list.push(features_tensor);
                        audio_embed_sizes_list.push(embed_size);
                        audio_frames_list.push(audio_frames);
                    }
                } else {
                    candle_core::bail!("No audios in `process_audio_for_sequences`");
                };
            }
        }

        if audio_features_list.is_empty() {
            return Ok((None, None, None));
        }

        // Pad sequences to same length
        let max_len = audio_features_list
            .iter()
            .map(|t| t.dim(0).unwrap_or(0))
            .max()
            .unwrap_or(0);

        let mut padded_features = Vec::new();
        for features in audio_features_list {
            let seq_len = features.dim(0)?;
            if seq_len < max_len {
                let padding =
                    Tensor::zeros((max_len - seq_len, AUDIO_FEATURE_SIZE), DType::F32, device)?;
                let padded = Tensor::cat(&[features, padding], 0)?;
                padded_features.push(padded);
            } else {
                padded_features.push(features);
            }
        }

        // Stack into batch tensor
        let input_audio_embeds = Tensor::stack(&padded_features, 0)?;

        // Create attention mask if multiple sequences
        let audio_attention_mask = if audio_frames_list.len() > 1 {
            Some(self.create_audio_attention_mask(&audio_frames_list, device)?)
        } else {
            None
        };

        Ok((
            Some(input_audio_embeds),
            Some(audio_embed_sizes_list),
            audio_attention_mask,
        ))
    }
}

impl Phi4MMInputsProcessor {
    fn pad_image(
        image: &DynamicImage,
        top: u32,
        bottom: u32,
        left: u32,
        right: u32,
        pad_color: Rgba<u8>,
    ) -> DynamicImage {
        // Calculate the new dimensions
        let new_width = image.width() + left + right;
        let new_height = image.height() + top + bottom;

        // Create a new image with the new dimensions and fill it with the pad color
        let mut new_image = DynamicImage::new_rgb8(new_width, new_height);
        for x in 0..new_width {
            for y in 0..new_height {
                new_image.put_pixel(x, y, pad_color);
            }
        }

        // Paste the original image into the center of the new image
        new_image
            .copy_from(image, 0, 0)
            .expect("Failed to copy image");

        new_image
    }

    fn compute_target_ratios(min_num: u32, max_num: u32) -> Vec<(u32, u32)> {
        let mut ratios: HashSet<(u32, u32)> = HashSet::new();
        for n in min_num..=max_num {
            for i in 1..=n {
                for j in 1..=n {
                    if i * j >= min_num && i * j <= max_num {
                        ratios.insert((i, j));
                    }
                }
            }
        }
        let mut sorted_ratios: Vec<(u32, u32)> = ratios.into_iter().collect();
        sorted_ratios.sort_by_key(|&(i, j)| i * j);
        sorted_ratios
    }

    fn find_closest_aspect_ratio(
        aspect_ratio: f64,
        target_ratios: Vec<(u32, u32)>,
        width: u32,
        height: u32,
        image_size: usize,
    ) -> (u32, u32) {
        let mut best_ratio_diff = f64::INFINITY;
        let mut best_ratio = (1, 1);
        let area = width * height;
        for ratio in target_ratios {
            let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
            let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();
            if ratio_diff < best_ratio_diff {
                best_ratio_diff = ratio_diff;
                best_ratio = ratio;
            } else if ratio_diff == best_ratio_diff
                && area as f64
                    > 0.5 * image_size as f64 * image_size as f64 * ratio.0 as f64 * ratio.1 as f64
            {
                best_ratio = ratio;
            }
        }
        best_ratio
    }

    fn dynamic_preprocess(
        &self,
        mut image: DynamicImage,
        min_num: usize,
        max_num: usize,
        image_size: usize,
        mask_size: usize,
        device: &Device,
    ) -> Result<(DynamicImage, Tensor)> {
        let (orig_w, orig_h) = image.dimensions();

        let w_crop_num = (orig_w as f64 / image_size as f64).ceil();
        let h_crop_num = (orig_h as f64 / image_size as f64).ceil();
        let (target_aspect_ratio, target_width, target_height) =
            if w_crop_num * h_crop_num > max_num as f64 {
                let aspect_ratio = orig_w as f64 / orig_h as f64;
                let target_ratios = Self::compute_target_ratios(min_num as u32, max_num as u32);

                let target_aspect_ratio = Self::find_closest_aspect_ratio(
                    aspect_ratio,
                    target_ratios,
                    orig_w,
                    orig_h,
                    image_size,
                );

                let target_width = image_size * target_aspect_ratio.0 as usize;
                let target_height = image_size * target_aspect_ratio.1 as usize;

                (
                    (target_aspect_ratio.0 as f64, target_aspect_ratio.1 as f64),
                    target_width,
                    target_height,
                )
            } else {
                let target_width = (image_size as f64 * w_crop_num) as usize;
                let target_height = (image_size as f64 * h_crop_num) as usize;
                let target_aspect_ratio = (w_crop_num, h_crop_num);

                (target_aspect_ratio, target_width, target_height)
            };

        let ratio_width = target_width as f64 / orig_w as f64;
        let ratio_height = target_height as f64 / orig_h as f64;
        let (new_size, padding_width, padding_height) = if ratio_width < ratio_height {
            (
                (target_width, (orig_h as f64 * ratio_width) as usize),
                0_usize,
                target_height - (orig_h as f64 * ratio_width) as usize,
            )
        } else {
            (
                ((orig_w as f64 * ratio_height) as usize, target_height),
                target_width - (orig_w as f64 * ratio_height) as usize,
                0_usize,
            )
        };

        // Guard against extreme aspect ratios resulting in too-small dimensions
        if new_size.1.min(target_height) < 10 || new_size.0.min(target_width) < 10 {
            candle_core::bail!(
                "Image aspect ratio too extreme; resulting size below minimum threshold",
            );
        }

        let mut attention_mask = Tensor::ones(
            (
                (mask_size as f64 * target_aspect_ratio.1) as usize,
                (mask_size as f64 * target_aspect_ratio.0) as usize,
            ),
            DType::U32,
            device,
        )?;
        if padding_width >= 14 {
            attention_mask = attention_mask.slice_assign(
                &[
                    0..attention_mask.dim(0)?,
                    (attention_mask.dim(1)? - padding_width / 14)..attention_mask.dim(1)?,
                ],
                &Tensor::zeros(
                    (attention_mask.dim(0)?, padding_width / 14),
                    DType::U32,
                    device,
                )?,
            )?;
        }
        if padding_height >= 14 {
            attention_mask = attention_mask.slice_assign(
                &[
                    (attention_mask.dim(0)? - padding_height / 14)..attention_mask.dim(0)?,
                    0..attention_mask.dim(1)?,
                ],
                &Tensor::zeros(
                    (padding_height / 14, attention_mask.dim(1)?),
                    DType::U32,
                    device,
                )?,
            )?;
        }

        // Ensure the attention mask is non-empty
        let mask_sum: u32 = attention_mask.sum_all()?.to_scalar::<u32>()?;
        if mask_sum == 0 {
            candle_core::bail!("dynamic_preprocess produced an attention mask with zero sum",);
        }

        image = image.resize_exact(new_size.0 as u32, new_size.1 as u32, FilterType::Nearest);
        image = Self::pad_image(
            &image,
            0,
            padding_height as u32,
            0,
            padding_width as u32,
            Rgba([255u8, 255, 255, 255]),
        );

        Ok((image, attention_mask))
    }
}

impl ImagePreProcessor for Phi4MMInputsProcessor {
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
        (_, _): (usize, usize),
    ) -> Result<PreprocessedImages> {
        // If no images, will not call this.
        assert!(!images.is_empty());
        assert!(videos.is_empty());

        // If >1 images, resize them all to the largest, potentially destroying aspect ratio
        let mut max_size = None;
        for image in images.iter() {
            if max_size.is_none() {
                max_size = Some((image.dimensions().0 as usize, image.dimensions().1 as usize))
            } else if max_size.is_some_and(|(x, _)| image.dimensions().0 as usize > x) {
                max_size = Some((image.dimensions().0 as usize, max_size.unwrap().1));
            } else if max_size.is_some_and(|(_, y)| image.dimensions().1 as usize > y) {
                max_size = Some((max_size.unwrap().0, image.dimensions().1 as usize));
            }
        }
        let (max_w, max_h) = max_size.unwrap();
        for image in images.iter_mut() {
            *image = image.resize_exact(max_w as u32, max_h as u32, FilterType::Nearest);
        }

        let mut image_sizes = Vec::new();
        let mut padded_images = Vec::new();
        let mut padded_masks = Vec::new();
        let mut num_img_tokens = Vec::new();
        for mut image in images {
            // Convert to rgb, default to true
            if config.do_convert_rgb.unwrap_or(true) {
                image = DynamicImage::ImageRgb8(image.to_rgb8());
            }

            let transforms = Transforms {
                input: &ToTensor,
                inner_transforms: &[&Normalize {
                    mean: vec![0.5, 0.5, 0.5],
                    std: vec![0.5, 0.5, 0.5],
                }],
            };
            // Dynamic HD
            let dyhd_base_resolution = DYHD_BASE_RESOLUTION;
            let base_resolution = dyhd_base_resolution;
            // over 384 and 448 resolution
            let mask_resolution = base_resolution / 14;
            let min_num = 1;

            let (elem, attention_mask) = self.dynamic_preprocess(
                image,
                min_num,
                config.dynamic_hd.unwrap(),
                base_resolution,
                mask_resolution,
                device,
            )?;

            let hd_image = elem.apply(transforms, device)?;
            let (img_h, img_w) = (hd_image.dim(1)?, hd_image.dim(2)?);
            let (mask_h, mask_w) = (attention_mask.dim(0)?, attention_mask.dim(1)?);

            // Resize with bicubic interpolation
            let global_image = hd_image
                .unsqueeze(0)?
                .interpolate2d(base_resolution, base_resolution)?;
            let global_attention_mask =
                Tensor::ones((1, mask_resolution, mask_resolution), DType::U32, device)?;

            let hd_image_reshape = hd_image
                .reshape((
                    1,
                    3,
                    (img_h as f32 / base_resolution as f32) as usize,
                    base_resolution,
                    (img_w as f32 / base_resolution as f32) as usize,
                    base_resolution,
                ))?
                .permute((0, 2, 4, 1, 3, 5))?
                .reshape(((), 3, base_resolution, base_resolution))?;

            let attention_mask_reshape = attention_mask
                .reshape((
                    1,
                    (mask_h as f32 / mask_resolution as f32) as usize,
                    mask_resolution,
                    (mask_w as f32 / mask_resolution as f32) as usize,
                    mask_resolution,
                ))?
                .permute((0, 1, 3, 2, 4))?
                .reshape(((), mask_resolution, mask_resolution))?;

            let downsample_attention_mask = {
                let h_indices =
                    Tensor::arange_step(0, attention_mask_reshape.dim(1)? as u32, 2, device)?;
                let w_indices =
                    Tensor::arange_step(0, attention_mask_reshape.dim(2)? as u32, 2, device)?;
                let selected = attention_mask_reshape
                    .index_select(&h_indices, 1)?
                    .index_select(&w_indices, 2)?;

                let mask = selected
                    .reshape((
                        1,
                        mask_h / mask_resolution,
                        mask_w / mask_resolution,
                        mask_resolution / 2 + mask_resolution % 2,
                        mask_resolution / 2 + mask_resolution % 2,
                    ))?
                    .permute((0, 1, 3, 2, 4))?;
                mask.reshape((mask.dim(1)? * mask.dim(2)?, mask.dim(3)? * mask.dim(4)?))?
            };

            let img_tokens = 256
                + 1
                + downsample_attention_mask.sum_all()?.to_scalar::<u32>()? as usize
                + downsample_attention_mask
                    .i((.., 0))?
                    .sum_all()?
                    .to_scalar::<u32>()? as usize
                + 16;

            let hd_image_reshape = Tensor::cat(&[global_image, hd_image_reshape], 0)?;
            let hd_mask_reshape = Tensor::cat(&[global_attention_mask, attention_mask_reshape], 0)?;

            image_sizes.push((img_h as u32, img_w as u32));
            padded_images.push(hd_image_reshape);
            padded_masks.push(hd_mask_reshape);
            num_img_tokens.push(img_tokens);
        }
        Ok(PreprocessedImages {
            pixel_values: Tensor::stack(&padded_images, 0)?,
            pixel_attention_mask: Some(Tensor::stack(&padded_masks, 0)?),
            image_sizes: None,
            num_img_tokens: Some(num_img_tokens),
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
