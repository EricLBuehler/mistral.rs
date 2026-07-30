#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalKind};
use std::{
    any::Any,
    collections::{hash_map::DefaultHasher, HashSet},
    hash::{Hash, Hasher},
    ops::Range,
    sync::Arc,
};

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use mistralrs_vision::{ApplyTransforms, Normalize, ToTensor, Transforms};
use regex::Regex;
use tokenizers::Tokenizer;

use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use rustfft::{
    num_complex::{Complex32, Complex64},
    FftPlanner,
};

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
        ProcessorCreator,
    },
    sequence::{build_mm_features_from_ranges, Sequence},
};

use crate::vision_models::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    multimodal_layout::{
        MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout, PackedMultimodalLayout,
        RequestMultimodalLayout,
    },
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
const AUDIO_MEL_SAMPLE_RATE: usize = 16000;
const AUDIO_MEL_N_FFT: usize = 512;
const AUDIO_MEL_FMAX_HZ: f32 = 7690.;
const AUDIO_PREEMPHASIS: f32 = 0.97;
const AUDIO_SAMPLE_SCALE: f32 = 32768.;
const HAMMING_ALPHA: f64 = 0.54;
const HAMMING_BETA: f64 = 0.46;

struct Phi4PromptPlan {
    tokens: Vec<u32>,
    image_ranges: Vec<(usize, usize)>,
    audio_ranges: Vec<(usize, usize)>,
}

struct Phi4ImageBatch {
    pixel_values: Tensor,
    attention_mask: Tensor,
    image_sizes: Vec<(u32, u32)>,
    hashes: Vec<u64>,
}

struct Phi4AudioBatch {
    input_embeds: Tensor,
    embed_sizes: Vec<usize>,
    feature_lens: Vec<usize>,
    vision_modes: Vec<bool>,
    attention_mask: Option<Tensor>,
    hashes: Vec<u64>,
}

fn expand_phi4_placeholders(
    input_ids: &[u32],
    image_token_counts: &[usize],
    audio_token_counts: &[usize],
) -> Result<Phi4PromptPlan> {
    let mut tokens = Vec::new();
    let mut image_ranges = Vec::with_capacity(image_token_counts.len());
    let mut audio_ranges = Vec::with_capacity(audio_token_counts.len());
    let mut image_index = 0usize;
    let mut audio_index = 0usize;

    for &token in input_ids {
        if token == IMAGE_SPECIAL_TOKEN_ID as u32 {
            let count = *image_token_counts.get(image_index).ok_or_else(|| {
                candle_core::Error::msg("Phi4MM has more image placeholders than image inputs")
            })?;
            if count == 0 {
                candle_core::bail!("Phi4MM image placeholder cannot be empty");
            }
            let start = tokens.len();
            tokens.extend(std::iter::repeat_n(token, count));
            image_ranges.push((start, count));
            image_index += 1;
        } else if token == AUDIO_SPECIAL_TOKEN_ID as u32 {
            let count = *audio_token_counts.get(audio_index).ok_or_else(|| {
                candle_core::Error::msg("Phi4MM has more audio placeholders than audio inputs")
            })?;
            if count == 0 {
                candle_core::bail!("Phi4MM audio placeholder cannot be empty");
            }
            let start = tokens.len();
            tokens.extend(std::iter::repeat_n(token, count));
            audio_ranges.push((start, count));
            audio_index += 1;
        } else {
            tokens.push(token);
        }
    }

    if image_index != image_token_counts.len() {
        candle_core::bail!(
            "Phi4MM has {image_index} image placeholders but {} image inputs",
            image_token_counts.len()
        );
    }
    if audio_index != audio_token_counts.len() {
        candle_core::bail!(
            "Phi4MM has {audio_index} audio placeholders but {} audio inputs",
            audio_token_counts.len()
        );
    }

    Ok(Phi4PromptPlan {
        tokens,
        image_ranges,
        audio_ranges,
    })
}

fn phi4_request_layout(
    sequence_id: usize,
    tokens: &[u32],
    query: Range<usize>,
    features: &[MultiModalFeature],
) -> Result<RequestMultimodalLayout> {
    if query.start > query.end || query.end > tokens.len() {
        candle_core::bail!(
            "Phi4MM packed query {query:?} exceeds {} tokens",
            tokens.len()
        );
    }
    let has_image = features
        .iter()
        .any(|feature| feature.kind == MultimodalKind::Image);
    let mut items = Vec::with_capacity(features.len());

    for feature in features {
        if feature.hashes.len() != 1 || feature.item_range.len() != 1 {
            candle_core::bail!(
                "Phi4MM packed layout requires one media item and hash per placeholder span"
            );
        }
        let end = feature
            .offset
            .checked_add(feature.length)
            .ok_or_else(|| candle_core::Error::msg("Phi4MM placeholder range overflow"))?;
        if end > tokens.len() {
            candle_core::bail!(
                "Phi4MM {:?} placeholder {}..{end} exceeds {} tokens",
                feature.kind,
                feature.offset,
                tokens.len()
            );
        }
        let expected_token = match feature.kind {
            MultimodalKind::Image => IMAGE_SPECIAL_TOKEN_ID as u32,
            MultimodalKind::Audio => AUDIO_SPECIAL_TOKEN_ID as u32,
            MultimodalKind::Video => {
                candle_core::bail!("Phi4MM does not support video layout items")
            }
        };
        if tokens[feature.offset..end]
            .iter()
            .any(|&token| token != expected_token)
        {
            candle_core::bail!(
                "Phi4MM {:?} placeholder contains a non-placeholder token",
                feature.kind
            );
        }
        let source_output = match feature.kind {
            MultimodalKind::Audio if has_image => 1,
            MultimodalKind::Image | MultimodalKind::Audio => 0,
            MultimodalKind::Video => unreachable!(),
        };
        let placeholder = feature.offset..end;
        items.push(MultimodalItemLayout::new(
            MultimodalEncoderKey {
                kind: feature.kind,
                hash: feature.hashes[0],
            },
            feature.item_range.start,
            placeholder.clone(),
            feature.attention_policy,
            vec![MultimodalEmbeddingMap::contiguous(
                placeholder,
                0,
                source_output,
            )?],
        )?);
    }

    Ok(RequestMultimodalLayout {
        sequence_id,
        query,
        items,
    })
}

fn phi4_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Phi4MM packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if query_len != seq.get_toks().len() {
                candle_core::bail!("Phi4MM packed prefill requires the complete uncached prompt");
            }
            phi4_request_layout(*seq.id(), seq.get_toks(), 0..query_len, seq.mm_features())
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

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

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Phi4MMInputProcessor requires a specified tokenizer.")
        })?;
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        self.prepare_prompt_plans(&tokenizer, input_seqs, device, config, paged_attn_metadata)
            .map_err(anyhow::Error::new)
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
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Phi4MMInputProcessor requires a specified tokenizer.")
        })?;
        let config_any = other_config
            .clone()
            .expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config_any.downcast_ref().expect("Downcast failed.");

        if is_prompt {
            self.prepare_prompt_plans(
                &tokenizer,
                input_seqs,
                device,
                config,
                paged_attn_metadata.as_mut(),
            )
            .map_err(anyhow::Error::new)?;
        }

        let has_media = is_prompt
            && input_seqs
                .iter()
                .any(|seq| seq.has_images() || seq.has_audios());
        if !has_media {
            return text_models_inputs_processor::TextInputsProcessor
                .process_inputs(
                    Some(tokenizer.clone()),
                    input_seqs,
                    is_prompt,
                    is_xlora,
                    device,
                    no_kv_cache,
                    last_n_context_len,
                    return_raw_logits,
                    sliding_window,
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
                        recurrent_batch_kind,
                        adapter_leases,
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
                            audio_feature_lens: None,
                            audio_vision_modes: None,
                            audio_attention_mask: None,
                            image_hashes: vec![],
                            audio_hashes: vec![],
                            packed_layout: None,
                        }),
                        paged_attn_meta,
                        flash_meta,
                        recurrent_batch_kind,
                        adapter_leases,
                    });
                    InputProcessorOutput {
                        inputs,
                        seq_indices,
                    }
                });
        }

        let image_batch = self
            .process_image_batch(input_seqs, config, device)
            .map_err(anyhow::Error::new)?;
        let audio_batch = self
            .process_audio_batch(input_seqs, device)
            .map_err(anyhow::Error::new)?;
        let toks = input_seqs
            .iter()
            .map(|seq| seq.get_toks().to_vec())
            .collect::<Vec<_>>();

        let result = if is_prompt {
            get_prompt_input(
                toks.iter().map(Vec::as_slice).collect(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
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
                sliding_window,
            )
        };

        result.and_then(|metadata| {
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
            let packed_layout = if is_prompt && flash_meta.packed {
                let query_lens = paged_attn_meta
                    .as_ref()
                    .and_then(|metadata| metadata.query_lens.as_deref())
                    .ok_or_else(|| {
                        anyhow::Error::msg("packed Phi4MM prefill requires logical query lengths")
                    })?;
                let layout =
                    phi4_packed_layout(input_seqs, query_lens).map_err(anyhow::Error::new)?;
                if layout.token_count() != input.dim(1)? {
                    anyhow::bail!(
                        "Phi4MM packed layout has {} tokens but input has {}",
                        layout.token_count(),
                        input.dim(1)?
                    );
                }
                Some(layout)
            } else {
                None
            };
            let (input_image_embeds, image_attention_mask, image_sizes, image_hashes) =
                match image_batch {
                    Some(batch) => (
                        Some(batch.pixel_values),
                        Some(batch.attention_mask),
                        Some(batch.image_sizes),
                        batch.hashes,
                    ),
                    None => (None, None, None, Vec::new()),
                };
            let (
                input_audio_embeds,
                audio_embed_sizes,
                audio_feature_lens,
                audio_vision_modes,
                audio_attention_mask,
                audio_hashes,
            ) = match audio_batch {
                Some(batch) => (
                    Some(batch.input_embeds),
                    Some(batch.embed_sizes),
                    Some(batch.feature_lens),
                    Some(batch.vision_modes),
                    batch.attention_mask,
                    batch.hashes,
                ),
                None => (None, None, None, None, None, Vec::new()),
            };
            let pixel_values = input_image_embeds.clone();
            let inputs: Box<dyn Any> = Box::new(ModelInputs {
                input_ids: input,
                seqlen_offsets: positions,
                context_lens,
                position_ids,
                pixel_values: pixel_values.clone(),
                model_specific_args: Box::new(Phi4MMVisionSpecificArgs {
                    input_image_embeds,
                    image_attention_mask,
                    image_sizes,
                    input_audio_embeds,
                    audio_embed_sizes,
                    audio_feature_lens,
                    audio_vision_modes,
                    audio_attention_mask,
                    image_hashes,
                    audio_hashes,
                    packed_layout,
                }),
                paged_attn_meta,
                flash_meta,
                recurrent_batch_kind: if is_prompt {
                    crate::pipeline::RecurrentBatchKind::Prefill
                } else {
                    crate::pipeline::RecurrentBatchKind::Decode
                },
                adapter_leases: crate::vision_models::adapter_leases(input_seqs, &seq_indices),
            });
            Ok(InputProcessorOutput {
                inputs,
                seq_indices,
            })
        })
    }
}

impl Phi4MMInputsProcessor {
    fn prepare_prompt_plans(
        &self,
        tokenizer: &Tokenizer,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        config: &PreProcessorConfig,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        let image_pattern =
            Regex::new(COMPATIBLE_IMAGE_SPECIAL_TOKEN_PATTERN).map_err(candle_core::Error::wrap)?;
        let audio_pattern =
            Regex::new(COMPATIBLE_AUDIO_SPECIAL_TOKEN_PATTERN).map_err(candle_core::Error::wrap)?;

        for seq in input_seqs {
            if seq.multimodal.has_changed_prompt && !seq.mm_features().is_empty() {
                continue;
            }
            let images = seq.clone_images().unwrap_or_default();
            let audios = seq.clone_audios().unwrap_or_default();
            if images.is_empty() && audios.is_empty() {
                continue;
            }

            let raw_image_hashes = suffix_hashes(
                seq.multimodal.image_hashes().unwrap_or_default(),
                images.len(),
                "image",
            )?
            .to_vec();
            let image_hashes = images
                .iter()
                .zip(&raw_image_hashes)
                .map(|(image, &hash)| phi4_image_hash(hash, image))
                .collect::<Vec<_>>();
            let raw_audio_hashes = suffix_hashes(
                seq.multimodal.audio_hashes().unwrap_or_default(),
                audios.len(),
                "audio",
            )?
            .to_vec();
            let audio_hashes = audios
                .iter()
                .zip(&raw_audio_hashes)
                .map(|(audio, &hash)| phi4_audio_hash(hash, audio))
                .collect::<Vec<_>>();

            let image_token_counts = if images.is_empty() {
                Vec::new()
            } else {
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
                } = self.preprocess(images, vec![], config, device, (usize::MAX, usize::MAX))?;
                let attention_mask = pixel_attention_mask.ok_or_else(|| {
                    candle_core::Error::msg("Phi4MM image preprocessing omitted its attention mask")
                })?;
                let image_sizes = image_sizes_all.ok_or_else(|| {
                    candle_core::Error::msg("Phi4MM image preprocessing omitted image sizes")
                })?;
                let token_counts = num_img_tokens.ok_or_else(|| {
                    candle_core::Error::msg("Phi4MM image preprocessing omitted token counts")
                })?;
                if pixel_values.dim(0)? != image_hashes.len()
                    || attention_mask.dim(0)? != image_hashes.len()
                    || image_sizes.len() != image_hashes.len()
                    || token_counts.len() != image_hashes.len()
                {
                    candle_core::bail!("Phi4MM image preprocessing metadata length mismatch");
                }
                let flattened_sizes = image_sizes
                    .iter()
                    .flat_map(|&(height, width)| [height, width])
                    .collect::<Vec<_>>();
                seq.multimodal.cached_pixel_values = Some(pixel_values);
                seq.multimodal.cached_pixel_attention_mask = Some(attention_mask);
                seq.multimodal.cached_spatial_shapes = Some(Tensor::from_vec(
                    flattened_sizes,
                    (image_sizes.len(), 2),
                    device,
                )?);
                token_counts
            };

            let audio_token_counts = audios
                .iter()
                .map(|audio| self.audio_token_count(audio))
                .collect::<Result<Vec<_>>>()?;

            let mut prompt = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(candle_core::Error::wrap)?;
            prompt = image_pattern
                .replace_all(&prompt, IMAGE_SPECIAL_TOKEN)
                .into_owned();
            prompt = audio_pattern
                .replace_all(&prompt, AUDIO_SPECIAL_TOKEN)
                .into_owned();
            let singleton_tokens = tokenizer
                .encode_fast(prompt, false)
                .map_err(candle_core::Error::wrap)?
                .get_ids()
                .to_vec();
            let plan = expand_phi4_placeholders(
                &singleton_tokens,
                &image_token_counts,
                &audio_token_counts,
            )?;

            let mut features = build_mm_features_from_ranges(
                &plan.image_ranges,
                &image_hashes,
                MultimodalKind::Image,
            );
            features.extend(build_mm_features_from_ranges(
                &plan.audio_ranges,
                &audio_hashes,
                MultimodalKind::Audio,
            ));
            features.sort_by_key(|feature| feature.offset);
            if features.len() != image_hashes.len() + audio_hashes.len() {
                candle_core::bail!("Phi4MM multimodal item metadata length mismatch");
            }

            let expanded_prompt = tokenizer
                .decode(&plan.tokens, false)
                .map_err(candle_core::Error::wrap)?;
            seq.set_initial_prompt(expanded_prompt);
            seq.set_toks_and_reallocate(plan.tokens, paged_attn_metadata.as_deref_mut());
            seq.set_mm_features(features);
            seq.multimodal.has_changed_prompt = true;
        }
        Ok(())
    }

    fn audio_token_count(&self, audio: &crate::AudioInput) -> Result<usize> {
        let samples = audio.to_mono();
        let (samples, sample_rate) =
            self.resample_audio_with_rubato(&samples, audio.sample_rate)?;
        let (window, hop) = match sample_rate {
            8000 => (200, 80),
            16000 => (400, 160),
            _ => candle_core::bail!("Unsupported sample rate: {sample_rate}"),
        };
        if samples.len() < window {
            candle_core::bail!(
                "Phi4MM audio has {} samples but needs at least {window}",
                samples.len()
            );
        }
        let feature_len = (samples.len() - window) / hop + 1;
        Ok(self.compute_audio_embed_size(
            feature_len * self.audio_feat_stride,
            self.audio_compression_rate,
            self.audio_downsample_rate,
        ))
    }

    fn process_image_batch(
        &self,
        input_seqs: &mut [&mut Sequence],
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<Option<Phi4ImageBatch>> {
        let mut images = Vec::new();
        let mut masks = Vec::new();
        let mut image_sizes = Vec::new();
        let mut hashes = Vec::new();

        for seq in input_seqs {
            if !seq.has_images() {
                continue;
            }

            let cached = match (
                &seq.multimodal.cached_pixel_values,
                &seq.multimodal.cached_pixel_attention_mask,
                &seq.multimodal.cached_spatial_shapes,
            ) {
                (Some(pixel_values), Some(attention_mask), Some(image_sizes)) => Some((
                    pixel_values.clone(),
                    attention_mask.clone(),
                    tensor_image_sizes(image_sizes)?,
                )),
                (None, None, None) => None,
                _ => candle_core::bail!("Phi4MM image preprocessing cache is incomplete"),
            };

            let (pixel_values, attention_mask, sizes, selected_hashes) =
                if let Some((pixel_values, attention_mask, sizes)) = cached {
                    let available = pixel_values.dim(0)?;
                    if attention_mask.dim(0)? != available || sizes.len() != available {
                        candle_core::bail!("Phi4MM cached image metadata length mismatch");
                    }
                    let range = if seq.is_chunked_prefill_view() {
                        seq.active_local_multimodal_item_range(MultimodalKind::Image, available)
                            .ok_or_else(|| {
                                candle_core::Error::msg(
                                    "Phi4MM image chunk has no active image item range",
                                )
                            })?
                    } else {
                        local_uncached_item_start(seq, MultimodalKind::Image, available)..available
                    };
                    let active_hashes: Vec<u64> = if seq.is_chunked_prefill_view() {
                        let active_images = seq.take_images().ok_or_else(|| {
                            candle_core::Error::msg("Phi4MM active image inputs are unavailable")
                        })?;
                        let raw_hashes = seq.image_hashes().unwrap_or_default();
                        if active_images.len() != raw_hashes.len() {
                            candle_core::bail!("Phi4MM active image hash count mismatch");
                        }
                        active_images
                            .iter()
                            .zip(raw_hashes)
                            .map(|(image, &hash)| phi4_image_hash(hash, image))
                            .collect()
                    } else {
                        let retained_images = seq.images().unwrap_or_default();
                        if retained_images.len() != available {
                            candle_core::bail!(
                                "Phi4MM cached image count does not match retained images"
                            );
                        }
                        let raw_hashes = suffix_hashes(
                            seq.multimodal.image_hashes().unwrap_or_default(),
                            available,
                            "image",
                        )?;
                        retained_images[range.clone()]
                            .iter()
                            .zip(&raw_hashes[range.clone()])
                            .map(|(image, &hash)| phi4_image_hash(hash, image))
                            .collect()
                    };
                    if active_hashes.len() != range.len() {
                        candle_core::bail!("Phi4MM active image hash count mismatch");
                    }
                    (
                        pixel_values.narrow(0, range.start, range.len())?,
                        attention_mask.narrow(0, range.start, range.len())?,
                        sizes[range].to_vec(),
                        active_hashes,
                    )
                } else {
                    let active_images = seq.take_images().ok_or_else(|| {
                        candle_core::Error::msg("Phi4MM image inputs are unavailable")
                    })?;
                    let raw_hashes = if seq.is_chunked_prefill_view() {
                        seq.image_hashes().unwrap_or_default()
                    } else {
                        suffix_hashes(
                            seq.multimodal.image_hashes().unwrap_or_default(),
                            active_images.len(),
                            "image",
                        )?
                    };
                    if active_images.len() != raw_hashes.len() {
                        candle_core::bail!("Phi4MM active image hash count mismatch");
                    }
                    let active_hashes = active_images
                        .iter()
                        .zip(raw_hashes)
                        .map(|(image, &hash)| phi4_image_hash(hash, image))
                        .collect();
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
                        rows: _,
                        cols: _,
                        pixel_values_list: _,
                        tgt_sizes: _,
                        image_sizes_all,
                        num_crops: _,
                    } = self.preprocess(
                        active_images,
                        vec![],
                        config,
                        device,
                        (usize::MAX, usize::MAX),
                    )?;
                    (
                        pixel_values,
                        pixel_attention_mask.ok_or_else(|| {
                            candle_core::Error::msg(
                                "Phi4MM image preprocessing omitted its attention mask",
                            )
                        })?,
                        image_sizes_all.ok_or_else(|| {
                            candle_core::Error::msg(
                                "Phi4MM image preprocessing omitted image sizes",
                            )
                        })?,
                        active_hashes,
                    )
                };

            let item_count = pixel_values.dim(0)?;
            if attention_mask.dim(0)? != item_count
                || sizes.len() != item_count
                || selected_hashes.len() != item_count
            {
                candle_core::bail!("Phi4MM active image metadata length mismatch");
            }
            for item in 0..item_count {
                let image = pixel_values.get(item)?;
                let mask = attention_mask.get(item)?;
                if image.dim(0)? != mask.dim(0)? {
                    candle_core::bail!("Phi4MM image crop and mask counts differ");
                }
                images.push(image);
                masks.push(mask);
            }
            image_sizes.extend(sizes);
            hashes.extend(selected_hashes);
        }

        if images.is_empty() {
            return Ok(None);
        }
        let max_crops = images
            .iter()
            .map(|image| image.dim(0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let images = images
            .into_iter()
            .map(|image| {
                let crops = image.dim(0)?;
                image.pad_with_zeros(0, 0, max_crops - crops)
            })
            .collect::<Result<Vec<_>>>()?;
        let masks = masks
            .into_iter()
            .map(|mask| pad_phi4_image_mask(mask, max_crops))
            .collect::<Result<Vec<_>>>()?;

        Ok(Some(Phi4ImageBatch {
            pixel_values: Tensor::stack(&images, 0)?,
            attention_mask: Tensor::stack(&masks, 0)?,
            image_sizes,
            hashes,
        }))
    }

    fn process_audio_batch(
        &self,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
    ) -> Result<Option<Phi4AudioBatch>> {
        let mut feature_tensors = Vec::new();
        let mut embed_sizes = Vec::new();
        let mut feature_lens = Vec::new();
        let mut vision_modes = Vec::new();
        let mut audio_frames = Vec::new();
        let mut hashes = Vec::new();

        for seq in input_seqs {
            if !seq.has_audios() {
                continue;
            }
            let audios = seq
                .take_audios()
                .ok_or_else(|| candle_core::Error::msg("Phi4MM audio inputs are unavailable"))?;
            let available = audios.len();
            let retained_hashes = if seq.is_chunked_prefill_view() {
                seq.audio_hashes().unwrap_or_default()
            } else {
                suffix_hashes(
                    seq.multimodal.audio_hashes().unwrap_or_default(),
                    available,
                    "audio",
                )?
            };
            let start = if seq.is_chunked_prefill_view() {
                0
            } else {
                local_uncached_item_start(seq, MultimodalKind::Audio, available)
            };
            let audios = &audios[start..];
            let raw_hashes = &retained_hashes[start..];
            let vision_mode = seq
                .mm_features()
                .iter()
                .any(|feature| feature.kind == MultimodalKind::Image);
            if raw_hashes.len() != audios.len() {
                candle_core::bail!("Phi4MM active audio hash count mismatch");
            }

            for (audio, &raw_hash) in audios.iter().zip(raw_hashes) {
                let features = self.extract_audio_features(&audio.to_mono(), audio.sample_rate)?;
                if features.is_empty() {
                    candle_core::bail!("Phi4MM audio preprocessing produced no frames");
                }
                if features
                    .iter()
                    .any(|feature| feature.len() != AUDIO_FEATURE_SIZE)
                {
                    candle_core::bail!("Phi4MM audio preprocessing produced invalid feature width");
                }
                let feature_len = features.len();
                let frames = feature_len * self.audio_feat_stride;
                let embed_size = self.compute_audio_embed_size(
                    frames,
                    self.audio_compression_rate,
                    self.audio_downsample_rate,
                );
                let flattened = features.into_iter().flatten().collect::<Vec<_>>();
                feature_tensors.push(Tensor::from_vec(
                    flattened,
                    (feature_len, AUDIO_FEATURE_SIZE),
                    device,
                )?);
                feature_lens.push(feature_len);
                vision_modes.push(vision_mode);
                audio_frames.push(frames);
                embed_sizes.push(embed_size);
                hashes.push(phi4_audio_hash(raw_hash, audio));
            }
        }

        if feature_tensors.is_empty() {
            return Ok(None);
        }
        let max_features = feature_lens.iter().copied().max().unwrap();
        let feature_tensors = feature_tensors
            .into_iter()
            .map(|features| {
                let len = features.dim(0)?;
                features.pad_with_zeros(0, 0, max_features - len)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = (feature_tensors.len() > 1)
            .then(|| self.create_audio_attention_mask(&audio_frames, device))
            .transpose()?;
        Ok(Some(Phi4AudioBatch {
            input_embeds: Tensor::stack(&feature_tensors, 0)?,
            embed_sizes,
            feature_lens,
            vision_modes,
            attention_mask,
            hashes,
        }))
    }

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
        let (n_fft, win_length, hop_length) = match fs {
            8000 => (256, 200, 80),
            16000 => (512, 400, 160),
            _ => candle_core::bail!("Unsupported sample rate: {fs}"),
        };
        if wav.len() < win_length {
            candle_core::bail!(
                "Phi4MM audio has {} samples but needs at least {win_length}",
                wav.len()
            );
        }

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let window = (0..win_length)
            .map(|index| {
                HAMMING_ALPHA
                    - HAMMING_BETA
                        * (2. * std::f64::consts::PI * index as f64 / (win_length - 1) as f64).cos()
            })
            .collect::<Vec<_>>();
        let mel_filters = self.create_mel_filterbank(
            AUDIO_FEATURE_SIZE,
            AUDIO_MEL_N_FFT,
            AUDIO_MEL_SAMPLE_RATE as f32,
            AUDIO_MEL_FMAX_HZ,
        )?;

        let n_batch = (wav.len() - win_length) / hop_length + 1;
        let mut mel_features = Vec::with_capacity(n_batch);
        for frame_index in 0..n_batch {
            let start = frame_index * hop_length;
            let frame = &wav[start..start + win_length];
            let mut windowed = frame
                .iter()
                .enumerate()
                .zip(&window)
                .map(|((index, &sample), &window)| {
                    let previous = if index == 0 { sample } else { frame[index - 1] };
                    let sample = (sample - AUDIO_PREEMPHASIS * previous) * AUDIO_SAMPLE_SCALE;
                    Complex64::new(sample as f64 * window, 0.)
                })
                .collect::<Vec<_>>();
            windowed.resize(n_fft, Complex64::new(0., 0.));
            fft.process(&mut windowed);

            let mut spectrum = windowed[..n_fft / 2 + 1]
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect::<Vec<_>>();
            if fs == 8000 {
                let bins = spectrum.len();
                spectrum.truncate(bins - 1);
                spectrum.resize(spectrum.len() + bins, Complex32::new(0., 0.));
            }
            if spectrum.len() != AUDIO_MEL_N_FFT / 2 + 1 {
                candle_core::bail!("Phi4MM audio spectrum has an invalid width");
            }
            let power_spectrum = spectrum
                .iter()
                .map(|value| value.norm_sqr())
                .collect::<Vec<_>>();
            let mut mel_frame = vec![0.; AUDIO_FEATURE_SIZE];
            for (mel_index, filter) in mel_filters.iter().enumerate() {
                let power = power_spectrum
                    .iter()
                    .zip(filter)
                    .map(|(&power, &coefficient)| power * coefficient)
                    .sum::<f32>();
                mel_frame[mel_index] = power.max(1.).ln();
            }
            mel_features.push(mel_frame);
        }
        Ok(mel_features)
    }

    fn create_mel_filterbank(
        &self,
        n_mels: usize,
        n_fft: usize,
        sample_rate: f32,
        fmax: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let bank_width = n_fft / 2 + 1;
        let sample_rate = sample_rate as f64;
        let fmax = fmax as f64;
        let fmin = 0_f64;
        if !(fmin < fmax && fmax <= sample_rate / 2.) {
            candle_core::bail!("Phi4MM mel filter frequency range is invalid");
        }
        let hz_to_mel = |frequency: f64| 1127. * (1. + frequency / 700.).ln();
        let bin_to_mel =
            |bin: usize| 1127. * (1. + bin as f64 * sample_rate / (n_fft as f64 * 700.)).ln();
        let frequency_to_bin =
            |frequency: f64| ((frequency * n_fft as f64 / sample_rate) + 0.5) as usize;
        let first_bin = frequency_to_bin(fmin) + 1;
        let last_bin = frequency_to_bin(fmax).max(first_bin);
        let mel_low = hz_to_mel(fmin);
        let mel_high = hz_to_mel(fmax);
        let mel_centers = (0..=n_mels + 1)
            .map(|index| mel_low + (mel_high - mel_low) * index as f64 / (n_mels + 1) as f64)
            .collect::<Vec<_>>();
        let mel_step = (mel_high - mel_low) / (n_mels + 1) as f64;

        let mut filters = Vec::with_capacity(n_mels);
        for mel_index in 0..n_mels {
            let mut filter = vec![0.; bank_width];
            let left = mel_centers[mel_index];
            let center = mel_centers[mel_index + 1];
            let right = mel_centers[mel_index + 2];
            for (bin, coefficient) in filter.iter_mut().enumerate().take(last_bin).skip(first_bin) {
                let mel = bin_to_mel(bin);
                if left < mel && mel < right {
                    *coefficient = (1. - (center - mel).abs() / mel_step) as f32;
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
}

fn suffix_hashes<'a>(hashes: &'a [u64], count: usize, kind: &str) -> Result<&'a [u64]> {
    if hashes.len() < count {
        candle_core::bail!(
            "Phi4MM has {} {kind} hashes but {count} retained {kind} inputs",
            hashes.len()
        );
    }
    Ok(&hashes[hashes.len() - count..])
}

fn phi4_image_hash(raw_hash: u64, image: &DynamicImage) -> u64 {
    let mut hasher = DefaultHasher::new();
    raw_hash.hash(&mut hasher);
    image.width().hash(&mut hasher);
    image.height().hash(&mut hasher);
    std::mem::discriminant(&image.color()).hash(&mut hasher);
    hasher.finish()
}

fn phi4_audio_hash(raw_hash: u64, audio: &crate::AudioInput) -> u64 {
    let mut hasher = DefaultHasher::new();
    raw_hash.hash(&mut hasher);
    audio.channels.hash(&mut hasher);
    hasher.finish()
}

fn local_uncached_item_start(seq: &Sequence, kind: MultimodalKind, available: usize) -> usize {
    let total_items = seq
        .mm_features()
        .iter()
        .filter(|feature| feature.kind == kind)
        .map(|feature| feature.item_range.end)
        .max()
        .unwrap_or(available);
    let retained_start = total_items.saturating_sub(available);
    seq.count_prefix_cached_mm_items_by_kind(kind)
        .saturating_sub(retained_start)
        .min(available)
}

fn tensor_image_sizes(image_sizes: &Tensor) -> Result<Vec<(u32, u32)>> {
    let rows = image_sizes.to_vec2::<u32>()?;
    rows.into_iter()
        .map(|row| match row.as_slice() {
            [height, width] => Ok((*height, *width)),
            _ => candle_core::bail!("Phi4MM cached image size must have two dimensions"),
        })
        .collect()
}

fn pad_phi4_image_mask(mask: Tensor, max_crops: usize) -> Result<Tensor> {
    let crops = mask.dim(0)?;
    if crops == max_crops {
        return Ok(mask);
    }
    let padding = Tensor::ones(
        (max_crops - crops, mask.dim(1)?, mask.dim(2)?),
        mask.dtype(),
        mask.device(),
    )?;
    Tensor::cat(&[mask, padding], 0)
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
        images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_, _): (usize, usize),
    ) -> Result<PreprocessedImages> {
        // If no images, will not call this.
        assert!(!images.is_empty());
        assert!(videos.is_empty());

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
        let max_crops = padded_images
            .iter()
            .map(|image| image.dim(0))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        padded_images = padded_images
            .into_iter()
            .map(|image| {
                let crops = image.dim(0)?;
                image.pad_with_zeros(0, 0, max_crops - crops)
            })
            .collect::<Result<Vec<_>>>()?;
        padded_masks = padded_masks
            .into_iter()
            .map(|mask| pad_phi4_image_mask(mask, max_crops))
            .collect::<Result<Vec<_>>>()?;
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};

    use crate::{
        paged_attention::block_hash::{MultimodalAttentionPolicy, MultimodalKind},
        vision_models::multimodal_layout::{
            MultimodalEncoderKey, MultimodalEncoderOutputs, PackedMultimodalLayout,
        },
    };

    use super::{
        expand_phi4_placeholders, pad_phi4_image_mask, phi4_audio_hash, phi4_image_hash,
        phi4_request_layout, MultiModalFeature, Phi4MMInputsProcessor, AUDIO_SPECIAL_TOKEN_ID,
        IMAGE_SPECIAL_TOKEN_ID,
    };

    fn feature(
        kind: MultimodalKind,
        hash: u64,
        item_index: usize,
        offset: usize,
        length: usize,
    ) -> MultiModalFeature {
        MultiModalFeature {
            kind,
            item_range: item_index..item_index + 1,
            hashes: vec![hash],
            offset,
            length,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }
    }

    fn output(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (1, values.len(), 1), &Device::Cpu).unwrap()
    }

    fn splice(
        requests: &[crate::vision_models::multimodal_layout::RequestMultimodalLayout],
        outputs: MultimodalEncoderOutputs,
    ) -> Vec<f32> {
        let layout = PackedMultimodalLayout::new(requests).unwrap();
        let text = Tensor::zeros(
            (1, layout.token_count(), 1),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    }

    #[test]
    fn expands_adjacent_media_as_distinct_items() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        let audio = AUDIO_SPECIAL_TOKEN_ID as u32;
        let plan = expand_phi4_placeholders(&[1, image, image, audio, audio, 2], &[2, 3], &[1, 2])
            .unwrap();
        assert_eq!(plan.image_ranges, vec![(1, 2), (3, 3)]);
        assert_eq!(plan.audio_ranges, vec![(6, 1), (7, 2)]);
        assert_eq!(
            plan.tokens,
            vec![1, image, image, image, image, image, audio, audio, audio, 2]
        );
    }

    #[test]
    fn rejects_placeholder_and_media_count_mismatches() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        assert!(expand_phi4_placeholders(&[image], &[], &[]).is_err());
        assert!(expand_phi4_placeholders(&[1], &[2], &[]).is_err());
        assert!(expand_phi4_placeholders(&[image], &[0], &[]).is_err());
    }

    #[test]
    fn eight_khz_fillzero_features_match_hf_oracle() {
        let processor = Phi4MMInputsProcessor {
            audio_compression_rate: 8,
            audio_downsample_rate: 1,
            audio_feat_stride: 1,
            eightk_method: "fillzero".to_string(),
        };
        let wav = (0..280)
            .map(|index| ((index % 17) - 8) as f32 / 16.)
            .collect::<Vec<_>>();
        let features = processor
            .extract_mel_spectrogram_rustfft(&wav, 8000)
            .unwrap();

        assert_eq!(features.len(), 2);
        assert!(features.iter().all(|frame| frame.len() == 80));
        let expected = [
            (0, 0, 13.901323),
            (0, 5, 14.708267),
            (0, 20, 15.498803),
            (0, 40, 16.623417),
            (0, 60, 24.281_36),
            (0, 65, 0.),
            (0, 79, 0.),
            (1, 0, 12.2515),
            (1, 5, 13.123133),
            (1, 20, 14.566305),
            (1, 40, 17.087433),
            (1, 60, 24.282797),
            (1, 65, 0.),
            (1, 79, 0.),
        ];
        for (frame, bin, expected) in expected {
            let actual = features[frame][bin];
            assert!(
                (actual - expected).abs() < 0.001,
                "frame {frame}, bin {bin}: expected {expected}, got {actual}"
            );
        }
        let sum = features.iter().flatten().sum::<f32>();
        assert!((sum - 2261.8604).abs() < 0.01, "feature sum: {sum}");
    }

    #[test]
    fn padded_crop_masks_remain_visible() {
        let mask = Tensor::zeros((1, 2, 2), candle_core::DType::U32, &Device::Cpu).unwrap();
        let padded = pad_phi4_image_mask(mask, 3)
            .unwrap()
            .to_vec3::<u32>()
            .unwrap();
        assert_eq!(padded[0], vec![vec![0, 0], vec![0, 0]]);
        assert_eq!(padded[1], vec![vec![1, 1], vec![1, 1]]);
        assert_eq!(padded[2], vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn encoder_hashes_include_preprocessing_shape_context() {
        let wide = image::DynamicImage::new_rgb8(4, 1);
        let tall = image::DynamicImage::new_rgb8(1, 4);
        assert_ne!(phi4_image_hash(1, &wide), phi4_image_hash(1, &tall));
        let mono = crate::AudioInput {
            samples: vec![0.; 4],
            sample_rate: 16000,
            channels: 1,
        };
        let stereo = crate::AudioInput {
            channels: 2,
            ..mono.clone()
        };
        assert_ne!(phi4_audio_hash(2, &mono), phi4_audio_hash(2, &stereo));
    }

    #[test]
    fn splices_unequal_text_image_audio_and_mixed_requests() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        let audio = AUDIO_SPECIAL_TOKEN_ID as u32;
        let text = phi4_request_layout(1, &[1, 2, 3], 0..3, &[]).unwrap();
        let image_request = phi4_request_layout(
            2,
            &[4, image, image, 5],
            0..4,
            &[feature(MultimodalKind::Image, 10, 0, 1, 2)],
        )
        .unwrap();
        let audio_request = phi4_request_layout(
            3,
            &[audio, audio, audio],
            0..3,
            &[feature(MultimodalKind::Audio, 20, 0, 0, 3)],
        )
        .unwrap();
        let mixed_request = phi4_request_layout(
            4,
            &[image, audio, audio],
            0..3,
            &[
                feature(MultimodalKind::Image, 30, 0, 0, 1),
                feature(MultimodalKind::Audio, 40, 0, 1, 2),
            ],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 10,
            },
            vec![output(&[10., 11.])],
        );
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Audio,
                hash: 20,
            },
            vec![output(&[20., 21., 22.]), output(&[120., 121., 122.])],
        );
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 30,
            },
            vec![output(&[30.])],
        );
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Audio,
                hash: 40,
            },
            vec![output(&[40., 41.]), output(&[140., 141.])],
        );

        assert_eq!(
            splice(
                &[text, image_request, audio_request, mixed_request],
                outputs
            ),
            vec![0., 0., 0., 0., 10., 11., 0., 20., 21., 22., 30., 140., 141.]
        );
    }

    #[test]
    fn same_audio_hash_selects_mode_per_request() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        let audio = AUDIO_SPECIAL_TOKEN_ID as u32;
        let speech = phi4_request_layout(
            1,
            &[audio, audio],
            0..2,
            &[feature(MultimodalKind::Audio, 77, 0, 0, 2)],
        )
        .unwrap();
        let vision = phi4_request_layout(
            2,
            &[image, audio, audio],
            0..3,
            &[
                feature(MultimodalKind::Image, 88, 0, 0, 1),
                feature(MultimodalKind::Audio, 77, 0, 1, 2),
            ],
        )
        .unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Audio,
                hash: 77,
            },
            vec![output(&[7., 8.]), output(&[70., 80.])],
        );
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 88,
            },
            vec![output(&[9.])],
        );

        assert_eq!(
            splice(&[speech, vision], outputs),
            vec![7., 8., 9., 70., 80.]
        );
    }

    #[test]
    fn causal_item_supports_before_partial_and_after_queries() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        let tokens = [1, image, image, image, 2];
        let media = [feature(MultimodalKind::Image, 5, 0, 1, 3)];
        let before = phi4_request_layout(1, &tokens, 0..1, &media).unwrap();
        let partial = phi4_request_layout(2, &tokens, 2..4, &media).unwrap();
        let after = phi4_request_layout(3, &tokens, 4..5, &media).unwrap();
        let mut outputs = HashMap::new();
        outputs.insert(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 5,
            },
            vec![output(&[5., 6., 7.])],
        );
        assert_eq!(
            splice(&[before, partial, after], outputs),
            vec![0., 6., 7., 0.]
        );
    }

    #[test]
    fn rejects_malformed_layout_metadata() {
        let image = IMAGE_SPECIAL_TOKEN_ID as u32;
        let mut missing_hash = feature(MultimodalKind::Image, 1, 0, 0, 1);
        missing_hash.hashes.clear();
        assert!(phi4_request_layout(1, &[image], 0..1, &[missing_hash]).is_err());
        assert!(
            phi4_request_layout(1, &[1], 0..1, &[feature(MultimodalKind::Image, 1, 0, 0, 1)])
                .is_err()
        );
        assert!(phi4_request_layout(
            1,
            &[image],
            0..1,
            &[feature(MultimodalKind::Image, 1, 0, 0, 2)]
        )
        .is_err());
        assert!(phi4_request_layout(1, &[image], 0..2, &[]).is_err());
    }
}
