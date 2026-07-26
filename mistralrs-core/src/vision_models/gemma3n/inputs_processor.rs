#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use crate::paged_attention::block_hash::{MultiModalFeature, MultimodalKind};
use std::{any::Any, collections::HashSet, ops::Range, sync::Arc};

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
        multimodal_layout::{
            MultimodalEmbeddingMap, MultimodalEncoderKey, MultimodalItemLayout,
            PackedMultimodalLayout, RequestMultimodalLayout,
        },
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

#[derive(Debug, PartialEq, Eq)]
struct Gemma3nActiveItem {
    local_index: usize,
    hash: u64,
    source: Range<usize>,
}

struct Gemma3nImageBatch {
    pixel_values: Tensor,
    hashes: Vec<u64>,
    source_ranges: Vec<Range<usize>>,
}

struct Gemma3nAudioBatch {
    mel: Tensor,
    mask: Tensor,
    hashes: Vec<u64>,
    source_ranges: Vec<Range<usize>>,
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

fn gemma3n_active_items(
    features: &[MultiModalFeature],
    kind: MultimodalKind,
    query: Range<usize>,
    available_items: usize,
) -> Result<Vec<Gemma3nActiveItem>> {
    if query.start > query.end {
        candle_core::bail!("Gemma 3n active query range is reversed");
    }
    let features = features
        .iter()
        .filter(|feature| feature.kind == kind)
        .collect::<Vec<_>>();
    let total_items = features
        .iter()
        .map(|feature| feature.item_range.end)
        .max()
        .unwrap_or(0);
    if available_items > total_items {
        candle_core::bail!(
            "Gemma 3n has {available_items} retained {kind:?} items but metadata describes {total_items}"
        );
    }
    let retained_start = total_items.saturating_sub(available_items);
    let mut local_indices = HashSet::new();
    let mut active = Vec::new();

    for feature in features {
        if feature.hashes.len() != 1 || feature.item_range.len() != 1 {
            candle_core::bail!("Gemma 3n requires one {kind:?} item and hash per placeholder span");
        }
        if feature.item_range.start < retained_start {
            continue;
        }
        let local_index = feature.item_range.start - retained_start;
        if local_index >= available_items || !local_indices.insert(local_index) {
            candle_core::bail!("Gemma 3n {kind:?} item coordinates are invalid");
        }
        let end = feature
            .offset
            .checked_add(feature.length)
            .ok_or_else(|| candle_core::Error::msg("Gemma 3n placeholder range overflow"))?;
        let overlap_start = feature.offset.max(query.start);
        let overlap_end = end.min(query.end);
        if overlap_start < overlap_end {
            active.push(Gemma3nActiveItem {
                local_index,
                hash: feature.hashes[0],
                source: overlap_start - feature.offset..overlap_end - feature.offset,
            });
        }
    }

    Ok(active)
}

fn gemma3n_mm_features(
    ranges: &[(usize, usize)],
    hashes: &[u64],
    kind: MultimodalKind,
) -> Vec<MultiModalFeature> {
    let mut features = build_mm_features_from_ranges(ranges, hashes, kind);
    for feature in &mut features {
        feature.splittable = true;
    }
    features
}

fn gemma3n_request_layout(
    sequence_id: usize,
    tokens: &[u32],
    query: Range<usize>,
    features: &[MultiModalFeature],
) -> Result<RequestMultimodalLayout> {
    if query.start > query.end || query.end > tokens.len() {
        candle_core::bail!(
            "Gemma 3n packed query {query:?} exceeds {} tokens",
            tokens.len()
        );
    }
    let mut items = Vec::with_capacity(features.len());
    for feature in features {
        if feature.hashes.len() != 1 || feature.item_range.len() != 1 {
            candle_core::bail!(
                "Gemma 3n packed prefill requires one encoder item per placeholder span"
            );
        }
        let token = match feature.kind {
            MultimodalKind::Image => IMAGE_TOKEN_ID,
            MultimodalKind::Audio => AUDIO_TOKEN_ID,
            MultimodalKind::Video => {
                candle_core::bail!("Gemma 3n does not support video layout items")
            }
        };
        let end = feature
            .offset
            .checked_add(feature.length)
            .ok_or_else(|| candle_core::Error::msg("Gemma 3n placeholder range overflow"))?;
        if feature.length == 0 || end > tokens.len() {
            candle_core::bail!("Gemma 3n placeholder range exceeds the prompt");
        }
        if tokens[feature.offset..end]
            .iter()
            .any(|&value| value != token)
        {
            candle_core::bail!(
                "Gemma 3n {:?} placeholder contains a non-placeholder token",
                feature.kind
            );
        }
        let placeholder = feature.offset..end;
        items.push(MultimodalItemLayout::new(
            MultimodalEncoderKey {
                kind: feature.kind,
                hash: feature.hashes[0],
            },
            feature.item_range.start,
            placeholder.clone(),
            feature.attention_policy,
            vec![MultimodalEmbeddingMap::contiguous(placeholder, 0, 0)?],
        )?);
    }
    Ok(RequestMultimodalLayout {
        sequence_id,
        query,
        items,
    })
}

fn gemma3n_packed_layout(
    input_seqs: &[&mut Sequence],
    query_lens: &[usize],
) -> Result<PackedMultimodalLayout> {
    if input_seqs.len() != query_lens.len() {
        candle_core::bail!("Gemma 3n packed multimodal metadata length mismatch");
    }
    let requests = input_seqs
        .iter()
        .zip(query_lens)
        .map(|(seq, &query_len)| {
            if seq.is_chunked_prefill_view()
                || seq.prefix_cache_len() != 0
                || query_len != seq.get_toks().len()
            {
                candle_core::bail!(
                    "Gemma 3n packed multimodal prefill requires the complete uncached prompt"
                );
            }
            gemma3n_request_layout(*seq.id(), seq.get_toks(), 0..query_len, seq.mm_features())
        })
        .collect::<Result<Vec<_>>>()?;
    PackedMultimodalLayout::new(&requests)
}

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

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> anyhow::Result<()> {
        let tokenizer = tokenizer.ok_or_else(|| {
            anyhow::Error::msg("Gemma3nImageProcessor requires a specified tokenizer.")
        })?;
        self.prepare_prompt_plans(&tokenizer, input_seqs, paged_attn_metadata)
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
            anyhow::Error::msg("Gemma3nImageProcessor requires a specified tokenizer.")
        })?;

        let config = other_config.expect("Need a PreProcessorConfig config.");
        let preprocessor_config: &PreProcessorConfig =
            config.downcast_ref().expect("Downcast failed.");

        if is_prompt {
            self.prepare_prompt_plans(&tokenizer, input_seqs, paged_attn_metadata.as_mut())
                .map_err(anyhow::Error::new)?;
        }

        let audio_batch = is_prompt
            .then(|| self.process_audio_batch(input_seqs, preprocessor_config, device))
            .transpose()
            .map_err(anyhow::Error::new)?
            .flatten();

        let image_batch = is_prompt
            .then(|| self.process_image_batch(input_seqs, preprocessor_config, device))
            .transpose()
            .map_err(anyhow::Error::new)?
            .flatten();

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

        let packed_layout = if is_prompt && flash_meta.packed {
            let query_lens = paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    anyhow::Error::msg("packed Gemma 3n prefill requires logical query lengths")
                })?;
            let layout =
                gemma3n_packed_layout(input_seqs, query_lens).map_err(anyhow::Error::new)?;
            if layout.token_count() != input.dim(1)? {
                anyhow::bail!(
                    "Gemma 3n packed layout has {} tokens but input has {}",
                    layout.token_count(),
                    input.dim(1)?
                );
            }
            Some(layout)
        } else {
            None
        };
        let (pixel_values, image_hashes, image_source_ranges) = match image_batch {
            Some(batch) => (Some(batch.pixel_values), batch.hashes, batch.source_ranges),
            None => (None, Vec::new(), Vec::new()),
        };
        let (audio_mel, audio_mel_mask, audio_hashes, audio_source_ranges) = match audio_batch {
            Some(batch) => (
                Some(batch.mel),
                Some(batch.mask),
                batch.hashes,
                batch.source_ranges,
            ),
            None => (None, None, Vec::new(), Vec::new()),
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
                image_source_ranges,
                audio_hashes,
                audio_source_ranges,
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
    }
}

impl Gemma3nImageProcessor {
    fn prepare_prompt_plans(
        &self,
        tokenizer: &Tokenizer,
        input_seqs: &mut [&mut Sequence],
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        for seq in input_seqs {
            if seq.multimodal.has_changed_prompt && !seq.mm_features().is_empty() {
                continue;
            }
            let image_count = seq.images().map_or(0, <[_]>::len);
            let audio_count = seq.audios().map_or(0, <[_]>::len);
            let image_hashes = seq.multimodal.image_hashes().unwrap_or_default().to_vec();
            let audio_hashes = seq.multimodal.audio_hashes().unwrap_or_default().to_vec();
            if image_hashes.len() != image_count {
                candle_core::bail!(
                    "Gemma 3n has {image_count} images but {} image hashes",
                    image_hashes.len()
                );
            }
            if audio_hashes.len() != audio_count {
                candle_core::bail!(
                    "Gemma 3n has {audio_count} audios but {} audio hashes",
                    audio_hashes.len()
                );
            }
            if image_count > 0 && !self.supports_images {
                candle_core::bail!("This Gemma 3n processor does not support images");
            }
            if audio_count > 0 && !self.supports_audio {
                candle_core::bail!("This Gemma 3n processor does not support audio");
            }

            let mut prompt = tokenizer
                .decode(seq.get_toks(), false)
                .map_err(candle_core::Error::wrap)?;
            let raw_image_count = prompt.match_indices(IMAGE_TOKEN).count();
            let raw_audio_count = prompt.match_indices(AUDIO_TOKEN).count();
            if raw_image_count != image_count {
                candle_core::bail!(
                    "Gemma 3n has {raw_image_count} image placeholders but {image_count} images"
                );
            }
            if raw_audio_count != audio_count {
                candle_core::bail!(
                    "Gemma 3n has {raw_audio_count} audio placeholders but {audio_count} audios"
                );
            }
            if image_count == 0 && audio_count == 0 {
                continue;
            }

            if image_count > 0 {
                prompt = prompt.replace(IMAGE_TOKEN, &self.full_image_sequence);
            }
            if audio_count > 0 {
                prompt = prompt.replace(AUDIO_TOKEN, &self.create_full_audio_sequence());
            }
            let ids = tokenizer
                .encode_fast(prompt.clone(), false)
                .map_err(candle_core::Error::wrap)?
                .get_ids()
                .to_vec();
            let image_ranges = find_image_placeholder_ranges(&ids, IMAGE_TOKEN_ID);
            let audio_ranges = find_image_placeholder_ranges(&ids, AUDIO_TOKEN_ID);
            if image_ranges.len() != image_count
                || image_ranges
                    .iter()
                    .any(|&(_, length)| length != self.image_token_count())
            {
                candle_core::bail!("Gemma 3n expanded image placeholder metadata is invalid");
            }
            if audio_ranges.len() != audio_count
                || audio_ranges
                    .iter()
                    .any(|&(_, length)| length != self.audio_seq_length)
            {
                candle_core::bail!("Gemma 3n expanded audio placeholder metadata is invalid");
            }

            let mut features =
                gemma3n_mm_features(&image_ranges, &image_hashes, MultimodalKind::Image);
            features.extend(gemma3n_mm_features(
                &audio_ranges,
                &audio_hashes,
                MultimodalKind::Audio,
            ));
            features.sort_by_key(|feature| feature.offset);
            seq.set_initial_prompt(prompt);
            seq.set_toks_and_reallocate(ids, paged_attn_metadata.as_deref_mut());
            seq.set_mm_features(features);
            seq.multimodal.has_changed_prompt = true;
        }
        Ok(())
    }

    fn image_token_count(&self) -> usize {
        self.full_image_sequence.matches(IMAGE_TOKEN).count()
    }

    fn active_query(seq: &Sequence) -> Result<Range<usize>> {
        if seq.is_chunked_prefill_view() {
            return seq.active_prompt_query_range().ok_or_else(|| {
                candle_core::Error::msg("Gemma 3n chunk is missing its active query range")
            });
        }
        let len = seq.get_toks().len();
        Ok(seq.prefix_cache_len().min(len)..len)
    }

    fn process_image_batch(
        &self,
        input_seqs: &mut [&mut Sequence],
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<Option<Gemma3nImageBatch>> {
        let mut images = Vec::new();
        let mut hashes = Vec::new();
        let mut source_ranges = Vec::new();
        for seq in input_seqs {
            let retained = seq.clone_images().unwrap_or_default();
            if retained.is_empty() {
                continue;
            }
            let active = gemma3n_active_items(
                seq.mm_features(),
                MultimodalKind::Image,
                Self::active_query(seq)?,
                retained.len(),
            )?;
            for item in active {
                images.push(retained[item.local_index].clone());
                hashes.push(item.hash);
                source_ranges.push(item.source);
            }
            if !seq.is_chunked_prefill_view() {
                let _ = seq.take_images();
            }
        }
        if images.is_empty() {
            return Ok(None);
        }

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
        } = self.preprocess(images, vec![], config, device, (usize::MAX, usize::MAX))?;
        if pixel_values.dim(0)? != hashes.len() || hashes.len() != source_ranges.len() {
            candle_core::bail!("Gemma 3n active image metadata length mismatch");
        }
        Ok(Some(Gemma3nImageBatch {
            pixel_values,
            hashes,
            source_ranges,
        }))
    }

    fn process_audio_batch(
        &self,
        input_seqs: &mut [&mut Sequence],
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<Option<Gemma3nAudioBatch>> {
        let mut audios = Vec::new();
        let mut hashes = Vec::new();
        let mut source_ranges = Vec::new();
        for seq in input_seqs {
            let retained = seq.clone_audios().unwrap_or_default();
            if retained.is_empty() {
                continue;
            }
            let active = gemma3n_active_items(
                seq.mm_features(),
                MultimodalKind::Audio,
                Self::active_query(seq)?,
                retained.len(),
            )?;
            for item in active {
                audios.push(retained[item.local_index].clone());
                hashes.push(item.hash);
                source_ranges.push(item.source);
            }
            if !seq.is_chunked_prefill_view() {
                let _ = seq.take_audios();
            }
        }
        if audios.is_empty() {
            return Ok(None);
        }

        let processor = AudioProcessor::new(config);
        let mut mels = Vec::with_capacity(audios.len());
        let mut masks = Vec::with_capacity(audios.len());
        for audio in audios {
            let (mel, mask) = processor
                .process_audio(&audio, device)
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            if mel.dim(0)? != 1
                || mask.dim(0)? != 1
                || mel.dim(1)? == 0
                || mel.dim(1)? != mask.dim(1)?
            {
                candle_core::bail!("Gemma 3n audio preprocessing returned invalid dimensions");
            }
            mels.push(mel);
            masks.push(mask);
        }
        let max_frames = mels
            .iter()
            .map(|mel| mel.dim(1))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap();
        let mut padded_mels = Vec::with_capacity(mels.len());
        let mut padded_masks = Vec::with_capacity(masks.len());
        for (mel, mask) in mels.into_iter().zip(masks) {
            let frames = mel.dim(1)?;
            padded_mels.push(mel.pad_with_zeros(1, 0, max_frames - frames)?);
            if frames == max_frames {
                padded_masks.push(mask);
            } else {
                let padding = Tensor::ones((1, max_frames - frames), mask.dtype(), mask.device())?;
                padded_masks.push(Tensor::cat(&[mask, padding], 1)?);
            }
        }
        if hashes.len() != padded_mels.len() || hashes.len() != source_ranges.len() {
            candle_core::bail!("Gemma 3n active audio metadata length mismatch");
        }
        Ok(Some(Gemma3nAudioBatch {
            mel: Tensor::cat(&padded_mels, 0)?,
            mask: Tensor::cat(&padded_masks, 0)?,
            hashes,
            source_ranges,
        }))
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::{DType, Device};

    use crate::{
        paged_attention::block_hash::MultimodalAttentionPolicy,
        vision_models::multimodal_layout::MultimodalEncoderOutputs,
    };

    use super::*;

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
            splittable: true,
        }
    }

    fn output(values: &[f32]) -> Tensor {
        Tensor::from_slice(values, (values.len(), 1), &Device::Cpu).unwrap()
    }

    #[test]
    fn causal_placeholder_metadata_is_splittable() {
        let features = gemma3n_mm_features(&[(2, 4), (9, 3)], &[11, 12], MultimodalKind::Image);

        assert_eq!(features.len(), 2);
        assert!(features.iter().all(|feature| feature.splittable));
    }

    #[test]
    fn active_items_keep_exact_prefix_and_chunk_coordinates() {
        let features = vec![
            feature(MultimodalKind::Image, 11, 0, 2, 4),
            feature(MultimodalKind::Audio, 21, 0, 7, 2),
            feature(MultimodalKind::Image, 12, 1, 10, 3),
        ];
        assert_eq!(
            gemma3n_active_items(&features, MultimodalKind::Image, 4..12, 2).unwrap(),
            vec![
                Gemma3nActiveItem {
                    local_index: 0,
                    hash: 11,
                    source: 2..4,
                },
                Gemma3nActiveItem {
                    local_index: 1,
                    hash: 12,
                    source: 0..2,
                },
            ]
        );
        assert!(
            gemma3n_active_items(&features, MultimodalKind::Image, 6..10, 2)
                .unwrap()
                .is_empty()
        );

        let suffix_features = vec![
            feature(MultimodalKind::Image, 31, 0, 0, 2),
            feature(MultimodalKind::Image, 32, 1, 5, 3),
            feature(MultimodalKind::Image, 33, 2, 10, 4),
        ];
        assert_eq!(
            gemma3n_active_items(&suffix_features, MultimodalKind::Image, 6..12, 2).unwrap(),
            vec![
                Gemma3nActiveItem {
                    local_index: 0,
                    hash: 32,
                    source: 1..3,
                },
                Gemma3nActiveItem {
                    local_index: 1,
                    hash: 33,
                    source: 0..2,
                },
            ]
        );
    }

    #[test]
    fn packed_layout_splices_heterogeneous_media_by_key() {
        let text_tokens = [1, 2];
        let image_tokens = [1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2];
        let audio_tokens = [AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID];
        let mixed_tokens = [IMAGE_TOKEN_ID, 9, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID];
        let requests = vec![
            gemma3n_request_layout(1, &text_tokens, 0..text_tokens.len(), &[]).unwrap(),
            gemma3n_request_layout(
                2,
                &image_tokens,
                0..image_tokens.len(),
                &[feature(MultimodalKind::Image, 11, 0, 1, 2)],
            )
            .unwrap(),
            gemma3n_request_layout(
                3,
                &audio_tokens,
                0..audio_tokens.len(),
                &[feature(MultimodalKind::Audio, 21, 0, 0, 3)],
            )
            .unwrap(),
            gemma3n_request_layout(
                4,
                &mixed_tokens,
                0..mixed_tokens.len(),
                &[
                    feature(MultimodalKind::Image, 31, 0, 0, 1),
                    feature(MultimodalKind::Audio, 32, 0, 2, 2),
                ],
            )
            .unwrap(),
        ];
        let layout = PackedMultimodalLayout::new(&requests).unwrap();
        let text = Tensor::zeros((1, layout.token_count(), 1), DType::F32, &Device::Cpu).unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 11,
                },
                vec![output(&[11., 12.])],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Audio,
                    hash: 21,
                },
                vec![output(&[21., 22., 23.])],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Image,
                    hash: 31,
                },
                vec![output(&[31.])],
            ),
            (
                MultimodalEncoderKey {
                    kind: MultimodalKind::Audio,
                    hash: 32,
                },
                vec![output(&[32., 33.])],
            ),
        ]);

        let result = layout
            .splice_embeddings(&text, &outputs)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(
            result,
            vec![0., 0., 0., 11., 12., 0., 21., 22., 23., 31., 0., 32., 33.]
        );
    }

    #[test]
    fn partial_layout_uses_encoder_source_offsets() {
        let tokens = [
            1,
            2,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            3,
        ];
        let request = gemma3n_request_layout(
            1,
            &tokens,
            4..8,
            &[feature(MultimodalKind::Image, 11, 0, 2, 6)],
        )
        .unwrap();
        let layout = PackedMultimodalLayout::new(&[request]).unwrap();
        let text = Tensor::zeros((1, 4, 1), DType::F32, &Device::Cpu).unwrap();
        let outputs: MultimodalEncoderOutputs = HashMap::from([(
            MultimodalEncoderKey {
                kind: MultimodalKind::Image,
                hash: 11,
            },
            vec![output(&[10., 11., 12., 13., 14., 15.])],
        )]);

        assert_eq!(
            layout
                .splice_embeddings(&text, &outputs)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![12., 13., 14., 15.]
        );
    }

    #[test]
    fn malformed_gemma3n_layout_metadata_fails_closed() {
        let tokens = [IMAGE_TOKEN_ID, IMAGE_TOKEN_ID];
        let mut missing_hash = feature(MultimodalKind::Image, 11, 0, 0, 2);
        missing_hash.hashes.clear();
        assert!(gemma3n_request_layout(1, &tokens, 0..2, &[missing_hash]).is_err());

        let mut grouped = feature(MultimodalKind::Image, 11, 0, 0, 2);
        grouped.item_range = 0..2;
        assert!(gemma3n_request_layout(1, &tokens, 0..2, &[grouped]).is_err());

        assert!(gemma3n_request_layout(
            1,
            &[IMAGE_TOKEN_ID, 7],
            0..2,
            &[feature(MultimodalKind::Image, 11, 0, 0, 2)]
        )
        .is_err());
        assert!(gemma3n_request_layout(
            1,
            &tokens,
            0..2,
            &[feature(MultimodalKind::Video, 11, 0, 0, 2)]
        )
        .is_err());
        assert!(gemma3n_request_layout(1, &tokens, 0..3, &[]).is_err());
        assert!(gemma3n_active_items(
            &[
                feature(MultimodalKind::Image, 11, 0, 0, 1),
                feature(MultimodalKind::Image, 12, 0, 1, 1),
            ],
            MultimodalKind::Image,
            0..2,
            1,
        )
        .is_err());
    }
}
