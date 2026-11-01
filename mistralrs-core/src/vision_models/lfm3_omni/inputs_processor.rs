use std::{any::Any, ops::Range, sync::Arc};

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType, InputsProcessorValidationError, MessagesAction, Processor,
    },
    sequence::{find_placeholder_delimited_ranges, Sequence},
    vision_models::{
        grid_processor::{GridProcessor, GridVisionSpecificArgs},
        lfm3_vl::{Config as VisionConfig, Lfm3VlProcessor},
        preprocessor_config::PreProcessorConfig,
        qwen2vl::validated_mm_features,
        ModelInputs,
    },
};

use super::audio_processing::AudioProcessor;

pub(crate) const AUDIO_TOKEN: &str = "<|audio|>";
pub(crate) const AUDIO_OUT_TOKEN: &str = "<|audio_out|>";
pub(crate) const AUDIO_START_TOKEN: &str = "<|audio_start|>";
pub(crate) const MIXED_START_TOKEN: &str = "<|mixed_start|>";
const IMAGE_TOKEN: &str = "<|image|>";
const VIDEO_TOKEN: &str = "<|video|>";
const AUDIO_DELIMITER_TOKENS: usize = 2;

pub(crate) struct AudioItem {
    pub features: Tensor,
    pub hash: u64,
    pub batch_index: usize,
    pub token_range: Range<usize>,
}

pub(crate) struct Lfm3OmniSpecificArgs {
    pub vision: Option<GridVisionSpecificArgs>,
    pub audio: Vec<AudioItem>,
    pub sequence_ids: Vec<usize>,
}

pub(crate) struct Lfm3OmniProcessor {
    vision: Lfm3VlProcessor,
    audio: AudioProcessor,
}

impl Lfm3OmniProcessor {
    pub(crate) fn new(
        config: &VisionConfig,
        preprocessor: &PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Self {
        Self {
            vision: Lfm3VlProcessor::new(config, preprocessor, max_edge),
            audio: AudioProcessor::new(),
        }
    }
}

impl Processor for Lfm3OmniProcessor {
    // Processor returns Arc<dyn InputsProcessor>, whose trait does not require Send or Sync.
    #[allow(clippy::arc_with_non_send_sync)]
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Lfm3OmniInputsProcessor {
            vision: self.vision.inputs_processor(),
            audio: self.audio.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[
            IMAGE_TOKEN,
            VIDEO_TOKEN,
            AUDIO_TOKEN,
            AUDIO_OUT_TOKEN,
            GridProcessor::PLACEHOLDER,
        ]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

struct Lfm3OmniInputsProcessor {
    vision: Arc<dyn InputsProcessor>,
    audio: AudioProcessor,
}

#[derive(Clone, Copy)]
pub(crate) enum AudioPlaceholderStyle {
    Bare,
    VisionDelimited,
}

pub(crate) struct AudioInputLayout<'a> {
    pub inputs: &'a ModelInputs,
    pub seq_indices: &'a [usize],
    pub token_id: u32,
    pub style: AudioPlaceholderStyle,
}

pub(crate) fn process_audio_items(
    processor: &AudioProcessor,
    layout: AudioInputLayout<'_>,
    input_seqs: &[&mut Sequence],
    device: &Device,
) -> Result<Vec<AudioItem>> {
    let rows = layout
        .inputs
        .input_ids
        .to_device(&Device::Cpu)?
        .to_vec2::<u32>()?;
    let mut packed_offset = 0;
    let mut audio = Vec::new();
    for (batch_index, &seq_index) in layout.seq_indices.iter().enumerate() {
        let seq = &input_seqs[seq_index];
        let query = audio_query(seq);
        let (row_index, offset, tokens) = if layout.inputs.flash_meta.packed {
            let query_len = layout
                .inputs
                .paged_attn_meta
                .as_ref()
                .and_then(|metadata| metadata.query_lens.as_ref())
                .context("Packed LFM3 audio input requires query lengths")?[batch_index];
            let offset = packed_offset;
            packed_offset += query_len;
            (0, offset, &rows[0][offset..packed_offset])
        } else {
            (batch_index, 0, rows[batch_index].as_slice())
        };
        let positions = tokens
            .iter()
            .enumerate()
            .filter_map(|(position, &token)| (token == layout.token_id).then_some(position))
            .collect::<Vec<_>>();
        let items = seq
            .mm_features()
            .iter()
            .filter(|feature| {
                feature.kind == MultimodalKind::Audio && feature.overlaps(query.start, query.end)
            })
            .collect::<Vec<_>>();
        let delimiter_tokens = match layout.style {
            AudioPlaceholderStyle::Bare => 0,
            AudioPlaceholderStyle::VisionDelimited => AUDIO_DELIMITER_TOKENS,
        };
        let expected_tokens = items
            .iter()
            .map(|item| item.length - delimiter_tokens)
            .sum::<usize>();
        anyhow::ensure!(
            positions.len() == expected_tokens,
            "LFM3 audio spans do not match the active audio clips"
        );
        let audios = seq.audios().unwrap_or_default();
        let total_items = seq
            .mm_features()
            .iter()
            .filter(|feature| feature.kind == MultimodalKind::Audio)
            .map(|feature| feature.item_range.end)
            .max()
            .unwrap_or(0);
        let retained_start = total_items.saturating_sub(audios.len());
        // Bare audio clips can touch, so feature lengths preserve their boundaries.
        let mut cursor = 0;
        for item in items {
            let length = item.length - delimiter_tokens;
            let positions = &positions[cursor..cursor + length];
            let start = positions[0];
            anyhow::ensure!(
                positions.iter().copied().eq(start..start + length),
                "LFM3 audio clip placeholders must be contiguous"
            );
            cursor += length;
            let index = item
                .item_range
                .start
                .checked_sub(retained_start)
                .context("LFM3 audio clip was removed before encoding")?;
            let clip = audios.get(index).context("LFM3 audio clip is missing")?;
            let features = processor.process_audio(clip, device)?;
            anyhow::ensure!(
                AudioProcessor::num_encoder_tokens(features.dim(2)?) == length,
                "LFM3 audio encoder length does not match its placeholders"
            );
            audio.push(AudioItem {
                features,
                hash: item.hashes[0],
                batch_index: row_index,
                token_range: start + offset..start + offset + length,
            });
        }
    }
    Ok(audio)
}

struct MediaTokens {
    audio: u32,
    audio_out: u32,
    image: u32,
    video: u32,
    start: u32,
    end: u32,
    audio_start: u32,
    mixed_start: u32,
}

impl MediaTokens {
    fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |token: &str| {
            tokenizer
                .token_to_id(token)
                .with_context(|| format!("LFM3-Omni tokenizer is missing {token}"))
        };
        Ok(Self {
            audio: id(AUDIO_TOKEN)?,
            audio_out: id(AUDIO_OUT_TOKEN)?,
            image: id(IMAGE_TOKEN)?,
            video: id(VIDEO_TOKEN)?,
            start: id(GridProcessor::VISION_START)?,
            end: id(GridProcessor::VISION_END)?,
            audio_start: id(AUDIO_START_TOKEN)?,
            mixed_start: id(MIXED_START_TOKEN)?,
        })
    }
}

fn expand_audio_placeholders(
    tokens: &[u32],
    media: &MediaTokens,
    audio_lengths: &[usize],
) -> Result<Vec<u32>> {
    if tokens.contains(&media.audio_out) {
        return Err(InputsProcessorValidationError(
            "LFM3-Omni speech history with <|audio_out|> requires audio codes and is not supported"
                .to_owned(),
        )
        .into());
    }
    let placeholders = tokens.iter().filter(|&&token| token == media.audio).count();
    if placeholders != audio_lengths.len() {
        return Err(InputsProcessorValidationError(format!(
            "LFM3-Omni prompt contains {placeholders} audio placeholders for {} clips",
            audio_lengths.len()
        ))
        .into());
    }
    let mut lengths = audio_lengths.iter();
    let mut expanded = Vec::new();
    for (index, &token) in tokens.iter().enumerate() {
        if token != media.audio && token != media.image && token != media.video {
            expanded.push(token);
            continue;
        }
        let wrapped = index > 0
            && tokens[index - 1] == media.start
            && tokens.get(index + 1) == Some(&media.end);
        if !wrapped {
            expanded.push(media.start);
        }
        let length = if token == media.audio {
            *lengths.next().expect("audio placeholder count checked")
        } else {
            1
        };
        expanded.extend(std::iter::repeat_n(token, length));
        if !wrapped {
            expanded.push(media.end);
        }
    }
    Ok(expanded)
}

fn audio_query(seq: &Sequence) -> Range<usize> {
    seq.active_prompt_query_range().unwrap_or_else(|| {
        let len = seq.prompt_position_source_toks().len();
        seq.prefix_cache_len().min(len)..len
    })
}

impl InputsProcessor for Lfm3OmniInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        device: &Device,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        if input_seqs
            .iter()
            .all(|seq| seq.multimodal.has_changed_prompt)
        {
            return Ok(());
        }
        let tokenizer = tokenizer.context("LFM3-Omni requires a tokenizer")?;
        let media = MediaTokens::new(&tokenizer)?;
        let prepare = input_seqs
            .iter()
            .map(|seq| !seq.multimodal.has_changed_prompt)
            .collect::<Vec<_>>();
        for (seq, &prepare) in input_seqs.iter_mut().zip(&prepare) {
            if !prepare {
                continue;
            }
            let lengths = seq
                .audios()
                .unwrap_or_default()
                .iter()
                .map(AudioProcessor::num_tokens_for_audio)
                .collect::<Result<Vec<_>>>()?;
            let mut expanded = expand_audio_placeholders(seq.get_toks(), &media, &lengths)?;
            if let Some(output) = seq.audio_output() {
                let marker = match output.mode {
                    crate::ChatAudioMode::Audio => media.audio_start,
                    crate::ChatAudioMode::TextAudio => media.mixed_start,
                };
                if expanded.last() != Some(&marker) {
                    expanded.push(marker);
                }
            }
            seq.set_toks_and_reallocate(expanded, paged_attn_metadata.as_deref_mut());
        }
        self.vision.prepare_for_paged_prompt_planning(
            Some(tokenizer.clone()),
            input_seqs,
            device,
            other_config,
            paged_attn_metadata,
        )?;
        for (seq, prepare) in input_seqs.iter_mut().zip(prepare) {
            if !prepare {
                continue;
            }
            let ranges = find_placeholder_delimited_ranges(
                seq.get_toks(),
                media.audio,
                media.start,
                media.end,
            );
            let mut features = seq.mm_features().to_vec();
            features.extend(validated_mm_features(
                &ranges,
                seq.audio_hashes().unwrap_or_default(),
                MultimodalKind::Audio,
            )?);
            features.sort_by_key(|feature| feature.offset);
            seq.set_mm_features(features);
            seq.set_initial_prompt(
                tokenizer
                    .decode(seq.get_toks(), false)
                    .map_err(anyhow::Error::msg)?,
            );
            seq.multimodal.has_changed_prompt = true;
        }
        Ok(())
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
    ) -> Result<InputProcessorOutput> {
        if is_prompt {
            self.prepare_for_paged_prompt_planning(
                tokenizer.clone(),
                input_seqs,
                device,
                other_config.clone(),
                paged_attn_metadata.as_mut(),
            )?;
        }
        let mut output = self.vision.process_inputs(
            tokenizer.clone(),
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
        )?;
        let mut inputs = *output
            .inputs
            .downcast::<ModelInputs>()
            .expect("vision inputs");
        let audio = if is_prompt && input_seqs.iter().any(|seq| seq.has_audios()) {
            let tokenizer = tokenizer.context("LFM3-Omni requires a tokenizer")?;
            process_audio_items(
                &self.audio,
                AudioInputLayout {
                    inputs: &inputs,
                    seq_indices: &output.seq_indices,
                    token_id: MediaTokens::new(&tokenizer)?.audio,
                    style: AudioPlaceholderStyle::VisionDelimited,
                },
                input_seqs,
                device,
            )?
        } else {
            Vec::new()
        };
        let vision = *inputs
            .model_specific_args
            .downcast::<GridVisionSpecificArgs>()
            .expect("grid vision arguments");
        let sequence_ids = output
            .seq_indices
            .iter()
            .map(|&index| *input_seqs[index].id())
            .collect();
        inputs.model_specific_args = Box::new(Lfm3OmniSpecificArgs {
            vision: Some(vision),
            audio,
            sequence_ids,
        });
        output.inputs = Box::new(inputs);
        Ok(output)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit, AddedToken,
    };
    use tokio::sync::{mpsc::channel, Mutex};

    use crate::{
        sampler::Sampler,
        sequence::{MultimodalData, SeqStepType, SequenceGroup, SequenceRecognizer},
        vision_models::grid_processor::GridProcessorOptions,
        AudioInput,
    };

    const AUDIO_PROMPT: &str = "x <|audio|> x <|audio|> x";

    pub(crate) fn tokenizer() -> Arc<Tokenizer> {
        let model = WordLevel::builder()
            .vocab(
                [("[UNK]".to_owned(), 0), ("x".to_owned(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.add_special_tokens(
            &[
                AUDIO_TOKEN,
                AUDIO_OUT_TOKEN,
                "<|audio_start|>",
                "<|mixed_start|>",
                IMAGE_TOKEN,
                VIDEO_TOKEN,
                GridProcessor::VISION_START,
                GridProcessor::VISION_END,
            ]
            .map(|token| AddedToken::from(token, true)),
        );
        Arc::new(tokenizer)
    }

    pub(crate) fn sequence(tokenizer: &Tokenizer, prompt: &str, id: usize) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler = Sampler::new(
            None,
            0,
            None,
            None,
            None,
            None,
            None,
            32,
            1.0,
            0.0,
            HashMap::new(),
            vec![],
        )
        .unwrap();
        let tokens = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
        let audio = [400, 1_600]
            .into_iter()
            .map(|samples| AudioInput {
                samples: vec![0.0; samples],
                sample_rate: 16_000,
                channels: 1,
            })
            .collect();
        Sequence::new_waiting(
            tokens,
            prompt.to_owned(),
            id,
            0,
            1,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None))),
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            Some(audio),
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            false,
            vec![],
            None,
        )
    }

    pub(crate) fn process_prompt(
        processor: &dyn InputsProcessor,
        tokenizer: Arc<Tokenizer>,
        sequences: &mut [&mut Sequence],
        config: Option<Arc<dyn Any>>,
    ) -> Result<ModelInputs> {
        let output = processor.process_inputs(
            Some(tokenizer),
            sequences,
            true,
            false,
            &Device::Cpu,
            false,
            None,
            false,
            None,
            config,
            None,
            None,
        )?;
        Ok(*output.inputs.downcast::<ModelInputs>().unwrap())
    }

    fn processor() -> Lfm3OmniInputsProcessor {
        let vision = GridProcessor::with_options(
            None,
            GridProcessorOptions {
                image_token: IMAGE_TOKEN,
                video_token: VIDEO_TOKEN,
                mrope: false,
                config: Some(PreProcessorConfig {
                    do_resize: Some(false),
                    patch_size: Some(2),
                    merge_size: Some(2),
                    temporal_patch_size: Some(2),
                    ..Default::default()
                }),
                ..Default::default()
            },
        );
        Lfm3OmniInputsProcessor {
            vision: vision.inputs_processor(),
            audio: AudioProcessor::new(),
        }
    }

    fn media() -> MediaTokens {
        MediaTokens {
            audio: 10,
            audio_out: 15,
            image: 11,
            video: 12,
            start: 13,
            end: 14,
            audio_start: 16,
            mixed_start: 17,
        }
    }

    #[test]
    fn audio_expansion_preserves_existing_spans_and_wraps_bare_media() -> Result<()> {
        let tokens = [1, 13, 10, 14, 2, 11, 3, 10, 4, 12, 5];
        assert_eq!(
            expand_audio_placeholders(&tokens, &media(), &[2, 3])?,
            [1, 13, 10, 10, 14, 2, 13, 11, 14, 3, 13, 10, 10, 10, 14, 4, 13, 12, 14, 5]
        );
        Ok(())
    }

    #[test]
    fn adjacent_audio_clips_remain_separate_cache_features() -> Result<()> {
        let tokens = expand_audio_placeholders(&[10, 10], &media(), &[2, 1])?;
        let ranges = find_placeholder_delimited_ranges(&tokens, 10, 13, 14);
        let features = validated_mm_features(&ranges, &[41, 42], MultimodalKind::Audio)?;
        assert_eq!(ranges, [(0, 4), (4, 3)]);
        assert_eq!(features[0].item_range, 0..1);
        assert_eq!(features[1].item_range, 1..2);
        assert_eq!(features[0].hashes, [41]);
        assert_eq!(features[1].hashes, [42]);
        assert!(!features[0].splittable);
        Ok(())
    }

    #[test]
    fn audio_planning_is_idempotent_and_forward_inputs_keep_clip_identity() -> Result<()> {
        let tokenizer = tokenizer();
        let processor = processor();
        let mut seq = sequence(&tokenizer, AUDIO_PROMPT, 0);
        let hashes = seq.audio_hashes().unwrap().to_vec();
        let config: Arc<dyn Any> = Arc::new(PreProcessorConfig::default());
        for _ in 0..2 {
            processor.prepare_for_paged_prompt_planning(
                Some(tokenizer.clone()),
                &mut [&mut seq],
                &Device::Cpu,
                Some(config.clone()),
                None,
            )?;
            assert_eq!(seq.mm_features().len(), 2);
            assert_eq!(seq.get_toks().len(), 10);
        }
        let inputs = process_prompt(&processor, tokenizer, &mut [&mut seq], Some(config))?;
        let args = *inputs
            .model_specific_args
            .downcast::<Lfm3OmniSpecificArgs>()
            .unwrap();
        assert_eq!(args.audio.len(), 2);
        assert_eq!(args.audio[0].hash, hashes[0]);
        assert_eq!(args.audio[1].hash, hashes[1]);
        assert_eq!(args.audio[0].token_range, 2..3);
        assert_eq!(args.audio[1].token_range, 6..8);
        assert_eq!(args.audio[0].features.dims(), &[1, 128, 1]);
        assert_eq!(args.audio[1].features.dims(), &[1, 128, 9]);

        seq.set_prefix_cache_len(5);
        let query = audio_query(&seq);
        assert_eq!(query, 5..10);
        let active = seq
            .mm_features()
            .iter()
            .filter(|item| item.overlaps(query.start, query.end))
            .collect::<Vec<_>>();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].item_range, 1..2);
        assert_eq!(active[0].hashes, [hashes[1]]);
        Ok(())
    }

    #[test]
    fn mixed_media_preserves_offsets_and_speech_markers() -> Result<()> {
        let tokenizer = tokenizer();
        let processor = processor();
        for mode in [
            None,
            Some(crate::ChatAudioMode::Audio),
            Some(crate::ChatAudioMode::TextAudio),
        ] {
            let prompt = if mode.is_some() {
                "x <|audio|> <|image|> <|video|> <|audio|> x"
            } else {
                "x <|audio|> x <|image|> x <|audio|> x"
            };
            let mut seq = sequence(&tokenizer, prompt, 42);
            seq.set_audio_output(mode.map(|mode| crate::ChatAudioRequest {
                mode,
                format: crate::ChatAudioFormat::Wav,
            }));
            seq.multimodal = MultimodalData::new(
                Some(vec![image::DynamicImage::new_rgb8(4, 4)]),
                seq.clone_audios(),
                mode.map(|_| {
                    vec![crate::VideoInput::from_frames(
                        vec![image::DynamicImage::new_rgb8(4, 4); 4],
                        2.0,
                        None,
                    )]
                }),
                None,
                None,
                None,
            );
            let config: Arc<dyn Any> = Arc::new(PreProcessorConfig::default());
            let planned = if mode.is_some() {
                processor.prepare_for_paged_prompt_planning(
                    Some(tokenizer.clone()),
                    &mut [&mut seq],
                    &Device::Cpu,
                    Some(config.clone()),
                    None,
                )?;
                Some(seq.get_toks().to_vec())
            } else {
                None
            };
            let inputs =
                process_prompt(&processor, tokenizer.clone(), &mut [&mut seq], Some(config))?;
            if let Some(mode) = mode {
                let planned = planned.unwrap();
                assert_eq!(seq.get_toks(), planned);
                let marker = tokenizer
                    .token_to_id(match mode {
                        crate::ChatAudioMode::Audio => AUDIO_START_TOKEN,
                        crate::ChatAudioMode::TextAudio => MIXED_START_TOKEN,
                    })
                    .unwrap();
                assert_eq!(planned.last(), Some(&marker));
                assert_eq!(planned.iter().filter(|&&token| token == marker).count(), 1);
            }
            let rows = inputs.input_ids.to_vec2::<u32>()?;
            let args = *inputs
                .model_specific_args
                .downcast::<Lfm3OmniSpecificArgs>()
                .unwrap();
            assert_eq!(args.sequence_ids, [42]);
            assert_eq!(inputs.pixel_values.unwrap().dims(), &[4, 24]);
            let vision = args.vision.unwrap();
            if mode.is_some() {
                assert_eq!(vision.pixel_values_videos.unwrap().dims(), &[8, 24]);
                assert!(seq
                    .mm_features()
                    .iter()
                    .any(|feature| feature.kind == MultimodalKind::Video));
            } else {
                assert_eq!(vision.continuous_img_pad, vec![vec![(6, 7)]]);
                assert_eq!(args.audio[0].token_range, 2..3);
                assert_eq!(args.audio[1].token_range, 10..12);
                assert_eq!(
                    seq.mm_features()
                        .iter()
                        .map(|feature| feature.kind)
                        .collect::<Vec<_>>(),
                    [
                        MultimodalKind::Audio,
                        MultimodalKind::Image,
                        MultimodalKind::Audio
                    ]
                );
            }
            assert_eq!(args.audio.len(), 2);
            let audio_id = tokenizer.token_to_id(AUDIO_TOKEN).unwrap();
            for item in args.audio {
                assert!(rows[item.batch_index][item.token_range]
                    .iter()
                    .all(|&token| token == audio_id));
            }
        }
        Ok(())
    }

    #[test]
    fn batched_audio_rows_follow_sequence_id_order() -> Result<()> {
        let tokenizer = tokenizer();
        let processor = processor();
        let mut first = sequence(&tokenizer, AUDIO_PROMPT, 17);
        let mut second = sequence(&tokenizer, AUDIO_PROMPT, 5);
        let inputs = process_prompt(
            &processor,
            tokenizer,
            &mut [&mut second, &mut first],
            Some(Arc::new(PreProcessorConfig::default())),
        )?;
        let args = *inputs
            .model_specific_args
            .downcast::<Lfm3OmniSpecificArgs>()
            .unwrap();
        assert_eq!(args.sequence_ids, [5, 17]);
        assert_eq!(
            args.audio
                .iter()
                .map(|item| item.batch_index)
                .collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
        assert_eq!(args.audio[0].token_range, args.audio[2].token_range);
        assert_eq!(args.audio[1].token_range, args.audio[3].token_range);
        Ok(())
    }
}
