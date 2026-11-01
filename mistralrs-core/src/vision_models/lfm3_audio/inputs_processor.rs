use std::{any::Any, sync::Arc};

use anyhow::{Context, Result};
use candle_core::Device;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::MultimodalKind,
    pipeline::{
        text_models_inputs_processor::{self, PagedAttentionMeta, TextInputsProcessor},
        InputProcessorOutput, InputsProcessor, InputsProcessorType, InputsProcessorValidationError,
        MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        lfm3_omni::{
            audio_processing::AudioProcessor,
            inputs_processor::{
                process_audio_items, AudioInputLayout, AudioPlaceholderStyle, Lfm3OmniSpecificArgs,
                AUDIO_OUT_TOKEN, AUDIO_START_TOKEN, AUDIO_TOKEN, MIXED_START_TOKEN,
            },
        },
        qwen2vl::validated_mm_features,
        ModelInputs,
    },
};

pub(crate) struct Lfm3AudioProcessor {
    audio: AudioProcessor,
}

impl Lfm3AudioProcessor {
    pub(crate) fn new() -> Self {
        Self {
            audio: AudioProcessor::new(),
        }
    }
}

impl Processor for Lfm3AudioProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Lfm3AudioInputsProcessor {
            audio: self.audio.clone(),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[AUDIO_TOKEN, AUDIO_OUT_TOKEN]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

struct Lfm3AudioInputsProcessor {
    audio: AudioProcessor,
}

struct AudioTokens {
    input: u32,
    output: u32,
    audio_start: u32,
    mixed_start: u32,
}

impl AudioTokens {
    fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |token| {
            tokenizer
                .token_to_id(token)
                .with_context(|| format!("LFM3-Audio tokenizer is missing {token}"))
        };
        Ok(Self {
            input: id(AUDIO_TOKEN)?,
            output: id(AUDIO_OUT_TOKEN)?,
            audio_start: id(AUDIO_START_TOKEN)?,
            mixed_start: id(MIXED_START_TOKEN)?,
        })
    }
}

struct ExpandedAudioClips {
    tokens: Vec<u32>,
    ranges: Vec<(usize, usize)>,
}

fn expand_audio_clips(
    tokens: &[u32],
    audio: &AudioTokens,
    lengths: &[usize],
) -> Result<ExpandedAudioClips> {
    if tokens.contains(&audio.output) {
        return Err(InputsProcessorValidationError(
            "LFM3-Audio speech history with <|audio_out|> requires audio codes and is not supported".into(),
        ).into());
    }
    let count = tokens.iter().filter(|&&token| token == audio.input).count();
    if count != lengths.len() {
        return Err(InputsProcessorValidationError(format!(
            "LFM3-Audio prompt contains {count} audio placeholders for {} clips",
            lengths.len(),
        ))
        .into());
    }
    let mut lengths = lengths.iter();
    let mut expanded = Vec::new();
    let mut ranges = Vec::new();
    for &token in tokens {
        if token == audio.input {
            let length = *lengths.next().expect("audio placeholder count checked");
            ranges.push((expanded.len(), length));
            expanded.extend(std::iter::repeat_n(token, length));
        } else {
            expanded.push(token);
        }
    }
    Ok(ExpandedAudioClips {
        tokens: expanded,
        ranges,
    })
}

impl InputsProcessor for Lfm3AudioInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    fn prepare_for_paged_prompt_planning(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _device: &Device,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<&mut PagedAttentionMeta>,
    ) -> Result<()> {
        let tokenizer = tokenizer.context("LFM3-Audio requires a tokenizer")?;
        let audio = AudioTokens::new(&tokenizer)?;
        for seq in input_seqs {
            if seq.multimodal.has_changed_prompt {
                continue;
            }
            if seq.has_images() || seq.has_videos() {
                return Err(InputsProcessorValidationError(
                    "LFM3-Audio accepts text and audio inputs; use LFM3-Omni for images or video"
                        .into(),
                )
                .into());
            }
            let lengths = seq
                .audios()
                .unwrap_or_default()
                .iter()
                .map(AudioProcessor::num_tokens_for_audio)
                .collect::<Result<Vec<_>>>()?;
            let ExpandedAudioClips {
                tokens: mut expanded,
                ranges,
            } = expand_audio_clips(seq.get_toks(), &audio, &lengths)?;
            if let Some(output) = seq.audio_output() {
                let marker = match output.mode {
                    crate::ChatAudioMode::Audio => audio.audio_start,
                    crate::ChatAudioMode::TextAudio => audio.mixed_start,
                };
                if expanded.last() != Some(&marker) {
                    expanded.push(marker);
                }
            }
            let features = validated_mm_features(
                &ranges,
                seq.audio_hashes().unwrap_or_default(),
                MultimodalKind::Audio,
            )?;
            seq.set_toks_and_reallocate(expanded, paged_attn_metadata.as_deref_mut());
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
        anyhow::ensure!(!no_kv_cache, "LFM3-Audio requires KV caching");
        anyhow::ensure!(!is_xlora, "LFM3-Audio does not support X-LoRA");
        if is_prompt {
            self.prepare_for_paged_prompt_planning(
                tokenizer.clone(),
                input_seqs,
                device,
                other_config.clone(),
                paged_attn_metadata.as_mut(),
            )?;
        }
        let output = TextInputsProcessor.process_inputs(
            tokenizer.clone(),
            input_seqs,
            is_prompt,
            false,
            device,
            no_kv_cache,
            last_n_context_len,
            return_raw_logits,
            sliding_window,
            other_config,
            paged_attn_metadata,
            mapper,
        )?;
        let text = *output
            .inputs
            .downcast::<text_models_inputs_processor::ModelInputs>()
            .expect("text inputs");
        let mut inputs = ModelInputs {
            input_ids: text.input_ids,
            seqlen_offsets: text.seqlen_offsets,
            context_lens: text.context_lens,
            position_ids: text.position_ids,
            pixel_values: None,
            model_specific_args: Box::new(()),
            paged_attn_meta: text.paged_attn_meta,
            flash_meta: text.flash_meta,
            recurrent_batch_kind: text.recurrent_batch_kind,
            adapter_leases: text.adapter_leases,
        };
        let audio = if is_prompt && input_seqs.iter().any(|seq| seq.has_audios()) {
            let tokenizer = tokenizer.context("LFM3-Audio requires a tokenizer")?;
            process_audio_items(
                &self.audio,
                AudioInputLayout {
                    inputs: &inputs,
                    seq_indices: &output.seq_indices,
                    token_id: AudioTokens::new(&tokenizer)?.input,
                    style: AudioPlaceholderStyle::Bare,
                },
                input_seqs,
                device,
            )?
        } else {
            Vec::new()
        };
        let sequence_ids = output
            .seq_indices
            .iter()
            .map(|&index| *input_seqs[index].id())
            .collect();
        inputs.model_specific_args = Box::new(Lfm3OmniSpecificArgs {
            vision: None,
            audio,
            sequence_ids,
        });
        Ok(InputProcessorOutput {
            inputs: Box::new(inputs),
            seq_indices: output.seq_indices,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        sequence::MultimodalData,
        vision_models::lfm3_omni::inputs_processor::tests::{process_prompt, sequence, tokenizer},
    };
    const AUDIO_PROMPT: &str = "x <|audio|><|audio|> x";

    #[test]
    fn adjacent_audio_clips_keep_bare_spans_for_all_output_modes() -> Result<()> {
        let tokenizer = tokenizer();
        let processor = Lfm3AudioProcessor::new().inputs_processor();
        for mode in [
            None,
            Some(crate::ChatAudioMode::Audio),
            Some(crate::ChatAudioMode::TextAudio),
        ] {
            let mut seq = sequence(&tokenizer, AUDIO_PROMPT, 7);
            seq.set_audio_output(mode.map(|mode| crate::ChatAudioRequest {
                mode,
                format: crate::ChatAudioFormat::Wav,
            }));
            let hashes = seq.audio_hashes().unwrap().to_vec();
            processor.prepare_for_paged_prompt_planning(
                Some(tokenizer.clone()),
                &mut [&mut seq],
                &Device::Cpu,
                None,
                None,
            )?;
            let planned = seq.get_toks().to_vec();
            assert_eq!(planned.len(), 5 + usize::from(mode.is_some()));
            assert_eq!(
                seq.mm_features()
                    .iter()
                    .map(|feature| (feature.offset, feature.length))
                    .collect::<Vec<_>>(),
                [(1, 1), (2, 2)]
            );
            assert!(seq.mm_features().iter().all(|feature| !feature.splittable));
            let inputs =
                process_prompt(processor.as_ref(), tokenizer.clone(), &mut [&mut seq], None)?;
            assert_eq!(seq.get_toks(), planned);
            assert!(inputs.pixel_values.is_none());
            let args = *inputs
                .model_specific_args
                .downcast::<Lfm3OmniSpecificArgs>()
                .unwrap();
            assert!(args.vision.is_none());
            assert_eq!(args.sequence_ids, [7]);
            assert_eq!(args.audio.len(), 2);
            assert_eq!(args.audio[0].token_range, 1..2);
            assert_eq!(args.audio[1].token_range, 2..4);
            assert_eq!(args.audio[0].hash, hashes[0]);
            assert_eq!(args.audio[1].hash, hashes[1]);
            assert_eq!(args.audio[0].features.dims(), [1, 128, 1]);
            assert_eq!(args.audio[1].features.dims(), [1, 128, 9]);
            if let Some(mode) = mode {
                let marker = match mode {
                    crate::ChatAudioMode::Audio => AUDIO_START_TOKEN,
                    crate::ChatAudioMode::TextAudio => MIXED_START_TOKEN,
                };
                assert_eq!(planned.last(), tokenizer.token_to_id(marker).as_ref());
            }
        }
        Ok(())
    }

    #[test]
    fn text_only_speech_markers_precede_scheduled_chunk_boundaries() -> Result<()> {
        use crate::{
            pipeline::prompt_chunks::build_prompt_chunk_plan, speculative::SpeculativePrefixReplay,
        };

        const CHUNK_TOKENS: usize = 32;
        let tokenizer = tokenizer();
        let processor = Lfm3AudioProcessor::new().inputs_processor();
        for mode in [crate::ChatAudioMode::Audio, crate::ChatAudioMode::TextAudio] {
            let marker = tokenizer
                .token_to_id(match mode {
                    crate::ChatAudioMode::Audio => AUDIO_START_TOKEN,
                    crate::ChatAudioMode::TextAudio => MIXED_START_TOKEN,
                })
                .unwrap();
            for original_len in [28, 31, 32, 33, 34] {
                let mut seq = sequence(&tokenizer, AUDIO_PROMPT, 7);
                seq.multimodal = MultimodalData::new(None, None, None, None, None, None);
                seq.set_toks_and_reallocate(vec![1; original_len], None);
                assert!(!seq.needs_prompt_planning());
                seq.set_audio_output(Some(crate::ChatAudioRequest {
                    mode,
                    format: crate::ChatAudioFormat::Wav,
                }));
                if seq.needs_prompt_planning() {
                    processor.prepare_for_paged_prompt_planning(
                        Some(tokenizer.clone()),
                        &mut [&mut seq],
                        &Device::Cpu,
                        None,
                        None,
                    )?;
                }
                let chunks = build_prompt_chunk_plan(
                    seq.get_toks().len(),
                    0,
                    CHUNK_TOKENS,
                    None,
                    SpeculativePrefixReplay::NotRequired,
                    seq.mm_features(),
                );
                let scheduled_end = chunks.last().unwrap().end;
                let inputs =
                    process_prompt(processor.as_ref(), tokenizer.clone(), &mut [&mut seq], None)?;
                assert_eq!(scheduled_end, original_len + 1);
                assert_eq!(seq.prompt_tokens(), scheduled_end);
                assert_eq!(seq.get_toks()[scheduled_end - 1], marker);
                assert_eq!(inputs.context_lens, [(scheduled_end - 1, 1)]);
                assert_eq!(inputs.seqlen_offsets, [0]);
                assert_eq!(inputs.position_ids, [scheduled_end]);
                assert_eq!(
                    inputs.input_ids.to_vec2::<u32>()?[0][scheduled_end - 1],
                    marker
                );
                if original_len == CHUNK_TOKENS {
                    assert_eq!(chunks.len(), 2);
                    assert_eq!(chunks[0].end, CHUNK_TOKENS);
                    assert_eq!(chunks[1].start, CHUNK_TOKENS);
                }
            }
        }
        Ok(())
    }
}
