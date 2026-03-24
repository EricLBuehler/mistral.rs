#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::Device;
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
    vision_models::ModelInputs,
};

use super::audio_processing::VoxtralAudioProcessor;
use super::config::VoxtralConfig;
use super::VoxtralSpecificArgs;

/// BOS token ID for Mistral tekken tokenizer.
const BOS_TOKEN_ID: u32 = 1;
/// [STREAMING_PAD] token ID (rank 32 in tekken special tokens).
const STREAMING_PAD_TOKEN_ID: u32 = 32;
/// Number of left-pad streaming tokens (from tekken.json audio config).
const N_LEFT_PAD_TOKENS: usize = 32;
/// Number of delay tokens (transcription_delay_ms / frame_rate alignment).
const N_DELAY_TOKENS: usize = 6;

pub struct VoxtralProcessor {
    audio_processor: VoxtralAudioProcessor,
}

impl VoxtralProcessor {
    pub fn new(cfg: &VoxtralConfig) -> Self {
        let enc_args = &cfg.multimodal.whisper_model_args.encoder_args;
        Self {
            audio_processor: VoxtralAudioProcessor::new(&enc_args.audio_encoding_args),
        }
    }
}

/// Audio output tokens per mel frame: conv stride (2) × adapter downsample (4).
const AUDIO_LENGTH_PER_TOK: usize = 8;
/// Number of right-pad silence tokens added to audio (from audio_processing.rs).
/// Subtracting from the generation cap prevents generating into silence region.
const N_RIGHT_PAD_TOKENS: usize = 17;

impl Processor for VoxtralProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(VoxtralInputsProcessor {
            audio_processor: VoxtralAudioProcessor::new_from_processor(&self.audio_processor),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}

struct VoxtralInputsProcessor {
    audio_processor: VoxtralAudioProcessor,
}

impl InputsProcessor for VoxtralInputsProcessor {
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
        _other_config: Option<Arc<dyn Any>>,
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
        let Some(_tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "VoxtralInputsProcessor requires a specified tokenizer.",
            ));
        };

        // Early call (add_request): replaces tokens for scheduler allocation.
        // Step call (pipeline.step): processes audio into mel features for forward pass.
        // Prompt is always [BOS, PAD*(N_LEFT_PAD + N_DELAY)] = 39 tokens.
        let mel_features = if is_prompt {
            let mut mel_accum = Vec::new();
            for seq in input_seqs.iter_mut() {
                // Early call: replace tokens with [BOS, PAD*N] for scheduler
                if !seq.multimodal.has_changed_prompt {
                    if seq.has_audios() {
                        let n_pad = N_LEFT_PAD_TOKENS + N_DELAY_TOKENS;
                        let mut prompt_tokens = Vec::with_capacity(1 + n_pad);
                        prompt_tokens.push(BOS_TOKEN_ID);
                        prompt_tokens.extend(std::iter::repeat_n(STREAMING_PAD_TOKEN_ID, n_pad));
                        seq.set_toks_and_reallocate(prompt_tokens, paged_attn_metadata.as_mut());
                        seq.multimodal.has_changed_prompt = true;
                    }
                } else {
                    // Step call: take and process audios into mel features
                    if let Some(audios) = seq.take_audios() {
                        for audio in &audios {
                            let mel = self
                                .audio_processor
                                .process_audio(audio, device)
                                .expect("Audio processing failed");
                            // Cap generation at num_audio_tokens matching HF behavior.
                            // num_audio_tokens = ceil(mel_frames / audio_length_per_tok)
                            // Subtract right-pad silence tokens to stop at speech boundary.
                            let mel_frames = mel.dims()[1];
                            let num_audio_tokens = mel_frames
                                .div_ceil(AUDIO_LENGTH_PER_TOK)
                                .saturating_sub(N_RIGHT_PAD_TOKENS);
                            seq.set_max_len(num_audio_tokens);
                            mel_accum.push(mel);
                        }
                    }
                }
            }
            if !mel_accum.is_empty() {
                let t = candle_core::Tensor::cat(&mel_accum, 1).map_err(anyhow::Error::from)?;
                Some(t)
            } else {
                None
            }
        } else {
            None
        };

        // Standard text input processing
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

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(VoxtralSpecificArgs {
                mel_features,
                n_delay_tokens: Some(N_DELAY_TOKENS as f32),
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
