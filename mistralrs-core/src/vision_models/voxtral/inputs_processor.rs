#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, sync::Arc};

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    paged_attention::block_hash::{MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind},
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
use super::{VoxtralAudioCacheKey, VoxtralAudioRequest, VoxtralSpecificArgs};

/// BOS token ID for Mistral tekken tokenizer.
const BOS_TOKEN_ID: u32 = 1;
/// [STREAMING_PAD] token ID (rank 32 in tekken special tokens).
const STREAMING_PAD_TOKEN_ID: u32 = 32;
/// Number of left-pad streaming tokens (from tekken.json audio config).
const N_LEFT_PAD_TOKENS: usize = 32;
/// Number of delay tokens (transcription_delay_ms / frame_rate alignment).
const N_DELAY_TOKENS: usize = 6;
const AUDIO_ENCODER_DOWNSAMPLE_FACTOR: usize = 2;

pub struct VoxtralProcessor {
    audio_processor: VoxtralAudioProcessor,
    audio_length_per_tok: usize,
}

impl VoxtralProcessor {
    pub fn new(cfg: &VoxtralConfig) -> Self {
        let enc_args = &cfg.multimodal.whisper_model_args.encoder_args;
        Self {
            audio_processor: VoxtralAudioProcessor::new(&enc_args.audio_encoding_args),
            audio_length_per_tok: AUDIO_ENCODER_DOWNSAMPLE_FACTOR
                * cfg
                    .multimodal
                    .whisper_model_args
                    .downsample_args
                    .downsample_factor,
        }
    }
}

/// Number of right-pad silence tokens added to audio (from audio_processing.rs).
/// Subtracting from the generation cap prevents generating into silence region.
const N_RIGHT_PAD_TOKENS: usize = 17;

impl Processor for VoxtralProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(VoxtralInputsProcessor {
            audio_processor: VoxtralAudioProcessor::new_from_processor(&self.audio_processor),
            audio_length_per_tok: self.audio_length_per_tok,
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
    audio_length_per_tok: usize,
}

fn audio_prompt_tokens() -> Vec<u32> {
    let n_pad = N_LEFT_PAD_TOKENS + N_DELAY_TOKENS;
    let mut prompt_tokens = Vec::with_capacity(1 + n_pad);
    prompt_tokens.push(BOS_TOKEN_ID);
    prompt_tokens.extend(std::iter::repeat_n(STREAMING_PAD_TOKEN_ID, n_pad));
    prompt_tokens
}

fn audio_prompt_feature(hashes: Vec<u64>) -> anyhow::Result<MultiModalFeature> {
    if hashes.is_empty() {
        anyhow::bail!("Voxtral audio prompt requires at least one audio hash");
    }
    Ok(MultiModalFeature {
        kind: MultimodalKind::Audio,
        item_range: 0..hashes.len(),
        hashes,
        offset: 0,
        length: 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS,
        attention_policy: MultimodalAttentionPolicy::Causal,
        splittable: false,
    })
}

fn prepare_audio_prompt(
    seq: &mut Sequence,
    paged_attn_metadata: Option<&mut PagedAttentionMeta>,
) -> anyhow::Result<()> {
    let hashes = seq
        .audio_hashes()
        .ok_or_else(|| anyhow::anyhow!("Voxtral audio prompt is missing audio hashes"))?
        .to_vec();
    let feature = audio_prompt_feature(hashes)?;
    if seq.mm_features().is_empty() {
        seq.set_mm_features(vec![feature]);
    } else if seq.mm_features().len() != 1
        || seq.mm_features()[0].kind != feature.kind
        || seq.mm_features()[0].item_range != feature.item_range
        || seq.mm_features()[0].hashes != feature.hashes
        || seq.mm_features()[0].offset != feature.offset
        || seq.mm_features()[0].length != feature.length
        || seq.mm_features()[0].attention_policy != feature.attention_policy
        || seq.mm_features()[0].splittable != feature.splittable
    {
        anyhow::bail!("Voxtral audio prompt has inconsistent multimodal feature metadata");
    }

    seq.set_toks_and_reallocate(audio_prompt_tokens(), paged_attn_metadata);
    seq.multimodal.has_changed_prompt = true;
    Ok(())
}

fn batch_mel_features(mels: &[Tensor]) -> anyhow::Result<Option<Tensor>> {
    let Some(first) = mels.first() else {
        return Ok(None);
    };
    let (first_batch, _, mel_bins) = first.dims3()?;
    if first_batch != 1 {
        anyhow::bail!("Voxtral request mel features must have batch size 1");
    }

    let mut max_frames = 0;
    for mel in mels {
        let (batch, frames, bins) = mel.dims3()?;
        if batch != 1 {
            anyhow::bail!("Voxtral request mel features must have batch size 1");
        }
        if frames == 0 {
            anyhow::bail!("Voxtral request mel features cannot be empty");
        }
        if bins != mel_bins {
            anyhow::bail!("Voxtral mel bin mismatch: expected {mel_bins}, received {bins}");
        }
        max_frames = max_frames.max(frames);
    }

    let padded = mels
        .iter()
        .map(|mel| {
            let frames = mel.dim(1)?;
            mel.pad_with_zeros(1, 0, max_frames - frames)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;
    Ok(Some(Tensor::cat(&padded, 0)?))
}

impl InputsProcessor for VoxtralInputsProcessor {
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
    ) -> anyhow::Result<()> {
        let Some(_tokenizer) = tokenizer else {
            return Err(anyhow::Error::msg(
                "VoxtralInputsProcessor requires a specified tokenizer.",
            ));
        };

        for seq in input_seqs.iter_mut() {
            if seq.multimodal.has_changed_prompt || !seq.has_audios() {
                continue;
            }
            prepare_audio_prompt(seq, paged_attn_metadata.as_deref_mut())?;
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

        let mut processed_mels = (0..input_seqs.len())
            .map(|_| None)
            .collect::<Vec<Option<Tensor>>>();
        if is_prompt {
            for (seq_idx, seq) in input_seqs.iter_mut().enumerate() {
                if !seq.multimodal.has_changed_prompt {
                    if seq.has_audios() {
                        prepare_audio_prompt(seq, paged_attn_metadata.as_mut())?;
                    }
                    continue;
                }

                let hashes = seq.audio_hashes().map(<[u64]>::to_vec).unwrap_or_default();
                let Some(audios) = seq.take_audios() else {
                    continue;
                };
                if audios.is_empty() {
                    continue;
                }
                if hashes.len() != audios.len() {
                    anyhow::bail!(
                        "Voxtral audio/hash cardinality mismatch: {} audios for {} hashes",
                        audios.len(),
                        hashes.len()
                    );
                }

                let mut request_mels = Vec::with_capacity(audios.len());
                for audio in &audios {
                    request_mels.push(self.audio_processor.process_audio(audio, device)?);
                }
                let request_mel = Tensor::cat(&request_mels, 1)?;
                let mel_frames = request_mel.dim(1)?;
                if self.audio_length_per_tok == 0 {
                    anyhow::bail!("Voxtral audio downsample factor cannot be zero");
                }
                let num_audio_tokens = mel_frames
                    .div_ceil(self.audio_length_per_tok)
                    .saturating_sub(N_RIGHT_PAD_TOKENS);
                seq.set_max_len(num_audio_tokens);
                processed_mels[seq_idx] = Some(request_mel);
            }
        }

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

        let mut request_mels = Vec::new();
        let mut mel_lengths = Vec::new();
        let mut audio_requests = Vec::new();
        for (logical_index, &seq_idx) in seq_indices.iter().enumerate() {
            let seq = &input_seqs[seq_idx];
            let Some(hashes) = seq.audio_hashes().filter(|hashes| !hashes.is_empty()) else {
                continue;
            };
            let mel_index = if let Some(mel) = processed_mels[seq_idx].take() {
                let index = request_mels.len();
                mel_lengths.push(mel.dim(1)?);
                request_mels.push(mel);
                Some(index)
            } else {
                None
            };
            audio_requests.push(VoxtralAudioRequest {
                logical_index,
                key: VoxtralAudioCacheKey {
                    sequence_id: *seq.id(),
                    hashes: hashes.to_vec(),
                },
                mel_index,
            });
        }
        let mel_features = batch_mel_features(&request_mels)?;

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(VoxtralSpecificArgs {
                mel_features,
                mel_lengths,
                audio_requests,
                n_delay_tokens: Some(N_DELAY_TOKENS as f32),
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

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    use super::{audio_prompt_feature, batch_mel_features, N_DELAY_TOKENS, N_LEFT_PAD_TOKENS};

    #[test]
    fn grouped_audio_feature_hashes_every_item_and_the_bos() -> anyhow::Result<()> {
        let feature = audio_prompt_feature(vec![11, 22, 33])?;
        assert_eq!(feature.item_range, 0..3);
        assert_eq!(feature.hashes, vec![11, 22, 33]);
        assert_eq!(feature.offset, 0);
        assert_eq!(feature.length, 1 + N_LEFT_PAD_TOKENS + N_DELAY_TOKENS);
        assert!(!feature.splittable);
        Ok(())
    }

    #[test]
    fn mel_batching_preserves_request_boundaries() -> anyhow::Result<()> {
        let first = Tensor::from_vec(vec![1f32; 4], (1, 2, 2), &Device::Cpu)?;
        let second = Tensor::from_vec(vec![2f32; 6], (1, 3, 2), &Device::Cpu)?;
        let batch = batch_mel_features(&[first, second])?.unwrap();

        assert_eq!(batch.dims(), &[2, 3, 2]);
        assert_eq!(
            batch.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?,
            vec![1., 1., 1., 1., 0., 0.]
        );
        assert_eq!(
            batch.narrow(0, 1, 1)?.flatten_all()?.to_vec1::<f32>()?,
            vec![2.; 6]
        );
        Ok(())
    }
}
