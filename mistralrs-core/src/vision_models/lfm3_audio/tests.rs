use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::{ShardedSafeTensors, ShardedVarBuilder};
use rand::SeedableRng;

use crate::{
    device_map::DummyDeviceMapper,
    paged_attention::AttentionImplementation,
    pipeline::{
        load_model_config, remap_lfm3_audio_legacy_weights, DeviceMappedModelLoader,
        Lfm3AudioLoader, MultimodalModel, MultimodalModelLoader, NormalLoadingMetadata,
        RecurrentBatchKind,
    },
    vision_models::lfm3_omni::{
        audio::{AudioEncoder, AudioProjector},
        audio_processing::{AudioProcessor, SAMPLE_RATE},
        codec::LiquidCodec,
        generation::{AudioSampling, DepthDecoder},
        inputs_processor::{AudioItem, Lfm3OmniSpecificArgs},
        tests::{
            compare_prompt, first_speech_hidden_and_depth, speech_forward, speech_sequence,
            ReferencePrompt,
        },
        Lfm3OmniModel,
    },
};

use super::config::AudioConfig;

const MAX_SEQUENCE_LENGTH: usize = 1024;
const AUDIO_HASH: u64 = 1;
const CHECKPOINT_PARAMETERS: usize = 2_013_919_106;
const SPEECH_SAMPLE_RATE: u32 = 24_000;
const SPEECH_SAMPLES_PER_FRAME: usize = 1920;
const SPEECH_SMOKE_SLOTS: usize = 24;
const COMPONENT_MAX_ERROR: f32 = 2e-4;
const TTS_MAX_ERROR: f32 = 2e-2;
const TTS_MEAN_ERROR: f64 = 2e-3;
const TTS_SEQUENCE_ID: usize = 1;

fn checkpoint() -> Result<(PathBuf, String, AudioConfig)> {
    let directory = PathBuf::from(std::env::var("LFM3_AUDIO_MODEL_DIR")?);
    let json = load_model_config(&directory.join("config.json"), false)?;
    let config = serde_json::from_str(&json)?;
    Ok((directory, json, config))
}

fn active_weight(name: &str) -> bool {
    !name.ends_with(".to_logits.weight")
        && [
            "lfm.",
            "encoder.lfm.lfm.",
            "encoder.downsample.",
            "mlp.",
            "codebook_embedding.",
            "depth_linear.",
            "depthformer.",
            "depth_embeddings.",
            "mask_embedding",
            "model.",
            "lm_head.",
            "decoder.",
            "from_quantizer.",
            "quantizer.project_out.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn weights(directory: &std::path::Path) -> Result<ShardedVarBuilder> {
    weights_on_device(directory, &Device::Cpu)
}

fn weights_on_device(directory: &std::path::Path, device: &Device) -> Result<ShardedVarBuilder> {
    Ok(unsafe {
        ShardedSafeTensors::sharded(
            &[
                directory.join("model.safetensors"),
                directory.join("audio_tokenizer/model.safetensors"),
            ],
            DType::F32,
            device,
            None,
            Arc::new(|name| active_weight(&name)),
        )?
    })
}

fn load_model() -> Result<(AudioConfig, Box<dyn MultimodalModel + Send + Sync>)> {
    let (directory, json, mut config) = checkpoint()?;
    let weights = weights(&directory)?;
    let parameters = weights
        .tensor_names()
        .context("missing weight index")?
        .iter()
        .map(|name| {
            weights
                .tensor_shape(name)
                .unwrap()
                .iter()
                .product::<usize>()
        })
        .sum::<usize>();
    assert_eq!(parameters, CHECKPOINT_PARAMETERS);
    let loader = Lfm3AudioLoader;
    let estimated = loader.non_mapped_size_in_bytes(&json, DType::F32, 1, None, None)?
        + loader
            .layer_sizes_in_bytes(&json, DType::F32, 1, None)?
            .iter()
            .sum::<usize>();
    assert_eq!(estimated, parameters * DType::F32.size_in_bytes());
    config.text_config.max_position_embeddings = MAX_SEQUENCE_LENGTH;
    let model = loader.load(
        &serde_json::to_string(&config)?,
        weights,
        NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: Device::Cpu,
            }),
            loading_isq: false,
            real_device: Device::Cpu,
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        },
        AttentionImplementation::Eager,
    )?;
    Ok((config, model))
}

fn compare_component(name: &str, actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims(), "{name}: shape");
    let error = (actual - expected)?.abs()?;
    let maximum = error.max_all()?.to_scalar::<f32>()?;
    let mean = error.mean_all()?.to_scalar::<f32>()?;
    eprintln!("{name}: max error={maximum}, mean error={mean}");
    anyhow::ensure!(
        maximum < COMPONENT_MAX_ERROR,
        "{name}: max={maximum}, mean={mean}"
    );
    Ok(())
}

fn compare_tts_tensor(name: &str, actual: &Tensor, expected: &Tensor) -> Result<()> {
    anyhow::ensure!(actual.dims() == expected.dims(), "{name}: shape differs");
    let actual = actual
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let mut maximum = 0.0f32;
    let mut total = 0.0f64;
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        anyhow::ensure!(
            actual.is_finite() && expected.is_finite(),
            "{name}: nonfinite value at {index}, actual={actual}, expected={expected}"
        );
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        total += f64::from(error);
    }
    let mean = total / actual.len() as f64;
    eprintln!("{name}: max error={maximum}, mean error={mean}");
    anyhow::ensure!(
        maximum < TTS_MAX_ERROR && mean < TTS_MEAN_ERROR,
        "{name}: max error={maximum}, mean error={mean}"
    );
    Ok(())
}

#[test]
#[ignore = "requires LFM3_AUDIO_MODEL_DIR and LFM3_AUDIO_TTS_AUDIT safetensors"]
fn real_audio_tts_hidden_and_depth_match_transformers() -> Result<()> {
    let device = match std::env::var("LFM3_AUDIO_TEST_DEVICE").as_deref() {
        Ok("metal") => Device::new_metal(0)?,
        Ok("cpu") | Err(_) => Device::Cpu,
        Ok(other) => anyhow::bail!("unsupported LFM3_AUDIO_TEST_DEVICE={other}"),
    };
    eprintln!("TTS parity device: {device:?}");
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_AUDIO_TTS_AUDIT")?, &Device::Cpu)?;
    let (directory, _, mut config) = checkpoint()?;
    config.text_config.max_position_embeddings = MAX_SEQUENCE_LENGTH;
    let weights = remap_lfm3_audio_legacy_weights(weights_on_device(&directory, &device)?);
    let model = Lfm3OmniModel::new_audio(
        &config,
        weights,
        true,
        NormalLoadingMetadata {
            mapper: Box::new(DummyDeviceMapper {
                nm_device: device.clone(),
            }),
            loading_isq: false,
            real_device: device.clone(),
            multi_progress: Arc::new(indicatif::MultiProgress::new()),
            matformer_slicing_config: None,
            rope_pairing: None,
        },
        AttentionImplementation::Eager,
    )?;
    let mut discrepancies = Vec::new();
    for name in ["tts", "readme", "simple"] {
        let input_ids = reference[&format!("{name}_input_ids")]
            .to_dtype(DType::U32)?
            .to_device(&device)?;
        let slot = {
            let mut cache = model.cache().hybrid();
            let slot = cache.allocate_seq(TTS_SEQUENCE_ID)?;
            cache.install_sequence_state_indices(&[(TTS_SEQUENCE_ID, slot)])?;
            slot
        };
        eprintln!("{name}: {} prompt tokens", input_ids.dim(1)?);
        speech_forward(
            &model,
            TTS_SEQUENCE_ID,
            &input_ids,
            0,
            RecurrentBatchKind::Prefill,
        )?;
        let (hidden, depth) = first_speech_hidden_and_depth(&model, TTS_SEQUENCE_ID)?;
        for (suffix, actual) in [("hidden", hidden), ("depth_logits", depth.clone())] {
            let key = format!("{name}_{suffix}");
            if let Err(error) = compare_tts_tensor(&key, &actual, &reference[&key]) {
                discrepancies.push(error.to_string());
            }
        }
        let depth = depth
            .to_device(&Device::Cpu)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        let top_k = AudioSampling::standalone().top_k.unwrap();
        for (book, logits) in depth.iter().enumerate() {
            let eos = logits[config.speech.audio_eos_code as usize];
            let rank = logits.iter().filter(|&&value| value > eos).count();
            eprintln!("{name}: codebook {book} audio EOS rank={rank}, logit={eos}");
            if !eos.is_finite() || rank < top_k {
                discrepancies.push(format!(
                    "{name}: codebook {book} EOS enters first-frame top {top_k}, rank={rank}, logit={eos}"
                ));
            }
        }
        model.release_generation_sequences(&[TTS_SEQUENCE_ID]);
        let mut cache = model.cache().hybrid();
        cache.release_seq(TTS_SEQUENCE_ID, slot)?;
        cache.reset()?;
    }
    anyhow::ensure!(discrepancies.is_empty(), "{}", discrepancies.join("; "));
    Ok(())
}

#[test]
#[ignore = "requires LFM3_AUDIO_MODEL_DIR and LFM3_AUDIO_PARITY safetensors"]
fn real_audio_encoder_depth_and_codec_match_transformers() -> Result<()> {
    let (directory, _, config) = checkpoint()?;
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_AUDIO_PARITY")?, &Device::Cpu)?;
    let weights = remap_lfm3_audio_legacy_weights(weights(&directory)?);
    let speech = &config.speech;
    let encoder = AudioEncoder::new(
        &speech.audio_encoder_config,
        weights.pp("model.audio_encoder"),
    )?;
    let projector = AudioProjector::new(
        speech.audio_encoder_config.hidden_size,
        config.text_config.hidden_size,
        weights.pp("model.audio_projector"),
    )?;
    let features = AudioProcessor::new().process_audio(
        &crate::AudioInput {
            samples: reference["audio_waveform"].to_vec1::<f32>()?,
            sample_rate: SAMPLE_RATE,
            channels: 1,
        },
        &Device::Cpu,
    )?;
    compare_component("audio frontend", &features, &reference["audio_features"])?;
    let encoded = encoder.forward(
        &reference["audio_features"],
        &[reference["audio_features"].dim(2)?],
    )?;
    compare_component(
        "audio projection",
        &projector.forward(&encoded)?.squeeze(0)?,
        &reference["audio_projected"],
    )?;
    let depth = DepthDecoder::new(
        &speech.depth_config,
        config.text_config.hidden_size,
        speech.num_codebooks,
        speech.codebook_size,
        weights.pp("model.depth_decoder"),
    )?;
    compare_component(
        "audio depth",
        &depth.forward(
            &reference["depth_hidden"],
            &reference["depth_codes"].to_dtype(DType::U32)?,
            &reference["depth_masked"].to_dtype(DType::U8)?,
        )?,
        &reference["depth_logits"],
    )?;
    let codec = LiquidCodec::new(speech.codec_config.clone(), weights)?;
    compare_component(
        "audio codec",
        &codec.decode(&reference["codec_codes"].to_dtype(DType::U32)?)?,
        &reference["codec_pcm"],
    )?;
    Ok(())
}

#[test]
#[ignore = "requires LFM3_AUDIO_MODEL_DIR and LFM3_AUDIO_PARITY safetensors"]
fn real_audio_text_asr_and_understanding_match_transformers() -> Result<()> {
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_AUDIO_PARITY")?, &Device::Cpu)?;
    let (config, model) = load_model()?;
    let mut discrepancies = Vec::new();
    for name in ["text", "asr", "understanding"] {
        let input_ids = reference[&format!("{name}_input_ids")].to_dtype(DType::U32)?;
        let mut args = *model
            .default_model_specific_args(&input_ids)
            .downcast::<Lfm3OmniSpecificArgs>()
            .expect("Audio model arguments");
        if name != "text" {
            let tokens = input_ids.flatten_all()?.to_vec1::<u32>()?;
            let start = tokens
                .iter()
                .position(|&token| token == config.speech.audio_token_id)
                .context("missing audio placeholders")?;
            let length = tokens[start..]
                .iter()
                .take_while(|&&token| token == config.speech.audio_token_id)
                .count();
            args.audio.push(AudioItem {
                features: reference["audio_features"].clone(),
                hash: AUDIO_HASH,
                batch_index: 0,
                token_range: start..start + length,
            });
        }
        compare_prompt(
            model.as_ref(),
            ReferencePrompt {
                name,
                input_ids: &input_ids,
                args: Box::new(args),
                logits: &reference[&format!("{name}_logits")],
                decode_input_ids: &reference[&format!("{name}_decode_input_ids")],
                decode_logits: &reference[&format!("{name}_decode_logits")],
            },
            &mut discrepancies,
        )?;
    }
    anyhow::ensure!(
        discrepancies.is_empty(),
        "{}",
        discrepancies
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    );
    Ok(())
}

#[test]
#[ignore = "requires LFM3_AUDIO_MODEL_DIR and LFM3_AUDIO_PARITY safetensors"]
fn real_audio_generates_speech_and_interleaved_wav() -> Result<()> {
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_AUDIO_PARITY")?, &Device::Cpu)?;
    let (_, model) = load_model()?;
    let eos = reference["stop_token_ids"]
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let limit = std::env::var("LFM3_AUDIO_SMOKE_SLOTS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(SPEECH_SMOKE_SLOTS);
    for (name, mode) in [
        ("speech", crate::ChatAudioMode::Audio),
        ("interleaved", crate::ChatAudioMode::TextAudio),
    ] {
        let input_ids = reference[&format!("{name}_input_ids")].to_dtype(DType::U32)?;
        let mut sequence = speech_sequence(&input_ids)?;
        sequence.set_audio_output(Some(crate::ChatAudioRequest {
            mode,
            format: crate::ChatAudioFormat::Wav,
        }));
        let sequence_id = *sequence.id();
        let slot = {
            let mut cache = model.cache().hybrid();
            let slot = cache.allocate_seq(sequence_id)?;
            cache.install_sequence_state_indices(&[(sequence_id, slot)])?;
            slot
        };
        let mut logits = speech_forward(
            model.as_ref(),
            sequence_id,
            &input_ids,
            0,
            RecurrentBatchKind::Prefill,
        )?;
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(0);
        let mut audio_slots = 0;
        let mut text_slots = 0;
        for step in 0..limit {
            let (scores, forced) = model.prepare_sampling(&sequence, logits, &mut rng)?;
            let token = if let Some(token) = forced {
                audio_slots += 1;
                token
            } else {
                text_slots += 1;
                scores.flatten_all()?.argmax(0)?.to_scalar::<u32>()?
            };
            model.accept_sample(&sequence, token)?;
            sequence.add_token(
                crate::sampler::Logprobs {
                    token,
                    logprob: 0.0,
                    bytes: None,
                    top_logprobs: None,
                },
                vec![],
                None,
            );
            if step + 1 == limit
                || model
                    .sampling_eos_tokens(&sequence, Some(&eos))
                    .is_some_and(|tokens| tokens.contains(&token))
            {
                break;
            }
            logits = speech_forward(
                model.as_ref(),
                sequence_id,
                &Tensor::new(&[[token]], &Device::Cpu)?,
                input_ids.dim(1)? + step,
                RecurrentBatchKind::Decode,
            )?;
        }
        model.finalize_response(&mut sequence)?;
        let response = sequence
            .generated_audio()
            .context("missing audio response")?;
        let wav = STANDARD.decode(response.data.as_ref().context("missing WAV data")?)?;
        let mut reader = hound::WavReader::new(std::io::Cursor::new(&wav))?;
        assert_eq!(reader.spec().sample_rate, SPEECH_SAMPLE_RATE);
        assert_eq!(reader.spec().channels, 1);
        let samples = reader
            .samples::<i16>()
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert!(!samples.is_empty() && samples.iter().any(|&sample| sample != 0));
        assert_eq!(samples.len() % SPEECH_SAMPLES_PER_FRAME, 0);
        assert!(samples.len() / SPEECH_SAMPLES_PER_FRAME <= audio_slots);
        if mode == crate::ChatAudioMode::TextAudio {
            assert!(text_slots > 0);
        }
        let path = format!("/tmp/mistralrs-audio-{name}-smoke.wav");
        std::fs::write(&path, &wav)?;
        eprintln!(
            "{name}: {audio_slots} audio slots, {text_slots} text slots, {} samples, saved {path}",
            samples.len()
        );
        model.release_generation_sequences(&[sequence_id]);
        let mut cache = model.cache().hybrid();
        cache.release_seq(sequence_id, slot)?;
        cache.reset()?;
    }
    Ok(())
}
