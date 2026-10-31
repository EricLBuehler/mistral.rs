use std::{any::Any, collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::ShardedSafeTensors;
use rand::SeedableRng;

use crate::{
    device_map::DummyDeviceMapper,
    paged_attention::{encoder_cache::CacheModality, AttentionImplementation},
    pipeline::{
        text_models_inputs_processor::FlashParams, DeviceMappedModelLoader, Lfm3OmniLoader,
        ModelForwardContext, MultimodalModel, NormalLoadingMetadata, RecurrentBatchKind,
        RecurrentMetadata,
    },
};

use super::{
    generation::{GenerationMarkers, GenerationMode, GenerationState},
    inputs_processor::{AudioItem, Lfm3OmniSpecificArgs},
    Config, GeneratedAudioInput, GenerationCache, Lfm3OmniModel, SequenceGeneration,
};

pub(crate) struct LogitsTolerance {
    maximum: f32,
    mean: f64,
    total_variation: f64,
}

// Cross-backend reduction differences need a wider logit bound; probability distance constrains their effect.
pub(crate) const REFERENCE_TOLERANCE: LogitsTolerance = LogitsTolerance {
    maximum: 2e-2,
    mean: 2e-3,
    total_variation: 1e-3,
};
const REPLAY_TOLERANCE: LogitsTolerance = LogitsTolerance {
    maximum: 5e-3,
    mean: 5e-4,
    total_variation: 1e-3,
};
const PARITY_MAX_SEQUENCE_LENGTH: usize = 1024;
const PARITY_SEQUENCE_ID: usize = 1;
const PARITY_AUDIO_HASH: u64 = 1;
const SPEECH_SAMPLE_RATE: u32 = 24_000;
const SPEECH_SAMPLES_PER_FRAME: usize = 1920;
const SPEECH_SMOKE_FRAMES: usize = 20;
const SPEECH_SMOKE_WAV: &str = "/tmp/mistralrs-omni-smoke.wav";
const SPEECH_REPLAY_FRAME: usize = 2;

fn generation_with_codes(entries: &[(usize, &[u32])]) -> SequenceGeneration {
    SequenceGeneration {
        state: GenerationState::new(
            GenerationMode::Audio,
            GenerationMarkers {
                audio_start: 10,
                mixed_start: 11,
                text_end: 12,
                audio_eos: 9,
                text_slots: 2,
                audio_slots: 3,
            },
        ),
        frames: vec![],
        code_history: entries
            .iter()
            .map(|&(position, codes)| (position, codes.to_vec()))
            .collect(),
    }
}

#[test]
fn generated_codes_replay_full_prompts_and_packed_chunks() {
    let cache = GenerationCache {
        hidden: HashMap::new(),
        sequences: [
            (
                1,
                generation_with_codes(&[(3, &[1, 2]), (5, &[3, 4]), (6, &[9, 9])]),
            ),
            (2, generation_with_codes(&[(2, &[5, 6]), (4, &[7, 8])])),
        ]
        .into_iter()
        .collect(),
    };
    let full = cache.audio_inputs(&[1], &[0], &[7], false);
    assert_eq!(
        full.iter()
            .map(|input| input.token_index)
            .collect::<Vec<_>>(),
        [3, 5, 6]
    );
    assert_eq!(full[2].codes, [9, 9]);
    let packed = cache.audio_inputs(&[2, 1], &[2, 4], &[3, 3], true);
    assert_eq!(
        packed,
        [
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 0,
                codes: vec![5, 6]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 2,
                codes: vec![7, 8]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 4,
                codes: vec![3, 4]
            },
            GeneratedAudioInput {
                batch_index: 0,
                token_index: 5,
                codes: vec![9, 9]
            },
        ]
    );
    let decode = cache.audio_inputs(&[2, 1], &[4, 5], &[1, 1], false);
    assert_eq!(decode[0].batch_index, 0);
    assert_eq!(decode[1].batch_index, 1);
    assert_eq!(decode[1].token_index, 0);
    assert_eq!(decode[1].codes, [3, 4]);
    assert_eq!(cache.audio_inputs(&[1], &[0], &[7], false), full);
}

#[test]
fn releasing_finished_speech_preserves_other_sequences() -> Result<()> {
    let mut cache = GenerationCache {
        hidden: [
            (1, Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?),
            (2, Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?),
        ]
        .into_iter()
        .collect(),
        sequences: [
            (1, generation_with_codes(&[(3, &[1, 2])])),
            (2, generation_with_codes(&[(4, &[3, 4])])),
        ]
        .into_iter()
        .collect(),
    };
    cache.release_sequences(&[1]);
    assert!(!cache.hidden.contains_key(&1));
    assert!(!cache.sequences.contains_key(&1));
    assert!(cache.hidden.contains_key(&2));
    assert_eq!(cache.sequences[&2].code_history[&4], [3, 4]);
    Ok(())
}

fn active_weight(name: &str) -> bool {
    [
        "model.",
        "lm_head.",
        "decoder.",
        "from_quantizer.",
        "quantizer.project_out.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

pub(crate) struct ReferencePrompt<'a> {
    pub name: &'a str,
    pub input_ids: &'a Tensor,
    pub args: Box<dyn Any>,
    pub logits: &'a Tensor,
    pub decode_input_ids: &'a Tensor,
    pub decode_logits: &'a Tensor,
}

pub(crate) fn recurrent_metadata(
    model: &dyn MultimodalModel,
    kind: RecurrentBatchKind,
) -> Result<RecurrentMetadata> {
    let cache = model.cache().hybrid();
    Ok(RecurrentMetadata::new(
        kind,
        cache
            .state_indices()
            .context("missing recurrent indices")?
            .clone(),
        cache.state_indices_host().map(ToOwned::to_owned),
    ))
}

pub(crate) fn compare_prompt(
    model: &dyn MultimodalModel,
    reference: ReferencePrompt<'_>,
    discrepancies: &mut Vec<anyhow::Error>,
) -> Result<()> {
    let slot = {
        let mut cache = model.cache().hybrid();
        let slot = cache.allocate_seq(PARITY_SEQUENCE_ID)?;
        cache.install_sequence_state_indices(&[(PARITY_SEQUENCE_ID, slot)])?;
        slot
    };
    let length = reference.input_ids.dim(1)?;
    let spans = [(length - 1, 1)];
    let positions = [length];
    let flash = FlashParams::empty(true);
    let mut context =
        ModelForwardContext::new(&[0], &spans, &positions, None, &flash).with_recurrent_metadata(
            Some(recurrent_metadata(model, RecurrentBatchKind::Prefill)?),
        );
    let logits = model.forward(reference.input_ids, None, reference.args, &mut context)?;
    if let Err(error) = compare_logits(
        reference.name,
        &logits,
        reference.logits,
        &REFERENCE_TOLERANCE,
    ) {
        discrepancies.push(error);
    }
    for step in 0..reference.decode_input_ids.dim(1)? {
        let input_ids = reference
            .decode_input_ids
            .narrow(1, step, 1)?
            .to_dtype(DType::U32)?;
        let offsets = [length + step];
        let positions = [length + step + 1];
        let mut context = ModelForwardContext::new(&offsets, &[(0, 1)], &positions, None, &flash)
            .with_recurrent_metadata(Some(recurrent_metadata(model, RecurrentBatchKind::Decode)?));
        let logits = model.forward(
            &input_ids,
            None,
            model.default_model_specific_args(&input_ids),
            &mut context,
        )?;
        if let Err(error) = compare_logits(
            &format!("{} decode {step}", reference.name),
            &logits,
            &reference.decode_logits.narrow(1, step, 1)?,
            &REFERENCE_TOLERANCE,
        ) {
            discrepancies.push(error);
        }
    }
    let mut cache = model.cache().hybrid();
    cache.release_seq(PARITY_SEQUENCE_ID, slot)?;
    cache.reset()?;
    Ok(())
}

pub(crate) fn compare_logits(
    name: &str,
    actual: &Tensor,
    expected: &Tensor,
    tolerance: &LogitsTolerance,
) -> Result<()> {
    anyhow::ensure!(
        actual.dims() == expected.dims(),
        "{name}: logits shape differs"
    );
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    let mut maximum = 0.0f32;
    let mut total = 0.0f64;
    let mut count = 0;
    let mut actual_best = (0, f32::NEG_INFINITY);
    let mut expected_best = (0, f32::NEG_INFINITY);
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        anyhow::ensure!(
            actual.is_finite() && expected.is_finite(),
            "{name}: nonfinite token {index}"
        );
        let error = (actual - expected).abs();
        maximum = maximum.max(error);
        total += f64::from(error);
        count += 1;
        if actual > actual_best.1 {
            actual_best = (index, actual);
        }
        if expected > expected_best.1 {
            expected_best = (index, expected);
        }
    }
    let mean = total / count as f64;
    let probabilities = |logits: &[f32], maximum: f32| {
        let mut probabilities = logits
            .iter()
            .map(|&logit| f64::from(logit - maximum).exp())
            .collect::<Vec<_>>();
        let sum = probabilities.iter().sum::<f64>();
        probabilities.iter_mut().for_each(|value| *value /= sum);
        probabilities
    };
    let total_variation = probabilities(&actual, actual_best.1)
        .iter()
        .zip(probabilities(&expected, expected_best.1))
        .map(|(&actual, expected)| (actual - expected).abs())
        .sum::<f64>()
        / 2.0;
    eprintln!(
        "{name}: logits max error={maximum}, mean error={mean}, probability TV={total_variation}, top token={}",
        actual_best.0
    );
    anyhow::ensure!(
        actual_best.0 == expected_best.0,
        "{name}: next token differs"
    );
    anyhow::ensure!(
        maximum < tolerance.maximum
            && mean < tolerance.mean
            && total_variation < tolerance.total_variation,
        "{name}: max={maximum}, mean={mean}, probability TV={total_variation}"
    );
    Ok(())
}

fn load_checkpoint_model() -> Result<(Config, Lfm3OmniModel)> {
    let directory = PathBuf::from(std::env::var("LFM3_OMNI_MODEL_DIR")?);
    let config_json = crate::pipeline::load_model_config(&directory.join("config.json"), false)?;
    let mut config: Config = serde_json::from_str(&config_json)?;
    let builder = unsafe {
        ShardedSafeTensors::sharded(
            &[
                directory.join("model.safetensors"),
                directory.join("audio_tokenizer/model.safetensors"),
            ],
            DType::F32,
            &Device::Cpu,
            None,
            Arc::new(|name| active_weight(&name)),
        )?
    };
    let parameters = builder
        .tensor_names()
        .context("missing weight index")?
        .iter()
        .map(|name| {
            builder
                .tensor_shape(name)
                .unwrap()
                .iter()
                .product::<usize>()
        })
        .sum::<usize>();
    let loader = Lfm3OmniLoader;
    let estimated = loader.non_mapped_size_in_bytes(&config_json, DType::F32, 1, None, None)?
        + loader
            .layer_sizes_in_bytes(&config_json, DType::F32, 1, None)?
            .iter()
            .sum::<usize>();
    assert_eq!(
        estimated,
        parameters * DType::F32.size_in_bytes(),
        "loader storage must match active checkpoint weights"
    );
    config.vision.text_config.max_position_embeddings = PARITY_MAX_SEQUENCE_LENGTH;
    let metadata = NormalLoadingMetadata {
        mapper: Box::new(DummyDeviceMapper {
            nm_device: Device::Cpu,
        }),
        loading_isq: false,
        real_device: Device::Cpu,
        multi_progress: Arc::new(indicatif::MultiProgress::new()),
        matformer_slicing_config: None,
        rope_pairing: None,
    };
    let model = Lfm3OmniModel::new(
        &config,
        builder,
        true,
        metadata,
        AttentionImplementation::Eager,
    )?;
    Ok((config, model))
}

#[test]
#[ignore = "requires LFM3_OMNI_MODEL_DIR and LFM3_OMNI_PARITY safetensors"]
fn real_omni_text_and_audio_logits_match_transformers() -> Result<()> {
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_OMNI_PARITY")?, &Device::Cpu)?;
    let (config, model) = load_checkpoint_model()?;
    let mut discrepancies = Vec::new();
    let input_ids = reference["text_input_ids"].to_dtype(DType::U32)?;
    compare_prompt(
        &model,
        ReferencePrompt {
            name: "text",
            input_ids: &input_ids,
            args: model.default_model_specific_args(&input_ids),
            logits: &reference["text_logits"],
            decode_input_ids: &reference["text_decode_input_ids"],
            decode_logits: &reference["text_decode_logits"],
        },
        &mut discrepancies,
    )?;

    let input_ids = reference["audio_input_ids"].to_dtype(DType::U32)?;
    let mut args = *model
        .default_model_specific_args(&input_ids)
        .downcast::<Lfm3OmniSpecificArgs>()
        .expect("Omni default arguments");
    let tokens = input_ids.flatten_all()?.to_vec1::<u32>()?;
    let start = tokens
        .iter()
        .position(|&token| token == config.speech.audio_token_id)
        .context("reference prompt has no audio placeholders")?;
    let count = tokens[start..]
        .iter()
        .take_while(|&&token| token == config.speech.audio_token_id)
        .count();
    args.audio.push(AudioItem {
        features: reference["audio_features"].clone(),
        hash: PARITY_AUDIO_HASH,
        batch_index: 0,
        token_range: start..start + count,
    });
    compare_prompt(
        &model,
        ReferencePrompt {
            name: "audio",
            input_ids: &input_ids,
            args: Box::new(args),
            logits: &reference["audio_logits"],
            decode_input_ids: &reference["audio_decode_input_ids"],
            decode_logits: &reference["audio_decode_logits"],
        },
        &mut discrepancies,
    )?;
    model
        .encoder_cache()
        .context("missing encoder cache")?
        .lock()
        .unwrap()
        .insert(
            CacheModality::Audio,
            PARITY_AUDIO_HASH,
            vec![reference["audio_projected"].clone()],
        );
    let mut args = *model
        .default_model_specific_args(&input_ids)
        .downcast::<Lfm3OmniSpecificArgs>()
        .expect("Omni default arguments");
    args.audio.push(AudioItem {
        features: reference["audio_features"].clone(),
        hash: PARITY_AUDIO_HASH,
        batch_index: 0,
        token_range: start..start + count,
    });
    compare_prompt(
        &model,
        ReferencePrompt {
            name: "audio with reference projection",
            input_ids: &input_ids,
            args: Box::new(args),
            logits: &reference["audio_logits"],
            decode_input_ids: &reference["audio_decode_input_ids"],
            decode_logits: &reference["audio_decode_logits"],
        },
        &mut discrepancies,
    )?;
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

pub(crate) fn speech_sequence(input_ids: &Tensor) -> Result<crate::sequence::Sequence> {
    use crate::{
        sampler::Sampler,
        sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer},
    };

    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    let sampler = Sampler::new(
        None,
        0,
        None,
        None,
        None,
        None,
        None,
        1,
        1.0,
        0.0,
        HashMap::new(),
        vec![],
    )?;
    let group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
        1, false, true, None,
    )));
    let mut sequence = Sequence::new_waiting(
        input_ids.flatten_all()?.to_vec1::<u32>()?,
        "Say hello.".to_owned(),
        PARITY_SEQUENCE_ID,
        0,
        0,
        sender,
        sampler,
        vec![],
        vec![],
        None,
        false,
        false,
        group,
        0,
        0,
        SequenceRecognizer::None,
        None,
        None,
        None,
        None,
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
        Some(0),
    );
    sequence.set_audio_output(Some(crate::ChatAudioRequest {
        mode: crate::ChatAudioMode::Audio,
        format: crate::ChatAudioFormat::Wav,
    }));
    Ok(sequence)
}

pub(crate) fn speech_forward(
    model: &dyn MultimodalModel,
    sequence_id: usize,
    input_ids: &Tensor,
    offset: usize,
    kind: RecurrentBatchKind,
) -> Result<Tensor> {
    let mut args = *model
        .default_model_specific_args(input_ids)
        .downcast::<Lfm3OmniSpecificArgs>()
        .expect("Omni default arguments");
    args.sequence_ids = vec![sequence_id];
    let length = input_ids.dim(1)?;
    let offsets = [offset];
    let positions = [offset + length];
    let spans = [(length - 1, 1)];
    let flash = FlashParams::empty(true);
    let mut context = ModelForwardContext::new(&offsets, &spans, &positions, None, &flash)
        .with_recurrent_metadata(Some(recurrent_metadata(model, kind)?));
    Ok(model.forward(input_ids, None, Box::new(args), &mut context)?)
}

pub(crate) fn first_speech_hidden_and_depth(
    model: &Lfm3OmniModel,
    sequence_id: usize,
) -> Result<(Tensor, Tensor)> {
    let hidden = model
        .generation
        .lock()
        .unwrap()
        .hidden
        .get(&sequence_id)
        .context("missing speech hidden state")?
        .clone();
    let books = model.codebook_embedding.embeddings().dim(0)? / model.codebook_size;
    let shape = (hidden.dim(0)?, books);
    let depth = model.depth_decoder.forward(
        &hidden,
        &Tensor::zeros(shape, DType::U32, hidden.device())?,
        &Tensor::ones(shape, DType::U8, hidden.device())?,
    )?;
    for (name, tensor) in [("speech hidden", &hidden), ("all-masked depth", &depth)] {
        let values = tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let finite = values.iter().filter(|value| value.is_finite()).count();
        let minimum = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::INFINITY, f32::min);
        let maximum = values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(f32::NEG_INFINITY, f32::max);
        let first_nonfinite = values.iter().position(|value| !value.is_finite());
        eprintln!(
            "{name}: shape={:?}, finite={finite}/{}, range=[{minimum}, {maximum}], first nonfinite={first_nonfinite:?}",
            tensor.dims(),
            values.len()
        );
    }
    Ok((hidden, depth))
}

#[test]
#[ignore = "requires real Omni weights and speech_input_ids in LFM3_OMNI_PARITY"]
fn real_omni_generates_speech_wav() -> Result<()> {
    let reference =
        candle_core::safetensors::load(std::env::var("LFM3_OMNI_PARITY")?, &Device::Cpu)?;
    let (config, model) = load_checkpoint_model()?;
    let input_ids = reference["speech_input_ids"].to_dtype(DType::U32)?;
    let mut sequence = speech_sequence(&input_ids)?;
    let sequence_id = *sequence.id();
    let mut slot = {
        let mut cache = model.cache().hybrid();
        let slot = cache.allocate_seq(sequence_id)?;
        cache.install_sequence_state_indices(&[(sequence_id, slot)])?;
        slot
    };
    let frame_limit = std::env::var("LFM3_OMNI_SMOKE_FRAMES")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(SPEECH_SMOKE_FRAMES);
    anyhow::ensure!(frame_limit > 0, "speech smoke needs at least one frame");
    let mut logits = speech_forward(
        &model,
        sequence_id,
        &input_ids,
        0,
        RecurrentBatchKind::Prefill,
    )?;
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(0);
    let mut frames = 0;
    for step in 0..frame_limit {
        let (_, forced_token) = model.prepare_sampling(&sequence, logits, &mut rng)?;
        assert_eq!(forced_token, Some(config.speech.audio_out_token_id));
        let terminal = {
            let generation = model.generation.lock().unwrap();
            let state = generation
                .sequences
                .get(&sequence_id)
                .context("missing speech state")?;
            frames = state.frames.len();
            !state.state.is_audio()
        };
        model.accept_sample(&sequence, config.speech.audio_out_token_id)?;
        sequence.add_token(
            crate::sampler::Logprobs {
                token: config.speech.audio_out_token_id,
                logprob: 0.0,
                bytes: None,
                top_logprobs: None,
            },
            vec![],
            None,
        );
        if terminal || step + 1 == frame_limit {
            break;
        }
        let input = Tensor::new(&[[config.speech.audio_out_token_id]], &Device::Cpu)?;
        logits = speech_forward(
            &model,
            sequence_id,
            &input,
            input_ids.dim(1)? + step,
            RecurrentBatchKind::Decode,
        )?;
        if step + 1 == SPEECH_REPLAY_FRAME {
            model.reset_model_specific_state();
            model.reset_model_specific_state_for_sequences(&[sequence_id]);
            {
                let mut cache = model.cache().hybrid();
                cache.release_seq(sequence_id, slot)?;
                cache.reset()?;
                slot = cache.allocate_seq(sequence_id)?;
                cache.install_sequence_state_indices(&[(sequence_id, slot)])?;
            }
            let replay = Tensor::from_slice(
                sequence.get_toks(),
                (1, sequence.get_toks().len()),
                &Device::Cpu,
            )?;
            let replayed =
                speech_forward(&model, sequence_id, &replay, 0, RecurrentBatchKind::Prefill)?;
            compare_logits("speech replay", &replayed, &logits, &REPLAY_TOLERANCE)?;
            logits = replayed;
        }
    }
    assert!(frames > 0, "speech terminated before emitting audio");
    let codes = model.generation.lock().unwrap().sequences[&sequence_id]
        .frames
        .clone();
    let codes = Tensor::from_vec(
        codes.into_iter().flatten().collect::<Vec<_>>(),
        (1, frames, config.speech.num_codebooks),
        &Device::Cpu,
    )?
    .transpose(1, 2)?
    .contiguous()?;
    let raw_pcm = model
        .codec
        .decode(&codes)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert!(
        raw_pcm.iter().all(|sample| sample.is_finite()),
        "speech PCM contains nonfinite samples"
    );
    assert!(
        raw_pcm.iter().any(|&sample| sample != 0.0),
        "speech PCM is silent"
    );
    model.finalize_response(&mut sequence)?;
    let response = sequence
        .generated_audio()
        .context("missing generated audio")?;
    let wav = STANDARD.decode(response.data.as_ref().context("missing audio data")?)?;
    let mut reader = hound::WavReader::new(std::io::Cursor::new(&wav))?;
    assert_eq!(reader.spec().sample_rate, SPEECH_SAMPLE_RATE);
    assert_eq!(reader.spec().channels, 1);
    let pcm = reader
        .samples::<i16>()
        .collect::<std::result::Result<Vec<_>, _>>()?;
    assert_eq!(pcm.len(), frames * SPEECH_SAMPLES_PER_FRAME);
    assert!(
        pcm.iter().any(|&sample| sample != 0),
        "speech output is silent"
    );
    let output =
        std::env::var("LFM3_OMNI_SMOKE_WAV").unwrap_or_else(|_| SPEECH_SMOKE_WAV.to_owned());
    std::fs::write(&output, &wav)?;
    eprintln!(
        "speech: {frames} frames, {} samples, saved {output}",
        pcm.len()
    );
    let mut cache = model.cache().hybrid();
    cache.release_seq(sequence_id, slot)?;
    cache.reset()?;
    Ok(())
}
