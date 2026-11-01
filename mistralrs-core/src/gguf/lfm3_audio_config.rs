use anyhow::{bail, ensure, Result};
use mistralrs_quant::GgufArchive;
use serde_json::{json, Value};

use super::{
    lfm3_vl_config::{metadata_f64, prepare_lfm3_vl_config, token_id},
    multimodal_binding_utils::{
        metadata_string, metadata_usize, validate_architecture, TensorInventory,
    },
    normal_config::{synthesize_normal_config_value, validate_lfm3_metadata},
};
use crate::{
    pipeline::sanitize_quantized_weight_source_config,
    vision_models::{lfm3_audio::config::AudioConfig, lfm3_omni::config::Config},
    NormalLoaderType,
};

// llama.cpp's conversion/lfm3.py rejects other codec and MaskGIT settings instead of writing metadata.
const ROPE_THETA: f64 = 1_000_000.0;
const CODEBOOKS: usize = 12;
const CODEBOOK_SIZE: usize = 2017;
const MEL_BINS: usize = 128;
const MASKGIT_SCHEDULE: [usize; 3] = [4, 4, 4];
const INTERLEAVED_TEXT: usize = 6;
const INTERLEAVED_AUDIO: usize = 12;
const CODEC_HIDDEN: usize = 512;
const CODEC_QUANTIZER_DIM: usize = 256;
const CODEC_FSQ_LEVELS: [usize; 4] = [8, 7, 6, 6];
const CODEC_FFT_SIZE: usize = 1280;
const CODEC_HOP: usize = 320;
const CODEC_FRAME_STACK: usize = 6;
const CODEC_WINDOW: usize = 30;
const CODEC_SAMPLE_RATE: usize = 24_000;

pub(crate) fn prepare_lfm3_audio_config(
    external_config: Option<&str>,
    archive: &GgufArchive,
) -> Result<String> {
    prepare_config(external_config, archive, false)
}

pub(crate) fn prepare_lfm3_omni_config(
    external_config: Option<&str>,
    archive: &GgufArchive,
) -> Result<String> {
    prepare_config(external_config, archive, true)
}

fn prepare_config(
    external_config: Option<&str>,
    archive: &GgufArchive,
    omni: bool,
) -> Result<String> {
    validate_architecture(archive, "lfm3")?;
    validate_lfm3_metadata(archive.metadata())?;
    for (key, expected) in [
        ("clip.audio.projector_type", "lfm3a"),
        ("clip.gen.audio.projector_type", "lfm3a_gen"),
    ] {
        ensure!(
            metadata_string(archive, key)? == Some(expected),
            "LFM3 speech GGUF requires `{key}={expected}`"
        );
    }
    let speech = speech_config(archive)?;
    let mut config: Value = match external_config {
        Some(config) => {
            let mut config =
                serde_json::from_str(&sanitize_quantized_weight_source_config(config)?)?;
            if !omni {
                crate::vision_models::lfm3_audio::config::normalize_config(&mut config)?;
            }
            config
        }
        None => {
            let mut config: Value = if omni {
                serde_json::from_str(&prepare_lfm3_vl_config(None, archive)?)?
            } else {
                let names = archive.tensors().keys().cloned().collect::<Vec<_>>();
                let text = synthesize_normal_config_value(
                    &NormalLoaderType::Lfm3,
                    archive.metadata(),
                    &names,
                )?;
                json!({"tie_word_embeddings": text["tie_word_embeddings"], "text_config": text})
            };
            config["model_type"] = json!(if omni { "lfm3_omni" } else { "lfm3_audio" });
            config["architectures"] = json!([if omni {
                "Lfm3OmniForConditionalGeneration"
            } else {
                "Lfm3AudioForConditionalGeneration"
            }]);
            config
                .as_object_mut()
                .unwrap()
                .extend(speech.as_object().unwrap().clone());
            config
        }
    };
    for field in ["codec_config", "audio_sampling"] {
        if config.get(field).is_none_or(Value::is_null) {
            config[field] = speech[field].clone();
        }
    }
    let parsed = if omni {
        serde_json::from_value::<Config>(config.clone())?.speech
    } else {
        serde_json::from_value::<AudioConfig>(config.clone())?.speech
    };
    let parsed = serde_json::to_value(parsed)?;
    for field in [
        "codec_config",
        "num_codebooks",
        "codebook_size",
        "audio_eos_code",
        "maskgit_schedule",
        "interleaved_n_text",
        "interleaved_n_audio",
    ] {
        ensure!(
            parsed[field] == speech[field],
            "LFM3 GGUF configuration `{field}` disagrees with the converter's audio format"
        );
    }
    Ok(serde_json::to_string(&config)?)
}

fn speech_config(archive: &GgufArchive) -> Result<Value> {
    let inventory = TensorInventory::from_archive(archive);
    let depth_hidden = metadata_usize(archive, "clip.gen.audio.embedding_length")?;
    ensure!(
        inventory.shape("a.gen.code.embd.weight")? == [CODEBOOKS, CODEBOOK_SIZE, depth_hidden],
        "LFM3 GGUF depth embeddings disagree with the audio format"
    );
    let audio_sampling = if archive
        .metadata_value("clip.gen.audio.sampling.temp")
        .is_some()
    {
        json!({
            "temperature": metadata_f64(archive, "clip.gen.audio.sampling.temp")?,
            "top_k": archive.metadata_value("clip.gen.audio.sampling.top_k").map(|_| metadata_usize(archive, "clip.gen.audio.sampling.top_k")).transpose()?,
            "repetition_penalty": metadata_f64(archive, "clip.gen.audio.sampling.penalty_repeat")?,
            "repetition_window": metadata_usize(archive, "clip.gen.audio.sampling.penalty_last_n")?,
        })
    } else {
        Value::Null
    };
    Ok(json!({
        "audio_encoder_config": audio_encoder_config(archive, &inventory)?,
        "depth_config": depth_config(archive, &inventory)?,
        "codec_config": {
            "encoder_dim": CODEC_HIDDEN, "quantizer_dim": CODEC_QUANTIZER_DIM, "fsq_levels": CODEC_FSQ_LEVELS, "n_q": CODEBOOKS,
            "stft_n_fft": CODEC_FFT_SIZE, "stft_hop_length": CODEC_HOP, "stft_win_length": CODEC_FFT_SIZE, "stft_frame_stack": CODEC_FRAME_STACK,
            "decoder_window": CODEC_WINDOW, "sampling_rate": CODEC_SAMPLE_RATE,
        },
        "audio_sampling": audio_sampling,
        "num_codebooks": CODEBOOKS, "codebook_size": CODEBOOK_SIZE, "audio_eos_code": CODEBOOK_SIZE - 1,
        "maskgit_schedule": MASKGIT_SCHEDULE,
        "interleaved_n_text": INTERLEAVED_TEXT, "interleaved_n_audio": INTERLEAVED_AUDIO,
        "audio_token_id": token_id(archive, "<|audio|>")?,
        "audio_out_token_id": token_id(archive, "<|audio_out|>")?,
        "audio_start_token_id": token_id(archive, "<|audio_start|>")?,
        "text_end_token_id": token_id(archive, "<|text_end|>")?,
        "mixed_start_token_id": token_id(archive, "<|mixed_start|>")?,
    }))
}

pub(super) fn audio_encoder_config(
    archive: &GgufArchive,
    inventory: &TensorInventory<'_>,
) -> Result<Value> {
    let hidden = metadata_usize(archive, "clip.audio.embedding_length")?;
    let num_layers = metadata_usize(archive, "clip.audio.block_count")?;
    let layers = inventory.require_layers("a.blk.", "LFM3-Audio")?;
    ensure!(
        layers.len() == num_layers,
        "LFM3 GGUF audio encoder layer count disagrees with tensor inventory"
    );
    let [channels, _, _, _] = inventory.shape("a.conv1d.0.weight")? else {
        bail!("LFM3 GGUF audio downsampler must have rank 4");
    };
    let mut layer_types = Vec::with_capacity(num_layers);
    let mut conv_width = None;
    let mut conv_bias = None;
    for layer in layers {
        let prefix = format!("a.blk.{layer}");
        let conv = format!("{prefix}.conv_dw.weight");
        let attention = format!("{prefix}.attn_q.weight");
        match (inventory.contains(&conv), inventory.contains(&attention)) {
            (true, false) => {
                let [size, width] = inventory.shape(&conv)? else {
                    bail!("LFM3 GGUF encoder convolution `{conv}` must have rank 2");
                };
                ensure!(
                    *size == hidden && conv_width.is_none_or(|previous| previous == *width),
                    "LFM3 GGUF audio convolutions have inconsistent dimensions"
                );
                let bias = inventory.contains(&format!("{prefix}.conv_dw.bias"));
                ensure!(
                    conv_bias.is_none_or(|previous| previous == bias),
                    "LFM3 GGUF audio convolutions have inconsistent bias settings"
                );
                conv_width = Some(*width);
                conv_bias = Some(bias);
                layer_types.push("conv");
            }
            (false, true) => layer_types.push("full_attention"),
            _ => bail!("LFM3 GGUF audio encoder layer {layer} must contain exactly one operator"),
        }
    }
    let mel_bins = metadata_usize(archive, "clip.audio.num_mel_bins")?;
    ensure!(
        mel_bins == MEL_BINS,
        "LFM3 GGUF audio frontend requires {MEL_BINS} mel bins"
    );
    Ok(json!({
        "conv_channels": channels, "hidden_size": hidden, "num_mel_bins": mel_bins,
        "sliding_window": metadata_usize(archive, "clip.audio.window_size")?,
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": metadata_usize(archive, "clip.audio.feed_forward_length")?,
            "num_hidden_layers": num_layers,
            "num_attention_heads": metadata_usize(archive, "clip.audio.attention.head_count")?,
            "num_key_value_heads": metadata_usize(archive, "clip.audio.attention.head_count_kv")?,
            "norm_eps": metadata_f64(archive, "clip.audio.attention.layer_norm_epsilon")?,
            "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
            "conv_L_cache": conv_width.unwrap_or_default(), "conv_bias": conv_bias.unwrap_or(false),
            "block_auto_adjust_ff_dim": false, "block_multiple_of": 1, "block_ffn_dim_multiplier": null,
            "layer_types": layer_types,
        },
    }))
}

fn depth_config(archive: &GgufArchive, inventory: &TensorInventory<'_>) -> Result<Value> {
    let num_layers = metadata_usize(archive, "clip.gen.audio.block_count")?;
    ensure!(
        inventory
            .require_layers("a.gen.code.blk.", "LFM3-Audio")?
            .len()
            == num_layers,
        "LFM3 GGUF depth layer count disagrees with tensor inventory"
    );
    let [head_dim] = inventory.shape("a.gen.code.blk.0.attn_q_norm.weight")? else {
        bail!("LFM3 GGUF depth query norm must have rank 1");
    };
    Ok(json!({
        "hidden_size": metadata_usize(archive, "clip.gen.audio.embedding_length")?,
        "intermediate_size": metadata_usize(archive, "clip.gen.audio.feed_forward_length")?,
        "num_hidden_layers": num_layers,
        "num_attention_heads": metadata_usize(archive, "clip.gen.audio.attention.head_count")?,
        "num_key_value_heads": metadata_usize(archive, "clip.gen.audio.attention.head_count_kv")?,
        "norm_eps": metadata_f64(archive, "clip.gen.audio.attention.layer_norm_epsilon")?,
        "head_dim": head_dim,
        "layer_types": vec!["full_attention"; num_layers],
        "rope_parameters": {"full_attention": {"rope_theta": ROPE_THETA, "rope_type": "default", "partial_rotary_factor": 1.0}},
        "use_sandwich_norm": false, "conv_bias": false, "conv_L_cache": 0, "sliding_window": 0,
    }))
}
