use std::{collections::BTreeSet, fs, path::Path, sync::Arc};

use candle_core::{
    quantized::gguf_file::Value as GgufValue, DType, Device, Module, Result, Tensor, D,
};
use mistralrs_quant::{
    GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource, QuantMethod, ReplicatedLayer,
    ShardedVarBuilder,
};
use serde::Deserialize;

use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{Activation, RotaryEmbedding, Sdpa},
    sequence::Sequence,
    speculative::{
        DFlashConfig, SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx,
        DFLASH_DEFAULT_N_PREDICT, DFLASH_MAX_N_PREDICT,
    },
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::{
    config::TextConfig,
    text::{rms_norm_f32, DFlashTargetCapture, DFlashTargetCaptureBatch, TextModel},
};

const DFLASH_MODEL_TYPE: &str = "muse_glimmer_assistant";
const DFLASH_GGUF_ARCHITECTURE: &str = "dflash";
const DFLASH_BLOCK_SIZE: usize = 16;

fn proposal_block_fits(base_len: usize, max_position_embeddings: usize) -> bool {
    base_len
        .checked_add(DFLASH_BLOCK_SIZE)
        .is_some_and(|end| end <= max_position_embeddings)
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
}

fn default_rope_theta() -> f64 {
    500_000.0
}

#[derive(Clone, Debug, Deserialize)]
struct DFlashModelConfig {
    #[serde(default)]
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    max_position_embeddings: usize,
    sliding_window: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    hidden_act: String,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    #[serde(default = "default_rope_parameters")]
    rope_parameters: RopeParameters,
}

fn default_rope_parameters() -> RopeParameters {
    RopeParameters {
        rope_theta: default_rope_theta(),
    }
}

impl DFlashModelConfig {
    fn validate(&self, target: &TextConfig) -> Result<()> {
        if self.model_type != DFLASH_MODEL_TYPE {
            candle_core::bail!(
                "DFlash model_type mismatch: expected `{DFLASH_MODEL_TYPE}`, got `{}`",
                self.model_type
            );
        }
        if self.hidden_size != target.hidden_size {
            candle_core::bail!(
                "DFlash hidden size mismatch: assistant {}, target {}",
                self.hidden_size,
                target.hidden_size
            );
        }
        if self.block_size != DFLASH_BLOCK_SIZE {
            candle_core::bail!(
                "DFlash block size {} is unsupported; expected {DFLASH_BLOCK_SIZE}",
                self.block_size
            );
        }
        if self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self.head_dim == 0
            || self.num_attention_heads * self.head_dim > self.hidden_size
        {
            candle_core::bail!("DFlash attention dimensions are incompatible");
        }
        if self.intermediate_size == 0
            || self.max_position_embeddings == 0
            || self.sliding_window == 0
            || !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
        {
            candle_core::bail!("DFlash dimensions and numeric parameters must be positive");
        }
        if self.mask_token_id as usize >= target.vocab_size {
            candle_core::bail!(
                "DFlash mask token {} exceeds target vocabulary {}",
                self.mask_token_id,
                target.vocab_size
            );
        }
        if self.target_layer_ids.is_empty()
            || self.target_layer_ids.len() * self.hidden_size == 0
            || self
                .target_layer_ids
                .iter()
                .any(|layer| *layer >= target.num_hidden_layers)
        {
            candle_core::bail!("DFlash target layer IDs are incompatible with the target model");
        }
        if !self
            .target_layer_ids
            .windows(2)
            .all(|layers| layers[0] < layers[1])
        {
            candle_core::bail!("DFlash target layer IDs must be strictly increasing");
        }
        if !self.layer_types.is_empty()
            && (self.layer_types.len() != self.num_hidden_layers
                || self
                    .layer_types
                    .iter()
                    .any(|layer_type| layer_type != "sliding_attention"))
        {
            candle_core::bail!("DFlash currently requires sliding attention in every layer");
        }
        if !self.hidden_act.is_empty() && self.hidden_act != "silu" {
            candle_core::bail!(
                "DFlash activation `{}` is unsupported; expected `silu`",
                self.hidden_act
            );
        }
        Ok(())
    }
}

pub(super) struct DFlashRuntime {
    model: DFlashModel,
    n_predict: usize,
    assistant: String,
}

pub(super) trait DFlashTargetWeights {
    fn device(&self) -> &Device;
    fn raw_embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor>;
    fn raw_lm_head(&self, hidden_states: &Tensor) -> Result<Tensor>;
}

impl DFlashTargetWeights for TextModel {
    fn device(&self) -> &Device {
        &self.device
    }

    fn raw_embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        TextModel::raw_embed_tokens(self, input_ids)
    }

    fn raw_lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        TextModel::raw_lm_head(self, hidden_states)
    }
}

fn raw_noise_embeddings(
    target: &impl DFlashTargetWeights,
    input_ids: &Tensor,
    assistant_device: &Device,
) -> Result<Tensor> {
    target
        .raw_embed_tokens(input_ids)?
        .to_device(assistant_device)
}

fn raw_candidate_logits(
    target: &impl DFlashTargetWeights,
    hidden_states: &Tensor,
) -> Result<Tensor> {
    target.raw_lm_head(&hidden_states.to_device(target.device())?)
}

impl DFlashRuntime {
    pub(super) fn load(
        config: DFlashConfig,
        target_cfg: &TextConfig,
        device: &Device,
        target_dtype: DType,
    ) -> Result<Self> {
        let assistant = config.model.clone();
        let path = config.resolve_path()?;
        let (mut model_config, vb) = if path.is_dir() {
            load_safetensors(&path, device, target_dtype)?
        } else if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            load_gguf(&path, device, target_dtype)?
        } else {
            candle_core::bail!(
                "DFlash assistant must be a safetensors directory or an official prequantized GGUF file; ISQ and UQFF assistants are unsupported"
            );
        };
        model_config.validate(target_cfg)?;
        model_config.max_position_embeddings = model_config
            .max_position_embeddings
            .min(target_cfg.max_position_embeddings);
        let n_predict = config.n_predict.unwrap_or(DFLASH_DEFAULT_N_PREDICT);
        if n_predict == 0 || n_predict > DFLASH_MAX_N_PREDICT {
            candle_core::bail!(
                "DFlash n_predict must be between 1 and {DFLASH_MAX_N_PREDICT}, got {n_predict}"
            );
        }
        let model = DFlashModel::new(&model_config, vb, device)?;
        Ok(Self {
            model,
            n_predict,
            assistant,
        })
    }

    pub(super) fn assistant(&self) -> &str {
        &self.assistant
    }

    pub(super) fn proposal_len(&self) -> usize {
        self.n_predict
    }

    pub(super) fn target_layer_ids(&self) -> &[usize] {
        &self.model.config.target_layer_ids
    }

    pub(super) fn bind_capture(
        &self,
        capture: DFlashTargetCaptureBatch,
        sequences: &[&Sequence],
        is_prompt: bool,
    ) -> Result<()> {
        if capture.rows.len() != sequences.len() {
            candle_core::bail!(
                "DFlash target capture batch {} does not match {} sequences",
                capture.rows.len(),
                sequences.len()
            );
        }
        let is_first_prompt_chunk = capture.is_first_prompt_chunk;
        for (capture, sequence) in capture.rows.into_iter().zip(sequences) {
            let pending = self.model.prepare_capture(capture)?;
            sequence.with_speculative_aux_state(
                || {
                    DFlashSequenceState::new(
                        self.model.layers.len(),
                        self.model.config.sliding_window,
                    )
                },
                |state| {
                    if is_prompt {
                        if is_first_prompt_chunk {
                            state.clear();
                        }
                        state.commit_capture(pending, None)?;
                    } else {
                        state.stage(pending)?;
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    pub(super) fn commit_capture(
        &self,
        sequences: &[&Sequence],
        rows: &[Option<usize>],
        expected_lens: &[usize],
    ) -> Result<()> {
        if sequences.len() != rows.len() || sequences.len() != expected_lens.len() {
            candle_core::bail!(
                "DFlash commit batch mismatch: sequences={}, rows={}, lengths={}",
                sequences.len(),
                rows.len(),
                expected_lens.len()
            );
        }
        for ((sequence, rows), expected_len) in sequences
            .iter()
            .zip(rows)
            .zip(expected_lens.iter().copied())
        {
            sequence.with_speculative_aux_state(
                || {
                    DFlashSequenceState::new(
                        self.model.layers.len(),
                        self.model.config.sliding_window,
                    )
                },
                |state| state.commit_pending(*rows, expected_len),
            )?;
        }
        Ok(())
    }

    pub(super) fn propose(
        &self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target: &impl DFlashTargetWeights,
    ) -> Result<SpeculativeProposalBatch> {
        let batch = ctx.sampled_tokens.len();
        if ctx.base_lens.len() != batch
            || ctx.sequences.len() != batch
            || ctx.seq_ids.len() != batch
        {
            candle_core::bail!(
                "DFlash proposal batch mismatch: sampled={batch}, lengths={}, sequences={}, ids={}",
                ctx.base_lens.len(),
                ctx.sequences.len(),
                ctx.seq_ids.len()
            );
        }
        let mut proposals = Vec::with_capacity(batch);
        for row in 0..batch {
            if !proposal_block_fits(
                ctx.base_lens[row],
                self.model.config.max_position_embeddings,
            ) {
                proposals.push(SpeculativeProposal::new(Vec::new()));
                continue;
            }
            let mut ids = vec![self.model.config.mask_token_id; DFLASH_BLOCK_SIZE];
            ids[0] = ctx.sampled_tokens[row];
            let ids = Tensor::from_vec(ids, (1, DFLASH_BLOCK_SIZE), target.device())?;
            let noise = raw_noise_embeddings(target, &ids, self.model.device())?;
            let hidden = ctx.sequences[row].with_speculative_aux_state(
                || {
                    DFlashSequenceState::new(
                        self.model.layers.len(),
                        self.model.config.sliding_window,
                    )
                },
                |state| self.model.forward(&noise, ctx.base_lens[row], state),
            )?;
            let logits = raw_candidate_logits(target, &hidden.narrow(1, 1, self.n_predict)?)?;
            let mut context = ctx.sequences[row].get_toks().to_vec();
            if !ctx.sampled_tokens_emitted {
                context.push(ctx.sampled_tokens[row]);
            }
            let logits = logits.squeeze(0)?;
            let mut tokens = Vec::with_capacity(self.n_predict);
            for draft_row in 0..self.n_predict {
                let row_logits = logits.get(draft_row)?.to_dtype(DType::F32)?;
                let sampled = ctx.sequences[row].sampler().sample(
                    row_logits,
                    &context,
                    false,
                    ctx.rng.clone(),
                    false,
                    batch > 1,
                )?;
                context.push(sampled.token);
                tokens.push(sampled.token);
            }
            proposals.push(SpeculativeProposal::with_logits(tokens, logits));
        }
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

fn load_safetensors(
    path: &Path,
    device: &Device,
    target_dtype: DType,
) -> Result<(DFlashModelConfig, ShardedVarBuilder)> {
    let config_path = path.join("config.json");
    let raw = fs::read_to_string(&config_path).map_err(|err| {
        candle_core::Error::msg(format!(
            "failed to read DFlash config {}: {err}",
            config_path.display()
        ))
    })?;
    let config = serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
    let mut weights = fs::read_dir(path)
        .map_err(|err| {
            candle_core::Error::msg(format!(
                "failed to list DFlash directory {}: {err}",
                path.display()
            ))
        })?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "safetensors")
        })
        .collect::<Vec<_>>();
    weights.sort();
    if weights.is_empty() {
        candle_core::bail!(
            "DFlash model directory {} contains no safetensors weights",
            path.display()
        );
    }
    let dtype = match target_dtype {
        DType::BF16 | DType::F16 | DType::F32 => target_dtype,
        _ => DType::BF16,
    };
    let vb = from_mmaped_safetensors(
        weights,
        Vec::new(),
        Some(dtype),
        device,
        Vec::new(),
        false,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    Ok((config, vb))
}

fn load_gguf(
    path: &Path,
    device: &Device,
    target_dtype: DType,
) -> Result<(DFlashModelConfig, ShardedVarBuilder)> {
    let archive = Arc::new(GgufArchive::open_file(path)?);
    let config = config_from_gguf(&archive)?;
    let bindings = bindings_from_gguf(&archive, &config)?;
    let dtype = match target_dtype {
        DType::BF16 | DType::F16 | DType::F32 => target_dtype,
        _ => DType::BF16,
    };
    let source = Arc::new(GgufWeightSource::new(archive, &bindings, dtype)?);
    Ok((config, source.sharded_var_builder(device.clone())))
}

fn config_from_gguf(archive: &GgufArchive) -> Result<DFlashModelConfig> {
    let architecture = metadata_string(archive, "general.architecture")?;
    if architecture != DFLASH_GGUF_ARCHITECTURE {
        candle_core::bail!(
            "DFlash GGUF architecture mismatch: expected `{DFLASH_GGUF_ARCHITECTURE}`, got `{architecture}`"
        );
    }
    let target_layer_ids = metadata_usize_array(archive, "dflash.target_layers")?
        .into_iter()
        .map(|layer| {
            layer.checked_sub(1).ok_or_else(|| {
                candle_core::Error::msg(
                    "DFlash GGUF target layer IDs must be one-indexed and positive",
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let num_hidden_layers = metadata_usize(archive, "dflash.block_count")?;
    let head_dim = metadata_usize(archive, "dflash.attention.key_length")?;
    let value_dim = metadata_usize(archive, "dflash.attention.value_length")?;
    if value_dim != head_dim {
        candle_core::bail!(
            "DFlash GGUF value length {value_dim} does not match key length {head_dim}"
        );
    }
    let sliding_pattern = metadata_bool_array(archive, "dflash.attention.sliding_window_pattern")?;
    if sliding_pattern.len() != num_hidden_layers || sliding_pattern.contains(&false) {
        candle_core::bail!(
            "DFlash GGUF currently requires sliding attention in every assistant layer"
        );
    }
    Ok(DFlashModelConfig {
        model_type: DFLASH_MODEL_TYPE.to_string(),
        hidden_size: metadata_usize(archive, "dflash.embedding_length")?,
        intermediate_size: metadata_usize(archive, "dflash.feed_forward_length")?,
        num_hidden_layers,
        num_attention_heads: metadata_usize(archive, "dflash.attention.head_count")?,
        num_key_value_heads: metadata_usize(archive, "dflash.attention.head_count_kv")?,
        head_dim,
        rms_norm_eps: metadata_f64(archive, "dflash.attention.layer_norm_rms_epsilon")?,
        max_position_embeddings: metadata_usize(archive, "dflash.context_length")?,
        sliding_window: metadata_usize(archive, "dflash.attention.sliding_window")?,
        layer_types: vec!["sliding_attention".to_string(); num_hidden_layers],
        hidden_act: "silu".to_string(),
        block_size: metadata_usize(archive, "dflash.block_size")?,
        mask_token_id: u32::try_from(metadata_usize(archive, "tokenizer.ggml.mask_token_id")?)
            .map_err(candle_core::Error::wrap)?,
        target_layer_ids,
        rope_parameters: RopeParameters {
            rope_theta: metadata_f64(archive, "dflash.rope.freq_base")?,
        },
    })
}

fn bindings_from_gguf(archive: &GgufArchive, config: &DFlashModelConfig) -> Result<GgufBindingMap> {
    let mut bindings = GgufBindingMap::new();
    let mut bind = |native: String, source: String| -> Result<()> {
        if !archive.contains_tensor(&source) {
            candle_core::bail!("DFlash GGUF is missing tensor `{source}`");
        }
        bindings.insert(native, GgufTensorBinding::tensor(source));
        Ok(())
    };
    bind("encoder.fc.weight".to_string(), "fc.weight".to_string())?;
    bind(
        "encoder.output_norm_enc.weight".to_string(),
        "enc.output_norm.weight".to_string(),
    )?;
    bind("norm.weight".to_string(), "output_norm.weight".to_string())?;
    for layer in 0..config.num_hidden_layers {
        let native = format!("layers.{layer}");
        let source = format!("blk.{layer}");
        for (target, role) in [
            ("input_layernorm.weight", "attn_norm.weight"),
            ("post_attention_layernorm.weight", "ffn_norm.weight"),
            ("self_attn.q_proj.weight", "attn_q.weight"),
            ("self_attn.k_proj.weight", "attn_k.weight"),
            ("self_attn.v_proj.weight", "attn_v.weight"),
            ("self_attn.o_proj.weight", "attn_output.weight"),
            ("self_attn.q_norm.weight", "attn_q_norm.weight"),
            ("self_attn.k_norm.weight", "attn_k_norm.weight"),
            ("mlp.gate_proj.weight", "ffn_gate.weight"),
            ("mlp.up_proj.weight", "ffn_up.weight"),
            ("mlp.down_proj.weight", "ffn_down.weight"),
        ] {
            bind(format!("{native}.{target}"), format!("{source}.{role}"))?;
        }
    }
    let consumed = bindings
        .iter()
        .filter_map(|(_, binding)| match binding {
            GgufTensorBinding::Tensor(source) => Some(source.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let available = archive.tensors().keys().cloned().collect::<BTreeSet<_>>();
    if consumed != available {
        let missing = available.difference(&consumed).cloned().collect::<Vec<_>>();
        let unknown = consumed.difference(&available).cloned().collect::<Vec<_>>();
        candle_core::bail!(
            "DFlash GGUF tensor inventory mismatch: unbound={missing:?}, unknown={unknown:?}"
        );
    }
    Ok(bindings)
}

fn metadata_string<'a>(archive: &'a GgufArchive, key: &str) -> Result<&'a str> {
    match archive.metadata_value(key) {
        Some(GgufValue::String(value)) => Ok(value),
        Some(_) => candle_core::bail!("DFlash GGUF metadata `{key}` must be a string"),
        None => candle_core::bail!("DFlash GGUF metadata `{key}` is required"),
    }
}

fn gguf_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U8(value) => Some(u64::from(*value)),
        GgufValue::U16(value) => Some(u64::from(*value)),
        GgufValue::U32(value) => Some(u64::from(*value)),
        GgufValue::U64(value) => Some(*value),
        GgufValue::I8(value) => u64::try_from(*value).ok(),
        GgufValue::I16(value) => u64::try_from(*value).ok(),
        GgufValue::I32(value) => u64::try_from(*value).ok(),
        GgufValue::I64(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

fn metadata_usize(archive: &GgufArchive, key: &str) -> Result<usize> {
    archive
        .metadata_value(key)
        .and_then(gguf_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            candle_core::Error::msg(format!(
                "DFlash GGUF metadata `{key}` must be a nonnegative integer"
            ))
        })
}

fn metadata_usize_array(archive: &GgufArchive, key: &str) -> Result<Vec<usize>> {
    let Some(GgufValue::Array(values)) = archive.metadata_value(key) else {
        candle_core::bail!("DFlash GGUF metadata `{key}` must be an integer array");
    };
    values
        .iter()
        .map(|value| {
            gguf_u64(value)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "DFlash GGUF metadata `{key}` must be an integer array"
                    ))
                })
        })
        .collect()
}

fn metadata_bool_array(archive: &GgufArchive, key: &str) -> Result<Vec<bool>> {
    let Some(GgufValue::Array(values)) = archive.metadata_value(key) else {
        candle_core::bail!("DFlash GGUF metadata `{key}` must be a boolean array");
    };
    values
        .iter()
        .map(|value| match value {
            GgufValue::Bool(value) => Ok(*value),
            _ => candle_core::bail!("DFlash GGUF metadata `{key}` must be a boolean array"),
        })
        .collect()
}

fn metadata_f64(archive: &GgufArchive, key: &str) -> Result<f64> {
    match archive.metadata_value(key) {
        Some(GgufValue::F32(value)) => Ok(f64::from(*value)),
        Some(GgufValue::F64(value)) => Ok(*value),
        Some(_) => candle_core::bail!("DFlash GGUF metadata `{key}` must be a float"),
        None => candle_core::bail!("DFlash GGUF metadata `{key}` is required"),
    }
}

struct DFlashModel {
    encoder: Arc<dyn QuantMethod>,
    encoder_norm: DFlashRmsNorm,
    layers: Vec<DFlashLayer>,
    norm: DFlashRmsNorm,
    config: DFlashModelConfig,
    device: Device,
}

impl DFlashModel {
    fn new(config: &DFlashModelConfig, vb: ShardedVarBuilder, device: &Device) -> Result<Self> {
        let vb = vb.set_device(device.clone());
        let encoder = ReplicatedLayer::new(
            config.target_layer_ids.len() * config.hidden_size,
            config.hidden_size,
            &None,
            false,
            vb.pp("encoder").pp("fc"),
        )?;
        let encoder_norm = DFlashRmsNorm::new(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("encoder").pp("output_norm_enc"),
        )?;
        let rotary = Arc::new(RotaryEmbedding::new(
            config.rope_parameters.rope_theta as f32,
            config.head_dim,
            config.max_position_embeddings,
            device,
            true,
            vb.dtype(),
        )?);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            layers.push(DFlashLayer::new(
                config,
                rotary.clone(),
                vb.pp("layers").pp(layer),
            )?);
        }
        let norm = DFlashRmsNorm::new(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            encoder,
            encoder_norm,
            layers,
            norm,
            config: config.clone(),
            device: device.clone(),
        })
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn prepare_capture(&self, capture: DFlashTargetCapture) -> Result<DFlashPendingCapture> {
        if capture.states.len() != self.config.target_layer_ids.len() {
            candle_core::bail!(
                "DFlash target capture has {} layers, expected {}",
                capture.states.len(),
                self.config.target_layer_ids.len()
            );
        }
        let len = capture
            .end
            .checked_sub(capture.start)
            .ok_or_else(|| candle_core::Error::msg("DFlash target capture range is reversed"))?;
        let states = capture
            .states
            .iter()
            .map(|state| {
                if state.dims() != [len, self.config.hidden_size] {
                    candle_core::bail!(
                        "DFlash target state shape {:?} does not match [{len}, {}]",
                        state.dims(),
                        self.config.hidden_size
                    );
                }
                state.to_device(&self.device)
            })
            .collect::<Result<Vec<_>>>()?;
        let encoded = self.encoder.forward(&Tensor::cat(&states, D::Minus1)?)?;
        let encoded = self.encoder_norm.forward(&encoded)?.unsqueeze(0)?;
        let positions = positions(capture.start, capture.end, &self.device)?;
        let layers = self
            .layers
            .iter()
            .map(|layer| layer.context_kv(&encoded, &positions))
            .collect::<Result<Vec<_>>>()?;
        Ok(DFlashPendingCapture {
            start: capture.start,
            end: capture.end,
            layers,
        })
    }

    fn forward(
        &self,
        noise: &Tensor,
        base_len: usize,
        state: &mut DFlashSequenceState,
    ) -> Result<Tensor> {
        if state.committed_end != base_len {
            candle_core::bail!(
                "DFlash cache length {} does not match target length {base_len}",
                state.committed_end
            );
        }
        if state.layers.len() != self.layers.len() {
            candle_core::bail!("DFlash sequence cache layer count changed");
        }
        let end = base_len
            .checked_add(DFLASH_BLOCK_SIZE)
            .ok_or_else(|| candle_core::Error::msg("DFlash proposal position overflow"))?;
        if end > self.config.max_position_embeddings {
            candle_core::bail!(
                "DFlash proposal end {end} exceeds context length {}",
                self.config.max_position_embeddings
            );
        }
        let positions = positions(base_len, end, &self.device)?;
        let mut hidden = noise.clone();
        for (layer, cache) in self.layers.iter().zip(&state.layers) {
            hidden = layer.forward(
                &hidden,
                cache,
                state.tail_start,
                state.committed_end,
                &positions,
            )?;
        }
        self.norm.forward(&hidden)
    }
}

fn positions(start: usize, end: usize, device: &Device) -> Result<Tensor> {
    let start = u32::try_from(start).map_err(candle_core::Error::wrap)?;
    let end = u32::try_from(end).map_err(candle_core::Error::wrap)?;
    Tensor::arange(start, end, device)
}

struct DFlashRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl DFlashRmsNorm {
    fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?,
            eps,
        })
    }
}

impl Module for DFlashRmsNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        rms_norm_f32(xs, Some((&self.weight, false)), self.eps)
    }
}

struct DFlashLayer {
    attention: DFlashAttention,
    mlp: DFlashMlp,
    input_layernorm: DFlashRmsNorm,
    post_attention_layernorm: DFlashRmsNorm,
}

impl DFlashLayer {
    fn new(
        config: &DFlashModelConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attention: DFlashAttention::new(config, rotary, vb.pp("self_attn"))?,
            mlp: DFlashMlp::new(config, vb.pp("mlp"))?,
            input_layernorm: DFlashRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: DFlashRmsNorm::new(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn context_kv(&self, context: &Tensor, positions: &Tensor) -> Result<DFlashLayerCache> {
        self.attention.context_kv(context, positions)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cache: &DFlashLayerCache,
        cache_start: usize,
        cache_end: usize,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let residual = hidden;
        let hidden = self.input_layernorm.forward(hidden)?;
        let hidden = self
            .attention
            .forward(&hidden, cache, cache_start, cache_end, positions)?;
        let hidden = (residual + hidden)?;
        let residual = &hidden;
        let hidden = self.post_attention_layernorm.forward(&hidden)?;
        residual + self.mlp.forward(&hidden)?
    }
}

struct DFlashAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: DFlashRmsNorm,
    k_norm: DFlashRmsNorm,
    rotary: Arc<RotaryEmbedding>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    sdpa_params: SdpaParams,
}

impl DFlashAttention {
    fn new(
        config: &DFlashModelConfig,
        rotary: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let query_size = config.num_attention_heads * config.head_dim;
        let kv_size = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            q_proj: ReplicatedLayer::new(
                config.hidden_size,
                query_size,
                &None,
                false,
                vb.pp("q_proj"),
            )?,
            k_proj: ReplicatedLayer::new(
                config.hidden_size,
                kv_size,
                &None,
                false,
                vb.pp("k_proj"),
            )?,
            v_proj: ReplicatedLayer::new(
                config.hidden_size,
                kv_size,
                &None,
                false,
                vb.pp("v_proj"),
            )?,
            o_proj: ReplicatedLayer::new(
                query_size,
                config.hidden_size,
                &None,
                false,
                vb.pp("o_proj"),
            )?,
            q_norm: DFlashRmsNorm::new(config.head_dim, config.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: DFlashRmsNorm::new(config.head_dim, config.rms_norm_eps, vb.pp("k_norm"))?,
            rotary,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            sliding_window: config.sliding_window,
            sdpa_params: SdpaParams {
                n_kv_groups: config.num_attention_heads / config.num_key_value_heads,
                softcap: None,
                softmax_scale: 1.0 / (config.head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    fn context_kv(&self, context: &Tensor, positions: &Tensor) -> Result<DFlashLayerCache> {
        let (batch, len, _) = context.dims3()?;
        let key = self
            .k_proj
            .forward(context)?
            .reshape((batch, len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self
            .rotary
            .forward_q(&self.k_norm.forward(&key)?, positions)?;
        let value = self
            .v_proj
            .forward(context)?
            .reshape((batch, len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        Ok(DFlashLayerCache { key, value })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cache: &DFlashLayerCache,
        cache_start: usize,
        cache_end: usize,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let (batch, query_len, _) = hidden.dims3()?;
        if batch != 1 || query_len != DFLASH_BLOCK_SIZE {
            candle_core::bail!(
                "DFlash noise shape must be [1, {DFLASH_BLOCK_SIZE}, hidden], got {:?}",
                hidden.dims()
            );
        }
        let query = self
            .q_proj
            .forward(hidden)?
            .reshape((batch, query_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query = self
            .rotary
            .forward_q(&self.q_norm.forward(&query)?, positions)?;
        let noise = self.context_kv(hidden, positions)?;
        let key = if cache.key.dim(2)? == 0 {
            noise.key
        } else {
            Tensor::cat(&[&cache.key, &noise.key], 2)?
        };
        let value = if cache.value.dim(2)? == 0 {
            noise.value
        } else {
            Tensor::cat(&[&cache.value, &noise.value], 2)?
        };
        let mask = dflash_mask(
            cache_start,
            cache_end,
            query_len,
            self.sliding_window,
            hidden.dtype(),
            hidden.device(),
        )?;
        let output = Sdpa.run_attention(
            &query.contiguous()?,
            &key.contiguous()?,
            &value.contiguous()?,
            &AttentionMask::Custom(mask),
            None,
            &self.sdpa_params,
        )?;
        self.o_proj.forward(&output.transpose(1, 2)?.reshape((
            batch,
            query_len,
            self.num_heads * self.head_dim,
        ))?)
    }
}

fn dflash_mask(
    cache_start: usize,
    cache_end: usize,
    query_len: usize,
    sliding_window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let cache_len = cache_end
        .checked_sub(cache_start)
        .ok_or_else(|| candle_core::Error::msg("DFlash cache position range is reversed"))?;
    let kv_len = cache_len
        .checked_add(query_len)
        .ok_or_else(|| candle_core::Error::msg("DFlash attention length overflow"))?;
    let mut mask = Vec::with_capacity(query_len * kv_len);
    for query in cache_end..cache_end + query_len {
        for key in cache_start..cache_end + query_len {
            mask.push(if query.abs_diff(key) <= sliding_window {
                0.0f32
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::from_vec(mask, (1, 1, query_len, kv_len), device)?.to_dtype(dtype)
}

struct DFlashMlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
}

impl DFlashMlp {
    fn new(config: &DFlashModelConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: ReplicatedLayer::new(
                config.hidden_size,
                config.intermediate_size,
                &None,
                false,
                vb.pp("gate_proj"),
            )?,
            up_proj: ReplicatedLayer::new(
                config.hidden_size,
                config.intermediate_size,
                &None,
                false,
                vb.pp("up_proj"),
            )?,
            down_proj: ReplicatedLayer::new(
                config.intermediate_size,
                config.hidden_size,
                &None,
                false,
                vb.pp("down_proj"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(hidden)?;
        let up = self.up_proj.forward(hidden)?;
        self.down_proj
            .forward(&crate::ops::mul_and_act(&gate, &up, Activation::Silu)?)
    }
}

struct DFlashLayerCache {
    key: Tensor,
    value: Tensor,
}

impl DFlashLayerCache {
    fn empty(device: &Device, dtype: DType, num_kv_heads: usize, head_dim: usize) -> Result<Self> {
        Ok(Self {
            key: Tensor::zeros((1, num_kv_heads, 0, head_dim), dtype, device)?,
            value: Tensor::zeros((1, num_kv_heads, 0, head_dim), dtype, device)?,
        })
    }

    fn prefix(&self, rows: usize) -> Result<Self> {
        Ok(Self {
            key: self.key.narrow(2, 0, rows)?,
            value: self.value.narrow(2, 0, rows)?,
        })
    }

    fn append(&mut self, next: &Self, sliding_window: usize) -> Result<()> {
        self.key = Tensor::cat(&[&self.key, &next.key], 2)?;
        self.value = Tensor::cat(&[&self.value, &next.value], 2)?;
        let len = self.key.dim(2)?;
        if len > sliding_window {
            self.key = self.key.narrow(2, len - sliding_window, sliding_window)?;
            self.value = self.value.narrow(2, len - sliding_window, sliding_window)?;
        }
        Ok(())
    }
}

struct DFlashPendingCapture {
    start: usize,
    end: usize,
    layers: Vec<DFlashLayerCache>,
}

impl DFlashPendingCapture {
    fn len(&self) -> usize {
        self.end - self.start
    }

    fn prefix(&self, rows: usize) -> Result<Self> {
        Ok(Self {
            start: self.start,
            end: self.start + rows,
            layers: self
                .layers
                .iter()
                .map(|layer| layer.prefix(rows))
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

struct DFlashSequenceState {
    layers: Vec<DFlashLayerCache>,
    tail_start: usize,
    committed_end: usize,
    pending: Option<DFlashPendingCapture>,
    sliding_window: usize,
}

impl DFlashSequenceState {
    fn new(layer_count: usize, sliding_window: usize) -> Self {
        Self {
            layers: Vec::with_capacity(layer_count),
            tail_start: 0,
            committed_end: 0,
            pending: None,
            sliding_window,
        }
    }

    fn clear(&mut self) {
        self.layers.clear();
        self.tail_start = 0;
        self.committed_end = 0;
        self.pending = None;
    }

    fn stage(&mut self, pending: DFlashPendingCapture) -> Result<()> {
        if self.pending.is_some() {
            candle_core::bail!("DFlash target capture was replaced before commit");
        }
        self.pending = Some(pending);
        Ok(())
    }

    fn commit_pending(&mut self, rows: Option<usize>, expected_end: usize) -> Result<()> {
        let Some(pending) = self.pending.take() else {
            if rows.is_none() && self.committed_end == expected_end {
                return Ok(());
            }
            candle_core::bail!(
                "DFlash has no pending target capture for expected length {expected_end}"
            );
        };
        let rows = rows.unwrap_or_else(|| pending.len());
        if rows > pending.len() {
            candle_core::bail!(
                "DFlash commit requested {rows} rows from {} captured rows",
                pending.len()
            );
        }
        if pending.start + rows != expected_end {
            candle_core::bail!(
                "DFlash committed target end {} does not match expected {expected_end}",
                pending.start + rows
            );
        }
        self.commit_capture(pending.prefix(rows)?, Some(expected_end))
    }

    fn commit_capture(
        &mut self,
        pending: DFlashPendingCapture,
        expected_end: Option<usize>,
    ) -> Result<()> {
        if pending.layers.is_empty() {
            candle_core::bail!("DFlash target capture has no assistant layers");
        }
        if self.layers.is_empty() {
            self.layers = pending
                .layers
                .iter()
                .map(|layer| {
                    DFlashLayerCache::empty(
                        layer.key.device(),
                        layer.key.dtype(),
                        layer.key.dim(1)?,
                        layer.key.dim(3)?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
        }
        if self.layers.len() != pending.layers.len() {
            candle_core::bail!("DFlash target capture layer count changed");
        }
        if self.committed_end != 0 && pending.start < self.committed_end {
            candle_core::bail!(
                "DFlash target capture starts at {}, before committed end {}",
                pending.start,
                self.committed_end
            );
        }
        if pending.start > self.committed_end {
            for layer in &mut self.layers {
                *layer = DFlashLayerCache::empty(
                    layer.key.device(),
                    layer.key.dtype(),
                    layer.key.dim(1)?,
                    layer.key.dim(3)?,
                )?;
            }
            self.tail_start = pending.start;
        }
        for (cache, next) in self.layers.iter_mut().zip(&pending.layers) {
            cache.append(next, self.sliding_window)?;
        }
        self.committed_end = pending.end;
        let retained = self.layers[0].key.dim(2)?;
        self.tail_start = self.committed_end.saturating_sub(retained);
        if expected_end.is_some_and(|expected| expected != self.committed_end) {
            candle_core::bail!("DFlash target commit ended at an unexpected position");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::HashMap};

    use mistralrs_quant::ShardedSafeTensors;

    use super::*;

    struct FakeRawTarget {
        embed_calls: Cell<usize>,
        head_calls: Cell<usize>,
    }

    impl DFlashTargetWeights for FakeRawTarget {
        fn device(&self) -> &Device {
            &Device::Cpu
        }

        fn raw_embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
            self.embed_calls.set(self.embed_calls.get() + 1);
            input_ids.to_dtype(DType::F32)
        }

        fn raw_lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
            self.head_calls.set(self.head_calls.get() + 1);
            hidden_states.clone().contiguous()
        }
    }

    fn pending(start: usize, values: &[f32]) -> Result<DFlashPendingCapture> {
        let len = values.len();
        let key = Tensor::from_vec(values.to_vec(), (1, 1, len, 1), &Device::Cpu)?;
        Ok(DFlashPendingCapture {
            start,
            end: start + len,
            layers: vec![DFlashLayerCache {
                key: key.clone(),
                value: key,
            }],
        })
    }

    #[test]
    fn proposal_helpers_use_raw_target_weights_without_transforms() -> Result<()> {
        let target = FakeRawTarget {
            embed_calls: Cell::new(0),
            head_calls: Cell::new(0),
        };
        let ids = Tensor::new(&[[1u32, 2, 3]], &Device::Cpu)?;
        let embeddings = raw_noise_embeddings(&target, &ids, &Device::Cpu)?;
        assert_eq!(embeddings.to_vec2::<f32>()?, [vec![1.0, 2.0, 3.0]]);

        let hidden = Tensor::new(&[[0.25f32, -0.5]], &Device::Cpu)?;
        let logits = raw_candidate_logits(&target, &hidden)?;
        assert_eq!(logits.to_vec2::<f32>()?, [vec![0.25, -0.5]]);
        assert_eq!(target.embed_calls.get(), 1);
        assert_eq!(target.head_calls.get(), 1);
        Ok(())
    }

    #[test]
    fn proposal_block_boundary_falls_back_without_overflow() {
        assert!(proposal_block_fits(48, 64));
        assert!(!proposal_block_fits(49, 64));
        assert!(!proposal_block_fits(usize::MAX, 64));
    }

    #[test]
    fn tiny_assistant_forward_matches_hf_formula_fixture() -> Result<()> {
        let config = DFlashModelConfig {
            model_type: DFLASH_MODEL_TYPE.to_string(),
            hidden_size: 2,
            intermediate_size: 3,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 64,
            sliding_window: 8,
            layer_types: vec!["sliding_attention".to_string()],
            hidden_act: "silu".to_string(),
            block_size: DFLASH_BLOCK_SIZE,
            mask_token_id: 31,
            target_layer_ids: vec![0],
            rope_parameters: RopeParameters { rope_theta: 100.0 },
        };
        let matrix = |values: &[f32], rows, columns| {
            Tensor::from_vec(values.to_vec(), (rows, columns), &Device::Cpu)
        };
        let vector = |values: &[f32]| Tensor::from_vec(values.to_vec(), values.len(), &Device::Cpu);
        let mut weights = HashMap::new();
        weights.insert(
            "encoder.fc.weight".to_string(),
            matrix(&[1.0, 0.5, -0.25, 0.75], 2, 2)?,
        );
        weights.insert(
            "encoder.output_norm_enc.weight".to_string(),
            vector(&[1.1, 0.9])?,
        );
        weights.insert(
            "layers.0.input_layernorm.weight".to_string(),
            vector(&[0.8, 1.2])?,
        );
        weights.insert(
            "layers.0.post_attention_layernorm.weight".to_string(),
            vector(&[1.05, 0.95])?,
        );
        weights.insert(
            "layers.0.self_attn.q_proj.weight".to_string(),
            matrix(&[0.7, 0.1, -0.2, 0.8], 2, 2)?,
        );
        weights.insert(
            "layers.0.self_attn.k_proj.weight".to_string(),
            matrix(&[0.6, -0.3, 0.2, 0.5], 2, 2)?,
        );
        weights.insert(
            "layers.0.self_attn.v_proj.weight".to_string(),
            matrix(&[0.4, 0.7, -0.5, 0.2], 2, 2)?,
        );
        weights.insert(
            "layers.0.self_attn.o_proj.weight".to_string(),
            matrix(&[0.9, 0.1, -0.1, 0.8], 2, 2)?,
        );
        weights.insert(
            "layers.0.self_attn.q_norm.weight".to_string(),
            vector(&[1.1, 0.9])?,
        );
        weights.insert(
            "layers.0.self_attn.k_norm.weight".to_string(),
            vector(&[0.95, 1.05])?,
        );
        weights.insert(
            "layers.0.mlp.gate_proj.weight".to_string(),
            matrix(&[0.3, -0.1, -0.2, 0.25, 0.15, 0.2], 3, 2)?,
        );
        weights.insert(
            "layers.0.mlp.up_proj.weight".to_string(),
            matrix(&[0.2, 0.05, 0.1, -0.3, -0.25, 0.1], 3, 2)?,
        );
        weights.insert(
            "layers.0.mlp.down_proj.weight".to_string(),
            matrix(&[0.4, -0.2, 0.1, -0.1, 0.3, 0.25], 2, 3)?,
        );
        weights.insert("norm.weight".to_string(), vector(&[1.0, 1.1])?);
        let vb = ShardedSafeTensors::wrap(weights, DType::F32, Device::Cpu);
        let model = DFlashModel::new(&config, vb, &Device::Cpu)?;

        let capture = DFlashTargetCapture {
            states: vec![matrix(&[0.2, -0.4, 0.6, 0.1], 2, 2)?],
            start: 0,
            end: 2,
        };
        let mut state = DFlashSequenceState::new(1, config.sliding_window);
        state.commit_capture(model.prepare_capture(capture)?, None)?;
        let noise = (0..DFLASH_BLOCK_SIZE)
            .flat_map(|row| [0.1 + row as f32 * 0.03, -0.2 + row as f32 * 0.02])
            .collect::<Vec<_>>();
        let noise = Tensor::from_vec(noise, (1, DFLASH_BLOCK_SIZE, 2), &Device::Cpu)?;
        let output = model.forward(&noise, 2, &mut state)?.flatten_all()?;
        let values = output.to_vec1::<f32>()?;
        let expected_first = [
            -0.217_300_98,
            -1.537_129_5,
            -0.139_752_22,
            -1.547_987_8,
            -0.072_869_04,
            -1.553_532,
            0.079_409_845,
            -1.553_139_2,
        ];
        for (actual, expected) in values.iter().zip(expected_first) {
            assert!((actual - expected).abs() < 2e-4, "{actual} != {expected}");
        }
        let sum = values.iter().sum::<f32>();
        assert!((sum - -8.508_78).abs() < 5e-4, "unexpected sum {sum}");
        Ok(())
    }

    #[test]
    fn gguf_target_layers_are_one_indexed() {
        let values = [2usize, 14, 26, 38, 50];
        let converted = values
            .into_iter()
            .map(|layer| layer.checked_sub(1).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(converted, [1, 13, 25, 37, 49]);
    }

    #[test]
    fn bidirectional_sliding_mask_is_inclusive() -> Result<()> {
        let mask = dflash_mask(0, 2048, 16, 2048, DType::F32, &Device::Cpu)?;
        let mask = mask.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;
        assert_eq!(mask[0][0], 0.0);
        assert!(mask[15][0].is_infinite());
        assert_eq!(mask[15][15], 0.0);
        assert_eq!(mask[15][2048 + 15], 0.0);
        Ok(())
    }

    #[test]
    fn prompt_and_baseline_none_commits_bind_the_exact_rows() -> Result<()> {
        let mut state = DFlashSequenceState::new(1, 16);
        state.commit_capture(pending(0, &[0.0, 1.0, 2.0])?, None)?;
        state.commit_pending(None, 3)?;
        assert_eq!(state.committed_end, 3);
        assert_eq!(state.layers[0].key.dim(2)?, 3);

        state.stage(pending(3, &[3.0])?)?;
        state.commit_pending(None, 4)?;
        assert_eq!(state.committed_end, 4);
        assert_eq!(
            state.layers[0].key.flatten_all()?.to_vec1::<f32>()?,
            [0.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn verification_commit_discards_unaccepted_rows() -> Result<()> {
        let mut state = DFlashSequenceState::new(1, 16);
        state.commit_capture(pending(0, &[0.0, 1.0, 2.0, 3.0])?, None)?;
        state.stage(pending(4, &[4.0, 5.0, 6.0, 7.0])?)?;
        state.commit_pending(Some(2), 6)?;
        assert_eq!(state.committed_end, 6);
        assert!(state.pending.is_none());
        assert_eq!(
            state.layers[0].key.flatten_all()?.to_vec1::<f32>()?,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        Ok(())
    }

    #[test]
    fn mixed_batch_fallback_commits_only_the_normal_decode_row() -> Result<()> {
        let mut state = DFlashSequenceState::new(1, 16);
        state.commit_capture(pending(0, &[0.0, 1.0, 2.0])?, None)?;
        state.stage(pending(3, &[3.0])?)?;
        state.commit_pending(None, 4)?;
        assert_eq!(state.committed_end, 4);
        assert_eq!(
            state.layers[0].key.flatten_all()?.to_vec1::<f32>()?,
            [0.0, 1.0, 2.0, 3.0]
        );

        state.stage(pending(4, &[4.0, 5.0])?)?;
        let err = state.commit_pending(None, 5).unwrap_err().to_string();
        assert!(err.contains("does not match expected 5"));
        assert_eq!(state.committed_end, 4);
        Ok(())
    }

    #[test]
    fn capacity_fallback_keeps_capture_state_ready_for_baseline_decode() -> Result<()> {
        let mut state = DFlashSequenceState::new(1, 16);
        state.commit_capture(pending(0, &[0.0, 1.0, 2.0, 3.0])?, None)?;
        state.stage(pending(4, &[4.0, 5.0, 6.0, 7.0])?)?;
        state.commit_pending(Some(2), 6)?;

        state.stage(pending(6, &[6.0])?)?;
        state.commit_pending(None, 7)?;
        assert_eq!(state.committed_end, 7);
        assert_eq!(
            state.layers[0].key.flatten_all()?.to_vec1::<f32>()?,
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn configured_window_controls_committed_tail() -> Result<()> {
        let mut state = DFlashSequenceState::new(1, 3);
        state.commit_capture(pending(0, &[0.0, 1.0, 2.0, 3.0, 4.0])?, None)?;
        assert_eq!(state.tail_start, 2);
        assert_eq!(state.committed_end, 5);
        assert_eq!(
            state.layers[0].key.flatten_all()?.to_vec1::<f32>()?,
            [2.0, 3.0, 4.0]
        );
        Ok(())
    }

    #[test]
    fn official_safetensors_config_matches_target_capture_contract() -> Result<()> {
        let assistant: DFlashModelConfig = serde_json::from_str(
            r#"{
                "model_type":"muse_glimmer_assistant",
                "hidden_size":6656,
                "intermediate_size":19968,
                "num_hidden_layers":5,
                "num_attention_heads":32,
                "num_key_value_heads":8,
                "head_dim":128,
                "rms_norm_eps":1e-5,
                "max_position_embeddings":131072,
                "sliding_window":2048,
                "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
                "hidden_act":"silu",
                "block_size":16,
                "mask_token_id":201818,
                "target_layer_ids":[1,13,25,37,49],
                "rope_parameters":{"rope_theta":500000.0}
            }"#,
        )
        .map_err(candle_core::Error::msg)?;
        let mut target: TextConfig = serde_json::from_str("{}").map_err(candle_core::Error::msg)?;
        target.max_position_embeddings = 32_768;
        assistant.validate(&target)?;
        let effective_context = assistant
            .max_position_embeddings
            .min(target.max_position_embeddings);
        assert!(proposal_block_fits(32_752, effective_context));
        assert!(!proposal_block_fits(32_753, effective_context));
        assert_eq!(assistant.target_layer_ids, [1, 13, 25, 37, 49]);
        assert_eq!(assistant.block_size - 1, DFLASH_MAX_N_PREDICT);
        Ok(())
    }

    #[test]
    #[ignore = "requires the official Muse-Glimmer DFlash GGUF sidecar"]
    fn official_q4_sidecar_loads() -> Result<()> {
        let path = std::env::var("MISTRALRS_MUSE_DFLASH_GGUF")
            .map_err(|_| candle_core::Error::msg("MISTRALRS_MUSE_DFLASH_GGUF is not set"))?;
        let target: TextConfig = serde_json::from_str("{}").map_err(candle_core::Error::msg)?;
        let runtime = DFlashRuntime::load(
            DFlashConfig::new(path, None, None),
            &target,
            &Device::Cpu,
            DType::F32,
        )?;
        assert_eq!(runtime.target_layer_ids(), [1, 13, 25, 37, 49]);
        assert_eq!(runtime.proposal_len(), 15);
        Ok(())
    }
}
