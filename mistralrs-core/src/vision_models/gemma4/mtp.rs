use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::ShardedVarBuilder;
use rand_isaac::Isaac64Rng;
use serde::Deserialize;

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::DeviceMapper,
    get_mut_arcmutex,
    layers::{Activation, RotaryEmbedding},
    paged_attention::PagedAttention,
    pipeline::text_models_inputs_processor::{
        FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
    },
    sequence::Sequence,
    speculative::{
        MtpConfig, SpeculativeKvCache, SpeculativeProposal, SpeculativeProposalBatch,
        SpeculativeProposeBatchCtx, SpeculativeProposer, TargetTokenEmbedder,
    },
    utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor},
};

use super::{
    config::Gemma4TextConfig,
    text::{first_kv_shared_layer_idx, ProportionalRotaryEmbedding},
};

#[derive(Debug, Clone, Deserialize)]
struct Gemma4AssistantConfig {
    #[serde(default)]
    model_type: String,
    backbone_hidden_size: usize,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default)]
    use_ordered_embeddings: bool,
    #[serde(default = "default_num_centroids")]
    num_centroids: usize,
    #[serde(default = "default_centroid_top_k")]
    centroid_intermediate_top_k: usize,
    #[serde(default)]
    dtype: Option<String>,
    text_config: Gemma4TextConfig,
}

fn default_num_centroids() -> usize {
    2048
}

fn default_centroid_top_k() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
struct AssistantGenerationConfig {
    num_assistant_tokens: Option<usize>,
}

pub struct Gemma4MtpRuntime {
    model: Gemma4MtpModel,
    n_predict: usize,
}

impl Gemma4MtpRuntime {
    pub fn load(
        config: MtpConfig,
        target_cfg: &Gemma4TextConfig,
        device: &Device,
        mapper: &dyn DeviceMapper,
        silent: bool,
    ) -> Result<Self> {
        let path = config.resolve_path()?;
        let config_path = path.join("config.json");
        let raw_config = fs::read_to_string(&config_path).map_err(|e| {
            candle_core::Error::Msg(format!(
                "failed to read MTP config at {}: {e}",
                config_path.display()
            ))
        })?;
        let assistant_cfg: Gemma4AssistantConfig =
            serde_json::from_str(&raw_config).map_err(candle_core::Error::msg)?;

        if assistant_cfg.model_type != "gemma4_assistant" {
            candle_core::bail!(
                "MTP model_type mismatch: expected `gemma4_assistant`, got `{}`",
                assistant_cfg.model_type
            );
        }
        if assistant_cfg.backbone_hidden_size != target_cfg.hidden_size {
            candle_core::bail!(
                "MTP backbone hidden size mismatch: assistant {}, target {}",
                assistant_cfg.backbone_hidden_size,
                target_cfg.hidden_size
            );
        }
        if assistant_cfg.text_config.vocab_size != target_cfg.vocab_size {
            candle_core::bail!(
                "MTP vocab size mismatch: assistant {}, target {}",
                assistant_cfg.text_config.vocab_size,
                target_cfg.vocab_size
            );
        }
        if !assistant_cfg.tie_word_embeddings {
            candle_core::bail!("MTP currently expects tied assistant word embeddings.");
        }
        if !assistant_cfg.use_ordered_embeddings {
            candle_core::bail!("MTP currently requires ordered centroid embeddings.");
        }

        let mut weight_paths = fs::read_dir(&path)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "failed to list MTP model directory {}: {e}",
                    path.display()
                ))
            })?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        weight_paths.sort();
        if weight_paths.is_empty() {
            candle_core::bail!(
                "MTP model directory {} has no safetensors weights.",
                path.display()
            );
        }

        let dtype = dtype_from_config(assistant_cfg.dtype.as_deref());
        let vb = from_mmaped_safetensors(
            weight_paths,
            Vec::new(),
            Some(dtype),
            device,
            Vec::new(),
            silent,
            None,
            |_| true,
            Arc::new(|_| DeviceForLoadTensor::Base),
        )?;

        let n_predict = match config.n_predict {
            Some(n) => n,
            None => read_generation_n_predict(&path)?.unwrap_or(6),
        };
        let model = Gemma4MtpModel::new(&assistant_cfg, target_cfg, vb, device, mapper)?;
        Ok(Self { model, n_predict })
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_tokens(
        &self,
        sampled_tokens: &[u32],
        sampled_tokens_emitted: bool,
        target_embedder: &TargetTokenEmbedder<'_>,
        target_hiddens: Tensor,
        seq_ids: &[usize],
        base_lens: &[usize],
        sequences: &[&Sequence],
        rng: Arc<Mutex<Isaac64Rng>>,
        cache: SpeculativeKvCache<'_>,
    ) -> Result<Vec<SpeculativeProposal>> {
        let batch = sampled_tokens.len();
        if batch == 0 {
            return Ok(Vec::new());
        }
        if seq_ids.len() != batch || base_lens.len() != batch || sequences.len() != batch {
            candle_core::bail!(
                "MTP batch shape mismatch: sampled={}, seq_ids={}, base_lens={}, sequences={}",
                batch,
                seq_ids.len(),
                base_lens.len(),
                sequences.len()
            );
        }
        if target_hiddens.dim(0)? != batch {
            candle_core::bail!(
                "MTP hidden batch mismatch: hidden={}, sampled={}",
                target_hiddens.dim(0)?,
                batch
            );
        }

        match cache {
            SpeculativeKvCache::Paged { metadata, kv_cache } => {
                let input_metadata =
                    make_mtp_decode_metadata(seq_ids, base_lens, metadata, self.model.device())?;
                let cache = Gemma4MtpStepCache::Paged {
                    kv_cache,
                    input_metadata: &input_metadata,
                };
                self.propose_tokens_with_cache(
                    sampled_tokens,
                    sampled_tokens_emitted,
                    target_embedder,
                    target_hiddens,
                    base_lens,
                    sequences,
                    rng,
                    &cache,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn propose_tokens_with_cache(
        &self,
        sampled_tokens: &[u32],
        sampled_tokens_emitted: bool,
        target_embedder: &TargetTokenEmbedder<'_>,
        target_hiddens: Tensor,
        base_lens: &[usize],
        sequences: &[&Sequence],
        rng: Arc<Mutex<Isaac64Rng>>,
        cache: &Gemma4MtpStepCache<'_>,
    ) -> Result<Vec<SpeculativeProposal>> {
        let batch = sampled_tokens.len();
        let mut contexts = sequences
            .iter()
            .zip(sampled_tokens.iter().copied())
            .map(|(seq, sampled)| {
                let mut context = seq.get_toks().to_vec();
                if !sampled_tokens_emitted {
                    context.push(sampled);
                }
                context
            })
            .collect::<Vec<_>>();
        let mut last_token =
            Tensor::from_vec(sampled_tokens.to_vec(), (batch, 1), self.model.device())?;
        let mut hidden = target_hiddens;
        let mut tokens = Vec::with_capacity(self.n_predict);
        let mut logits = Vec::with_capacity(self.n_predict);
        for _ in 0..self.n_predict {
            let input_embed = target_embedder(&last_token)?;
            let (_argmax_token, draft_logits, next_hidden) =
                self.model.step(input_embed, hidden, base_lens, cache)?;
            let draft_token = sample_draft_tokens(&draft_logits, sequences, &mut contexts, &rng)?;
            tokens.push(draft_token.clone());
            logits.push(draft_logits);
            last_token = draft_token;
            hidden = next_hidden;
        }
        if tokens.is_empty() {
            return Ok((0..batch)
                .map(|_| SpeculativeProposal::new(Vec::new()))
                .collect());
        }
        let tokens = Tensor::cat(&tokens, D::Minus1)?;
        let tokens = tokens.to_vec2::<u32>()?;
        let logits = Tensor::cat(&logits, 1)?;
        tokens
            .into_iter()
            .enumerate()
            .map(|(row, tokens)| Ok(SpeculativeProposal::with_logits(tokens, logits.get(row)?)))
            .collect()
    }
}

fn sample_draft_tokens(
    logits: &Tensor,
    sequences: &[&Sequence],
    contexts: &mut [Vec<u32>],
    rng: &Arc<Mutex<Isaac64Rng>>,
) -> Result<Tensor> {
    let batch = sequences.len();
    if contexts.len() != batch {
        candle_core::bail!(
            "MTP sampling context batch mismatch: contexts={}, sequences={batch}",
            contexts.len()
        );
    }

    let mut tokens = Vec::with_capacity(batch);
    for (row, seq) in sequences.iter().enumerate() {
        let row_logits = logits.get(row)?.squeeze(0)?.to_dtype(DType::F32)?;
        let sampled = seq.sampler().sample(
            row_logits,
            &contexts[row],
            false,
            rng.clone(),
            false,
            batch > 1,
        )?;
        contexts[row].push(sampled.token);
        tokens.push(sampled.token);
    }

    Tensor::from_vec(tokens, (batch, 1), logits.device())
}

impl SpeculativeProposer for Gemma4MtpRuntime {
    fn proposal_len(&self) -> usize {
        self.n_predict
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let target_hiddens = ctx.target_hiddens.ok_or_else(|| {
            candle_core::Error::Msg(
                "MTP requires target hidden state for speculative proposal.".to_string(),
            )
        })?;
        let target_embedder = target_embedder.ok_or_else(|| {
            candle_core::Error::Msg(
                "MTP requires a target token embedder for speculative proposal.".to_string(),
            )
        })?;
        let proposals = self.propose_tokens(
            ctx.sampled_tokens,
            ctx.sampled_tokens_emitted,
            target_embedder,
            target_hiddens,
            ctx.seq_ids,
            ctx.base_lens,
            ctx.sequences,
            ctx.rng,
            ctx.cache,
        )?;
        Ok(SpeculativeProposalBatch::new(proposals))
    }
}

fn dtype_from_config(dtype: Option<&str>) -> DType {
    match dtype {
        Some("float32" | "f32") => DType::F32,
        Some("float16" | "f16") => DType::F16,
        Some("bfloat16" | "bf16") => DType::BF16,
        _ => DType::BF16,
    }
}

fn read_generation_n_predict(path: &Path) -> Result<Option<usize>> {
    let path = path.join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "failed to read MTP generation config at {}: {e}",
            path.display()
        ))
    })?;
    let cfg: AssistantGenerationConfig =
        serde_json::from_str(&raw).map_err(candle_core::Error::msg)?;
    Ok(cfg.num_assistant_tokens)
}

struct Gemma4MtpModel {
    pre_projection: Linear,
    post_projection: Linear,
    layers: Vec<Gemma4MtpDecoderLayer>,
    norm: Gemma4MtpRmsNorm,
    lm_head_weight: Tensor,
    masked_embedding: Gemma4MtpMaskedEmbedding,
    device: Device,
}

enum Gemma4MtpStepCache<'a> {
    Paged {
        kv_cache: &'a [(Tensor, Tensor)],
        input_metadata: &'a PagedAttentionInputMetadata,
    },
}

struct Gemma4MtpRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Gemma4MtpRmsNorm {
    fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(size, "weight")?.to_dtype(DType::F32)?,
            eps,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let output_dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.powf(2.0)?.mean_keepdim(D::Minus1)?;
        let scale = (&variance + self.eps)?.recip()?.sqrt()?;
        let xs = xs.broadcast_mul(&scale)?;
        xs.broadcast_mul(&self.weight)?.to_dtype(output_dtype)
    }
}

impl Gemma4MtpModel {
    fn new(
        cfg: &Gemma4AssistantConfig,
        target_cfg: &Gemma4TextConfig,
        vb: ShardedVarBuilder,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<Self> {
        let text_cfg = &cfg.text_config;
        let donor_indices = donor_indices(target_cfg, text_cfg)?;

        let pre_projection = linear_no_bias(
            2 * cfg.backbone_hidden_size,
            text_cfg.hidden_size,
            vb.pp("pre_projection"),
        )?;
        let post_projection = linear_no_bias(
            text_cfg.hidden_size,
            cfg.backbone_hidden_size,
            vb.pp("post_projection"),
        )?;

        let vb_m = vb.pp("model");
        let lm_head_weight = vb_m.get(
            (text_cfg.vocab_size, text_cfg.hidden_size),
            "embed_tokens.weight",
        )?;
        let norm =
            Gemma4MtpRmsNorm::new(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let mut layers = Vec::with_capacity(text_cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for (layer_idx, donor_idx) in donor_indices.into_iter().enumerate() {
            layers.push(Gemma4MtpDecoderLayer::new(
                text_cfg,
                layer_idx,
                donor_idx,
                vb_l.pp(layer_idx),
                device,
                mapper,
            )?);
        }

        let masked_embedding = Gemma4MtpMaskedEmbedding::new(
            text_cfg.hidden_size,
            text_cfg.vocab_size,
            cfg.num_centroids,
            cfg.centroid_intermediate_top_k,
            vb.pp("masked_embedding"),
        )?;

        Ok(Self {
            pre_projection,
            post_projection,
            layers,
            norm,
            lm_head_weight,
            masked_embedding,
            device: device.clone(),
        })
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn step(
        &self,
        input_embed: Tensor,
        target_hidden: Tensor,
        positions: &[usize],
        cache: &Gemma4MtpStepCache<'_>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut hidden_states = Tensor::cat(&[input_embed, target_hidden], D::Minus1)?;
        hidden_states = hidden_states.apply(&self.pre_projection)?;

        // Each MTP step is a single query over the target donor KV cache, not
        // a causal prefill over newly produced draft tokens.
        let flash = FlashParams::empty(false);
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, positions, cache, &flash)?;
        }

        let draft_hidden_states = self.norm.forward(&hidden_states)?;
        let backbone_hidden_states = draft_hidden_states.apply(&self.post_projection)?;
        let (token, logits) = self
            .masked_embedding
            .get_logits_and_top_token(&draft_hidden_states, &self.lm_head_weight)?;
        Ok((token, logits, backbone_hidden_states))
    }
}

fn donor_indices(
    target_cfg: &Gemma4TextConfig,
    draft_cfg: &Gemma4TextConfig,
) -> Result<Vec<usize>> {
    let first_shared = first_kv_shared_layer_idx(target_cfg);
    let target_layer_types = &target_cfg.layer_types[..first_shared];
    let mut result = Vec::with_capacity(draft_cfg.num_hidden_layers);
    for (draft_idx, draft_layer_type) in draft_cfg.layer_types.iter().enumerate() {
        let Some(target_idx) = target_layer_types
            .iter()
            .rposition(|layer_type| layer_type == draft_layer_type)
        else {
            candle_core::bail!(
                "MTP draft layer {draft_idx} has type `{draft_layer_type}` but the target has no non-shared donor layer of that type."
            );
        };
        result.push(target_idx);
    }
    Ok(result)
}

struct Gemma4MtpDecoderLayer {
    self_attn: Gemma4MtpAttention,
    mlp: Gemma4MtpMlp,
    input_layernorm: Gemma4MtpRmsNorm,
    post_attention_layernorm: Gemma4MtpRmsNorm,
    pre_feedforward_layernorm: Gemma4MtpRmsNorm,
    post_feedforward_layernorm: Gemma4MtpRmsNorm,
    layer_scalar: Option<Tensor>,
}

impl Gemma4MtpDecoderLayer {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        donor_layer_idx: usize,
        vb: ShardedVarBuilder,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<Self> {
        let self_attn =
            Gemma4MtpAttention::new(cfg, layer_idx, donor_layer_idx, vb.pp("self_attn"), device)?;
        let mlp = Gemma4MtpMlp::new(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_activation,
            vb.pp("mlp"),
        )?;
        let input_layernorm = Gemma4MtpRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = Gemma4MtpRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = Gemma4MtpRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = Gemma4MtpRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_feedforward_layernorm"), false),
        )?;
        let layer_scalar = vb.get((1,), "layer_scalar").ok();
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            layer_scalar,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        positions: &[usize],
        cache: &Gemma4MtpStepCache<'_>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let normed = self.input_layernorm.forward(xs)?;
        let attn = self
            .self_attn
            .forward(&normed, positions, cache, flash_params)?;
        let attn = self.post_attention_layernorm.forward(&attn)?;
        let xs = (attn + residual)?;

        let residual = xs.clone();
        let normed = self.pre_feedforward_layernorm.forward(&xs)?;
        let mlp = self.mlp.forward(&normed)?;
        let mlp = self.post_feedforward_layernorm.forward(&mlp)?;
        let mut xs = (mlp + residual)?;
        if let Some(scalar) = &self.layer_scalar {
            xs = xs.broadcast_mul(scalar)?;
        }
        Ok(xs)
    }
}

struct Gemma4MtpAttention {
    q_proj: Linear,
    o_proj: Linear,
    q_norm: Gemma4MtpRmsNorm,
    rotary_emb_global: Option<ProportionalRotaryEmbedding>,
    rotary_emb_local: Option<RotaryEmbedding>,
    paged_attn: PagedAttention,
    sdpa_params: SdpaParams,
    donor_layer_idx: usize,
    num_heads: usize,
    head_dim: usize,
}

impl Gemma4MtpAttention {
    fn new(
        cfg: &Gemma4TextConfig,
        layer_idx: usize,
        donor_layer_idx: usize,
        vb: ShardedVarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let is_sliding = cfg.layer_types[layer_idx] == "sliding_attention";
        let head_dim = if is_sliding {
            cfg.head_dim
        } else {
            cfg.global_head_dim
        };
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = if is_sliding {
            cfg.num_key_value_heads
        } else {
            cfg.num_global_key_value_heads
                .unwrap_or(cfg.num_key_value_heads)
        };
        let q_proj = linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        let q_norm = Gemma4MtpRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

        let rotary_emb_local = if is_sliding {
            Some(RotaryEmbedding::new(
                cfg.rope_local_base_freq() as f32,
                head_dim,
                cfg.max_position_embeddings,
                device,
                true,
                vb.dtype(),
            )?)
        } else {
            None
        };
        let rotary_emb_global = if is_sliding {
            None
        } else {
            Some(ProportionalRotaryEmbedding::new(
                cfg.rope_theta as f32,
                head_dim,
                cfg.partial_rotary_factor(),
                cfg.max_position_embeddings,
                device,
                true,
                vb.dtype(),
            )?)
        };
        let paged_attn = PagedAttention::new(head_dim, device, None)?;
        let sdpa_params = SdpaParams {
            n_kv_groups: num_heads / num_kv_heads,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: if is_sliding {
                Some(cfg.effective_sliding_window())
            } else {
                None
            },
            sinks: None,
        };
        Ok(Self {
            q_proj,
            o_proj,
            q_norm,
            rotary_emb_global,
            rotary_emb_local,
            paged_attn,
            sdpa_params,
            donor_layer_idx,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        positions: &[usize],
        cache: &Gemma4MtpStepCache<'_>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let mut q = xs.apply(&self.q_proj)?;
        q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
        q = self.q_norm.forward(&q)?;
        q = if let Some(rotary) = &self.rotary_emb_local {
            rotary.forward_q(&q, positions)?
        } else {
            self.rotary_emb_global
                .as_ref()
                .expect("global rotary missing")
                .forward_q(&q, positions)?
        };
        let attn = match cache {
            Gemma4MtpStepCache::Paged {
                kv_cache,
                input_metadata,
            } => {
                let (key_cache, value_cache) =
                    kv_cache.get(self.donor_layer_idx).ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "MTP donor layer {} is missing from the target paged KV cache",
                            self.donor_layer_idx
                        ))
                    })?;
                self.paged_attn.forward_donor_cache(
                    &q,
                    key_cache,
                    value_cache,
                    &AttentionMask::None,
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?
            }
        };
        let attn = attn.reshape((b_sz, q_len, ()))?;
        attn.apply(&self.o_proj)
    }
}

struct Gemma4MtpMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Gemma4MtpMlp {
    fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.activation)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Gemma4MtpMaskedEmbedding {
    centroids: Linear,
    token_ordering: Tensor,
    hidden_size: usize,
    num_centroids: usize,
    centroid_top_k: usize,
    vocab_size_per_centroid: usize,
    num_selected: usize,
}

impl Gemma4MtpMaskedEmbedding {
    fn new(
        hidden_size: usize,
        vocab_size: usize,
        num_centroids: usize,
        centroid_top_k: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let centroids = linear_no_bias(hidden_size, num_centroids, vb.pp("centroids"))?;
        let token_ordering = vb
            .get((vocab_size,), "token_ordering")?
            .to_dtype(DType::U32)?;
        Ok(Self {
            centroids,
            token_ordering,
            hidden_size,
            num_centroids,
            centroid_top_k,
            vocab_size_per_centroid: vocab_size / num_centroids,
            num_selected: centroid_top_k * (vocab_size / num_centroids),
        })
    }

    fn get_logits_and_top_token(
        &self,
        hidden_states: &Tensor,
        lm_head_weight: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let hidden_states = hidden_states.reshape(((), self.hidden_size))?;
        let centroid_logits = hidden_states.apply(&self.centroids)?;
        let top_k_indices = topk_indices_u32(&centroid_logits, self.centroid_top_k)?;
        let clusters = self
            .token_ordering
            .reshape((self.num_centroids, self.vocab_size_per_centroid))?;
        let selected = clusters
            .index_select(&top_k_indices.flatten_all()?.to_dtype(DType::U32)?, 0)?
            .reshape((hidden_states.dim(0)?, self.num_selected))?;
        let selected_embeddings = lm_head_weight
            .index_select(&selected.flatten_all()?.to_dtype(DType::U32)?, 0)?
            .reshape((hidden_states.dim(0)?, self.num_selected, self.hidden_size))?
            .to_dtype(DType::F32)?;
        let logits = hidden_states
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .broadcast_mul(&selected_embeddings)?
            .sum(D::Minus1)?;
        // Match HF's ordered-embedding fallback so speculative q is not
        // artificially concentrated on the selected centroid tokens.
        let mask_value = logits.min_all()?.to_scalar::<f32>()? - 1.0;
        let argmax = logits.argmax(D::Minus1)?;
        let token = selected
            .gather(&argmax.unsqueeze(1)?, D::Minus1)?
            .to_dtype(DType::U32)?;
        let vocab_size = self.num_centroids * self.vocab_size_per_centroid;
        let full_logits = Tensor::full(
            mask_value,
            (hidden_states.dim(0)?, vocab_size),
            logits.device(),
        )?
        .scatter(&selected.to_dtype(DType::U32)?, &logits, D::Minus1)?;
        debug_assert_eq!(token.dims(), &[hidden_states.dim(0)?, 1]);
        Ok((token, full_logits.unsqueeze(1)?))
    }
}

fn topk_indices_u32(logits: &Tensor, top_k: usize) -> Result<Tensor> {
    let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let width = rows
        .first()
        .map(Vec::len)
        .ok_or_else(|| candle_core::Error::Msg("empty top-k logits".into()))?;
    if top_k > width {
        candle_core::bail!("top-k {top_k} exceeds logits width {width}");
    }

    let mut indices = Vec::with_capacity(rows.len() * top_k);
    for row in &rows {
        for (idx, _) in topk_pairs(row, top_k) {
            indices.push(idx);
        }
    }
    Tensor::from_vec(indices, (rows.len(), top_k), logits.device())
}

fn topk_pairs(values: &[f32], top_k: usize) -> Vec<(u32, f32)> {
    let mut ranked = values
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx as u32, value))
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(top_k.min(ranked.len()));
    ranked
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

fn make_mtp_decode_metadata(
    seq_ids: &[usize],
    context_lens: &[usize],
    paged_meta: &PagedAttentionMeta,
    device: &Device,
) -> Result<PagedAttentionInputMetadata> {
    if seq_ids.len() != context_lens.len() {
        candle_core::bail!(
            "MTP metadata batch mismatch: seq_ids={}, context_lens={}",
            seq_ids.len(),
            context_lens.len()
        );
    }

    let kv_mgr = get_mut_arcmutex!(paged_meta.kv_cache_manager);
    let full_tables = seq_ids
        .iter()
        .map(|seq_id| {
            kv_mgr
                .get_block_ids(*seq_id)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "MTP sequence {seq_id} has no paged attention blocks"
                    ))
                })
                .map(|ids| ids.to_vec())
        })
        .collect::<Result<Vec<_>>>()?;
    drop(kv_mgr);

    let mut block_tables = Vec::with_capacity(seq_ids.len());
    let mut context_lens_windowed = Vec::with_capacity(seq_ids.len());
    let mut slot_mappings = Vec::with_capacity(seq_ids.len());
    for (full_table, context_len) in full_tables.iter().zip(context_lens.iter().copied()) {
        let (block_table, context_len_windowed) =
            if let Some(sliding_window) = paged_meta.sliding_window {
                // Keep paged MTP aligned with the normal-cache inclusive SWA mask.
                let window_start = context_len.saturating_sub(sliding_window + 1);
                let slide_idx = window_start / paged_meta.block_size;
                let block_aligned_start = slide_idx * paged_meta.block_size;
                (
                    full_table.get(slide_idx..).unwrap_or(&[]).to_vec(),
                    context_len.saturating_sub(block_aligned_start),
                )
            } else {
                (full_table.clone(), context_len)
            };
        block_tables.push(block_table);
        context_lens_windowed.push(context_len_windowed);

        let block_pos = context_len.saturating_sub(1);
        let slot = full_table
            .get(block_pos / paged_meta.block_size)
            .copied()
            .unwrap_or(0)
            * paged_meta.block_size
            + block_pos % paged_meta.block_size;
        slot_mappings.push(slot as i64);
    }

    let batch = seq_ids.len();
    let slot_mappings = Tensor::from_vec(slot_mappings, (batch,), device)?;

    let windowed_block_tables = block_tables;
    let block_tables = table_tensor(&windowed_block_tables, device)?;
    let full_block_tables = table_tensor(&full_tables, device)?;
    let context_lens_tensor = Tensor::from_vec(
        context_lens_windowed
            .iter()
            .map(|len| usize_to_u32(*len, "windowed context length"))
            .collect::<Result<Vec<_>>>()?,
        (batch,),
        device,
    )?;
    let full_context_lens_tensor = Tensor::from_vec(
        context_lens
            .iter()
            .map(|len| usize_to_u32(*len, "full context length"))
            .collect::<Result<Vec<_>>>()?,
        (batch,),
        device,
    )?;

    let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) = paged_kv_tensors(
        &windowed_block_tables,
        &context_lens_windowed,
        paged_meta.block_size,
        device,
    )?;
    let batch_i32 = usize_to_i32(batch, "MTP batch size")?;
    let request_indices = Tensor::from_vec((0..batch_i32).collect::<Vec<_>>(), (batch,), device)?;
    let kv_tile_indices = Tensor::from_vec(vec![0i32; batch], (batch,), device)?;
    let o_indptr = Tensor::from_vec((0..=batch_i32).collect::<Vec<_>>(), (batch + 1,), device)?;
    let kv_chunk_size = Tensor::from_vec(
        vec![usize_to_i32(paged_meta.block_size, "paged block size")?],
        (1,),
        device,
    )?;

    let location = device.location();
    Ok(PagedAttentionInputMetadata {
        block_tables: Some(HashMap::from([(location, block_tables)])),
        context_lens: Some(HashMap::from([(location, context_lens_tensor)])),
        slot_mappings: HashMap::from([(location, slot_mappings)]),
        max_context_len: Some(context_lens_windowed.iter().copied().max().unwrap_or(0)),
        full_block_tables: Some(HashMap::from([(location, full_block_tables)])),
        full_context_lens: Some(HashMap::from([(location, full_context_lens_tensor)])),
        full_max_context_len: Some(context_lens.iter().copied().max().unwrap_or(0)),
        is_first_prompt_chunk: false,
        paged_kv_indptr: Some(HashMap::from([(location, paged_kv_indptr)])),
        paged_kv_indices: Some(HashMap::from([(location, paged_kv_indices)])),
        paged_kv_last_page_len: Some(HashMap::from([(location, paged_kv_last_page_len)])),
        paged_kv_request_indices: Some(HashMap::from([(location, request_indices)])),
        paged_kv_tile_indices: Some(HashMap::from([(location, kv_tile_indices)])),
        paged_kv_o_indptr: Some(HashMap::from([(location, o_indptr)])),
        paged_kv_chunk_size: Some(HashMap::from([(location, kv_chunk_size)])),
        num_cached_tokens: None,
        query_lens: None,
        cu_seqlens_q: None,
        cu_seqlens_kv: None,
    })
}

fn table_tensor(rows: &[Vec<usize>], device: &Device) -> Result<Tensor> {
    let max_len = rows.iter().map(Vec::len).max().unwrap_or(0).max(1);
    let mut values = Vec::with_capacity(rows.len() * max_len);
    for row in rows {
        for value in row {
            values.push(usize_to_u32(*value, "block table entry")?);
        }
        values.extend(std::iter::repeat_n(0u32, max_len.saturating_sub(row.len())));
    }
    Tensor::from_vec(values, (rows.len(), max_len), device)
}

fn paged_kv_tensors(
    tables: &[Vec<usize>],
    context_lens: &[usize],
    block_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut indptr = Vec::with_capacity(tables.len() + 1);
    let mut indices = Vec::new();
    let mut last_page_len = Vec::with_capacity(tables.len());
    indptr.push(0i32);
    let mut nnz = 0i32;
    for (table, context_len) in tables.iter().zip(context_lens.iter().copied()) {
        nnz = nnz
            .checked_add(usize_to_i32(table.len(), "paged table length")?)
            .ok_or_else(|| candle_core::Error::Msg("paged table nnz overflowed".to_string()))?;
        indptr.push(nnz);
        for value in table {
            indices.push(usize_to_i32(*value, "paged block index")?);
        }
        let len = if table.is_empty() {
            0
        } else {
            usize_to_i32(
                context_len.saturating_sub((table.len() - 1) * block_size),
                "paged last page length",
            )?
        };
        last_page_len.push(len);
    }
    let indptr = Tensor::from_vec(indptr, (tables.len() + 1,), device)?;
    let indices_len = indices.len();
    let indices = Tensor::from_vec(indices, (indices_len,), device)?;
    let last_page_len = Tensor::from_vec(last_page_len, (tables.len(),), device)?;
    Ok((indptr, indices, last_page_len))
}

fn usize_to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("{name} exceeds u32::MAX: {value}")))
}

fn usize_to_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("{name} exceeds i32::MAX: {value}")))
}
