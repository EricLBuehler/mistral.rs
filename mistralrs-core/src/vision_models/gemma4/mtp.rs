use std::{collections::HashMap, fs, path::Path, sync::Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::Linear;
use mistralrs_quant::ShardedVarBuilder;
use serde::Deserialize;

use crate::{
    attention::{AttentionMask, SdpaParams},
    device_map::DeviceMapper,
    get_mut_arcmutex,
    layers::{Activation, RmsNorm, RotaryEmbedding},
    ops::TopKLastDimOp,
    paged_attention::PagedAttention,
    pipeline::text_models_inputs_processor::{
        FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
    },
    speculative::MtpConfig,
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
        let path = config.model.as_path()?;
        let config_path = path.join("config.json");
        let raw_config = fs::read_to_string(&config_path).map_err(|e| {
            candle_core::Error::Msg(format!(
                "failed to read Gemma4 MTP config at {}: {e}",
                config_path.display()
            ))
        })?;
        let assistant_cfg: Gemma4AssistantConfig =
            serde_json::from_str(&raw_config).map_err(candle_core::Error::msg)?;

        if assistant_cfg.model_type != "gemma4_assistant" {
            candle_core::bail!(
                "Gemma4 MTP model_type mismatch: expected `gemma4_assistant`, got `{}`",
                assistant_cfg.model_type
            );
        }
        if assistant_cfg.backbone_hidden_size != target_cfg.hidden_size {
            candle_core::bail!(
                "Gemma4 MTP backbone hidden size mismatch: assistant {}, target {}",
                assistant_cfg.backbone_hidden_size,
                target_cfg.hidden_size
            );
        }
        if assistant_cfg.text_config.vocab_size != target_cfg.vocab_size {
            candle_core::bail!(
                "Gemma4 MTP vocab size mismatch: assistant {}, target {}",
                assistant_cfg.text_config.vocab_size,
                target_cfg.vocab_size
            );
        }
        if !assistant_cfg.tie_word_embeddings {
            candle_core::bail!("Gemma4 MTP currently expects tied assistant word embeddings.");
        }
        if !assistant_cfg.use_ordered_embeddings {
            candle_core::bail!("Gemma4 MTP currently requires ordered centroid embeddings.");
        }

        let mut weight_paths = fs::read_dir(path)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "failed to list Gemma4 MTP model directory {}: {e}",
                    path.display()
                ))
            })?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        weight_paths.sort();
        if weight_paths.is_empty() {
            candle_core::bail!(
                "Gemma4 MTP model directory {} has no safetensors weights.",
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
            None => read_generation_n_predict(path)?.unwrap_or(6),
        };
        let model = Gemma4MtpModel::new(&assistant_cfg, target_cfg, vb, device, mapper)?;
        tracing::info!(
            "Gemma4 MTP loaded from {} with n_predict={n_predict}",
            path.display()
        );
        Ok(Self { model, n_predict })
    }

    pub fn propose(
        &self,
        sampled_token: u32,
        target_embedder: impl Fn(u32) -> Result<Tensor>,
        target_hidden: Tensor,
        seq_id: usize,
        base_len: usize,
        paged_meta: &PagedAttentionMeta,
        kv_cache: &[(Tensor, Tensor)],
    ) -> Result<Vec<u32>> {
        let input_metadata =
            make_mtp_decode_metadata(seq_id, base_len, paged_meta, self.model.device())?;
        let mut last_token = sampled_token;
        let mut hidden = target_hidden;
        let mut tokens = Vec::with_capacity(self.n_predict);
        for _ in 0..self.n_predict {
            let input_embed = target_embedder(last_token)?;
            let (draft_token, next_hidden) =
                self.model
                    .step(input_embed, hidden, base_len, kv_cache, &input_metadata)?;
            tokens.push(draft_token);
            last_token = draft_token;
            hidden = next_hidden;
        }
        Ok(tokens)
    }
}

fn dtype_from_config(dtype: Option<&str>) -> DType {
    match dtype {
        Some("float32" | "f32") => DType::F32,
        Some("float16" | "f16") => DType::F16,
        Some("bfloat16" | "bf16") | _ => DType::BF16,
    }
}

fn read_generation_n_predict(path: &Path) -> Result<Option<usize>> {
    let path = path.join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "failed to read Gemma4 MTP generation config at {}: {e}",
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
    norm: RmsNorm,
    lm_head_weight: Tensor,
    masked_embedding: Gemma4MtpMaskedEmbedding,
    device: Device,
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
        let norm = RmsNorm::new(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb_m.pp("norm"))?;

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
        position: usize,
        kv_cache: &[(Tensor, Tensor)],
        input_metadata: &PagedAttentionInputMetadata,
    ) -> Result<(u32, Tensor)> {
        let mut hidden_states = Tensor::cat(&[input_embed, target_hidden], D::Minus1)?;
        hidden_states = hidden_states.apply(&self.pre_projection)?;

        let flash = FlashParams::empty(true);
        for layer in &self.layers {
            hidden_states =
                layer.forward(&hidden_states, position, kv_cache, input_metadata, &flash)?;
        }

        let draft_hidden_states = hidden_states.apply(&self.norm)?;
        let backbone_hidden_states = draft_hidden_states.apply(&self.post_projection)?;
        let token = self
            .masked_embedding
            .get_top_token(&draft_hidden_states, &self.lm_head_weight)?;
        Ok((token, backbone_hidden_states))
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
                "Gemma4 MTP draft layer {draft_idx} has type `{draft_layer_type}` but the target has no non-shared donor layer of that type."
            );
        };
        result.push(target_idx);
    }
    Ok(result)
}

struct Gemma4MtpDecoderLayer {
    self_attn: Gemma4MtpAttention,
    mlp: Gemma4MtpMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm,
    post_feedforward_layernorm: RmsNorm,
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
        let input_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("input_layernorm"), false),
        )?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("post_attention_layernorm"), false),
        )?;
        let pre_feedforward_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("pre_feedforward_layernorm"), false),
        )?;
        let post_feedforward_layernorm = RmsNorm::new(
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
        position: usize,
        kv_cache: &[(Tensor, Tensor)],
        input_metadata: &PagedAttentionInputMetadata,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let normed = xs.apply(&self.input_layernorm)?;
        let attn = self
            .self_attn
            .forward(&normed, position, kv_cache, input_metadata, flash_params)?
            .apply(&self.post_attention_layernorm)?;
        let xs = (attn + residual)?;

        let residual = xs.clone();
        let mlp = self
            .mlp
            .forward(&xs.apply(&self.pre_feedforward_layernorm)?)?
            .apply(&self.post_feedforward_layernorm)?;
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
    q_norm: RmsNorm,
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
        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;

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
        position: usize,
        kv_cache: &[(Tensor, Tensor)],
        input_metadata: &PagedAttentionInputMetadata,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let mut q = xs.apply(&self.q_proj)?;
        q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
        q = q.apply(&self.q_norm)?;
        q = if let Some(rotary) = &self.rotary_emb_local {
            rotary.forward_q(&q, &[position])?
        } else {
            self.rotary_emb_global
                .as_ref()
                .expect("global rotary missing")
                .forward_q(&q, &[position])?
        };
        let (key_cache, value_cache) = kv_cache.get(self.donor_layer_idx).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Gemma4 MTP donor layer {} is missing from the target paged KV cache",
                self.donor_layer_idx
            ))
        })?;
        let attn = self.paged_attn.forward_donor_cache(
            &q,
            key_cache,
            value_cache,
            &AttentionMask::None,
            input_metadata,
            &self.sdpa_params,
            Some(flash_params),
        )?;
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
    vocab_size: usize,
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
            vocab_size,
            num_centroids,
            centroid_top_k,
            vocab_size_per_centroid: vocab_size / num_centroids,
            num_selected: centroid_top_k * (vocab_size / num_centroids),
        })
    }

    fn get_top_token(&self, hidden_states: &Tensor, lm_head_weight: &Tensor) -> Result<u32> {
        let hidden_states = hidden_states.reshape(((), self.hidden_size))?;
        let centroid_logits = hidden_states.apply(&self.centroids)?;
        let top_k_indices = centroid_logits.topk(self.centroid_top_k)?.indices;
        let clusters = self
            .token_ordering
            .reshape((self.num_centroids, self.vocab_size_per_centroid))?;
        let selected = clusters
            .index_select(&top_k_indices.flatten_all()?.to_dtype(DType::U32)?, 0)?
            .reshape((hidden_states.dim(0)?, self.num_selected))?;
        let selected_embeddings = lm_head_weight
            .index_select(&selected.flatten_all()?.to_dtype(DType::U32)?, 0)?
            .reshape((hidden_states.dim(0)?, self.num_selected, self.hidden_size))?;
        let logits = hidden_states
            .unsqueeze(1)?
            .broadcast_mul(&selected_embeddings)?
            .sum(D::Minus1)?;
        let argmax = logits.argmax(D::Minus1)?;
        let token = selected
            .gather(&argmax.unsqueeze(1)?, D::Minus1)?
            .squeeze(0)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        debug_assert!(token < self.vocab_size as u32);
        Ok(token)
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let ws = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(ws, None))
}

fn make_mtp_decode_metadata(
    seq_id: usize,
    context_len: usize,
    paged_meta: &PagedAttentionMeta,
    device: &Device,
) -> Result<PagedAttentionInputMetadata> {
    let kv_mgr = get_mut_arcmutex!(paged_meta.kv_cache_manager);
    let full_table = kv_mgr
        .get_block_ids(seq_id)
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Gemma4 MTP sequence {seq_id} has no paged attention blocks"
            ))
        })?
        .to_vec();
    drop(kv_mgr);

    let (block_table, context_len_windowed) =
        if let Some(sliding_window) = paged_meta.sliding_window {
            let window_start = context_len.saturating_sub(sliding_window);
            let slide_idx = window_start / paged_meta.block_size;
            let block_aligned_start = slide_idx * paged_meta.block_size;
            (
                full_table.get(slide_idx..).unwrap_or(&[]).to_vec(),
                context_len.saturating_sub(block_aligned_start),
            )
        } else {
            (full_table.clone(), context_len)
        };

    let block_pos = context_len.saturating_sub(1);
    let slot = full_table
        .get(block_pos / paged_meta.block_size)
        .copied()
        .unwrap_or(0)
        * paged_meta.block_size
        + block_pos % paged_meta.block_size;
    let slot_mappings = Tensor::from_vec(vec![slot as i64], (1,), device)?;

    let block_tables = table_tensor(&block_table, device)?;
    let full_block_tables = table_tensor(&full_table, device)?;
    let context_lens = Tensor::from_vec(vec![context_len_windowed as u32], (1,), device)?;
    let full_context_lens = Tensor::from_vec(vec![context_len as u32], (1,), device)?;

    let (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len) = paged_kv_tensors(
        &block_table,
        context_len_windowed,
        paged_meta.block_size,
        device,
    )?;
    let request_indices = Tensor::from_vec(vec![0i32], (1,), device)?;
    let kv_tile_indices = Tensor::from_vec(vec![0i32], (1,), device)?;
    let o_indptr = Tensor::from_vec(vec![0i32, 1], (2,), device)?;
    let kv_chunk_size = Tensor::from_vec(vec![paged_meta.block_size as i32], (1,), device)?;

    let location = device.location();
    Ok(PagedAttentionInputMetadata {
        block_tables: Some(HashMap::from([(location.clone(), block_tables)])),
        context_lens: Some(HashMap::from([(location.clone(), context_lens)])),
        slot_mappings: HashMap::from([(location.clone(), slot_mappings)]),
        max_context_len: Some(context_len_windowed),
        full_block_tables: Some(HashMap::from([(location.clone(), full_block_tables)])),
        full_context_lens: Some(HashMap::from([(location.clone(), full_context_lens)])),
        full_max_context_len: Some(context_len),
        is_first_prompt_chunk: false,
        paged_kv_indptr: Some(HashMap::from([(location.clone(), paged_kv_indptr)])),
        paged_kv_indices: Some(HashMap::from([(location.clone(), paged_kv_indices)])),
        paged_kv_last_page_len: Some(HashMap::from([(location.clone(), paged_kv_last_page_len)])),
        paged_kv_request_indices: Some(HashMap::from([(location.clone(), request_indices)])),
        paged_kv_tile_indices: Some(HashMap::from([(location.clone(), kv_tile_indices)])),
        paged_kv_o_indptr: Some(HashMap::from([(location.clone(), o_indptr)])),
        paged_kv_chunk_size: Some(HashMap::from([(location, kv_chunk_size)])),
        num_cached_tokens: None,
        query_lens: None,
        cu_seqlens_q: None,
        cu_seqlens_kv: None,
    })
}

fn table_tensor(table: &[usize], device: &Device) -> Result<Tensor> {
    let table = if table.is_empty() {
        vec![0u32]
    } else {
        table.iter().map(|x| *x as u32).collect::<Vec<_>>()
    };
    let len = table.len();
    Tensor::from_vec(table, (1, len), device)
}

fn paged_kv_tensors(
    table: &[usize],
    context_len: usize,
    block_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor)> {
    let num_blocks = table.len();
    let indptr = Tensor::from_vec(vec![0i32, num_blocks as i32], (2,), device)?;
    let indices = Tensor::from_vec(
        table.iter().map(|x| *x as i32).collect::<Vec<_>>(),
        (num_blocks,),
        device,
    )?;
    let last_page_len = if num_blocks == 0 {
        0
    } else {
        context_len.saturating_sub((num_blocks - 1) * block_size) as i32
    };
    let last_page_len = Tensor::from_vec(vec![last_page_len], (1,), device)?;
    Ok((indptr, indices, last_page_len))
}
