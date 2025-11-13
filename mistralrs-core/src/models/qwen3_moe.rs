#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, FusedExperts, QuantMethod, QuantizedConfig, ReplicatedLayer,
    RowParallelLayer, ShardedVarBuilder,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    attention::SdpaParams,
    device_map::DeviceMapper,
    layers::{self, embedding, Activation, CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    paged_attention::{AttentionImplementation, ModelConfigMetadata, PagedAttention},
    pipeline::{
        extract_logits,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel, KvCache, NormalCache, NormalCacheType, NormalLoadingMetadata,
        NormalModel,
    },
    serde_default_fn,
    utils::{progress::NiceProgressBar, unvarbuilder::UnVarBuilder},
};

macro_rules! sliding_window {
    ($layer_idx:expr, $cfg:expr) => {
        if !($cfg.sliding_window.is_some()
            && $cfg.use_sliding_window
            && $layer_idx >= $cfg.max_window_layers)
        {
            None
        } else {
            $cfg.sliding_window
        }
    };
}

serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) sliding_window: Option<usize>,
    pub(crate) head_dim: Option<usize>,
    pub(crate) quantization_config: Option<QuantizedConfig>,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) max_window_layers: usize,
    pub(crate) use_sliding_window: bool,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_experts: usize,
    pub(crate) mlp_only_layers: Vec<usize>,
    pub(crate) decoder_sparse_step: usize,
    pub(crate) norm_topk_prob: bool,
    pub(crate) num_experts_per_tok: usize,
}

impl Config {
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let q_proj = ColumnParallelLayer::new(
            hidden_sz,
            num_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("q_proj"), loading_isq),
        )?;
        let kv_shard = mistralrs_quant::compute_kv_shard(
            cfg.num_key_value_heads,
            cfg.hidden_size / cfg.num_attention_heads,
            comm,
        );
        let k_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("k_proj"), loading_isq),
        )?;
        let v_proj = ColumnParallelLayer::new_with_shard(
            hidden_sz,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            kv_shard,
            mapper.set_device(layer_idx, vb.pp("v_proj"), loading_isq),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            hidden_sz,
            &cfg.quantization_config,
            false,
            comm,
            mapper.set_device(layer_idx, vb.pp("o_proj"), loading_isq),
        )?;
        let sliding_window = sliding_window!(layer_idx, cfg);
        let q_norm = RmsNorm::new(
            cfg.head_dim(),
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("q_norm"), false),
        )?;
        let k_norm = RmsNorm::new(
            cfg.head_dim(),
            cfg.rms_norm_eps,
            mapper.set_device(layer_idx, vb.pp("k_norm"), false),
        )?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_heads / comm.world_size(),
            num_kv_heads: (num_kv_heads / comm.world_size()).max(1),
            head_dim,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: mistralrs_quant::compute_n_kv_groups(
                    cfg.num_key_value_heads,
                    cfg.num_attention_heads,
                    comm,
                ),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window,
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.q_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut q = MatMul.qmethod_matmul(&xs, &*self.q_proj)?;
        let mut k = MatMul.qmethod_matmul(&xs, &*self.k_proj)?;
        let mut v = MatMul.qmethod_matmul(&xs, &*self.v_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            q = q.to_dtype(original_dtype)?;
            k = k.to_dtype(original_dtype)?;
            v = v.to_dtype(original_dtype)?;
        }

        (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let mut attn_output = match &self.paged_attn {
            Some(paged_attn) => match metadata {
                Some(((key_cache, value_cache), input_metadata)) => paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?,
                None => {
                    // If we don't have metadata, we are most likely generating an imatrix so we don't want to populate that.
                    // Generating the dummy metadata with the assumption that we are not generating text (only processing prompts).
                    let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                    // Sanity check.
                    assert!(attention_mask.is_some());
                    paged_attn.forward(
                        &q,
                        &k,
                        &v,
                        attention_mask,
                        None,
                        None,
                        &input_metadata,
                        &self.sdpa_params,
                        Some(flash_params),
                    )?
                }
            },
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(
                    &q,
                    &k,
                    &v,
                    attention_mask,
                    Some(flash_params),
                    &self.sdpa_params,
                )?
            }
        };

        if let Some(t) = self.q_proj.quantized_act_type() {
            attn_output = attn_output.to_dtype(t)?;
        }
        attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        let mut res = MatMul.qmethod_matmul(&attn_output, &*self.o_proj)?;
        if self.q_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

#[derive(Clone)]
struct Mlp {
    gate_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    down_proj: Arc<dyn QuantMethod>,
    act_fn: Activation,
}

impl Mlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        comm: &Arc<mistralrs_quant::Comm>,
        i_size: usize,
    ) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        let gate_proj = ColumnParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("gate_proj"),
        )?;
        let up_proj = RowParallelLayer::new(
            hidden_size,
            i_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("up_proj"),
        )?;
        let down_proj = ColumnParallelLayer::new(
            i_size,
            hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let mut xs = xs.clone();
        if let Some(t) = self.gate_proj.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        let mut current_hidden_states = MatMul
            .qmethod_matmul(&xs, &*self.gate_proj)?
            .apply(&self.act_fn)?;
        let rhs = MatMul.qmethod_matmul(&xs, &*self.up_proj)?;
        current_hidden_states = current_hidden_states.broadcast_mul(&rhs)?;
        let mut res = MatMul.qmethod_matmul(&current_hidden_states, &*self.down_proj)?;
        if self.gate_proj.quantized_act_type().is_some() {
            res = res.to_dtype(original_dtype)?;
        }
        Ok(res)
    }
}

struct FastMoeMlp {
    gate: Arc<dyn QuantMethod>,
    fused_gate_proj: Arc<dyn QuantMethod>,
    fused_up_proj: Arc<dyn QuantMethod>,
    fused_down_proj: Arc<dyn QuantMethod>,
    act: Activation,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FastMoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        _comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        if !vb.device().is_metal() {
            candle_core::bail!("FastMoeMlp requires Metal.");
        }

        let num_experts = cfg.num_experts;
        let gate = mistralrs_quant::linear_no_bias(
            cfg.hidden_size,
            num_experts,
            &cfg.quantization_config,
            vb.pp("gate").set_device(layer_device),
        )?;

        let FusedExperts {
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
        } = FusedExperts::new(
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.num_experts,
            &cfg.quantization_config,
            vb,
        )?;

        Ok(Self {
            gate,
            fused_gate_proj,
            fused_up_proj,
            fused_down_proj,
            act: cfg.hidden_act,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();

        let (b_size, seq_len, hidden_dim) = xs.dims3()?;

        let router_logits = self.gate.forward_autocast(xs)?;
        let routing_weights =
            candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;

        let indices = routing_weights.arg_sort_last_dim(false)?.narrow(
            D::Minus1,
            0,
            self.num_experts_per_tok,
        )?;
        let mut scores = routing_weights.gather(&indices.contiguous()?, D::Minus1)?;

        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        let ys = {
            let xs = xs.reshape((b_size, seq_len, 1, 1, hidden_dim))?;
            let gate = self
                .fused_gate_proj
                .gather_forward_autocast(&xs, &indices)?;
            let up = self.fused_up_proj.gather_forward_autocast(&xs, &indices)?;
            let xs = self
                .fused_down_proj
                .gather_forward_autocast(&(up * gate.apply(&self.act)?)?, &indices)?;
            xs.squeeze(D::Minus2)?
        };

        ys.to_dtype(DType::F32)?
            .broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((b_size, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

struct SlowMoeMlp {
    gate: candle_nn::Linear,
    experts: Vec<Mlp>,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl SlowMoeMlp {
    fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        layer_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let num_experts = cfg.num_experts;
        let gate = layers::linear_no_bias(
            cfg.hidden_size,
            num_experts,
            vb.pp("gate").set_device(layer_device),
        )?;

        let experts_vb = vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            experts.push(Mlp::new(
                cfg,
                experts_vb.pp(i),
                comm,
                cfg.moe_intermediate_size,
            )?);
        }

        Ok(Self {
            gate,
            experts,
            norm_topk_prob: cfg.norm_topk_prob,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Shapes and device
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let device = xs.device();

        // Flatten tokens: xs2d: [N, H]
        let xs2d = xs.reshape(((), hidden_dim))?;

        // Router: logits -> softmax -> top-k indices and scores
        let router_logits = xs2d.apply(&self.gate)?; // [N, E]
        let routing_weights_full = candle_nn::ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?; // [N, E]
        let indices = routing_weights_full
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?; // [N, K]
        let mut scores = routing_weights_full.gather(&indices, D::Minus1)?; // [N, K]
        if self.norm_topk_prob {
            scores = scores.broadcast_div(&scores.sum_keepdim(D::Minus1)?)?;
        }

        // Grouping: build ids_to_sorted / ids_from_sorted in host memory
        let n_tokens = xs2d.dim(0)?;
        let k = indices.dim(D::Minus1)?;
        let n_experts = self.experts.len();
        let total_rows = n_tokens * k;
        let indices_cpu = indices.to_device(&Device::Cpu)?.to_dtype(DType::I64)?.to_vec2::<i64>()?; // [N,K]
        let mut ids_to_sorted: Vec<i64> = Vec::with_capacity(total_rows);
        let mut ids_from_sorted: Vec<i64> = vec![-1; total_rows];
        let mut tokens_per_expert: Vec<usize> = vec![0; n_experts];
        for e in 0..n_experts {
            for i in 0..n_tokens {
                let base = i * k;
                let row_indices = &indices_cpu[i];
                for slot in 0..k {
                    if (row_indices[slot] as usize) == e {
                        let row_idx = base + slot;
                        ids_from_sorted[row_idx] = ids_to_sorted.len() as i64;
                        ids_to_sorted.push(row_idx as i64);
                        tokens_per_expert[e] += 1;
                        break;
                    }
                }
            }
        }
        if ids_to_sorted.len() != total_rows {
            candle_core::bail!("ids_to_sorted size mismatch");
        }

        // Expand xs per top-k and gather to expert-grouped order
        let xs_k = xs2d
            .reshape((n_tokens, 1, hidden_dim))?
            .expand((n_tokens, k, hidden_dim))?
            .contiguous()? // [N, K, H]
            .reshape((total_rows, hidden_dim))?; // [N*K, H]
        let ids_to_sorted_t = Tensor::new(ids_to_sorted.as_slice(), &device)?;
        let grouped_in = xs_k.index_select(&ids_to_sorted_t, 0)?; // [N*K, H]

        // Prefix sums to slice per expert
        let mut prefix: Vec<usize> = Vec::with_capacity(n_experts + 1);
        prefix.push(0);
        for e in 0..n_experts {
            prefix.push(prefix[e] + tokens_per_expert[e]);
        }

        // Run each expert MLP on its contiguous slice (batched along rows)
        let mut out_chunks: Vec<Tensor> = Vec::with_capacity(n_experts);
        for e in 0..n_experts {
            let cnt = tokens_per_expert[e];
            if cnt == 0 { continue; }
            let off = prefix[e];
            let slice = grouped_in.narrow(0, off, cnt)?;   // [cnt, H]
            let slice_bth = slice.unsqueeze(0)?;            // [1, cnt, H]
            let out_e = self.experts[e].forward(&slice_bth)?.squeeze(0)?; // [cnt, H]
            out_chunks.push(out_e);
        }
        let grouped_out = if out_chunks.is_empty() {
            grouped_in.zeros_like()? // degenerate: no assignments
        } else {
            Tensor::cat(&out_chunks, 0)? // [N*K, H]
        };

        // Unpermute back to original row order and shape [N, K, H]
        let ids_from_sorted_t = Tensor::new(ids_from_sorted.as_slice(), &device)?;
        let y = grouped_out.index_select(&ids_from_sorted_t, 0)?; // [N*K, H]
        let y = y.reshape((n_tokens, k, hidden_dim))?;            // [N, K, H]

        // Combine with routing scores: weighted sum across K
        let y = y
            .to_dtype(DType::F32)?
            .broadcast_mul(&scores.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .to_dtype(xs.dtype())?; // [N, H]

        y.reshape((b_size, seq_len, hidden_dim))
    }

    /// Standalone grouped GEMM with explicit permute/unpermute and gather/scatter.
    ///
    /// Implements a MoE-style grouped matmul path analogous to ggml's non-mat-vec `mul_mat_id`:
    /// - Expand inputs per top-k slot, build permutation to group rows by expert
    /// - Apply a sequence of per-expert QuantMethod layers stage-by-stage on grouped slices
    /// - Unpermute back to original (token, top-k) row order
    ///
    /// Arguments:
    /// - `xs`: Input activations, shape [B, T, H] or [N, H] (N = B*T)
    /// - `indices`: Expert ids per token and top-k slot, shape [B, T, K] or [N, K]
    /// - `stages`: List of stages; each stage is a vec of per-expert layers of length E
    ///
    /// Returns: Tensor with shape [B, T, K, O] or [N, K, O], where O is last stage output dim.
    #[allow(clippy::needless_pass_by_value)]
    pub fn grouped_sequential_gemm(
        &self,
        xs: &Tensor,
        indices: &Tensor,
        stages: &[Vec<Arc<dyn QuantMethod>>],
    ) -> Result<Tensor> {
        use candle_core::DType;

        // Infer xs shape and flatten to [N, H]
        let (b, t, h, xs_2d, n_tokens, xs_is_3d) = match xs.rank() {
            3 => {
                let (b, t, h) = xs.dims3()?;
                let n = b * t;
                (b, t, h, xs.reshape((n, h))?, n, true)
            }
            2 => {
                let (n, h) = xs.dims2()?;
                (1, n, h, xs.clone(), n, false)
            }
            _ => candle_core::bail!("grouped_sequential_gemm: xs must be rank-2 or rank-3"),
        };

        // Indices must be [N, K] or [B, T, K]; flatten to [N, K]
        let (k, indices_2d) = match indices.rank() {
            3 => {
                let (ib, it, ik) = indices.dims3()?;
                let expected_n = ib * it;
                if xs_is_3d {
                    if !(ib == b && it == t) {
                        candle_core::bail!("indices must match xs batch/time dims");
                    }
                } else if expected_n != t {
                    candle_core::bail!(
                        "indices [B,T,K] must flatten to N == T when xs is [N,H]"
                    );
                }
                (ik, indices.reshape((expected_n, ik))?)
            }
            2 => {
                let (in_n, ik) = indices.dims2()?;
                if in_n != n_tokens {
                    candle_core::bail!("indices first dim (N) must match tokens count");
                }
                (ik, indices.clone())
            }
            _ => candle_core::bail!("grouped_sequential_gemm: indices must be rank-2 or rank-3"),
        };

        if stages.is_empty() {
            candle_core::bail!("stages must be non-empty");
        }
        let n_experts = stages[0].len();
        for s in stages.iter() {
            if s.len() != n_experts {
                candle_core::bail!("all stages must have {} experts", n_experts);
            }
        }

        // Copy indices to CPU and build mappings
        let indices_cpu = indices_2d
            .to_device(&Device::Cpu)?
            .to_dtype(DType::I64)?
            .to_vec2::<i64>()?; // [N, K]

        let total_rows = n_tokens * k;
        let mut ids_to_sorted: Vec<i64> = Vec::with_capacity(total_rows);
        let mut ids_from_sorted: Vec<i64> = vec![-1; total_rows];
        let mut tokens_per_expert: Vec<usize> = vec![0; n_experts];

        // Group rows by expert (expert-major traversal mirrors ggml path)
        for e in 0..n_experts {
            for i in 0..n_tokens {
                let base = i * k;
                let row_indices = &indices_cpu[i];
                for slot in 0..k {
                    if (row_indices[slot] as usize) == e {
                        let row_idx = base + slot;
                        ids_from_sorted[row_idx] = ids_to_sorted.len() as i64;
                        ids_to_sorted.push(row_idx as i64);
                        tokens_per_expert[e] += 1;
                        break;
                    }
                }
            }
        }
        if ids_to_sorted.len() != total_rows {
            candle_core::bail!("ids_to_sorted size mismatch");
        }

        // Expand xs per top-k slot: [N,H] -> [N,1,H] -> [N,K,H] -> [N*K,H]
        let xs_k = xs_2d
            .reshape((n_tokens, 1, h))?
            .expand((n_tokens, k, h))?
            .contiguous()?
            .reshape((total_rows, h))?;

        // Gather rows into expert-grouped order
        let ids_to_sorted_t = Tensor::new(ids_to_sorted.as_slice(), &Device::Cpu)?;
        let mut cur = xs_k.index_select(&ids_to_sorted_t, 0)?;

        // Prefix sums for expert offsets
        let mut prefix: Vec<usize> = Vec::with_capacity(n_experts + 1);
        prefix.push(0);
        for e in 0..n_experts {
            prefix.push(prefix[e] + tokens_per_expert[e]);
        }

        // Apply each stage per expert on its slice
        for stage in stages.iter() {
            let mut outs: Vec<Tensor> = Vec::with_capacity(n_experts);
            for e in 0..n_experts {
                let cnt = tokens_per_expert[e];
                if cnt == 0 {
                    continue;
                }
                let off = prefix[e];
                let slice = cur.narrow(0, off, cnt)?; // [cnt, in_dim]
                let out_e = stage[e].forward_autocast(&slice)?; // [cnt, out_dim]
                outs.push(out_e);
            }
            if !outs.is_empty() {
                cur = Tensor::cat(&outs, 0)?;
            }
        }

        // Unpermute: y[orig_row] = cur[ids_from_sorted[orig_row]]
        let ids_from_sorted_t = Tensor::new(ids_from_sorted.as_slice(), &Device::Cpu)?;
        let y = cur.index_select(&ids_from_sorted_t, 0)?; // [N*K, O]

        // Reshape back to [N,K,O] or [B,T,K,O]
        let o = y.dim(D::Minus1)?;
        let y = y.reshape((n_tokens, k, o))?;
        if xs_is_3d {
            y.reshape((b, t, k, o))
        } else {
            Ok(y)
        }
    }
}

enum MoeOrMlp {
    FastMoe(FastMoeMlp),
    SlowMoe(SlowMoeMlp),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FastMoe(m) => m.forward(xs),
            Self::SlowMoe(m) => m.forward(xs),
        }
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MoeOrMlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: ShardedVarBuilder,
        mapper: &dyn DeviceMapper,
        layer_idx: usize,
        loading_isq: bool,
        paged_attn: Option<PagedAttention>,
        real_device: Device,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        let self_attn = Attention::new(
            rotary_emb,
            cfg,
            mapper.set_device(layer_idx, vb.pp("self_attn"), loading_isq),
            mapper,
            layer_idx,
            loading_isq,
            paged_attn,
            comm,
        )?;
        let mlp = if !cfg.mlp_only_layers.contains(&layer_idx)
            && (cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0)
        {
            let vb = mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq);

            if vb.device().is_metal() {
                MoeOrMlp::FastMoe(FastMoeMlp::new(
                    cfg,
                    vb,
                    mapper
                        .device_for(layer_idx, false)
                        .cloned()
                        .unwrap_or(real_device),
                    comm,
                )?)
            } else {
                MoeOrMlp::SlowMoe(SlowMoeMlp::new(
                    cfg,
                    vb,
                    mapper
                        .device_for(layer_idx, false)
                        .cloned()
                        .unwrap_or(real_device),
                    comm,
                )?)
            }
        } else {
            MoeOrMlp::Mlp(Mlp::new(
                cfg,
                mapper.set_device(layer_idx, vb.pp("mlp"), loading_isq),
                comm,
                cfg.intermediate_size,
            )?)
        };
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
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(
            &xs,
            attention_mask,
            seqlen_offsets,
            kv_cache,
            metadata,
            flash_params,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }
}

pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Arc<dyn QuantMethod>,
    sliding_window: Option<usize>,
    device: Device,
    cache: EitherCache,
    max_seq_len: usize,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    cfg: ModelConfigMetadata,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");
        let vb_lm_head = vb.pp("lm_head");
        Self::new_inner(
            cfg,
            vb_m,
            vb_lm_head,
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )
    }

    pub fn new_inner(
        cfg: &Config,
        vb_m: ShardedVarBuilder,
        vb_lm_head: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization: {}.",
                quant_cfg.name(),
                quant_cfg.get_bits_name(&vb_m)
            );
        }
        let mapper = normal_loading_metadata.mapper;

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            mapper.set_nm_device(vb_m.pp("embed_tokens"), false),
            &cfg.quantization_config,
        )?;

        let head_dim = cfg.head_dim();
        let mut ropes = HashMap::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            ropes.insert(
                device.location(),
                Arc::new(RotaryEmbedding::new(
                    cfg.rope_theta as f32,
                    head_dim,
                    cfg.max_position_embeddings,
                    device,
                    is_gptx,
                    vb_m.dtype(),
                )?),
            );
        }

        let load_in_parallel =
            !(normal_loading_metadata.real_device.is_metal() && cfg.quantization_config.is_none());
        let vb_l = vb_m.pp("layers");
        let layers: Vec<DecoderLayer> = NiceProgressBar::<_, 'b'>(
            0..cfg.num_hidden_layers,
            "Loading repeating layers",
            &normal_loading_metadata.multi_progress,
        )
        .run(load_in_parallel, |layer_idx| {
            let device = mapper
                .device_for(layer_idx, false)
                .unwrap_or(&normal_loading_metadata.real_device);
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let paged_attn = match &attention_mechanism {
                AttentionImplementation::Eager => None,
                AttentionImplementation::PagedAttention => Some(
                    PagedAttention::new(head_dim, device, None)
                        .expect("PagedAttention creation failed"),
                ),
            };
            let comm = mapper
                .get_comm_for(layer_idx)
                .expect("Failed to get comm for layer");
            DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                &*mapper,
                layer_idx,
                normal_loading_metadata.loading_isq,
                paged_attn,
                normal_loading_metadata.real_device.clone(),
                &comm,
            )
        })?;
        let norm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            mapper.set_nm_device(vb_m.pp("norm"), false),
        )?;
        let lm_head = if !cfg.tie_word_embeddings {
            ReplicatedLayer::new(
                cfg.hidden_size,
                cfg.vocab_size,
                &cfg.quantization_config,
                false,
                mapper.set_nm_device(vb_lm_head, normal_loading_metadata.loading_isq),
            )?
        } else {
            ReplicatedLayer::from_linear(candle_nn::Linear::new(
                mapper.cast_nm_device(
                    embed_tokens.embeddings(),
                    normal_loading_metadata.loading_isq,
                )?,
                None,
            ))?
        };
        let cache_types = (0..cfg.num_hidden_layers)
            .map(|layer_idx| {
                sliding_window!(layer_idx, cfg)
                    .map(|window| NormalCacheType::SlidingWindow { window })
                    .unwrap_or(NormalCacheType::Normal {
                        max_seq_len: cfg.max_position_embeddings,
                    })
            })
            .collect::<Vec<_>>();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            sliding_window: cfg.sliding_window,
            device: normal_loading_metadata.real_device,
            cache: EitherCache::Normal(NormalCache::from_types(cache_types)),
            max_seq_len: cfg.max_position_embeddings,
            cfg: ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: (cfg.num_key_value_heads / mapper.get_comm_for(0)?.world_size())
                    .max(1),
                num_attn_heads: cfg.num_attention_heads / mapper.get_comm_for(0)?.world_size(),
                sliding_window: cfg.sliding_window,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
            },
            mapper,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward_embeds(
            input_ids,
            self.embed_tokens.forward(input_ids)?,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embeds(
        &self,
        input_ids: &Tensor,
        input_embeds: Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let mut xs = input_embeds;
        let cache = &mut self.cache.normal().0;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            input_ids,
            metadata
                .as_ref()
                .map(|(_, _)| &seqlen_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.sliding_window,
            xs.dtype(),
            self.cfg.num_attn_heads,
        )?;
        // PagedAttention prompt chunking
        let attention_mask = attention_mask.filter(|_| {
            metadata
                .as_ref()
                .map(|(_, meta)| meta.is_first_prompt_chunk)
                .unwrap_or(true)
        });
        for (i, layer) in self.layers.iter().enumerate() {
            xs = self.mapper.map(xs, i)?;
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offsets,
                &mut cache[i],
                metadata
                    .as_ref()
                    .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                flash_params,
            )?;
            // dbg!(&i);
        }
        let xs = xs.to_device(&self.device)?;
        let mut xs = xs.apply(&self.norm)?;
        if let Some(t) = self.lm_head.quantized_act_type() {
            xs = xs.to_dtype(t)?;
        }
        extract_logits(&MatMul.qmethod_matmul(&xs, &*self.lm_head)?, context_lens)
    }
}

impl IsqModel for Model {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        let mut tensors = Vec::new();
        tensors.push((&mut self.lm_head, None));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            tensors.push((&mut layer.self_attn.q_proj, Some(i)));
            tensors.push((&mut layer.self_attn.k_proj, Some(i)));
            tensors.push((&mut layer.self_attn.v_proj, Some(i)));
            tensors.push((&mut layer.self_attn.o_proj, Some(i)));
            match &mut layer.mlp {
                MoeOrMlp::Mlp(layer) => {
                    tensors.push((&mut layer.gate_proj, Some(i)));
                    tensors.push((&mut layer.up_proj, Some(i)));
                    tensors.push((&mut layer.down_proj, Some(i)));
                }
                MoeOrMlp::FastMoe(layer) => {
                    tensors.push((&mut layer.fused_gate_proj, Some(i)));
                    tensors.push((&mut layer.fused_up_proj, Some(i)));
                    tensors.push((&mut layer.fused_down_proj, Some(i)));
                }
                MoeOrMlp::SlowMoe(layer) => {
                    for expert in &mut layer.experts {
                        tensors.push((&mut expert.gate_proj, Some(i)));
                        tensors.push((&mut expert.up_proj, Some(i)));
                        tensors.push((&mut expert.down_proj, Some(i)));
                    }
                }
            }
        }
        (tensors, &*self.mapper)
    }

    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        let uvb_m = uvb.pp("model");
        uvb_m.pp("embed_tokens").add(&self.embed_tokens);
        uvb_m.pp("norm").add(&self.norm);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let uvb_l = uvb_m.pp("layers").pp(layer_idx);
            uvb_l.pp("input_layernorm").add(&layer.input_layernorm);
            uvb_l
                .pp("post_attention_layernorm")
                .add(&layer.post_attention_layernorm);
            uvb_l
                .pp("self_attn")
                .pp("q_norm")
                .add(&layer.self_attn.q_norm);
            uvb_l
                .pp("self_attn")
                .pp("k_norm")
                .add(&layer.self_attn.k_norm);
            if let MoeOrMlp::FastMoe(layer) = &layer.mlp {
                uvb_l.pp("mlp").pp("gate").add(&layer.gate);
            } else if let MoeOrMlp::SlowMoe(layer) = &layer.mlp {
                uvb_l.pp("mlp").pp("gate").add(&layer.gate);
            }
        }

        uvb.to_safetensors()
    }
}

impl NormalModel for Model {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        self.forward(
            input_ids,
            seqlen_offsets,
            context_lens,
            metadata,
            flash_params,
        )
    }
    fn xlora_forward(
        &self,
        _input_ids: &Tensor,
        _input_ids_full: &Tensor,
        _seqlen_offsets: &[usize],
        _seqlen_offsets_full: &[usize],
        _no_kv_cache: bool,
        _non_granular_state: &Option<crate::xlora_models::NonGranularState>,
        _context_lens: Vec<(usize, usize)>,
        _position_ids: Vec<usize>,
        _flash_params: &FlashParams,
        _flash_params_full: &FlashParams,
    ) -> Result<Tensor> {
        unimplemented!()
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn is_xlora(&self) -> bool {
        false
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg
    }
}

impl AnyMoeBaseModelMixin for Model {}
