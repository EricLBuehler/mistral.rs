#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{
    ColumnParallelLayer, QuantMethod, ReplicatedLayer, RowParallelLayer, ShardedVarBuilder,
};

use super::config::{QsaConfig, TextConfig};
use crate::{
    attention::{AttentionMask, SdpaParams},
    layers::{rms_norm_partial_rotary_neox, GemmaRmsNorm, Qwen3VLRotaryEmbedding, Sdpa},
    paged_attention::{load_fp8_attention_scales, PagedAttention},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        KvCache,
    },
};

/// Per-token aux cache row: `[indexer key | cos | sin]`.
pub(super) fn aux_dim(qsa: &QsaConfig, half_rot: usize) -> usize {
    qsa.head_dim + 2 * half_rot
}

/// How a forward runs its QSA layers, decided once per step on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QsaMode {
    /// Every query sees at most budget + ratio - 1 tokens: the selection keeps them all.
    Dense,
    /// Block-sparse attention; `max_blocks` bounds the complete blocks of any query.
    Sparse { max_blocks: usize },
}

impl QsaMode {
    pub(super) fn for_max_kv_len(qsa: &QsaConfig, max_kv_len: usize) -> Self {
        if max_kv_len <= qsa.max_selected_tokens() {
            Self::Dense
        } else {
            Self::Sparse {
                max_blocks: max_kv_len / qsa.compress_ratio,
            }
        }
    }

    #[cfg(any(feature = "cuda", test))]
    fn with_table_capacity(self, qsa: &QsaConfig, capacity: usize) -> Self {
        match self {
            Self::Dense => Self::Dense,
            Self::Sparse { max_blocks } => {
                let max_blocks = max_blocks.min(capacity / qsa.compress_ratio);
                if max_blocks <= qsa.block_topk() {
                    Self::Dense
                } else {
                    Self::Sparse { max_blocks }
                }
            }
        }
    }
}

/// Per-forward state shared by every QSA layer.
pub(super) struct QsaStep<'a> {
    pub mode: QsaMode,
    /// Absolute kv position of every flattened token, host side (non-paged path only).
    pub positions: &'a [Option<usize>],
    #[cfg(feature = "cuda")]
    pub layout: Option<&'a crate::cuda::qwen4_exp::TokenLayout>,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct QsaAuxCache {
    raw: Tensor,
    block_keys: Tensor,
}

pub(super) struct QsaAttention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
    pub(super) q_norm: GemmaRmsNorm,
    pub(super) k_norm: GemmaRmsNorm,
    index_qk_proj: Arc<dyn QuantMethod>,
    pub(super) index_q_norm: GemmaRmsNorm,
    pub(super) index_k_norm: GemmaRmsNorm,
    // (1 + w) in f32 for the fused block finalize
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    index_k_norm_f32: Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    half_rot: usize,
    qsa: QsaConfig,
    pub(super) rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    eps: f64,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    aux_cache: Mutex<Option<QsaAuxCache>>,
}

impl QsaAttention {
    /// `vb_sa` and `vb_norms` are the `self_attn` builders already placed on the layer device, the
    /// former with ISQ applied.
    pub(super) fn load(
        vb_sa: ShardedVarBuilder,
        vb_norms: ShardedVarBuilder,
        cfg: &TextConfig,
        qsa: QsaConfig,
        rotary_emb: Arc<Qwen3VLRotaryEmbedding>,
        use_paged_attention: bool,
        comm: &Arc<mistralrs_quant::Comm>,
    ) -> Result<Self> {
        if comm.world_size() != 1 {
            candle_core::bail!("Qwen4-Exp QSA attention does not support tensor parallelism yet");
        }
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let q_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_heads * head_dim * 2,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("q_proj"),
        )?;
        let k_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("k_proj"),
        )?;
        let v_proj = ColumnParallelLayer::new(
            cfg.hidden_size,
            num_kv_heads * head_dim,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("v_proj"),
        )?;
        let o_proj = RowParallelLayer::new(
            num_heads * head_dim,
            cfg.hidden_size,
            &cfg.quantization_config,
            false,
            comm,
            vb_sa.pp("o_proj"),
        )?;
        let index_qk_proj = ReplicatedLayer::new(
            cfg.hidden_size,
            (qsa.n_heads + 1) * qsa.head_dim,
            &cfg.quantization_config,
            false,
            vb_sa.pp("indexer").pp("index_qk_proj"),
        )?;
        #[cfg(feature = "cutile")]
        if use_paged_attention {
            mistralrs_quant::cutile::register_qsa_shape(mistralrs_quant::cutile::QsaWarmShape {
                ratio: qsa.compress_ratio,
                topk: qsa.block_topk(),
                n_q_heads: num_heads,
                n_kv_heads: num_kv_heads,
                block_size: crate::paged_attention::DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
            });
        }
        let paged_attn = if use_paged_attention {
            Some(PagedAttention::new_with_fp8_attention_scales(
                head_dim,
                vb_norms.device(),
                None,
                load_fp8_attention_scales(&vb_norms)?,
            )?)
        } else {
            None
        };
        let index_k_norm = GemmaRmsNorm::new(
            qsa.head_dim,
            cfg.rms_norm_eps,
            vb_norms.pp("indexer").pp("k_layernorm"),
        )?;
        let index_k_norm_f32 = (index_k_norm.original_weight().to_dtype(DType::F32)? + 1.0)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_norms.pp("q_norm"))?,
            k_norm: GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb_norms.pp("k_norm"))?,
            index_qk_proj,
            index_q_norm: GemmaRmsNorm::new(
                qsa.head_dim,
                cfg.rms_norm_eps,
                vb_norms.pp("indexer").pp("q_layernorm"),
            )?,
            index_k_norm,
            index_k_norm_f32,
            num_heads,
            num_kv_heads,
            head_dim,
            half_rot: cfg.rot_dim() / 2,
            qsa,
            rotary_emb,
            paged_attn,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
            eps: cfg.rms_norm_eps,
            aux_cache: Mutex::new(None),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward(
        &self,
        x: &Tensor,
        attention_mask: &AttentionMask,
        cos_sin: &(Tensor, Tensor),
        kv_cache: Option<&mut KvCache>,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
        step: &QsaStep<'_>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;
        let (q_gate, k, v) =
            crate::ops::qkv_projections(x, &*self.q_proj, &*self.k_proj, &*self.v_proj)?;
        let q_gate = q_gate.reshape((b_sz, seq_len, self.num_heads, self.head_dim * 2))?;
        let q = q_gate.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_gate
            .narrow(D::Minus1, self.head_dim, self.head_dim)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?;
        let cos_sin = (
            cos_sin.0.to_device(x.device())?,
            cos_sin.1.to_device(x.device())?,
        );
        let (q, k) = self.rotary_emb.forward_qk_norm(
            &cos_sin,
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
        )?;

        let index_qk = self.index_qk_proj.forward(x)?;
        let index_q_dim = self.qsa.n_heads * self.qsa.head_dim;
        let index_q = index_qk.narrow(D::Minus1, 0, index_q_dim)?.reshape((
            b_sz,
            seq_len,
            self.qsa.n_heads,
            self.qsa.head_dim,
        ))?;
        let raw_key = index_qk.narrow(D::Minus1, index_q_dim, self.qsa.head_dim)?;

        let y = match (&self.paged_attn, metadata) {
            (Some(paged_attn), Some(((key_cache, value_cache), input_metadata))) => self
                .forward_paged(PagedQsaInputs {
                    paged_attn,
                    q: &q,
                    k: &k,
                    v: &v,
                    index_q: &index_q,
                    raw_key: &raw_key,
                    cos_sin: &cos_sin,
                    attention_mask,
                    key_cache,
                    value_cache,
                    metadata: input_metadata,
                    flash_params,
                    step,
                })?,
            (Some(paged_attn), None) => {
                let input_metadata = PagedAttentionInputMetadata::dummy(q.device())?;
                let y = paged_attn.forward(
                    &q,
                    &k,
                    &v.transpose(1, 2)?.contiguous()?,
                    attention_mask,
                    None,
                    None,
                    &input_metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?;
                y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
            }
            (None, _) => self.forward_unpaged(UnpagedQsaInputs {
                q: &q,
                k: &k,
                v: &v,
                index_q: &index_q,
                raw_key: &raw_key,
                cos_sin: &cos_sin,
                attention_mask,
                kv_cache: kv_cache.ok_or_else(|| {
                    candle_core::Error::msg("unpaged Qwen4-Exp QSA requires a KV cache")
                })?,
                flash_params,
                step,
            })?,
        };

        let gate = candle_nn::ops::sigmoid(&gate.to_dtype(y.dtype())?)?;
        self.o_proj.forward(&y.broadcast_mul(&gate)?)
    }

    /// Normalized, rotated indexer queries `[b, s, heads, head_dim]`.
    fn index_query(&self, index_q: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
        rms_norm_partial_rotary_neox(
            index_q,
            self.index_q_norm.weight(),
            cos,
            sin,
            self.eps as f32,
        )
    }

    #[cfg(feature = "cuda")]
    fn aux_cache(
        &self,
        key_cache: &Tensor,
        block_size: usize,
        dtype: DType,
    ) -> Result<QsaAuxCache> {
        if !block_size.is_multiple_of(self.qsa.compress_ratio) {
            candle_core::bail!(
                "Qwen4-Exp QSA paged block size {block_size} must be divisible by compression ratio {}",
                self.qsa.compress_ratio
            );
        }
        let mut aux = self.aux_cache.lock().expect("QSA aux cache poisoned");
        let num_blocks = key_cache.dim(0)?;
        if let Some(existing) = aux.as_ref() {
            if existing.raw.dim(0)? == num_blocks * block_size
                && existing.raw.device().same_device(key_cache.device())
            {
                return Ok(existing.clone());
            }
        }
        let half_rot = self.rotary_emb_half_dim();
        let cache = QsaAuxCache {
            raw: Tensor::zeros(
                (num_blocks * block_size, aux_dim(&self.qsa, half_rot)),
                dtype,
                key_cache.device(),
            )?,
            block_keys: Tensor::zeros(
                (
                    num_blocks * block_size / self.qsa.compress_ratio,
                    self.qsa.head_dim,
                ),
                dtype,
                key_cache.device(),
            )?,
        };
        *aux = Some(cache.clone());
        Ok(cache)
    }

    fn rotary_emb_half_dim(&self) -> usize {
        self.half_rot
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn forward_paged(&self, inputs: PagedQsaInputs<'_>) -> Result<Tensor> {
        let PagedQsaInputs {
            paged_attn,
            q,
            k,
            v,
            index_q,
            raw_key,
            cos_sin,
            attention_mask,
            key_cache,
            value_cache,
            metadata,
            flash_params,
            step,
        } = inputs;
        let (b_sz, _, seq_len, _) = q.dims4()?;
        #[cfg(feature = "cuda")]
        if q.device().is_cuda() {
            use crate::cuda::qwen4_exp as kernels;
            let block_size = metadata
                .block_size
                .ok_or_else(|| candle_core::Error::msg("QSA requires a paged block size"))?;
            let location = q.device().location();
            let block_tables = metadata
                .block_tables
                .as_ref()
                .and_then(|tables| tables.get(&location))
                .ok_or_else(|| candle_core::Error::msg("QSA requires paged block tables"))?;
            let layout = step
                .layout
                .ok_or_else(|| candle_core::Error::msg("QSA requires a token layout"))?;
            // Multi-token decode metadata carries one block table per query row; QSA wants one per sequence
            let block_tables = &per_sequence_block_tables(block_tables, layout.n_seqs)?;
            // Prompt metadata carries per-token positions in `context_lens`; only decode has lengths
            // Decode-row lengths are per query row; the last row of each sequence holds its kv length
            let kv_lens = &match &layout.kv_lens {
                Some(kv_lens) => kv_lens.clone(),
                None => last_row_per_sequence(
                    metadata
                        .context_lens
                        .as_ref()
                        .and_then(|lens| lens.get(&location))
                        .ok_or_else(|| {
                            candle_core::Error::msg("QSA requires paged context lengths")
                        })?,
                    layout.n_seqs,
                )?,
            };
            let slot_mapping = metadata
                .slot_mappings
                .get(&location)
                .ok_or_else(|| candle_core::Error::msg("QSA requires paged slot mappings"))?;
            if key_cache.dtype() != q.dtype() {
                candle_core::bail!(
                    "Qwen4-Exp QSA needs a {:?} KV cache, got {:?} (FP8 KV cache is unsupported)",
                    q.dtype(),
                    key_cache.dtype()
                );
            }
            if !matches!(block_tables.dtype(), DType::U32 | DType::I32)
                || !matches!(kv_lens.dtype(), DType::U32 | DType::I32)
            {
                candle_core::bail!("QSA expects 32-bit paged block tables and context lengths");
            }
            let paged = kernels::PagedView {
                block_tables,
                kv_lens,
                block_size,
            };
            let shape = kernels::QsaShape {
                ratio: self.qsa.compress_ratio,
                topk: self.qsa.block_topk(),
                half_rot: self.rotary_emb_half_dim(),
                n_index_heads: self.qsa.n_heads,
            };
            let n_tokens = b_sz * seq_len;
            let aux = self.aux_cache(&key_cache, block_size, q.dtype())?;
            kernels::qsa_aux_write(
                &raw_key.reshape((n_tokens, self.qsa.head_dim))?,
                &cos_sin.0.reshape((n_tokens, shape.half_rot))?,
                &cos_sin.1.reshape((n_tokens, shape.half_rot))?,
                slot_mapping,
                &aux.raw,
            )?;
            kernels::qsa_finalize(kernels::QsaFinalizeArgs {
                aux: &aux.raw,
                block_keys: &aux.block_keys,
                layout,
                paged: &paged,
                norm_weight: &self.index_k_norm_f32,
                shape: &shape,
                eps: self.eps,
            })?;
            // Table capacity is part of the graph key; live context lengths can change during replay.
            let mode = step
                .mode
                .with_table_capacity(&self.qsa, block_tables.dim(1)? * block_size);
            if matches!(mode, QsaMode::Dense) && layout.kv_lens.is_some() {
                let y = paged_attn.forward(
                    q,
                    k,
                    &v.transpose(1, 2)?.contiguous()?,
                    attention_mask,
                    Some(key_cache.clone()),
                    Some(value_cache.clone()),
                    metadata,
                    &self.sdpa_params,
                    Some(flash_params),
                )?;
                return y.transpose(1, 2)?.reshape((b_sz, seq_len, ()));
            }
            let mut key_cache = key_cache.clone();
            let mut value_cache = value_cache.clone();
            paged_attn.write_cache(
                &k.transpose(1, 2)?.contiguous()?,
                &v.contiguous()?,
                &mut key_cache,
                &mut value_cache,
                metadata,
            )?;
            let selection = if let QsaMode::Sparse { max_blocks } = mode {
                let index_q = self.index_query(index_q, cos_sin)?.reshape((
                    n_tokens,
                    self.qsa.n_heads,
                    self.qsa.head_dim,
                ))?;
                Some(
                    match kernels::qsa_select_cutile(
                        &index_q,
                        &aux.block_keys,
                        layout,
                        &paged,
                        &shape,
                        max_blocks,
                    )? {
                        Some(selection) => selection,
                        None => kernels::qsa_select(
                            &index_q,
                            &aux.block_keys,
                            layout,
                            &paged,
                            &shape,
                            max_blocks,
                        )?,
                    },
                )
            } else {
                None
            };
            let q_tokens = q.transpose(1, 2)?.contiguous()?.reshape((
                n_tokens,
                self.num_heads,
                self.head_dim,
            ))?;
            let attn_args = kernels::QsaAttentionArgs {
                q: &q_tokens,
                key_cache: &key_cache,
                value_cache: &value_cache,
                layout,
                paged: &paged,
                selection: selection
                    .as_ref()
                    .map(|(selected, counts)| (selected, counts)),
                shape: &shape,
                n_kv_heads: self.num_kv_heads,
                scale: self.sdpa_params.softmax_scale,
            };
            let y = match kernels::qsa_attention_cutile(&attn_args)? {
                Some(y) => y,
                None => kernels::qsa_attention(attn_args)?,
            };
            return y.reshape((b_sz, seq_len, ()));
        }
        candle_core::bail!(
            "Qwen4-Exp paged attention requires CUDA; run with paged attention disabled on this device"
        )
    }

    fn forward_unpaged(&self, inputs: UnpagedQsaInputs<'_>) -> Result<Tensor> {
        let UnpagedQsaInputs {
            q,
            k,
            v,
            index_q,
            raw_key,
            cos_sin,
            attention_mask,
            kv_cache,
            flash_params,
            step,
        } = inputs;
        let (b_sz, _, seq_len, _) = q.dims4()?;
        let half_rot = self.rotary_emb_half_dim();
        let aux_width = aux_dim(&self.qsa, half_rot);
        // The indexer state rides along as an extra key head so it batches and truncates with the cache
        let aux = Tensor::cat(
            &[
                raw_key.contiguous()?,
                cos_sin.0.to_dtype(k.dtype())?,
                cos_sin.1.to_dtype(k.dtype())?,
            ],
            D::Minus1,
        )?
        .pad_with_zeros(D::Minus1, 0, self.head_dim - aux_width)?
        .unsqueeze(1)?;
        let k_all = Tensor::cat(&[k.clone(), aux], 1)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let (cache_k, cache_v) = kv_cache.append(&k_all, &v)?;
        let keys = cache_k.narrow(1, 0, self.num_kv_heads)?.contiguous()?;
        let aux = cache_k
            .i((.., self.num_kv_heads, .., ..aux_width))?
            .contiguous()?;

        let y = match step.mode {
            QsaMode::Dense => Sdpa.run_attention(
                q,
                &keys,
                &cache_v,
                attention_mask,
                Some(flash_params),
                &self.sdpa_params,
            )?,
            QsaMode::Sparse { .. } => {
                let index_q = self.index_query(index_q, cos_sin)?;
                let mask = self.selection_mask(&index_q, &aux, step, b_sz, seq_len, q.dtype())?;
                Sdpa.run_attention_noflash(
                    q,
                    &keys,
                    &cache_v,
                    Some(&mask),
                    &self.sdpa_params,
                    false,
                )?
            }
        };
        y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))
    }

    /// Additive `[b, 1, s, kv]` mask admitting each query's selected blocks and incomplete tail.
    fn selection_mask(
        &self,
        index_q: &Tensor,
        aux: &Tensor,
        step: &QsaStep<'_>,
        b_sz: usize,
        seq_len: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let ratio = self.qsa.compress_ratio;
        let topk = self.qsa.block_topk();
        let half_rot = self.rotary_emb_half_dim();
        let kv_len = aux.dim(1)?;
        let n_blocks = kv_len / ratio;
        let d = self.qsa.head_dim;
        let block_starts = aux.narrow(1, 0, n_blocks * ratio)?;
        let pooled = block_starts
            .narrow(D::Minus1, 0, d)?
            .contiguous()?
            .reshape((b_sz, n_blocks, ratio, d))?
            .to_dtype(DType::F32)?
            .mean(2)?
            .to_dtype(aux.dtype())?;
        let starts = block_starts
            .reshape((b_sz, n_blocks, ratio, aux.dim(D::Minus1)?))?
            .i((.., .., 0, ..))?;
        let cos = starts.narrow(D::Minus1, d, half_rot)?.contiguous()?;
        let sin = starts
            .narrow(D::Minus1, d + half_rot, half_rot)?
            .contiguous()?;
        let block_keys = rms_norm_partial_rotary_neox(
            &pooled.unsqueeze(2)?,
            self.index_k_norm.weight(),
            &cos,
            &sin,
            self.eps as f32,
        )?
        .squeeze(2)?;
        // [b, s, heads, d] x [b, blocks, d] -> [b, s, heads, blocks]
        let scores = index_q
            .to_dtype(DType::F32)?
            .reshape((b_sz, seq_len * self.qsa.n_heads, d))?
            .matmul(&block_keys.to_dtype(DType::F32)?.transpose(1, 2)?)?
            .relu()?
            .reshape((b_sz, seq_len, self.qsa.n_heads, n_blocks))?
            .sum(2)?;
        let scores = (scores / (d as f64).sqrt())?
            .to_device(&Device::Cpu)?
            .to_vec3::<f32>()?;
        let mut mask = vec![f32::NEG_INFINITY; b_sz * seq_len * kv_len];
        for b in 0..b_sz {
            for i in 0..seq_len {
                let Some(pos) = step.positions[b * seq_len + i] else {
                    continue;
                };
                let row = &mut mask[(b * seq_len + i) * kv_len..(b * seq_len + i + 1) * kv_len];
                let nb = (pos + 1) / ratio;
                let mut blocks = (0..nb).collect::<Vec<_>>();
                if nb > topk {
                    let s = &scores[b][i];
                    blocks.select_nth_unstable_by(topk, |x, y| s[*y].total_cmp(&s[*x]));
                    blocks.truncate(topk);
                }
                for block in blocks {
                    row[block * ratio..(block + 1) * ratio].fill(0.0);
                }
                row[nb * ratio..=pos].fill(0.0);
            }
        }
        Tensor::from_vec(mask, (b_sz, 1, seq_len, kv_len), index_q.device())?.to_dtype(dtype)
    }
}

struct PagedQsaInputs<'a> {
    paged_attn: &'a PagedAttention,
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    index_q: &'a Tensor,
    raw_key: &'a Tensor,
    cos_sin: &'a (Tensor, Tensor),
    attention_mask: &'a AttentionMask,
    key_cache: Tensor,
    value_cache: Tensor,
    metadata: &'a PagedAttentionInputMetadata,
    flash_params: &'a FlashParams,
    step: &'a QsaStep<'a>,
}

struct UnpagedQsaInputs<'a> {
    q: &'a Tensor,
    k: &'a Tensor,
    v: &'a Tensor,
    index_q: &'a Tensor,
    raw_key: &'a Tensor,
    cos_sin: &'a (Tensor, Tensor),
    attention_mask: &'a AttentionMask,
    kv_cache: &'a mut KvCache,
    flash_params: &'a FlashParams,
    step: &'a QsaStep<'a>,
}

#[cfg(feature = "cuda")]
fn per_sequence_block_tables(block_tables: &Tensor, n_seqs: usize) -> Result<Tensor> {
    let (rows, width) = block_tables.dims2()?;
    if rows == n_seqs {
        return Ok(block_tables.clone());
    }
    if n_seqs == 0 || !rows.is_multiple_of(n_seqs) {
        candle_core::bail!("QSA block tables have {rows} rows for {n_seqs} sequences");
    }
    block_tables
        .reshape((n_seqs, rows / n_seqs, width))?
        .i((.., 0, ..))?
        .contiguous()
}

#[cfg(feature = "cuda")]
fn last_row_per_sequence(lens: &Tensor, n_seqs: usize) -> Result<Tensor> {
    let rows = lens.dim(0)?;
    if rows == n_seqs {
        return Ok(lens.clone());
    }
    if n_seqs == 0 || !rows.is_multiple_of(n_seqs) {
        candle_core::bail!("QSA context lengths have {rows} rows for {n_seqs} sequences");
    }
    let q_len = rows / n_seqs;
    lens.reshape((n_seqs, q_len))?
        .i((.., q_len - 1))?
        .contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_is_dense_until_the_budget_is_exceeded() {
        let qsa = QsaConfig {
            n_heads: 4,
            head_dim: 128,
            budget: 2048,
            compress_ratio: 4,
        };
        assert_eq!(QsaMode::for_max_kv_len(&qsa, 2051), QsaMode::Dense);
        assert_eq!(
            QsaMode::for_max_kv_len(&qsa, 2052),
            QsaMode::Sparse { max_blocks: 513 }
        );
        let decode = QsaMode::Sparse { max_blocks: 65536 };
        assert_eq!(decode.with_table_capacity(&qsa, 2048), QsaMode::Dense);
        assert_eq!(decode.with_table_capacity(&qsa, 2051), QsaMode::Dense);
        assert_eq!(
            decode.with_table_capacity(&qsa, 2080),
            QsaMode::Sparse { max_blocks: 520 }
        );
        assert_eq!(
            QsaMode::Dense.with_table_capacity(&qsa, 16384),
            QsaMode::Dense
        );
    }
}
