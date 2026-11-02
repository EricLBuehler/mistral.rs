#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
#[cfg(feature = "cuda")]
use mistralrs_quant::embedding::EmbeddingGatherMode;
use mistralrs_quant::{
    embedding::{EmbeddingPacking, EmbeddingSource, ManagedEmbedding},
    nvfp4::embedding::Nvfp4Embedding,
    safetensors::MmapedSafetensors,
    GgufArchive, GgufEmbedding, QuantMethod, ReplicatedLayer, ShardedVarBuilder,
};

use super::config::{PleConfig, TextConfig};
use crate::layers::grouped_rms_norm;
#[cfg(feature = "cuda")]
use crate::memory_plan::WeightAllocationBudget;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const LAYER_SEED_PRIME: u64 = 10007;
const GATE_MAGNITUDE_FLOOR: f64 = 1e-6;
const TABLE_SHARD_PREFIX: &str = "shard_";
const GGUF_TABLE_NAME: &str = "per_layer_token_embd.weight";
const TABLE_PACKING: EmbeddingPacking = EmbeddingPacking::Symmetric {
    preferred_bits: 8,
    fallback_bits: 4,
};

/// Device bytes of the resident n-gram table at the precision the device map plans for.
#[cfg(feature = "cuda")]
pub(crate) fn planned_resident_table_bytes(cfg: &TextConfig, ple: &PleConfig) -> usize {
    let rows = PleHash::new(cfg, ple, 0).vocab_sizes.iter().sum::<u64>() as usize;
    TABLE_PACKING.planned_bytes(rows, ple.head_dim(), DType::BF16)
}

fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// Hash constants of one PLE module; matches `Qwen4ExpTextNGramEmbedding`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PleHash {
    pub multipliers: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    pub eos: u32,
}

impl PleHash {
    pub(super) fn new(cfg: &TextConfig, ple: &PleConfig, ple_layer_index: u64) -> Self {
        if let Some(hash) = &cfg.ple_hash {
            return Self {
                multipliers: hash.layer_multipliers.clone(),
                vocab_sizes: hash.head_vocab_sizes.clone(),
                offsets: hash.head_offsets.clone(),
                heads_per_ngram: ple.heads_per_ngram,
                eos: ple.eos_token_id,
            };
        }
        let max_long = (1u64 << 63) - 1;
        let multiplier_max = max_long / (cfg.vocab_size as u64).max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = cfg
            .seed
            .wrapping_add(LAYER_SEED_PRIME.wrapping_mul(ple_layer_index));
        let multipliers = (0..ple.ngram_size as u64)
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        let num_heads = ple.num_heads();
        let mut vocab_sizes = Vec::with_capacity(num_heads);
        let mut offsets = Vec::with_capacity(num_heads);
        let mut total = 0u64;
        for head in 0..num_heads {
            let global = ple_layer_index as usize * num_heads + head;
            let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1);
            vocab_sizes.push(size);
            offsets.push(total);
            total += size;
        }
        Self {
            multipliers,
            vocab_sizes,
            offsets,
            heads_per_ngram: ple.heads_per_ngram,
            eos: ple.eos_token_id,
        }
    }

    pub(super) fn num_heads(&self) -> usize {
        self.vocab_sizes.len()
    }

    /// Table rows of `tokens[local]`; `history` precedes `tokens`, oldest first, `None` for no token.
    pub(super) fn rows(
        &self,
        tokens: &[u32],
        local: usize,
        history: &[Option<u32>],
        out: &mut [u64],
    ) {
        let ngram = self.multipliers.len();
        let mut ctx = vec![self.eos as u64; ngram];
        ctx[0] = tokens[local] as u64;
        let mut cut = false;
        for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
            let tok = if local >= s {
                Some(tokens[local - s])
            } else {
                let back = s - local;
                history.len().checked_sub(back).and_then(|idx| history[idx])
            };
            cut = cut || tok.is_none_or(|tok| tok == self.eos);
            *slot = if cut {
                self.eos as u64
            } else {
                tok.unwrap() as u64
            };
        }
        for n in 2..=ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&self.multipliers)
                .fold(0u64, |mixed, (tok, mult)| mixed ^ tok.wrapping_mul(*mult));
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                out[h] = mixed % self.vocab_sizes[h] + self.offsets[h];
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ffi(&self) -> Result<crate::cuda::qwen4_exp::Q4PleHashParams> {
        use crate::cuda::qwen4_exp::{PLE_MAX_HEADS, PLE_MAX_NGRAM};
        if self.multipliers.len() > PLE_MAX_NGRAM || self.num_heads() > PLE_MAX_HEADS {
            candle_core::bail!("PLE n-gram configuration exceeds the CUDA kernel limits");
        }
        let mut params = crate::cuda::qwen4_exp::Q4PleHashParams {
            multipliers: [0; PLE_MAX_NGRAM],
            vocab_sizes: [1; PLE_MAX_HEADS],
            offsets: [0; PLE_MAX_HEADS],
            ngram_size: self.multipliers.len() as i32,
            heads_per_ngram: self.heads_per_ngram as i32,
            num_heads: self.num_heads() as i32,
            eos: self.eos,
        };
        params.multipliers[..self.multipliers.len()].copy_from_slice(&self.multipliers);
        params.vocab_sizes[..self.num_heads()].copy_from_slice(&self.vocab_sizes);
        params.offsets[..self.num_heads()].copy_from_slice(&self.offsets);
        Ok(params)
    }
}

pub(crate) fn table_source(
    raw: Option<Arc<MmapedSafetensors>>,
    archive: Option<Arc<GgufArchive>>,
    prefix: &str,
    columns: usize,
) -> Result<EmbeddingSource> {
    if let Some(archive) = archive {
        let table = GgufEmbedding::new(archive, GGUF_TABLE_NAME)?;
        if table.columns() != columns {
            candle_core::bail!(
                "GGUF n-gram table width {} differs from {columns}",
                table.columns()
            );
        }
        return Ok(EmbeddingSource::gguf(table));
    }
    let raw = raw.ok_or_else(|| {
        candle_core::Error::msg(
        "Qwen4-Exp PLE requires memory-mapped safetensors or GGUF weights (unset MISTRALRS_NO_MMAP)"
    )
    })?;
    let mut names = Vec::new();
    loop {
        let name = format!("{prefix}.{TABLE_SHARD_PREFIX}{}.weight", names.len());
        if raw.get(&name).is_err() {
            break;
        }
        names.push(name);
    }
    let Some(first) = names.first() else {
        candle_core::bail!("Qwen4-Exp PLE table `{prefix}.{TABLE_SHARD_PREFIX}N.weight` not found");
    };
    if DType::try_from(raw.get(first)?.dtype())? == DType::U8 {
        let shards = names
            .into_iter()
            .map(|weight| {
                let scale = format!("{weight}_scale");
                (weight, scale)
            })
            .collect();
        let table = Nvfp4Embedding::new(raw, shards, columns, &format!("{prefix}.weight_scale_2"))?;
        Ok(EmbeddingSource::nvfp4(table))
    } else {
        EmbeddingSource::dense_shards(raw, names, columns)?.with_packing(TABLE_PACKING)
    }
}

/// PLE state before a multi-token speculative verify plus that step's inputs, so a rejected tail can be
/// dropped by rebuilding the state from the accepted prefix.
#[derive(Clone)]
pub(super) struct PleStash {
    pub slots: Vec<u32>,
    // [b, hc * hidden, state_len]
    pub conv: Tensor,
    // [b, context] token history as `id + 1`
    pub history: Tensor,
    // [b, q, hc * hidden]
    pub conv_inputs: Tensor,
    // [b, q] as `id + 1`
    pub tokens: Tensor,
}

impl PleStash {
    pub(super) fn into_tensors(self) -> Vec<Tensor> {
        vec![self.conv, self.history, self.conv_inputs, self.tokens]
    }

    pub(super) fn from_tensors(tensors: &[Tensor], slots: Vec<u32>) -> Result<Self> {
        let [conv, history, conv_inputs, tokens] = tensors else {
            candle_core::bail!("PLE stash expects 4 tensors, got {}", tensors.len());
        };
        Ok(Self {
            slots,
            conv: conv.clone(),
            history: history.clone(),
            conv_inputs: conv_inputs.clone(),
            tokens: tokens.clone(),
        })
    }

    /// Rewrites each `(batch_idx, keep_rows)` sequence's state to the pre-verify state advanced by its kept rows.
    pub(super) fn rollback(
        &self,
        rows: &[(usize, usize)],
        conv_pool: &Tensor,
        history_pool: &Tensor,
    ) -> Result<()> {
        let state_len = self.conv.dim(2)?;
        let context = self.history.dim(1)?;
        for &(batch_idx, keep) in rows {
            let slot = *self.slots.get(batch_idx).ok_or_else(|| {
                candle_core::Error::msg(format!("PLE stash has no batch row {batch_idx}"))
            })? as usize;
            let conv = Tensor::cat(
                &[
                    self.conv.i(batch_idx)?.t()?,
                    self.conv_inputs
                        .i(batch_idx)?
                        .narrow(0, 0, keep)?
                        .to_dtype(self.conv.dtype())?,
                ],
                0,
            )?;
            let conv = conv
                .narrow(0, conv.dim(0)? - state_len, state_len)?
                .t()?
                .unsqueeze(0)?
                .contiguous()?;
            conv_pool.slice_set(&conv, 0, slot)?;
            let history = Tensor::cat(
                &[
                    self.history.i(batch_idx)?,
                    self.tokens.i(batch_idx)?.narrow(0, 0, keep)?,
                ],
                0,
            )?;
            let history = history
                .narrow(0, history.dim(0)? - context, context)?
                .unsqueeze(0)?
                .contiguous()?;
            history_pool.slice_set(&history, 0, slot)?;
        }
        Ok(())
    }
}

/// Hybrid-cache PLE rows: conv history `[slots, hc * hidden, state_len]`, token history as `id + 1` f32.
pub(super) struct PleState<'a> {
    pub conv: &'a Tensor,
    pub history: &'a Tensor,
    pub slots: &'a Tensor,
}

/// Where each logical sequence sits on the flattened token axis.
pub(super) struct PleBatch<'a> {
    pub seqs: &'a [(usize, usize)],
    #[cfg(feature = "cuda")]
    pub layout: Option<&'a crate::cuda::qwen4_exp::TokenLayout>,
}

/// Hashed n-gram embedding injected into every hyper-connection stream (HF `Qwen4ExpTextPLELayer`).
pub(super) struct PleLayer {
    table: ManagedEmbedding,
    #[cfg(feature = "cuda")]
    table_resource_id: String,
    #[cfg(feature = "cuda")]
    pub(super) allow_resident_table: bool,
    hash: PleHash,
    #[cfg(feature = "cuda")]
    hash_ffi: crate::cuda::qwen4_exp::Q4PleHashParams,
    key_proj: Arc<dyn QuantMethod>,
    value_proj: Arc<dyn QuantMethod>,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    // [hc * hidden, kernel]
    conv_weight: Tensor,
    cfg: PleConfig,
    hc: usize,
    hidden: usize,
    eps: f64,
}

impl PleLayer {
    pub(super) fn load(cfg: &TextConfig, ple: &PleConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hc_hidden = cfg.hc_hidden_size();
        let device = vb.device().clone();
        let hash = PleHash::new(cfg, ple, 0);
        let table_vb = vb.pp("ple_embedding").pp("ngram_embedding");
        let table_resource_id = table_vb.prefix();
        let table = ManagedEmbedding::new(
            table_source(
                table_vb.raw_safetensors(),
                table_vb.raw_gguf(),
                &table_resource_id,
                ple.head_dim(),
            )?,
            &device,
        )?;
        if let Some(raw) = vb.raw_safetensors() {
            verify_hash_buffers(&raw, &vb.pp("ple_embedding").prefix(), &hash)?;
        }
        let norm = |name: &str| -> Result<Tensor> {
            vb.pp(name).get(hc_hidden, "weight")?.to_dtype(DType::F32)? + 1.0
        };
        let conv_weight = vb
            .pp("conv1d")
            .get((hc_hidden, 1, ple.conv_kernel_size), "weight")?
            .squeeze(1)?
            .contiguous()?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            hash_ffi: hash.ffi()?,
            table,
            #[cfg(feature = "cuda")]
            table_resource_id,
            #[cfg(feature = "cuda")]
            allow_resident_table: true,
            hash,
            key_proj: ReplicatedLayer::new(
                ple.embed_dim,
                hc_hidden,
                &cfg.quantization_config,
                false,
                vb.pp("key_proj"),
            )?,
            value_proj: ReplicatedLayer::new(
                ple.embed_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                vb.pp("value_proj"),
            )?,
            norm_key: norm("norm_key")?,
            norm_query: norm("norm_query")?,
            norm_conv: norm("norm_conv")?,
            conv_weight,
            cfg: ple.clone(),
            hc: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn gathers_on_device(&self) -> bool {
        self.table.gathers_on_device()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn reclaimable_weight_cache_bytes(&self) -> Option<(Device, usize)> {
        self.table
            .reclaimable_weight_cache_bytes()
            .map(|bytes| (self.conv_weight.device().clone(), bytes))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn table_resource_id(&self) -> &str {
        &self.table_resource_id
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepare_table(
        &mut self,
        budget: &dyn Fn(&Device) -> Result<WeightAllocationBudget>,
    ) -> Result<()> {
        let device = self.conv_weight.device();
        self.table.prepare(
            device,
            budget,
            self.allow_resident_table,
            crate::utils::progress::progress_suppressed(),
        )
    }

    /// `hidden_states + ple(hidden_states, tokens)`, advancing the per-sequence state. `conv_inputs_out`
    /// receives the `[tokens, hc * hidden]` rows appended to the conv history.
    pub(super) fn forward(
        &self,
        hidden_states: &Tensor,
        tokens: &Tensor,
        state: &PleState<'_>,
        batch: &PleBatch<'_>,
        conv_inputs_out: Option<&mut Option<Tensor>>,
    ) -> Result<Tensor> {
        let dims = hidden_states.dims().to_vec();
        let width = dims[dims.len() - 1];
        let flat = hidden_states.reshape(((), width))?;
        let n_tokens = flat.dim(0)?;
        #[cfg(feature = "cuda")]
        if let (true, Some(layout)) = (self.gathers_on_device(), batch.layout) {
            let rows = crate::cuda::qwen4_exp::ple_hash(
                tokens,
                layout,
                state.history,
                state.slots,
                &self.hash_ffi,
            )?;
            let mode = if layout.kv_lens.is_some() {
                EmbeddingGatherMode::Prefill
            } else {
                EmbeddingGatherMode::Decode
            };
            let emb = self
                .table
                .gather(&rows, flat.dtype().into(), mode)?
                .reshape((n_tokens, self.cfg.embed_dim))?;
            let (gated, normed) = self.gate(&emb, &flat)?;
            if let Some(out) = conv_inputs_out {
                *out = Some(normed.clone());
            }
            let out = crate::cuda::qwen4_exp::ple_conv(crate::cuda::qwen4_exp::PleConvArgs {
                normed: &normed,
                gated: &gated,
                residual: &flat,
                weight: &self.conv_weight,
                state_pool: state.conv,
                slots: state.slots,
                layout,
                dilation: self.cfg.conv_dilation(),
            })?;
            return out.reshape(dims);
        }
        let emb = self.host_embeddings(tokens, state, batch.seqs, n_tokens)?;
        let (gated, normed) = self.gate(&emb.to_dtype(flat.dtype())?, &flat)?;
        if let Some(out) = conv_inputs_out {
            *out = Some(normed.clone());
        }
        let out = self.conv_host(&flat, &gated, &normed, state, batch.seqs)?;
        out.reshape(dims)
    }

    /// `(gated_value, norm_conv(gated_value))`, both `[tokens, hc * hidden]`.
    fn gate(&self, emb: &Tensor, flat: &Tensor) -> Result<(Tensor, Tensor)> {
        let key = self.key_proj.forward(emb)?;
        let value = self.value_proj.forward(emb)?;
        #[cfg(feature = "cuda")]
        if flat.device().is_cuda() {
            return crate::cuda::qwen4_exp::ple_gate(
                &key,
                flat,
                &value,
                crate::cuda::qwen4_exp::PleGateWeights {
                    key: &self.norm_key,
                    query: &self.norm_query,
                    conv: &self.norm_conv,
                },
                self.hc,
                self.eps,
            );
        }
        let n = flat.dim(0)?;
        let key = grouped_rms_norm(&key, &self.norm_key, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let query = grouped_rms_norm(flat, &self.norm_query, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (self.hidden as f64).sqrt())?;
        let magnitude = gate.abs()?.maximum(GATE_MAGNITUDE_FLOOR)?.sqrt()?;
        let gate = candle_nn::ops::sigmoid(&(magnitude * gate.sign()?)?)?;
        let gated = gate
            .broadcast_mul(&value.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .reshape((n, self.hc * self.hidden))?
            .to_dtype(flat.dtype())?;
        let normed = grouped_rms_norm(&gated, &self.norm_conv, self.hc, self.eps)?;
        Ok((gated, normed))
    }

    fn host_embeddings(
        &self,
        tokens: &Tensor,
        state: &PleState<'_>,
        seqs: &[(usize, usize)],
        n_tokens: usize,
    ) -> Result<Tensor> {
        let tokens = tokens
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let slots = state.slots.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        let history = state.history.to_device(&Device::Cpu)?;
        let heads = self.hash.num_heads();
        let context = self.cfg.context_len();
        let mut rows = vec![self.hash.offsets[0]; n_tokens * heads];
        let mut new_history = Vec::with_capacity(seqs.len());
        for (seq, &(start, len)) in seqs.iter().enumerate() {
            let stored = history.i(slots[seq] as usize)?.to_vec1::<f32>()?;
            let mut hist = stored
                .iter()
                .map(|v| (*v > 0.0).then(|| *v as u32 - 1))
                .collect::<Vec<_>>();
            let seq_tokens = &tokens[start..start + len];
            for local in 0..len {
                let t = start + local;
                self.hash.rows(
                    seq_tokens,
                    local,
                    &hist,
                    &mut rows[t * heads..(t + 1) * heads],
                );
            }
            hist.extend(seq_tokens.iter().map(|t| Some(*t)));
            let tail = hist[hist.len() - context..]
                .iter()
                .map(|t| t.map_or(0.0, |t| t as f32 + 1.0))
                .collect::<Vec<_>>();
            new_history.push(tail);
        }
        for (seq, tail) in new_history.into_iter().enumerate() {
            let row = Tensor::from_vec(tail, (1, context), state.history.device())?;
            state.history.slice_set(&row, 0, slots[seq] as usize)?;
        }
        let emb = self.table.gather_host(&rows, state.conv.device())?;
        emb.reshape((n_tokens, heads * self.cfg.head_dim()))
    }

    fn conv_host(
        &self,
        flat: &Tensor,
        gated: &Tensor,
        normed: &Tensor,
        state: &PleState<'_>,
        seqs: &[(usize, usize)],
    ) -> Result<Tensor> {
        let slots = state.slots.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        let kernel = self.cfg.conv_kernel_size;
        let dilation = self.cfg.conv_dilation();
        let state_len = self.cfg.conv_state_len();
        let weight = self.conv_weight.to_dtype(DType::F32)?;
        let mut conv_out = Vec::with_capacity(seqs.len());
        for (seq, &(start, len)) in seqs.iter().enumerate() {
            let slot = slots[seq] as usize;
            let history = state.conv.i(slot)?.to_dtype(DType::F32)?.t()?;
            let x = normed.narrow(0, start, len)?.to_dtype(DType::F32)?;
            let padded = Tensor::cat(&[&history, &x], 0)?;
            let mut acc = Tensor::zeros(x.shape(), DType::F32, x.device())?;
            for k in 0..kernel {
                let offset = state_len - (kernel - 1 - k) * dilation;
                let tap = padded.narrow(0, offset, len)?;
                acc = (acc + tap.broadcast_mul(&weight.i((.., k))?.unsqueeze(0)?)?)?;
            }
            conv_out.push(candle_nn::ops::silu(&acc)?);
            let new_state = padded
                .narrow(0, padded.dim(0)? - state_len, state_len)?
                .t()?
                .to_dtype(state.conv.dtype())?
                .unsqueeze(0)?
                .contiguous()?;
            state.conv.slice_set(&new_state, 0, slot)?;
        }
        let conv = Tensor::cat(&conv_out, 0)?;
        if conv.dim(0)? != flat.dim(0)? {
            candle_core::bail!("Qwen4-Exp host PLE requires unpadded, contiguous sequences");
        }
        ((flat.to_dtype(DType::F32)? + gated.to_dtype(DType::F32)?)? + conv)?.to_dtype(flat.dtype())
    }
}

fn verify_hash_buffers(raw: &MmapedSafetensors, prefix: &str, hash: &PleHash) -> Result<()> {
    let read = |name: &str| -> Result<Option<Vec<u64>>> {
        let Ok(view) = raw.get(&format!("{prefix}.{name}")) else {
            return Ok(None);
        };
        if view.dtype() != safetensors::Dtype::I64 {
            candle_core::bail!("PLE buffer {name} must be int64");
        }
        Ok(Some(
            view.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as u64)
                .collect(),
        ))
    };
    for (name, expected) in [
        ("layer_multipliers", &hash.multipliers),
        ("ngram_heads_vocab_sizes", &hash.vocab_sizes),
        ("ngram_heads_offsets", &hash.offsets),
    ] {
        if let Some(stored) = read(name)? {
            if &stored != expected {
                candle_core::bail!(
                    "Qwen4-Exp PLE `{name}` in the checkpoint ({stored:?}) does not match the config ({expected:?})"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "ple_tests.rs"]
mod tests;
