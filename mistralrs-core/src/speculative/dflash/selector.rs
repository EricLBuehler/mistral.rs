use super::*;

pub(super) struct DynamicConv {
    // [2, kernel_size, hidden]
    pub(super) base_kernel: Tensor,
    pub(super) kernel_projection: Tensor,
    pub(super) kernel_size: usize,
    pub(super) group_size: usize,
}

impl DynamicConv {
    pub(super) fn load(vb: ShardedVarBuilder, cfg: &DFlashConfig) -> Result<Self> {
        let kernel_size = cfg
            .dflash_config
            .conv_kernel_size
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no conv_kernel_size"))?;
        let group_size = cfg
            .dflash_config
            .conv_group_size
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no conv_group_size"))?;
        let groups = cfg.hidden_size / group_size;
        Ok(Self {
            base_kernel: vb.get((2, kernel_size, cfg.hidden_size), "base_kernel")?,
            kernel_projection: vb
                .pp("kernel_projection")
                .get((2 * kernel_size * groups, cfg.hidden_size), "weight")?,
            kernel_size,
            group_size,
        })
    }

    // base + dynamic two-tap causal depthwise conv, grouped so one dynamic scalar covers group_size
    // channels: out_t = sum_o (base[o] + dyn[t, o]) * x_{t-o}
    pub(super) fn convolve(
        &self,
        hidden: &Tensor,
        dynamic: &Tensor,
        base: &Tensor,
    ) -> Result<Tensor> {
        let (b, len, h) = hidden.dims3()?;
        #[cfg(feature = "cuda")]
        if hidden.device().is_cuda()
            && self.kernel_size <= crate::cuda::dynamic_conv::MAX_DYNAMIC_CONV_KERNEL_SIZE
        {
            return crate::cuda::dynamic_conv::dynamic_conv(
                hidden,
                dynamic,
                base,
                self.kernel_size,
                self.group_size,
            );
        }
        let groups = h / self.group_size;
        let blocks = hidden.reshape((b, len, groups, self.group_size))?;
        // dynamic: [b, len, kernel_size, groups] -> broadcast over group_size
        let dynamic = dynamic.reshape((b, len, self.kernel_size, groups, 1))?;
        let mut output: Option<Tensor> = None;
        for offset in 0..self.kernel_size {
            let values = if offset == 0 {
                blocks.clone()
            } else {
                let kept = blocks.narrow(1, 0, len - offset)?;
                let pad = Tensor::zeros(
                    (b, offset, groups, self.group_size),
                    blocks.dtype(),
                    blocks.device(),
                )?;
                Tensor::cat(&[pad, kept], 1)?
            };
            let kernel = base
                .i(offset)?
                .reshape((1, 1, groups, self.group_size))?
                .to_dtype(hidden.dtype())?;
            let dyn_o = dynamic.i((.., .., offset))?;
            let term = values
                .broadcast_mul(&kernel)?
                .add(&values.broadcast_mul(&dyn_o)?)?;
            output = Some(match output {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        output
            .ok_or_else(|| candle_core::Error::msg("empty conv kernel"))?
            .reshape((b, len, h))
    }

    pub(super) fn prepare(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, len, h) = hidden.dims3()?;
        let groups = h / self.group_size;
        let dynamic = hidden
            .broadcast_matmul(&self.kernel_projection.t()?.to_dtype(hidden.dtype())?)?
            .reshape((b, len, 2, self.kernel_size, groups))?;
        let pre = self.convolve(hidden, &dynamic.i((.., .., 0))?, &self.base_kernel.i(0)?)?;
        Ok((pre, dynamic.i((.., .., 1))?))
    }

    pub(super) fn finish(&self, hidden: &Tensor, dynamic: &Tensor) -> Result<Tensor> {
        self.convolve(hidden, dynamic, &self.base_kernel.i(1)?)
    }
}

pub(super) struct CandidateSelector {
    // [vocab, rank] raw parameters (no `.weight` suffix in the checkpoint)
    pub(super) predecessor_codebook: Tensor,
    pub(super) successor_codebook: Tensor,
    pub(super) hidden_projection: Tensor,
    pub(super) top_k: usize,
    #[cfg(feature = "cuda")]
    pub(super) vocab_size: usize,
}

pub(super) fn resolve_dflash_sampling_policy(
    method: MtpDraftSamplingMethod,
    capability: std::result::Result<(), String>,
) -> Result<MtpDraftSamplingMethod> {
    match method {
        MtpDraftSamplingMethod::Auto => Ok(if capability.is_ok() {
            MtpDraftSamplingMethod::Probabilistic
        } else {
            MtpDraftSamplingMethod::Greedy
        }),
        MtpDraftSamplingMethod::Greedy => Ok(MtpDraftSamplingMethod::Greedy),
        MtpDraftSamplingMethod::Probabilistic => capability
            .map(|()| MtpDraftSamplingMethod::Probabilistic)
            .map_err(|reason| {
                candle_core::Error::msg(format!(
                    "probabilistic DFlash drafting is unavailable: {reason}"
                ))
            }),
    }
}

#[cfg(feature = "cuda")]
pub(super) struct CandidateSelectorCudaSpec {
    pub(super) device_is_cuda: bool,
    pub(super) logits_dtype: DType,
    pub(super) hidden_projection_dtype: DType,
    pub(super) predecessor_dtype: DType,
    pub(super) successor_dtype: DType,
    pub(super) top_k: usize,
    pub(super) vocab_size: usize,
    pub(super) predecessor_vocab_size: Option<usize>,
    pub(super) successor_vocab_size: Option<usize>,
}

#[cfg(feature = "cuda")]
pub(super) fn validate_candidate_selector_cuda(
    spec: CandidateSelectorCudaSpec,
) -> std::result::Result<(), String> {
    if !spec.device_is_cuda {
        return Err("the draft model is not on CUDA".to_string());
    }
    if !matches!(spec.logits_dtype, DType::BF16 | DType::F16 | DType::F32) {
        return Err(format!(
            "CUDA ranked top-k does not support {:?} logits",
            spec.logits_dtype
        ));
    }
    for (name, dtype) in [
        ("hidden projection", spec.hidden_projection_dtype),
        ("predecessor codebook", spec.predecessor_dtype),
        ("successor codebook", spec.successor_dtype),
    ] {
        if !matches!(dtype, DType::BF16 | DType::F32) {
            return Err(format!(
                "CUDA candidate selection does not support {dtype:?} {name}"
            ));
        }
    }
    if spec.vocab_size == 0 {
        return Err("the vocabulary is empty".to_string());
    }
    if spec.vocab_size > crate::ops::CUDA_TOPK_MAX_EXACT_PACKED_VOCAB {
        return Err(format!(
            "vocabulary size {} cannot be represented exactly by packed F32 indices",
            spec.vocab_size
        ));
    }
    for (name, codebook_vocab_size) in [
        ("predecessor", spec.predecessor_vocab_size),
        ("successor", spec.successor_vocab_size),
    ] {
        if codebook_vocab_size != Some(spec.vocab_size) {
            return Err(format!(
                "{name} codebook vocabulary {codebook_vocab_size:?} does not match logits vocabulary {}",
                spec.vocab_size
            ));
        }
    }
    let max_top_k = crate::ops::cuda_topk_ranked_packed_max_k(spec.vocab_size)
        .expect("nonempty representable vocabulary checked above")
        .min(crate::ops::CUDA_DFLASH_SELECTOR_MAX_K);
    if spec.top_k == 0 || spec.top_k > max_top_k {
        return Err(format!(
            "selector top_k={} must be in [1, {max_top_k}] for vocabulary {}",
            spec.top_k, spec.vocab_size
        ));
    }
    Ok(())
}

impl CandidateSelector {
    pub(super) fn load(vb: ShardedVarBuilder, cfg: &DFlashConfig) -> Result<Self> {
        let rank = cfg
            .dflash_config
            .selector_rank
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no selector_rank"))?;
        let top_k = cfg
            .dflash_config
            .selector_top_k
            .ok_or_else(|| candle_core::Error::msg("DFlash2 config has no selector_top_k"))?;
        Ok(Self {
            predecessor_codebook: vb.get_unchecked("predecessor_codebook")?,
            successor_codebook: vb.get_unchecked("successor_codebook")?,
            hidden_projection: vb
                .pp("hidden_projection")
                .get((rank, cfg.hidden_size), "weight")?,
            top_k,
            #[cfg(feature = "cuda")]
            vocab_size: cfg.vocab_size,
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn cuda_capability(
        &self,
        device_is_cuda: bool,
        logits_dtype: DType,
        vocab_size: usize,
    ) -> std::result::Result<(), String> {
        if vocab_size != self.vocab_size {
            return Err(format!(
                "logits vocabulary {vocab_size} does not match configured vocabulary {}",
                self.vocab_size
            ));
        }
        validate_candidate_selector_cuda(CandidateSelectorCudaSpec {
            device_is_cuda,
            logits_dtype,
            hidden_projection_dtype: self.hidden_projection.dtype(),
            predecessor_dtype: self.predecessor_codebook.dtype(),
            successor_dtype: self.successor_codebook.dtype(),
            top_k: self.top_k,
            vocab_size,
            predecessor_vocab_size: self.predecessor_codebook.dims().first().copied(),
            successor_vocab_size: self.successor_codebook.dims().first().copied(),
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn configured_cuda_capability(
        &self,
        device_is_cuda: bool,
        logits_dtype: DType,
    ) -> std::result::Result<(), String> {
        self.cuda_capability(device_is_cuda, logits_dtype, self.vocab_size)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn cuda_candidates(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
    ) -> Result<(crate::ops::RankedTopKPackedOutput, Tensor)> {
        let (batch, positions, vocab) = logits.dims3()?;
        let rows = batch * positions;
        self.cuda_capability(logits.device().is_cuda(), logits.dtype(), vocab)
            .map_err(candle_core::Error::msg)?;
        let logits = logits.reshape((rows, vocab))?.contiguous()?;
        let topk = crate::ops::cuda_topk_ranked_packed_batched(&logits, self.top_k)?;
        let projection_dtype = self.hidden_projection.dtype();
        let projected = hidden
            .reshape((rows, ()))?
            .to_dtype(projection_dtype)?
            .broadcast_matmul(&self.hidden_projection.t()?)?
            .contiguous()?;
        Ok((topk, projected))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn select_greedy_cuda(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &Tensor,
    ) -> Result<Tensor> {
        let (topk, projected) = self.cuda_candidates(hidden, logits)?;
        crate::ops::cuda_dflash_greedy_select(
            &topk,
            &projected,
            &self.predecessor_codebook.contiguous()?,
            &self.successor_codebook.contiguous()?,
            anchors,
        )
    }

    #[cfg(feature = "cuda")]
    pub(super) fn select_sample_cuda(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &Tensor,
        inverse_temperatures: &Tensor,
        uniforms: &Tensor,
    ) -> Result<crate::ops::DFlashSelectorSampleOutput> {
        let (topk, projected) = self.cuda_candidates(hidden, logits)?;
        crate::ops::cuda_dflash_sample_select(crate::ops::DFlashSelectorSampleInput {
            topk: &topk,
            projected_hidden: &projected,
            predecessor_codebook: &self.predecessor_codebook.contiguous()?,
            successor_codebook: &self.successor_codebook.contiguous()?,
            anchors,
            inverse_temperatures,
            uniforms,
        })
    }

    /// Greedy path walk over the per-position top-k candidates for every sequence at once; scores
    /// couple each candidate to the chosen predecessor through the low-rank codebooks gated by the
    /// position's hidden state. `hidden` is `[batch, n, hidden]` and `logits` is
    /// `[batch, n, vocab]`.
    pub(super) fn select_greedy_batch(
        &self,
        hidden: &Tensor,
        logits: &Tensor,
        anchors: &[u32],
    ) -> Result<Vec<Vec<u32>>> {
        let (batch, positions, vocab) = logits.dims3()?;
        let k = self.top_k;
        let rows = batch * positions;

        #[cfg(feature = "cuda")]
        if self
            .cuda_capability(logits.device().is_cuda(), logits.dtype(), vocab)
            .is_ok()
        {
            let anchors = Tensor::from_vec(anchors.to_vec(), (batch,), logits.device())?;
            return self
                .select_greedy_cuda(hidden, logits, &anchors)?
                .to_vec2::<u32>();
        }

        let logits = logits
            .reshape((rows, vocab))?
            .to_dtype(DType::F32)?
            .contiguous()?;

        let (unary, candidates) = topk_rows(&logits, k)?;
        let hproj = hidden
            .reshape((rows, ()))?
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.hidden_projection.t()?.to_dtype(DType::F32)?)?;
        let cand_flat = candidates.iter().flatten().copied().collect::<Vec<u32>>();
        let cand_ids = Tensor::from_vec(cand_flat.clone(), (rows * k,), logits.device())?;
        let succ = self
            .successor_codebook
            .to_dtype(DType::F32)?
            .index_select(&cand_ids, 0)?;
        let mut pred_ids = anchors.to_vec();
        pred_ids.extend_from_slice(&cand_flat);
        let pred_ids_t = Tensor::from_vec(pred_ids.clone(), (pred_ids.len(),), logits.device())?;
        let pred = self
            .predecessor_codebook
            .to_dtype(DType::F32)?
            .index_select(&pred_ids_t, 0)?;
        let rank = hproj.dim(1)?;
        let packed = Tensor::cat(
            &[
                hproj.flatten_all()?,
                succ.flatten_all()?,
                pred.flatten_all()?,
            ],
            0,
        )?
        .to_vec1::<f32>()?;
        let (h_len, s_len) = (rows * rank, rows * k * rank);
        let hproj: Vec<&[f32]> = packed[..h_len].chunks(rank).collect();
        let succ: Vec<&[f32]> = packed[h_len..h_len + s_len].chunks(rank).collect();
        let pred: Vec<&[f32]> = packed[h_len + s_len..].chunks(rank).collect();

        let mut paths = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut path = Vec::with_capacity(positions);
            let mut pred_row = b;
            for pos in 0..positions {
                let row = b * positions + pos;
                let mut best = f32::NEG_INFINITY;
                let mut best_idx = 0usize;
                for cand in 0..k {
                    let mut dot = 0f32;
                    let sv = &succ[row * k + cand];
                    let pv = &pred[pred_row];
                    let hv = &hproj[row];
                    for r in 0..rank {
                        dot += pv[r] * hv[r] * sv[r];
                    }
                    let score = unary[row][cand] + dot;
                    if score > best {
                        best = score;
                        best_idx = cand;
                    }
                }
                path.push(candidates[row][best_idx]);
                pred_row = batch + row * k + best_idx;
            }
            paths.push(path);
        }
        Ok(paths)
    }
}

pub(super) type TopkRows = (Vec<Vec<f32>>, Vec<Vec<u32>>);
// (is_causal, sliding_window, ctx_len, block)
pub(super) type MaskKey = (bool, Option<usize>, usize, usize);

pub(super) fn topk_rows(logits: &Tensor, k: usize) -> Result<TopkRows> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        let (rows, _) = logits.dims2()?;
        let ones = Tensor::ones((rows,), DType::F32, logits.device())?;
        let packed = crate::ops::cuda_topk_logits_f32_packed_batched(logits, k, &ones)?;
        let packed_rows = packed.packed.to_vec2::<f32>()?;
        let mut values = Vec::with_capacity(rows);
        let mut indices = Vec::with_capacity(rows);
        for row in packed_rows {
            values.push(row[..k].to_vec());
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            indices.push(row[k..2 * k].iter().map(|i| *i as u32).collect());
        }
        return Ok((values, indices));
    }
    let rows = logits.to_vec2::<f32>()?;
    let mut values = Vec::with_capacity(rows.len());
    let mut indices = Vec::with_capacity(rows.len());
    for row in rows {
        let mut order = (0..row.len()).collect::<Vec<_>>();
        order.select_nth_unstable_by(k - 1, |&a, &b| row[b].total_cmp(&row[a]));
        let top = &order[..k];
        values.push(top.iter().map(|&i| row[i]).collect());
        #[allow(clippy::cast_possible_truncation)]
        indices.push(top.iter().map(|&i| i as u32).collect());
    }
    Ok((values, indices))
}
