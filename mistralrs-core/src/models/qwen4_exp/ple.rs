use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Error, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::config::Config;

#[allow(dead_code)]
pub(crate) struct PleHasher {
    ngram_size: usize,
    heads_per_ngram: usize,
    eos_token_id: u32,
    multipliers: Vec<u64>,
    head_offsets: Vec<u64>,
    head_vocab_sizes: Vec<u64>,
}

#[allow(dead_code)]
impl PleHasher {
    pub(crate) fn new(config: &Config) -> Result<Self> {
        if config.ple_layer_ids.is_empty() {
            candle_core::bail!("Qwen4Exp PLE hashing requires a configured PLE layer");
        }
        config.validate()?;
        Ok(Self {
            ngram_size: config.ngram_size,
            heads_per_ngram: config.heads_per_ngram,
            eos_token_id: config.eos_token_id,
            multipliers: config.ple_layer_multipliers.clone(),
            head_offsets: config.ple_head_offsets.clone(),
            head_vocab_sizes: config.ple_head_vocab_sizes.clone(),
        })
    }

    pub(crate) fn head_count(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    pub(crate) fn rows_for_tokens(
        &self,
        tokens: &[u32],
        predecessors: &[Option<u32>],
    ) -> Result<Vec<u32>> {
        let n_prev = self.ngram_size - 1;
        let expected_predecessors = tokens
            .len()
            .checked_mul(n_prev)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE predecessor count overflow"))?;
        if predecessors.len() != expected_predecessors {
            candle_core::bail!(
                "Qwen4Exp PLE expected {expected_predecessors} predecessor entries, got {}",
                predecessors.len()
            );
        }

        let mut rows = Vec::with_capacity(tokens.len() * self.head_count());
        let mut context = vec![self.eos_token_id; self.ngram_size];
        for (token_index, &token) in tokens.iter().enumerate() {
            context[0] = token;
            let mut cut = false;
            for distance in 1..self.ngram_size {
                let predecessor = if cut {
                    None
                } else {
                    predecessors[token_index * n_prev + (n_prev - distance)]
                };
                cut = cut
                    || predecessor.is_none()
                    || predecessor.is_some_and(|value| value == self.eos_token_id);
                context[distance] = predecessor.filter(|_| !cut).unwrap_or(self.eos_token_id);
            }

            for ngram in 2..=self.ngram_size {
                let mut mixed = u64::from(context[0]).wrapping_mul(self.multipliers[0]);
                for (token, multiplier) in context[1..ngram].iter().zip(&self.multipliers[1..ngram])
                {
                    mixed ^= u64::from(*token).wrapping_mul(*multiplier);
                }
                let first_head = (ngram - 2) * self.heads_per_ngram;
                for head in first_head..first_head + self.heads_per_ngram {
                    let row = mixed % self.head_vocab_sizes[head] + self.head_offsets[head];
                    rows.push(u32::try_from(row).map_err(|_| {
                        Error::msg(format!("Qwen4Exp PLE row {row} does not fit u32"))
                    })?);
                }
            }
        }
        Ok(rows)
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PleSequenceSnapshot {
    history: Vec<u32>,
    initialized: bool,
    n_prev: usize,
    eos_token_id: u32,
}

#[allow(dead_code)]
pub(crate) struct PleSequenceState {
    histories: HashMap<usize, Vec<u32>>,
    n_prev: usize,
    eos_token_id: u32,
}

#[allow(dead_code)]
impl PleSequenceState {
    pub(crate) fn new(hasher: &PleHasher) -> Self {
        Self {
            histories: HashMap::new(),
            n_prev: hasher.ngram_size - 1,
            eos_token_id: hasher.eos_token_id,
        }
    }

    pub(crate) fn rows_for_chunk(
        &mut self,
        hasher: &PleHasher,
        sequence_id: usize,
        tokens: &[u32],
    ) -> Result<Vec<u32>> {
        if hasher.ngram_size - 1 != self.n_prev || hasher.eos_token_id != self.eos_token_id {
            candle_core::bail!("Qwen4Exp PLE sequence state is incompatible with the hasher");
        }
        let history = self
            .histories
            .entry(sequence_id)
            .or_insert_with(|| vec![self.eos_token_id; self.n_prev]);
        let mut predecessors = Vec::with_capacity(tokens.len() * self.n_prev);
        for &token in tokens {
            predecessors.extend(history.iter().copied().map(Some));
            history.rotate_left(1);
            if let Some(last) = history.last_mut() {
                *last = token;
            }
        }
        hasher.rows_for_tokens(tokens, &predecessors)
    }

    pub(crate) fn rows_for_packed_chunks(
        &mut self,
        hasher: &PleHasher,
        chunks: &[(usize, &[u32])],
    ) -> Result<Vec<u32>> {
        let row_count = chunks.iter().try_fold(0usize, |count, (_, tokens)| {
            let chunk_rows = tokens
                .len()
                .checked_mul(hasher.head_count())
                .ok_or_else(|| Error::msg("Qwen4Exp packed PLE row count overflow"))?;
            count
                .checked_add(chunk_rows)
                .ok_or_else(|| Error::msg("Qwen4Exp packed PLE row count overflow"))
        })?;
        let snapshots = chunks
            .iter()
            .map(|(sequence_id, _)| (*sequence_id, self.snapshot(*sequence_id)))
            .collect::<HashMap<_, _>>();
        let mut rows = Vec::with_capacity(row_count);
        for (sequence_id, tokens) in chunks {
            match self.rows_for_chunk(hasher, *sequence_id, tokens) {
                Ok(chunk_rows) => rows.extend(chunk_rows),
                Err(error) => {
                    for (sequence_id, snapshot) in snapshots {
                        self.restore(sequence_id, &snapshot)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(rows)
    }

    pub(crate) fn snapshot(&self, sequence_id: usize) -> PleSequenceSnapshot {
        let history = self.histories.get(&sequence_id);
        PleSequenceSnapshot {
            history: history
                .cloned()
                .unwrap_or_else(|| vec![self.eos_token_id; self.n_prev]),
            initialized: history.is_some(),
            n_prev: self.n_prev,
            eos_token_id: self.eos_token_id,
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &PleSequenceSnapshot) -> Result<()> {
        if snapshot.n_prev != self.n_prev || snapshot.eos_token_id != self.eos_token_id {
            candle_core::bail!("Qwen4Exp PLE snapshot is incompatible with the sequence state");
        }
        if snapshot.history.len() != self.n_prev {
            candle_core::bail!(
                "Qwen4Exp PLE snapshot expected {} predecessor tokens, got {}",
                self.n_prev,
                snapshot.history.len()
            );
        }
        Ok(())
    }

    pub(crate) fn restore(
        &mut self,
        sequence_id: usize,
        snapshot: &PleSequenceSnapshot,
    ) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        if snapshot.initialized {
            self.histories.insert(sequence_id, snapshot.history.clone());
        } else {
            self.histories.remove(&sequence_id);
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, sequence_id: usize) -> bool {
        self.histories.remove(&sequence_id).is_some()
    }

    pub(crate) fn reset(&mut self, sequence_id: usize) {
        self.release(sequence_id);
    }

    pub(crate) fn clear(&mut self) {
        self.histories.clear();
    }

    /// Whether predecessor history still exists for a sequence.
    pub(crate) fn has_sequence(&self, sequence_id: usize) -> bool {
        self.histories.contains_key(&sequence_id)
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PleConvSnapshot {
    history: Vec<f32>,
    initialized: bool,
    history_tokens: usize,
    channels: usize,
}

#[allow(dead_code)]
pub(crate) struct PleConvState {
    histories: HashMap<usize, Vec<f32>>,
    kernel_size: usize,
    dilation: usize,
    history_tokens: usize,
    channels: usize,
}

#[allow(dead_code)]
impl PleConvState {
    pub(crate) fn new(kernel_size: usize, dilation: usize, channels: usize) -> Result<Self> {
        if kernel_size == 0 || dilation == 0 || channels == 0 {
            candle_core::bail!(
                "Qwen4Exp PLE convolution kernel size, dilation, and channels must be non-zero"
            );
        }
        let history_tokens = (kernel_size - 1)
            .checked_mul(dilation)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE convolution history length overflow"))?;
        history_tokens
            .checked_mul(channels)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE convolution state size overflow"))?;
        Ok(Self {
            histories: HashMap::new(),
            kernel_size,
            dilation,
            history_tokens,
            channels,
        })
    }

    pub(crate) fn forward_chunk(
        &mut self,
        sequence_id: usize,
        input: &[f32],
        kernel: &[f32],
    ) -> Result<Vec<f32>> {
        let expected_kernel = self
            .kernel_size
            .checked_mul(self.channels)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE convolution kernel size overflow"))?;
        if kernel.len() != expected_kernel {
            candle_core::bail!(
                "Qwen4Exp PLE convolution expected {expected_kernel} kernel values, got {}",
                kernel.len()
            );
        }
        if !input.len().is_multiple_of(self.channels) {
            candle_core::bail!(
                "Qwen4Exp PLE convolution input length {} is not divisible by {} channels",
                input.len(),
                self.channels
            );
        }

        let history_values = self.history_tokens * self.channels;
        let history = self
            .histories
            .get(&sequence_id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; history_values]);
        let token_count = input.len() / self.channels;
        let mut output = vec![0.0; input.len()];
        for token in 0..token_count {
            for channel in 0..self.channels {
                let mut value = 0.0f32;
                for tap in 0..self.kernel_size {
                    let lookback = (self.kernel_size - 1 - tap) * self.dilation;
                    let padded_token = self.history_tokens + token - lookback;
                    let activation = if padded_token < self.history_tokens {
                        history[padded_token * self.channels + channel]
                    } else {
                        input[(padded_token - self.history_tokens) * self.channels + channel]
                    };
                    value += activation * kernel[tap * self.channels + channel];
                }
                output[token * self.channels + channel] = value / (1.0 + (-value).exp());
            }
        }

        let mut updated = Vec::with_capacity(history_values + input.len());
        updated.extend_from_slice(&history);
        updated.extend_from_slice(input);
        let keep_from = updated.len().saturating_sub(history_values);
        self.histories
            .insert(sequence_id, updated[keep_from..].to_vec());
        Ok(output)
    }

    pub(crate) fn forward_tensor_chunk(
        &mut self,
        sequence_id: usize,
        input: &Tensor,
        kernel: &Tensor,
    ) -> Result<Tensor> {
        let (token_count, channels) = input.dims2()?;
        if channels != self.channels {
            candle_core::bail!(
                "Qwen4Exp PLE convolution expected {} input channels, got {channels}",
                self.channels
            );
        }
        let (kernel_size, kernel_channels) = kernel.dims2()?;
        if kernel_size != self.kernel_size || kernel_channels != self.channels {
            candle_core::bail!(
                "Qwen4Exp PLE convolution expected kernel shape [{}, {}], got {:?}",
                self.kernel_size,
                self.channels,
                kernel.dims()
            );
        }
        if input.dtype() != kernel.dtype() {
            candle_core::bail!(
                "Qwen4Exp PLE convolution input and kernel dtypes differ: {:?} and {:?}",
                input.dtype(),
                kernel.dtype()
            );
        }
        if !input.device().same_device(kernel.device()) {
            candle_core::bail!(
                "Qwen4Exp PLE convolution input and kernel are on different devices"
            );
        }

        let history_values = self.history_tokens * self.channels;
        let history = self
            .histories
            .get(&sequence_id)
            .cloned()
            .unwrap_or_else(|| vec![0.0; history_values]);
        let history = Tensor::from_vec(
            history,
            (self.history_tokens, self.channels),
            input.device(),
        )?
        .to_dtype(input.dtype())?;
        let state_and_input = Tensor::cat(&[history, input.clone()], 0)?;
        let mut outputs = Vec::with_capacity(token_count);
        for token in 0..token_count {
            let mut taps = Vec::with_capacity(self.kernel_size);
            for tap in 0..self.kernel_size {
                let lookback = (self.kernel_size - 1 - tap) * self.dilation;
                taps.push(state_and_input.narrow(0, self.history_tokens + token - lookback, 1)?);
            }
            let window = Tensor::cat(&taps, 0)?;
            outputs.push((window * kernel)?.sum(0)?);
        }
        let output = if outputs.is_empty() {
            Tensor::zeros((0, self.channels), input.dtype(), input.device())?
        } else {
            Tensor::stack(&outputs, 0)?
        };
        let output = candle_nn::ops::silu(&output)?;

        let keep_from = state_and_input.dim(0)?.saturating_sub(self.history_tokens);
        let updated = state_and_input
            .narrow(0, keep_from, self.history_tokens)?
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        self.histories.insert(sequence_id, updated);
        Ok(output)
    }

    pub(crate) fn forward_packed_chunks(
        &mut self,
        chunks: &[(usize, &[f32])],
        kernel: &[f32],
    ) -> Result<Vec<f32>> {
        let output_values = chunks.iter().try_fold(0usize, |count, (_, input)| {
            count
                .checked_add(input.len())
                .ok_or_else(|| Error::msg("Qwen4Exp packed PLE convolution output size overflow"))
        })?;
        let snapshots = chunks
            .iter()
            .map(|(sequence_id, _)| (*sequence_id, self.snapshot(*sequence_id)))
            .collect::<HashMap<_, _>>();
        let mut output = Vec::with_capacity(output_values);
        for (sequence_id, input) in chunks {
            match self.forward_chunk(*sequence_id, input, kernel) {
                Ok(chunk_output) => output.extend(chunk_output),
                Err(error) => {
                    for (sequence_id, snapshot) in snapshots {
                        self.restore(sequence_id, &snapshot)?;
                    }
                    return Err(error);
                }
            }
        }
        Ok(output)
    }

    pub(crate) fn snapshot(&self, sequence_id: usize) -> PleConvSnapshot {
        let history = self.histories.get(&sequence_id);
        PleConvSnapshot {
            history: history
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.history_tokens * self.channels]),
            initialized: history.is_some(),
            history_tokens: self.history_tokens,
            channels: self.channels,
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &PleConvSnapshot) -> Result<()> {
        if snapshot.history_tokens != self.history_tokens || snapshot.channels != self.channels {
            candle_core::bail!("Qwen4Exp PLE convolution snapshot is incompatible with the state");
        }
        if snapshot.history.len() != self.history_tokens * self.channels {
            candle_core::bail!("Qwen4Exp PLE convolution snapshot has an invalid history length");
        }
        Ok(())
    }

    pub(crate) fn restore(&mut self, sequence_id: usize, snapshot: &PleConvSnapshot) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        if snapshot.initialized {
            self.histories.insert(sequence_id, snapshot.history.clone());
        } else {
            self.histories.remove(&sequence_id);
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, sequence_id: usize) -> bool {
        self.histories.remove(&sequence_id).is_some()
    }

    pub(crate) fn reset(&mut self, sequence_id: usize) {
        self.release(sequence_id);
    }

    pub(crate) fn clear(&mut self) {
        self.histories.clear();
    }

    /// Whether convolution history still exists for a sequence.
    pub(crate) fn has_sequence(&self, sequence_id: usize) -> bool {
        self.histories.contains_key(&sequence_id)
    }
}

#[allow(dead_code)]
pub(crate) struct PleEmbedding {
    table: Arc<dyn QuantMethod>,
    head_count: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl PleEmbedding {
    pub(crate) fn new(
        table: Arc<dyn QuantMethod>,
        head_count: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if head_count == 0 || head_dim == 0 {
            candle_core::bail!("Qwen4Exp PLE embedding head count and dimension must be non-zero");
        }
        Ok(Self {
            table,
            head_count,
            head_dim,
        })
    }

    pub(crate) fn gather(
        &self,
        rows: &[u32],
        batch: usize,
        tokens: usize,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let expected_rows = batch
            .checked_mul(tokens)
            .and_then(|count| count.checked_mul(self.head_count))
            .ok_or_else(|| Error::msg("Qwen4Exp PLE gather row count overflow"))?;
        if rows.len() != expected_rows {
            candle_core::bail!(
                "Qwen4Exp PLE gather expected {expected_rows} rows for [{batch}, {tokens}, {}], got {}",
                self.head_count,
                rows.len()
            );
        }

        let (_, table_device) = self.table.dtype_and_device();
        let row_ids = Tensor::from_vec(
            rows.to_vec(),
            (batch, tokens, self.head_count),
            &table_device,
        )?;
        let gathered = self
            .table
            .embedding_forward(&row_ids, dtype)
            .map_err(|error| {
                Error::msg(format!(
                "Qwen4Exp PLE embedding row gather is unsupported by the loaded weight: {error}"
            ))
            })?;
        let expected_shape = [batch, tokens, self.head_count, self.head_dim];
        if gathered.dims() != expected_shape {
            candle_core::bail!(
                "Qwen4Exp PLE table returned shape {:?}, expected {:?}",
                gathered.dims(),
                expected_shape
            );
        }
        gathered
            .to_device(device)?
            .reshape((batch, tokens, self.head_count * self.head_dim))
    }
}

#[allow(dead_code)]
pub(crate) struct PleLayer {
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    conv_kernel: Tensor,
    norm_eps: f64,
    hidden_size: usize,
    hc_count: usize,
}

#[allow(dead_code)]
impl PleLayer {
    /// PLE normalization gammas (key, query, and convolution input), exposed for ISQ
    /// residual handling.
    pub(crate) fn residual_norms(&self) -> [&Tensor; 3] {
        [&self.norm_key, &self.norm_query, &self.norm_conv]
    }

    pub(crate) fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let head_count = (config.ngram_size - 1)
            .checked_mul(config.heads_per_ngram)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE head count overflow"))?;
        if !config.ple_embed_dim.is_multiple_of(head_count) {
            candle_core::bail!(
                "Qwen4Exp ple_embed_dim {} must be divisible by {head_count} PLE heads",
                config.ple_embed_dim
            );
        }
        let embedding_size = config.ple_embed_dim;
        let wide_size = config
            .hidden_size
            .checked_mul(config.hc_count)
            .ok_or_else(|| Error::msg("Qwen4Exp PLE wide width overflow"))?;
        Ok(Self {
            key: ReplicatedLayer::new(
                embedding_size,
                wide_size,
                &config.quantization_config,
                false,
                vb.pp("key"),
            )?,
            value: ReplicatedLayer::new(
                embedding_size,
                config.hidden_size,
                &config.quantization_config,
                false,
                vb.pp("value"),
            )?,
            norm_key: vb.pp("norm_key").get(wide_size, "weight")?,
            norm_query: vb.pp("norm_query").get(wide_size, "weight")?,
            norm_conv: vb.pp("norm_conv").get(wide_size, "weight")?,
            // The GGUF converter stores the conv kernel as ne = [kernel, channels], which
            // the weight source presents row-major as [channels, kernel] — the same
            // convention as the GDN `ssm_conv1d`. Transpose once at load to the internal
            // [kernel, channels] working layout.
            conv_kernel: vb
                .pp("conv1d")
                .get((wide_size, config.ple_conv_kernel_size), "weight")?
                .t()?
                .contiguous()?,
            norm_eps: config.rms_norm_eps,
            hidden_size: config.hidden_size,
            hc_count: config.hc_count,
        })
    }

    fn grouped_norm(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let (batch, tokens, streams, hidden) = input.dims4()?;
        if streams != self.hc_count || hidden != self.hidden_size {
            candle_core::bail!(
                "Qwen4Exp PLE grouped norm expected [{batch}, {tokens}, {}, {}], got {:?}",
                self.hc_count,
                self.hidden_size,
                input.dims()
            );
        }
        let dtype = input.dtype();
        let input = input.to_dtype(DType::F32)?;
        let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
        input
            .broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?
            .reshape((batch, tokens, self.hc_count * self.hidden_size))?
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?
            .reshape((batch, tokens, self.hc_count, self.hidden_size))?
            .to_dtype(dtype)
    }

    pub(crate) fn prepare_convolution(
        &self,
        embeddings: &Tensor,
        hidden: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, tokens, streams, hidden_size) = hidden.dims4()?;
        if streams != self.hc_count || hidden_size != self.hidden_size {
            candle_core::bail!(
                "Qwen4Exp PLE expected hidden shape [batch, tokens, {}, {}], got {:?}",
                self.hc_count,
                self.hidden_size,
                hidden.dims()
            );
        }
        if embeddings.dims().len() != 3
            || embeddings.dim(0)? != batch
            || embeddings.dim(1)? != tokens
        {
            candle_core::bail!(
                "Qwen4Exp PLE embeddings and hidden states have incompatible shapes"
            );
        }

        let key = self.key.forward(embeddings)?.reshape((
            batch,
            tokens,
            self.hc_count,
            self.hidden_size,
        ))?;
        let value = self.value.forward(embeddings)?;
        let key = self.grouped_norm(&key, &self.norm_key)?;
        let query = self.grouped_norm(hidden, &self.norm_query)?;
        let score = ((key * query)?
            .sum_keepdim(D::Minus1)?
            .to_dtype(DType::F32)?
            / (self.hidden_size as f64).sqrt())?;
        let magnitude = score.abs()?.clamp(1e-6, f64::INFINITY)?.sqrt()?;
        let sign = (score.ge(0.0)?.to_dtype(DType::F32)? - score.le(0.0)?.to_dtype(DType::F32)?)?;
        let gate = candle_nn::ops::sigmoid(&(sign * magnitude)?)?.to_dtype(value.dtype())?;
        let gated = value.unsqueeze(2)?.broadcast_mul(&gate)?.reshape((
            batch,
            tokens,
            self.hc_count,
            self.hidden_size,
        ))?;
        let normalized = self.grouped_norm(&gated, &self.norm_conv)?;
        Ok((gated, normalized))
    }

    pub(crate) fn forward_chunk(
        &self,
        state: &mut PleConvState,
        sequence_id: usize,
        embeddings: &Tensor,
        hidden: &Tensor,
    ) -> Result<Tensor> {
        let (gated, normalized) = self.prepare_convolution(embeddings, hidden)?;
        let (batch, tokens, _, _) = normalized.dims4()?;
        if batch != 1 {
            candle_core::bail!("Qwen4Exp PLE stateful convolution expects one sequence per chunk");
        }
        let normalized = normalized.reshape((tokens, self.hc_count * self.hidden_size))?;
        let convolved = state
            .forward_tensor_chunk(sequence_id, &normalized, &self.conv_kernel)?
            .reshape((1, tokens, self.hc_count, self.hidden_size))?;
        hidden + (gated + convolved)?
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PleStateSnapshot {
    sequence: PleSequenceSnapshot,
    convolution: PleConvSnapshot,
}

#[allow(dead_code)]
pub(crate) struct PleState {
    sequences: PleSequenceState,
    convolution: PleConvState,
}

#[allow(dead_code)]
impl PleState {
    pub(crate) fn new(
        hasher: &PleHasher,
        kernel_size: usize,
        dilation: usize,
        channels: usize,
    ) -> Result<Self> {
        Ok(Self {
            sequences: PleSequenceState::new(hasher),
            convolution: PleConvState::new(kernel_size, dilation, channels)?,
        })
    }

    pub(crate) fn forward_packed_chunks(
        &mut self,
        hasher: &PleHasher,
        token_chunks: &[(usize, &[u32])],
        conv_chunks: &[(usize, &[f32])],
        kernel: &[f32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        if token_chunks.len() != conv_chunks.len() {
            candle_core::bail!(
                "Qwen4Exp packed PLE expected equal token and convolution chunk counts, got {} and {}",
                token_chunks.len(),
                conv_chunks.len()
            );
        }
        for ((token_sequence, tokens), (conv_sequence, input)) in
            token_chunks.iter().zip(conv_chunks)
        {
            if token_sequence != conv_sequence {
                candle_core::bail!(
                    "Qwen4Exp packed PLE sequence mismatch: token sequence {token_sequence}, convolution sequence {conv_sequence}"
                );
            }
            if !input.len().is_multiple_of(self.convolution.channels)
                || tokens.len() != input.len() / self.convolution.channels
            {
                candle_core::bail!(
                    "Qwen4Exp packed PLE sequence {token_sequence} has {} tokens but {} convolution values for {} channels",
                    tokens.len(),
                    input.len(),
                    self.convolution.channels
                );
            }
        }

        let snapshots = token_chunks
            .iter()
            .map(|(sequence_id, _)| (*sequence_id, self.snapshot(*sequence_id)))
            .collect::<HashMap<_, _>>();
        let rows = self
            .sequences
            .rows_for_packed_chunks(hasher, token_chunks)?;
        match self.convolution.forward_packed_chunks(conv_chunks, kernel) {
            Ok(output) => Ok((rows, output)),
            Err(error) => {
                for (sequence_id, snapshot) in snapshots {
                    self.restore(sequence_id, &snapshot)?;
                }
                Err(error)
            }
        }
    }

    pub(crate) fn forward_packed_tensor_chunks(
        &mut self,
        hasher: &PleHasher,
        embedding: &PleEmbedding,
        layer: &PleLayer,
        token_chunks: &[(usize, &[u32])],
        hidden_chunks: &[(usize, &Tensor)],
    ) -> Result<Tensor> {
        if token_chunks.is_empty() || token_chunks.len() != hidden_chunks.len() {
            candle_core::bail!(
                "Qwen4Exp packed PLE tensor execution requires equal non-empty token and hidden chunk counts"
            );
        }
        for ((token_sequence, tokens), (hidden_sequence, hidden)) in
            token_chunks.iter().zip(hidden_chunks)
        {
            if token_sequence != hidden_sequence {
                candle_core::bail!(
                    "Qwen4Exp packed PLE tensor sequence mismatch: token sequence {token_sequence}, hidden sequence {hidden_sequence}"
                );
            }
            let (batch, hidden_tokens, streams, hidden_size) = hidden.dims4()?;
            if batch != 1
                || hidden_tokens != tokens.len()
                || streams != layer.hc_count
                || hidden_size != layer.hidden_size
            {
                candle_core::bail!(
                    "Qwen4Exp packed PLE sequence {token_sequence} has incompatible token and hidden shapes"
                );
            }
        }

        let snapshots = token_chunks
            .iter()
            .map(|(sequence_id, _)| (*sequence_id, self.snapshot(*sequence_id)))
            .collect::<HashMap<_, _>>();
        let result = (|| {
            let mut outputs = Vec::with_capacity(token_chunks.len());
            for ((sequence_id, tokens), (_, hidden)) in token_chunks.iter().zip(hidden_chunks) {
                let rows = self
                    .sequences
                    .rows_for_chunk(hasher, *sequence_id, tokens)?;
                let gathered =
                    embedding.gather(&rows, 1, tokens.len(), hidden.dtype(), hidden.device())?;
                outputs.push(layer.forward_chunk(
                    &mut self.convolution,
                    *sequence_id,
                    &gathered,
                    hidden,
                )?);
            }
            Tensor::cat(&outputs, 1)
        })();
        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                for (sequence_id, snapshot) in snapshots {
                    self.restore(sequence_id, &snapshot)?;
                }
                Err(error)
            }
        }
    }

    pub(crate) fn snapshot(&self, sequence_id: usize) -> PleStateSnapshot {
        PleStateSnapshot {
            sequence: self.sequences.snapshot(sequence_id),
            convolution: self.convolution.snapshot(sequence_id),
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &PleStateSnapshot) -> Result<()> {
        self.sequences.validate_snapshot(&snapshot.sequence)?;
        self.convolution.validate_snapshot(&snapshot.convolution)
    }

    pub(crate) fn restore(
        &mut self,
        sequence_id: usize,
        snapshot: &PleStateSnapshot,
    ) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        let current_sequence = self.sequences.snapshot(sequence_id);
        self.sequences.restore(sequence_id, &snapshot.sequence)?;
        if let Err(error) = self.convolution.restore(sequence_id, &snapshot.convolution) {
            self.sequences.restore(sequence_id, &current_sequence)?;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, sequence_id: usize) -> bool {
        self.sequences.release(sequence_id) | self.convolution.release(sequence_id)
    }

    /// Reset one sequence to the absent state; histories reinitialize on next use.
    pub(crate) fn reset(&mut self, sequence_id: usize) {
        self.sequences.reset(sequence_id);
        self.convolution.reset(sequence_id);
    }

    pub(crate) fn clear(&mut self) {
        self.sequences.clear();
        self.convolution.clear();
    }

    /// Whether predecessor or convolution history still exists for a sequence.
    pub(crate) fn has_sequence(&self, sequence_id: usize) -> bool {
        self.sequences.has_sequence(sequence_id) || self.convolution.has_sequence(sequence_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

    use crate::models::qwen4_exp::config::tests::fixture_config;

    fn projection(weight: Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?))
    }

    fn ple_layer() -> Result<PleLayer> {
        let device = candle_core::Device::Cpu;
        Ok(PleLayer {
            key: projection(Tensor::from_vec(
                vec![
                    1.0f32, 0.0, 0.0, 1.0, // stream 0 key
                    1.0, 0.0, 0.0, 1.0, // stream 1 key
                ],
                (4, 2),
                &device,
            )?)?,
            value: projection(Tensor::from_vec(
                vec![
                    1.0f32, 0.0, // value dim 0
                    0.0, 1.0, // value dim 1
                ],
                (2, 2),
                &device,
            )?)?,
            norm_key: Tensor::ones(4, DType::F32, &device)?,
            norm_query: Tensor::ones(4, DType::F32, &device)?,
            norm_conv: Tensor::ones(4, DType::F32, &device)?,
            conv_kernel: Tensor::zeros((2, 4), DType::F32, &device)?,
            norm_eps: 1e-6,
            hidden_size: 2,
            hc_count: 2,
        })
    }

    #[test]
    fn ple_embedding_gather_preserves_token_and_head_order() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let table = Tensor::from_vec(
            vec![
                0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
            ],
            (6, 2),
            &device,
        )?;
        let embedding = PleEmbedding::new(projection(table)?, 2, 2)?;
        let gathered = embedding.gather(&[4, 1, 3, 5], 1, 2, DType::F32, &device)?;

        assert_eq!(gathered.dims(), &[1, 2, 4]);
        assert_eq!(
            gathered.flatten_all()?.to_vec1::<f32>()?,
            vec![4.0, 4.5, 1.0, 1.5, 3.0, 3.5, 5.0, 5.5]
        );
        Ok(())
    }

    #[test]
    fn ple_embedding_gather_supports_q8_0_without_dense_table_conversion() -> Result<()> {
        use candle_core::quantized::{GgmlDType, QTensor};
        use mistralrs_quant::GgufMatMul;

        let device = candle_core::Device::Cpu;
        let values = (0..8 * 32)
            .map(|index| (index as f32 - 64.0) / 17.0)
            .collect::<Vec<_>>();
        let dense = Tensor::from_vec(values, (8, 32), &device)?;
        let quantized = Arc::new(QTensor::quantize(&dense, GgmlDType::Q8_0)?);
        let table = Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: quantized.clone(),
            b: None,
        })?) as Arc<dyn QuantMethod>;
        let embedding = PleEmbedding::new(table, 2, 32)?;
        let rows = [7, 1, 3, 1];
        let actual = embedding.gather(&rows, 1, 2, DType::F32, &device)?;
        let expected = quantized
            .dequantize(&device)?
            .index_select(&Tensor::from_vec(rows.to_vec(), 4, &device)?, 0)?
            .reshape((1, 2, 64))?;

        let max_diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(max_diff <= 1e-6, "max_diff={max_diff}");
        Ok(())
    }

    #[test]
    fn ple_embedding_gather_rejects_wrong_row_inventory() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let table = Tensor::zeros((6, 2), DType::F32, &device)?;
        let embedding = PleEmbedding::new(projection(table)?, 2, 2)?;
        let error = embedding
            .gather(&[1, 2, 3], 1, 2, DType::F32, &device)
            .unwrap_err();
        assert!(error.to_string().contains("expected 4 rows"));
        Ok(())
    }

    #[test]
    fn ple_projection_gate_matches_small_reference() -> Result<()> {
        let layer = ple_layer()?;
        let embeddings = Tensor::from_vec(
            vec![1.0f32, 2.0, -1.0, 0.5],
            (1, 2, 2),
            &candle_core::Device::Cpu,
        )?;
        let hidden = Tensor::from_vec(
            vec![
                2.0f32, 1.0, 1.0, -2.0, // token 0 streams
                -0.5, 1.5, 2.0, 0.25, // token 1 streams
            ],
            (1, 2, 2, 2),
            &candle_core::Device::Cpu,
        )?;

        let (gated, normalized) = layer.prepare_convolution(&embeddings, &hidden)?;
        let gated = gated.flatten_all()?.to_vec1::<f32>()?;
        let normalized = normalized.flatten_all()?.to_vec1::<f32>()?;

        let eps = 1e-6f32;
        let mut expected_gated = Vec::new();
        let mut expected_normalized = Vec::new();
        let embedding_rows = [[1.0f32, 2.0], [-1.0, 0.5]];
        let hidden_rows = [[[2.0f32, 1.0], [1.0, -2.0]], [[-0.5, 1.5], [2.0, 0.25]]];
        for (embedding, streams) in embedding_rows.iter().zip(hidden_rows) {
            let keys = [[embedding[0], embedding[1]], [embedding[0], embedding[1]]];
            for (key, query) in keys.iter().zip(streams) {
                let key_rms = ((key[0] * key[0] + key[1] * key[1]) / 2.0 + eps).sqrt();
                let query_rms = ((query[0] * query[0] + query[1] * query[1]) / 2.0 + eps).sqrt();
                let score = (key[0] / key_rms * query[0] / query_rms
                    + key[1] / key_rms * query[1] / query_rms)
                    / 2.0f32.sqrt();
                let transformed = score.signum() * score.abs().max(1e-6).sqrt();
                let gate = 1.0 / (1.0 + (-transformed).exp());
                let values = [embedding[0] * gate, embedding[1] * gate];
                expected_gated.extend(values);
                let value_rms =
                    ((values[0] * values[0] + values[1] * values[1]) / 2.0 + eps).sqrt();
                expected_normalized.extend([values[0] / value_rms, values[1] / value_rms]);
            }
        }
        for (actual, expected) in gated.iter().zip(expected_gated) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        for (actual, expected) in normalized.iter().zip(expected_normalized) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn ple_layer_adds_gated_and_convolved_values_to_hidden() -> Result<()> {
        let layer = ple_layer()?;
        let embeddings = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2), &candle_core::Device::Cpu)?;
        let hidden = Tensor::from_vec(
            vec![2.0f32, 1.0, 1.0, -2.0],
            (1, 1, 2, 2),
            &candle_core::Device::Cpu,
        )?;
        let (gated, _) = layer.prepare_convolution(&embeddings, &hidden)?;
        let expected = (&hidden + &gated)?;
        let mut state = PleConvState::new(2, 3, 4)?;
        let actual = layer.forward_chunk(&mut state, 7, &embeddings, &hidden)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(state.snapshot(7).initialized);
        Ok(())
    }

    #[test]
    fn ple_projection_rejects_incompatible_hidden_shape() -> Result<()> {
        let layer = ple_layer()?;
        let embeddings = Tensor::zeros((1, 1, 2), DType::F32, &candle_core::Device::Cpu)?;
        let hidden = Tensor::zeros((1, 1, 1, 2), DType::F32, &candle_core::Device::Cpu)?;
        let error = layer.prepare_convolution(&embeddings, &hidden).unwrap_err();
        assert!(error.to_string().contains("expected hidden shape"));
        Ok(())
    }

    #[test]
    fn hash_rows_match_golden_wrapping_u64_vectors() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let rows = hasher.rows_for_tokens(&[7, 11], &[Some(5), Some(3), Some(7), Some(5)])?;
        assert_eq!(
            rows,
            vec![
                34, 134, 234, 334, 434, 534, 634, 734, 843, 943, 1043, 1143, 1243, 1343, 1443,
                1543, 96, 196, 296, 396, 496, 596, 696, 796, 899, 999, 1099, 1199, 1299, 1399,
                1499, 1599,
            ]
        );
        Ok(())
    }

    #[test]
    fn eos_and_missing_predecessors_reset_the_older_window() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let eos = config.eos_token_id;
        let after_eos = hasher.rows_for_tokens(&[13], &[Some(9), Some(eos)])?;
        let missing = hasher.rows_for_tokens(&[13], &[Some(9), None])?;
        assert_eq!(after_eos, missing);

        let older_nine = hasher.rows_for_tokens(&[13], &[Some(9), Some(eos)])?;
        let older_forty_two = hasher.rows_for_tokens(&[13], &[Some(42), Some(eos)])?;
        assert_eq!(older_nine, older_forty_two);
        Ok(())
    }

    #[test]
    fn sequence_state_makes_chunked_hashing_match_one_shot() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let tokens = [3, 5, 7, config.eos_token_id, 11, 13];

        let mut one_shot_state = PleSequenceState::new(&hasher);
        let one_shot = one_shot_state.rows_for_chunk(&hasher, 4, &tokens)?;

        let mut chunked_state = PleSequenceState::new(&hasher);
        let mut chunked = chunked_state.rows_for_chunk(&hasher, 4, &tokens[..2])?;
        chunked.extend(chunked_state.rows_for_chunk(&hasher, 4, &tokens[2..5])?);
        chunked.extend(chunked_state.rows_for_chunk(&hasher, 4, &tokens[5..])?);
        assert_eq!(chunked, one_shot);
        Ok(())
    }

    #[test]
    fn sequence_state_is_independent_and_resettable() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleSequenceState::new(&hasher);
        state.rows_for_chunk(&hasher, 1, &[3, 5])?;
        state.rows_for_chunk(&hasher, 2, &[41])?;

        let continued = state.rows_for_chunk(&hasher, 1, &[7])?;
        state.reset(1);
        let reset = state.rows_for_chunk(&hasher, 1, &[7])?;
        assert_ne!(continued, reset);

        state.clear();
        let sequence_two_reset = state.rows_for_chunk(&hasher, 2, &[43])?;
        let mut fresh = PleSequenceState::new(&hasher);
        assert_eq!(sequence_two_reset, fresh.rows_for_chunk(&hasher, 2, &[43])?);
        Ok(())
    }

    #[test]
    fn packed_chunks_match_independent_sequence_processing() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let chunks: &[(usize, &[u32])] = &[(7, &[3, 5]), (11, &[13]), (7, &[17])];

        let mut packed_state = PleSequenceState::new(&hasher);
        let packed = packed_state.rows_for_packed_chunks(&hasher, chunks)?;

        let mut expected_state = PleSequenceState::new(&hasher);
        let mut expected = Vec::new();
        for (sequence_id, tokens) in chunks {
            expected.extend(expected_state.rows_for_chunk(&hasher, *sequence_id, tokens)?);
        }
        assert_eq!(packed, expected);
        assert_eq!(packed_state.snapshot(7), expected_state.snapshot(7));
        assert_eq!(packed_state.snapshot(11), expected_state.snapshot(11));
        Ok(())
    }

    #[test]
    fn failed_packed_hashing_rolls_back_every_sequence() -> Result<()> {
        let hasher = PleHasher {
            ngram_size: 2,
            heads_per_ngram: 1,
            eos_token_id: 0,
            multipliers: vec![1, 0],
            head_offsets: vec![u64::from(u32::MAX) - 1],
            head_vocab_sizes: vec![3],
        };
        let mut state = PleSequenceState::new(&hasher);
        let before_one = state.snapshot(1);
        let before_two = state.snapshot(2);

        let error = state
            .rows_for_packed_chunks(&hasher, &[(1, &[0]), (2, &[2])])
            .unwrap_err();
        assert!(error.to_string().contains("does not fit u32"));
        assert_eq!(state.snapshot(1), before_one);
        assert_eq!(state.snapshot(2), before_two);
        Ok(())
    }

    #[test]
    fn snapshot_restore_rolls_back_speculative_history() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleSequenceState::new(&hasher);
        state.rows_for_chunk(&hasher, 1, &[3, 5])?;
        let snapshot = state.snapshot(1);

        state.rows_for_chunk(&hasher, 1, &[7, 11])?;
        state.restore(1, &snapshot)?;
        let restored = state.rows_for_chunk(&hasher, 1, &[13])?;

        let mut expected_state = PleSequenceState::new(&hasher);
        expected_state.rows_for_chunk(&hasher, 1, &[3, 5])?;
        let expected = expected_state.rows_for_chunk(&hasher, 1, &[13])?;
        assert_eq!(restored, expected);
        Ok(())
    }

    #[test]
    fn release_removes_only_the_requested_sequence() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleSequenceState::new(&hasher);
        state.rows_for_chunk(&hasher, 1, &[3, 5])?;
        state.rows_for_chunk(&hasher, 2, &[7, 11])?;
        let sequence_two_snapshot = state.snapshot(2);

        assert!(state.release(1));
        assert!(!state.release(1));
        assert_eq!(state.snapshot(2), sequence_two_snapshot);
        assert_eq!(
            state.snapshot(1).history,
            vec![config.eos_token_id; config.ngram_size - 1]
        );
        Ok(())
    }

    #[test]
    fn incompatible_snapshot_is_rejected_without_mutating_state() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleSequenceState::new(&hasher);
        state.rows_for_chunk(&hasher, 1, &[3, 5])?;
        let before = state.snapshot(1);
        let mut incompatible = before.clone();
        incompatible.n_prev += 1;

        let error = state.restore(1, &incompatible).unwrap_err();
        assert!(error.to_string().contains("snapshot is incompatible"));
        assert_eq!(state.snapshot(1), before);
        Ok(())
    }

    #[test]
    fn dilated_depthwise_convolution_matches_reference() -> Result<()> {
        let mut state = PleConvState::new(3, 2, 2)?;
        let input = [
            1.0, 10.0, // token 0
            2.0, 20.0, // token 1
            3.0, 30.0, // token 2
            4.0, 40.0, // token 3
            5.0, 50.0, // token 4
        ];
        let kernel = [1.0, 0.1, 10.0, 0.2, 100.0, 0.3];
        let output = state.forward_chunk(7, &input, &kernel)?;

        let expected_pre_silu: [f32; 10] = [
            100.0, 3.0, 200.0, 6.0, 310.0, 11.0, 420.0, 16.0, 531.0, 22.0,
        ];
        for (actual, expected) in output.iter().zip(expected_pre_silu) {
            let expected = expected / (1.0 + (-expected).exp());
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn convolution_chunking_matches_one_shot() -> Result<()> {
        let kernel = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125];
        let input = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let mut one_shot = PleConvState::new(3, 2, 2)?;
        let expected = one_shot.forward_chunk(1, &input, &kernel)?;

        let mut chunked = PleConvState::new(3, 2, 2)?;
        let mut actual = chunked.forward_chunk(1, &input[..2], &kernel)?;
        actual.extend(chunked.forward_chunk(1, &input[2..6], &kernel)?);
        actual.extend(chunked.forward_chunk(1, &input[6..], &kernel)?);
        assert_eq!(actual, expected);
        assert_eq!(chunked.snapshot(1), one_shot.snapshot(1));
        Ok(())
    }

    #[test]
    fn convolution_snapshot_restores_and_release_is_isolated() -> Result<()> {
        let kernel = [1.0, 1.0];
        let mut state = PleConvState::new(2, 2, 1)?;
        state.forward_chunk(1, &[2.0, 3.0], &kernel)?;
        state.forward_chunk(2, &[5.0], &kernel)?;
        let sequence_one = state.snapshot(1);
        let sequence_two = state.snapshot(2);

        state.forward_chunk(1, &[7.0], &kernel)?;
        state.restore(1, &sequence_one)?;
        assert_eq!(state.snapshot(1), sequence_one);
        assert_eq!(state.snapshot(2), sequence_two);
        assert!(state.release(1));
        assert!(!state.snapshot(1).initialized);
        assert_eq!(state.snapshot(2), sequence_two);
        Ok(())
    }

    #[test]
    fn tensor_convolution_matches_f32_reference_and_chunking() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let kernel_values = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125];
        let input_values = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        let kernel = Tensor::from_vec(kernel_values.to_vec(), (3, 2), &device)?;
        let input = Tensor::from_vec(input_values.to_vec(), (4, 2), &device)?;

        let mut reference = PleConvState::new(3, 2, 2)?;
        let expected = reference.forward_chunk(1, &input_values, &kernel_values)?;
        let mut one_shot = PleConvState::new(3, 2, 2)?;
        let actual = one_shot
            .forward_tensor_chunk(1, &input, &kernel)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        assert_eq!(one_shot.snapshot(1), reference.snapshot(1));

        let mut chunked = PleConvState::new(3, 2, 2)?;
        let first = chunked.forward_tensor_chunk(1, &input.narrow(0, 0, 1)?, &kernel)?;
        let middle = chunked.forward_tensor_chunk(1, &input.narrow(0, 1, 2)?, &kernel)?;
        let last = chunked.forward_tensor_chunk(1, &input.narrow(0, 3, 1)?, &kernel)?;
        let chunked_output = Tensor::cat(&[first, middle, last], 0)?;
        assert_eq!(chunked_output.flatten_all()?.to_vec1::<f32>()?, expected);
        assert_eq!(chunked.snapshot(1), one_shot.snapshot(1));
        Ok(())
    }

    #[test]
    fn tensor_convolution_validates_shape_without_mutating_state() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mut state = PleConvState::new(2, 1, 2)?;
        let kernel = Tensor::ones((2, 2), candle_core::DType::F32, &device)?;
        let valid = Tensor::from_vec(vec![2.0f32, 3.0], (1, 2), &device)?;
        state.forward_tensor_chunk(1, &valid, &kernel)?;
        let before = state.snapshot(1);

        let invalid = Tensor::ones((1, 3), candle_core::DType::F32, &device)?;
        let error = state
            .forward_tensor_chunk(1, &invalid, &kernel)
            .unwrap_err();
        assert!(error.to_string().contains("expected 2 input channels"));
        assert_eq!(state.snapshot(1), before);
        Ok(())
    }

    #[test]
    fn packed_convolution_matches_independent_sequence_processing() -> Result<()> {
        let kernel = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125];
        let chunks: &[(usize, &[f32])] = &[
            (7, &[1.0, -1.0, 2.0, -2.0]),
            (11, &[3.0, -3.0]),
            (7, &[4.0, -4.0]),
        ];

        let mut packed_state = PleConvState::new(3, 2, 2)?;
        let packed = packed_state.forward_packed_chunks(chunks, &kernel)?;

        let mut expected_state = PleConvState::new(3, 2, 2)?;
        let mut expected = Vec::new();
        for (sequence_id, input) in chunks {
            expected.extend(expected_state.forward_chunk(*sequence_id, input, &kernel)?);
        }
        assert_eq!(packed, expected);
        assert_eq!(packed_state.snapshot(7), expected_state.snapshot(7));
        assert_eq!(packed_state.snapshot(11), expected_state.snapshot(11));
        Ok(())
    }

    #[test]
    fn failed_packed_convolution_rolls_back_every_sequence() -> Result<()> {
        let kernel = [1.0, 1.0, 1.0, 1.0];
        let mut state = PleConvState::new(2, 1, 2)?;
        state.forward_chunk(1, &[2.0, 3.0], &kernel)?;
        let before_one = state.snapshot(1);
        let before_two = state.snapshot(2);

        let error = state
            .forward_packed_chunks(&[(1, &[5.0, 7.0]), (2, &[11.0])], &kernel)
            .unwrap_err();
        assert!(error.to_string().contains("not divisible by 2 channels"));
        assert_eq!(state.snapshot(1), before_one);
        assert_eq!(state.snapshot(2), before_two);
        Ok(())
    }

    #[test]
    fn packed_tensor_execution_matches_independent_chunks() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let hasher = PleHasher {
            ngram_size: 2,
            heads_per_ngram: 1,
            eos_token_id: 0,
            multipliers: vec![1, 3],
            head_offsets: vec![0],
            head_vocab_sizes: vec![6],
        };
        let table = Tensor::from_vec(
            vec![
                0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
            ],
            (6, 2),
            &device,
        )?;
        let embedding = PleEmbedding::new(projection(table)?, 1, 2)?;
        let layer = ple_layer()?;
        let first = Tensor::from_vec(
            vec![2.0f32, 1.0, 1.0, -2.0, -0.5, 1.5, 2.0, 0.25],
            (1, 2, 2, 2),
            &device,
        )?;
        let second = Tensor::from_vec(vec![1.0f32, 0.5, -1.0, 2.0], (1, 1, 2, 2), &device)?;
        let token_chunks: &[(usize, &[u32])] = &[(7, &[1, 2]), (11, &[3])];
        let hidden_chunks = [(7, &first), (11, &second)];

        let mut packed_state = PleState::new(&hasher, 2, 3, 4)?;
        let packed = packed_state.forward_packed_tensor_chunks(
            &hasher,
            &embedding,
            &layer,
            token_chunks,
            &hidden_chunks,
        )?;

        let mut independent_state = PleState::new(&hasher, 2, 3, 4)?;
        let one = independent_state.forward_packed_tensor_chunks(
            &hasher,
            &embedding,
            &layer,
            &token_chunks[..1],
            &hidden_chunks[..1],
        )?;
        let two = independent_state.forward_packed_tensor_chunks(
            &hasher,
            &embedding,
            &layer,
            &token_chunks[1..],
            &hidden_chunks[1..],
        )?;
        let expected = Tensor::cat(&[one, two], 1)?;
        assert_eq!(
            packed.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(packed_state.snapshot(7), independent_state.snapshot(7));
        assert_eq!(packed_state.snapshot(11), independent_state.snapshot(11));
        Ok(())
    }

    #[test]
    fn packed_tensor_execution_rolls_back_hash_and_convolution_on_gather_failure() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let hasher = PleHasher {
            ngram_size: 2,
            heads_per_ngram: 1,
            eos_token_id: 0,
            multipliers: vec![1, 0],
            head_offsets: vec![0],
            head_vocab_sizes: vec![8],
        };
        let table = Tensor::zeros((4, 2), DType::F32, &device)?;
        let embedding = PleEmbedding::new(projection(table)?, 1, 2)?;
        let layer = ple_layer()?;
        let first = Tensor::ones((1, 1, 2, 2), DType::F32, &device)?;
        let second = Tensor::ones((1, 1, 2, 2), DType::F32, &device)?;
        let mut state = PleState::new(&hasher, 2, 3, 4)?;
        let before_one = state.snapshot(1);
        let before_two = state.snapshot(2);

        let error = state
            .forward_packed_tensor_chunks(
                &hasher,
                &embedding,
                &layer,
                &[(1, &[1]), (2, &[7])],
                &[(1, &first), (2, &second)],
            )
            .unwrap_err();
        assert!(error.to_string().contains("index-select"));
        assert_eq!(state.snapshot(1), before_one);
        assert_eq!(state.snapshot(2), before_two);
        Ok(())
    }

    #[test]
    fn combined_packed_state_matches_independent_components() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let token_chunks: &[(usize, &[u32])] = &[(7, &[3, 5]), (11, &[13]), (7, &[17])];
        let conv_chunks: &[(usize, &[f32])] = &[
            (7, &[1.0, -1.0, 2.0, -2.0]),
            (11, &[3.0, -3.0]),
            (7, &[4.0, -4.0]),
        ];
        let kernel = [0.5, -0.25, 1.0, 0.75, -0.5, 0.125];

        let mut combined = PleState::new(&hasher, 3, 2, 2)?;
        let actual = combined.forward_packed_chunks(&hasher, token_chunks, conv_chunks, &kernel)?;

        let mut sequences = PleSequenceState::new(&hasher);
        let expected_rows = sequences.rows_for_packed_chunks(&hasher, token_chunks)?;
        let mut convolution = PleConvState::new(3, 2, 2)?;
        let expected_conv = convolution.forward_packed_chunks(conv_chunks, &kernel)?;
        assert_eq!(actual, (expected_rows, expected_conv));
        assert_eq!(combined.snapshot(7).sequence, sequences.snapshot(7));
        assert_eq!(combined.snapshot(7).convolution, convolution.snapshot(7));
        assert_eq!(combined.snapshot(11).sequence, sequences.snapshot(11));
        assert_eq!(combined.snapshot(11).convolution, convolution.snapshot(11));
        Ok(())
    }

    #[test]
    fn combined_state_rolls_back_hashing_when_convolution_fails() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleState::new(&hasher, 2, 1, 2)?;
        let kernel = [1.0, 1.0, 1.0, 1.0];
        state.forward_packed_chunks(&hasher, &[(1, &[3])], &[(1, &[2.0, 3.0])], &kernel)?;
        let before_one = state.snapshot(1);
        let before_two = state.snapshot(2);

        let error = state
            .forward_packed_chunks(
                &hasher,
                &[(1, &[5]), (2, &[7])],
                &[(1, &[5.0, 7.0]), (2, &[11.0, 13.0])],
                &[1.0],
            )
            .unwrap_err();
        assert!(error.to_string().contains("expected 4 kernel values"));
        assert_eq!(state.snapshot(1), before_one);
        assert_eq!(state.snapshot(2), before_two);
        Ok(())
    }

    #[test]
    fn combined_state_validates_chunk_ownership_before_mutation() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleState::new(&hasher, 2, 1, 1)?;
        let before = state.snapshot(1);

        let error = state
            .forward_packed_chunks(&hasher, &[(1, &[3])], &[(2, &[5.0])], &[1.0, 1.0])
            .unwrap_err();
        assert!(error.to_string().contains("sequence mismatch"));
        assert_eq!(state.snapshot(1), before);
        assert!(!state.snapshot(2).sequence.initialized);
        assert!(!state.snapshot(2).convolution.initialized);
        Ok(())
    }

    #[test]
    fn combined_snapshot_restore_release_and_clear_cover_both_states() -> Result<()> {
        let config = fixture_config();
        let hasher = PleHasher::new(&config)?;
        let mut state = PleState::new(&hasher, 2, 1, 1)?;
        let kernel = [0.5, 1.0];
        state.forward_packed_chunks(
            &hasher,
            &[(1, &[3, 5]), (2, &[7])],
            &[(1, &[2.0, 3.0]), (2, &[5.0])],
            &kernel,
        )?;
        let sequence_one = state.snapshot(1);
        let sequence_two = state.snapshot(2);

        state.forward_packed_chunks(&hasher, &[(1, &[11])], &[(1, &[7.0])], &kernel)?;
        state.restore(1, &sequence_one)?;
        assert_eq!(state.snapshot(1), sequence_one);
        assert_eq!(state.snapshot(2), sequence_two);

        assert!(state.release(1));
        assert!(!state.snapshot(1).sequence.initialized);
        assert!(!state.snapshot(1).convolution.initialized);
        assert_eq!(state.snapshot(2), sequence_two);
        state.clear();
        assert!(!state.snapshot(2).sequence.initialized);
        assert!(!state.snapshot(2).convolution.initialized);
        Ok(())
    }

    #[test]
    fn predecessor_inventory_must_match_tokens() {
        let config = fixture_config();
        let hasher = PleHasher::new(&config).unwrap();
        let error = hasher.rows_for_tokens(&[1, 2], &[Some(1)]).unwrap_err();
        assert!(error.to_string().contains("expected 4 predecessor entries"));
    }
}
