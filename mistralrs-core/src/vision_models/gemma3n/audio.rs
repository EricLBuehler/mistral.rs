#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{bail, DType, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, Linear, ModuleT};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::{conv1d_no_bias, conv2d_no_bias, linear_no_bias, RmsNorm};

use super::config::Gemma3nAudioConfig;

/// Gemma3n Cumulative Group Normalization
/// Applies Group Normalization cumulatively over the time dimension.
pub struct Gemma3nCumulativeGroupNorm {
    num_channels: usize,
    feature_dims: Vec<usize>,
    eps: f64,
    _use_scale: bool,
    _use_bias: bool,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    reduction_axes: Vec<usize>,
    channel_axis: usize,  // Track which axis has channels
}

impl Gemma3nCumulativeGroupNorm {
    pub fn new(
        num_channels: usize,
        feature_dims: Vec<usize>,
        eps: f64,
        use_scale: bool,
        use_bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let weight = if use_scale {
            Some(vb.get(num_channels, "weight")?)
        } else {
            None
        };

        let bias = if use_bias {
            Some(vb.get(num_channels, "bias")?)
        } else {
            None
        };

        // For input format [B, T, *feature_dims, C], normalize over feature_dims and C
        // Batch is at 0, Time at 1, so we reduce over dimensions 2 onwards
        let reduction_axes: Vec<usize> = (2..2 + feature_dims.len() + 1).collect();
        let channel_axis = 2 + feature_dims.len();  // Channels are at last dimension

        Ok(Self {
            num_channels,
            feature_dims,
            eps,
            _use_scale: use_scale,
            _use_bias: use_bias,
            weight,
            bias,
            reduction_axes,
            channel_axis,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        // Input format: [B, T, *feature_dims, C] matching PyTorch
        // Verify input shape
        let expected_suffix = self.feature_dims.iter().cloned().chain(std::iter::once(self.num_channels)).collect::<Vec<_>>();
        let actual_suffix = x.dims()[2..].to_vec();
        if actual_suffix != expected_suffix {
            bail!(
                "Input tensor shape suffix {:?} does not match expected suffix {:?}",
                actual_suffix, expected_suffix
            );
        }

        let input_dtype = x.dtype();
        let calc_dtype = DType::F32;
        let x_calc = x.to_dtype(calc_dtype)?;

        // Prepare broadcastable mask
        let mask_calc = if let Some(mask) = mask {
            // Mask should be [B, T]
            if mask.dims() != &[x.dims()[0], x.dims()[1]] {
                bail!(
                    "Mask shape {:?} must match input Batch/Time dimensions {:?}",
                    mask.dims(), &[x.dims()[0], x.dims()[1]]
                );
            }
            // Expand mask from [B, T] to [B, T, 1, ..., 1] for broadcasting
            let mut mask_shape = mask.dims().to_vec();
            for _ in 0..expected_suffix.len() {
                mask_shape.push(1);
            }
            mask.reshape(mask_shape)?.to_dtype(calc_dtype)?
        } else {
            Tensor::ones_like(&x_calc)?
        };

        // Mask the input for sum calculation: only valid elements contribute
        let x_masked_for_sum = x_calc.broadcast_mul(&mask_calc)?;

        // Cumulative Statistics Calculation
        // 1. Sum of values over reduction axes at each time step
        let mut sum_values_at_t = x_masked_for_sum.clone();
        for &axis in self.reduction_axes.iter().rev() {
            sum_values_at_t = sum_values_at_t.sum_keepdim(axis)?;
        }

        // 2. Cumulative sum of values over time
        let cum_sum_values = sum_values_at_t.cumsum(1)?;

        // 3. Count of valid elements
        let mut elements_in_group_at_t = mask_calc.clone();
        for &axis in self.reduction_axes.iter().rev() {
            elements_in_group_at_t = elements_in_group_at_t.sum_keepdim(axis)?;
        }

        // 4. Cumulative count of valid elements over time
        let cum_count_elements = elements_in_group_at_t.cumsum(1)?;
        let safe_cum_count_elements = cum_count_elements.clamp(1.0, f64::INFINITY)?;

        // 5. Cumulative mean
        let cum_mean = cum_sum_values.div(&safe_cum_count_elements)?;

        // 6. Sum of squared differences from the cumulative mean
        let squared_diff_from_mean = (x_calc.broadcast_sub(&cum_mean))?.sqr()?;
        let mut sum_sq_diff_at_t = squared_diff_from_mean.broadcast_mul(&mask_calc)?;
        for &axis in self.reduction_axes.iter().rev() {
            sum_sq_diff_at_t = sum_sq_diff_at_t.sum_keepdim(axis)?;
        }

        // 7. Cumulative sum of squared differences over time
        let cum_sum_sq_diff = sum_sq_diff_at_t.cumsum(1)?;

        // 8. Cumulative variance
        let cum_variance = cum_sum_sq_diff.div(&safe_cum_count_elements)?;

        // Normalize the input using rsqrt for efficiency
        let normalized_x = x_calc
            .broadcast_sub(&cum_mean)?
            .broadcast_mul(&(cum_variance.broadcast_add(&Tensor::new(self.eps as f32, cum_variance.device())?.to_dtype(calc_dtype)?)?.recip()?.sqrt())?)?;

        // Apply affine transformation
        let mut result = normalized_x;
        if let Some(weight) = &self.weight {
            let scale = weight.to_dtype(calc_dtype)?;
            let mut scale_shape = vec![1; x.dims().len()];
            scale_shape[x.dims().len() - 1] = self.num_channels;  // Channels at last dimension
            let scale = scale.reshape(scale_shape)?;
            result = result.broadcast_mul(&scale)?;
        }

        if let Some(bias) = &self.bias {
            let bias = bias.to_dtype(calc_dtype)?;
            let mut bias_shape = vec![1; x.dims().len()];
            bias_shape[x.dims().len() - 1] = self.num_channels;  // Channels at last dimension
            let bias = bias.reshape(bias_shape)?;
            result = result.broadcast_add(&bias)?;
        }

        // Zero out outputs for time steps that were originally masked (where mask_calc is 0)
        let final_output = result.broadcast_mul(&mask_calc)?;

        final_output.to_dtype(input_dtype)
    }
}

/// Relative Position Embedding for Gemma3n Audio
pub struct Gemma3nAudioRelativePositionEmbedding {
    _config: Gemma3nAudioConfig,
    num_heads: usize,
    head_dim: usize,
    max_backward: usize,
    max_forward: usize,
    pos_proj: Linear,
    inv_timescales: Tensor,
}

impl Gemma3nAudioRelativePositionEmbedding {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = config.conf_num_attention_heads;
        let channels = config.hidden_size;
        let head_dim = channels / num_heads;
        let max_backward = if config.conf_attention_context_left > 0 {
            config.conf_attention_context_left - 1
        } else {
            0
        };
        let max_forward = config.conf_attention_context_right;

        let pos_proj = linear_no_bias(channels, num_heads * head_dim, vb.pp("pos_proj"))?;

        // Create inverse timescales
        let min_timescale = 1.0_f32;
        let max_timescale = 1.0e4_f32;
        let num_timescales = channels / 2;
        let log_timescale_increment =
            (max_timescale / min_timescale).ln() / (num_timescales as f32 - 1.0).max(1.0);

        let inv_timescales = Tensor::arange(0, num_timescales as i64, vb.device())?
            .to_dtype(DType::F32)?
            .affine(-log_timescale_increment as f64, 0.0)?
            .exp()?
            .affine(min_timescale as f64, 0.0)?;

        let inv_timescales = inv_timescales.unsqueeze(0)?.unsqueeze(0)?;

        Ok(Self {
            _config: config.clone(),
            num_heads,
            head_dim,
            max_backward,
            max_forward,
            pos_proj,
            inv_timescales,
        })
    }

    fn get_timing_signal_1d_pos(&self, position: &Tensor, dtype: DType) -> Result<Tensor> {
        // position: [1, F_span]
        let position = position.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;

        let scaled_time = position.broadcast_mul(&self.inv_timescales)?;
        let sin_emb = scaled_time.sin()?;
        let cos_emb = scaled_time.cos()?;

        let timing_signal = Tensor::cat(&[sin_emb, cos_emb], D::Minus1)?;
        timing_signal.to_dtype(dtype)
    }

    fn relative_shift(
        &self,
        term_bd_before_shift: &Tensor,
        batch_size: usize,
        num_heads: usize,
        num_query_blocks: usize,
        query_block_size: usize,
        key_context_size: usize,
        max_span_plus_1: usize,
    ) -> Result<Tensor> {
        // term_bd_before_shift: [B, N, U, W, F]
        let pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1;

        // Pad the last dimension on the right
        let term_bd_padded = term_bd_before_shift.pad_with_zeros(
            D::Minus1,
            0,
            pad_amount_last_dim)?;

        // Reshape for slicing
        let term_bd_reshaped = term_bd_padded.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size * (key_context_size + 1),
        ))?;

        // Slice to effective [B, N, U, W * C]
        let term_bd_sliced = term_bd_reshaped.narrow(
            D::Minus1,
            0,
            query_block_size * key_context_size,
        )?;

        // Reshape back to [B, N, U, W, C]
        term_bd_sliced.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))
    }

    pub fn forward(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        // queries: [B, U, W, N, H]
        // keys:    [B, U, C, N, H]
        let (batch_size, num_query_blocks, query_block_size, num_heads, head_dim) = match queries.dims() {
            &[b, u, w, n, h] => (b, u, w, n, h),
            _ => bail!("Expected queries to have 5 dimensions"),
        };
        
        // Ensure keys have the same number of blocks as queries
        let keys = if keys.dim(1)? != num_query_blocks {
            if keys.dim(1)? > num_query_blocks {
                keys.narrow(1, 0, num_query_blocks)?
            } else {
                bail!("Keys have fewer blocks than queries: {} < {}", keys.dim(1)?, num_query_blocks)
            }
        } else {
            keys.clone()
        };

        let key_context_size = keys.dim(2)?;

        // Relative positions: [L, L-1, ..., -R]
        // Create position indices manually since we need negative step
        let mut pos_values = Vec::new();
        let mut pos = self.max_backward as i64;
        while pos >= -(self.max_forward as i64) {
            pos_values.push(pos);
            pos -= 1;
        }
        let pos_indices = Tensor::from_vec(pos_values, (1, self.max_backward + self.max_forward + 1), queries.device())?;

        let max_span_plus_1 = pos_indices.dim(1)?;

        let sin_emb_timing_signal = self.get_timing_signal_1d_pos(&pos_indices, queries.dtype())?;

        // Project sinusoidal embeddings
        let projected_sin_emb = self.pos_proj.forward(&sin_emb_timing_signal)?;
        let sin_emb = projected_sin_emb
            .reshape((1, max_span_plus_1, self.num_heads, self.head_dim))?
            .squeeze(0)?;

        // term_ac: Query-Key content interaction
        let queries_p = queries.transpose(1, 3)?.transpose(2, 3)?.contiguous()?; // [B, N, U, W, H]
        // For keys: [B, U, C, N, H] -> [B, N, U, H, C]
        // First swap U and N: [B, U, C, N, H] -> [B, N, C, U, H]
        // Then swap C and U: [B, N, C, U, H] -> [B, N, U, C, H]  
        // Finally swap C and H: [B, N, U, C, H] -> [B, N, U, H, C]
        let keys_p_t = keys
            .transpose(1, 3)?  // [B, U, C, N, H] -> [B, N, C, U, H]
            .transpose(2, 3)?  // [B, N, C, U, H] -> [B, N, U, C, H]
            .transpose(3, 4)?  // [B, N, U, C, H] -> [B, N, U, H, C]
            .contiguous()?;
        let term_ac = queries_p.matmul(&keys_p_t)?; // [B, N, U, W, C]

        // term_bd: Query-Position interaction
        let q_transposed = queries.transpose(1, 3)?.transpose(2, 3)?; // [B, N, U, W, H]
        let s_transposed = sin_emb.transpose(0, 2)?.transpose(0, 1)?; // [N, H, F]

        let q_reshaped = q_transposed.reshape((
            batch_size,
            num_heads,
            num_query_blocks * query_block_size,
            head_dim,
        ))?;

        // Broadcast s_transposed for batch dimension
        let s_broadcast = s_transposed
            .unsqueeze(0)?
            .broadcast_as((batch_size, num_heads, head_dim, max_span_plus_1))?
            .contiguous()?;

        let term_bd_unshifted_matmul = q_reshaped.contiguous()?.matmul(&s_broadcast)?;

        let term_bd_unshifted = term_bd_unshifted_matmul.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        ))?;

        // Apply relative shift
        let term_bd_shifted = self.relative_shift(
            &term_bd_unshifted,
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
            max_span_plus_1,
        )?;

        term_ac.broadcast_add(&term_bd_shifted)
    }
}

/// Gemma3n Audio Attention
pub struct Gemma3nAudioAttention {
    _config: Gemma3nAudioConfig,
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_future_horizon: usize,
    max_past_horizon: usize,
    attention_invalid_logits_value: f64,
    _attention_logits_soft_cap: f64,
    context_size: usize,
    relative_position_embedding: Gemma3nAudioRelativePositionEmbedding,
    per_dim_scale: Tensor,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    q_scale: f64,
    local_causal_valid_mask: Tensor,
    softcap: Tensor,
}

impl Gemma3nAudioAttention {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = config.conf_num_attention_heads;
        let hidden_size = config.hidden_size;
        let head_dim = hidden_size / num_heads;

        let chunk_size = config.conf_attention_chunk_size;
        let max_future_horizon = config.conf_attention_context_right;
        let max_past_horizon = if config.conf_attention_context_left > 0 {
            config.conf_attention_context_left - 1
        } else {
            0
        };
        let context_size = chunk_size + max_past_horizon + max_future_horizon;

        let relative_position_embedding =
            Gemma3nAudioRelativePositionEmbedding::new(config, vb.pp("relative_position_embedding"))?;
        // per_dim_scale is a learnable parameter, not zeros
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?;

        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("v_proj"))?;

        let q_scale = (head_dim as f64).powf(-0.5);
        // r_softplus_0 = 1.0 / softplus(0) = 1.0 / ln(1 + exp(0)) = 1.0 / ln(2)
        let r_softplus_0 = 1.0 / (1.0_f64 + 1.0_f64.exp()).ln();
        let q_scale = q_scale * r_softplus_0;

        // Create local causal mask
        // Create lower triangular mask
        let lower_causal_mask = {
            let mut mask_vec = vec![0u8; context_size * chunk_size];
            for i in 0..chunk_size {
                for j in 0..context_size {
                    if j <= i {
                        mask_vec[i * context_size + j] = 1;
                    }
                }
            }
            Tensor::from_vec(mask_vec, (chunk_size, context_size), vb.device())?
                .to_dtype(DType::U8)?
        };

        // Create upper triangular mask with offset
        let offset = -(max_past_horizon as i64 + max_future_horizon as i64);
        let upper_causal_mask = {
            let mut mask_vec = vec![0u8; chunk_size * context_size];
            for i in 0..chunk_size {
                for j in 0..context_size {
                    if i as i64 >= j as i64 + offset {
                        mask_vec[i * context_size + j] = 1;
                    }
                }
            }
            Tensor::from_vec(mask_vec, (chunk_size, context_size), vb.device())?
                .to_dtype(DType::U8)?
        };

        let local_causal_valid_mask = lower_causal_mask
            .broadcast_mul(&upper_causal_mask)?
            .to_dtype(DType::U8)?;

        let softcap = Tensor::new(config.conf_attention_logit_cap as f32, vb.device())?;

        Ok(Self {
            _config: config.clone(),
            num_heads,
            head_dim,
            chunk_size,
            max_future_horizon,
            max_past_horizon,
            attention_invalid_logits_value: config.conf_attention_invalid_logits_value,
            _attention_logits_soft_cap: config.conf_attention_logit_cap,
            context_size,
            relative_position_embedding,
            per_dim_scale,
            q_proj,
            k_proj,
            v_proj,
            q_scale,
            local_causal_valid_mask,
            softcap,
        })
    }

    fn pad_dim1(&self, x: &Tensor, left_pad: usize, right_pad: usize) -> Result<Tensor> {
        x.pad_with_zeros(1, left_pad, right_pad)
    }

    fn convert_to_block(&self, x: &Tensor, _padding_val: f64) -> Result<Tensor> {
        let shape = x.dims();
        let (b, t) = (shape[0], shape[1]);
        let num_blocks = (t + self.chunk_size - 1) / self.chunk_size;

        let padding_len = num_blocks * self.chunk_size - t;
        let x = if padding_len > 0 {
            self.pad_dim1(x, 0, padding_len)?
        } else {
            x.clone()
        };

        let mut new_shape = vec![b, num_blocks, self.chunk_size];
        new_shape.extend_from_slice(&shape[2..]);
        x.reshape(new_shape)
    }

    fn extract_block_context(&self, x: &Tensor) -> Result<Tensor> {
        let pad_left = self.max_past_horizon;
        let pad_right = self.max_future_horizon + self.chunk_size - 1;
        let x = self.pad_dim1(x, pad_left, pad_right)?;

        let frame_len = self.context_size;
        let frame_step = self.chunk_size;

        // Use unfold operation to extract sliding windows
        // x shape: (batch, time + padding, ...)
        let _batch_size = x.dim(0)?;
        let time_dim = x.dim(1)?;
        let num_windows = (time_dim - frame_len) / frame_step + 1;

        let mut windows = Vec::new();
        for i in 0..num_windows {
            let start_idx = i * frame_step;
            let _end_idx = start_idx + frame_len;
            let window = x.narrow(1, start_idx, frame_len)?;
            windows.push(window);
        }

        // Stack windows along dimension 1
        let result = Tensor::stack(&windows, 1)?;

        // If x has more than 2 dimensions, we need to transpose
        if x.dims().len() > 2 && result.dims().len() > 3 {
            result.transpose(1, 2)
        } else {
            Ok(result)
        }
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let query_states = self.q_proj.forward(x)?;
        let key_states = self.k_proj.forward(x)?;
        let value_states = self.v_proj.forward(x)?;

        let (b, t) = match x.dims() {
            &[b, t, _] => (b, t),
            _ => bail!("Expected input to have 3 dimensions"),
        };

        let query_states = query_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let key_states = key_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let value_states = value_states.reshape((b, t, self.num_heads, self.head_dim))?;

        // Apply per-dim scale using logaddexp(x, 0) = log(exp(x) + 1) = softplus(x)
        let per_dim_scale_sp = {
            let ones = Tensor::ones_like(&self.per_dim_scale)?.to_dtype(DType::F32)?;
            let exp_scale = self.per_dim_scale.to_dtype(DType::F32)?.exp()?;
            ones.broadcast_add(&exp_scale)?.log()?
        };

        let per_dim_scale_sp_broadcast = per_dim_scale_sp
            .reshape((1, 1, 1, self.head_dim))?
            .to_dtype(query_states.dtype())?;

        let query_states = query_states
            .affine(self.q_scale, 0.0)?
            .broadcast_mul(&per_dim_scale_sp_broadcast)?;

        let (batch_size, q_time) = (query_states.dim(0)?, query_states.dim(1)?);

        // Convert to blocks
        let query_blocks = self.convert_to_block(&query_states, 0.0)?;
        let mut key_blocks = self.extract_block_context(&key_states)?;
        let mut value_blocks = self.extract_block_context(&value_states)?;
        let num_query_blocks = query_blocks.dim(1)?;
        
        // Ensure key_blocks and value_blocks have the correct context size
        if key_blocks.dim(2)? != self.context_size {
            let current_context = key_blocks.dim(2)?;
            if current_context < self.context_size {
                // Pad with zeros
                let pad_amount = self.context_size - current_context;
                key_blocks = key_blocks.pad_with_zeros(2, 0, pad_amount)?;
                value_blocks = value_blocks.pad_with_zeros(2, 0, pad_amount)?;
            } else {
                // Truncate
                key_blocks = key_blocks.narrow(2, 0, self.context_size)?;
                value_blocks = value_blocks.narrow(2, 0, self.context_size)?;
            }
        }
        
        // Ensure the number of blocks matches between queries and keys/values
        let num_key_blocks = key_blocks.dim(1)?;
        if num_query_blocks != num_key_blocks {
            if num_query_blocks < num_key_blocks {
                // Truncate key/value blocks to match query blocks
                key_blocks = key_blocks.narrow(1, 0, num_query_blocks)?;
                value_blocks = value_blocks.narrow(1, 0, num_query_blocks)?;
            } else {
                // This case is problematic - we have more query blocks than key blocks
                // We'll pad the key/value blocks, though this might not be semantically correct
                let pad_blocks = num_query_blocks - num_key_blocks;
                let (batch_size, context_size, num_heads, head_dim) = 
                    match key_blocks.dims() {
                        &[b, _, c, n, h] => (b, c, n, h),
                        _ => bail!("Unexpected key_blocks dimensions")
                    };
                
                // Create zero padding with the same shape as one block
                let pad_shape = vec![batch_size, pad_blocks, context_size, num_heads, head_dim];
                let padding = Tensor::zeros(pad_shape.as_slice(), key_blocks.dtype(), key_blocks.device())?;
                
                // Concatenate padding to key and value blocks
                key_blocks = Tensor::cat(&[key_blocks, padding.clone()], 1)?;
                value_blocks = Tensor::cat(&[value_blocks, padding], 1)?;
            }
        }

        // Create validity mask
        let original_valid_mask = mask.eq(0.0)?.to_dtype(DType::U8)?; // True for valid

        // Extract blocks from validity mask
        let extracted_valid_mask_blocks = self.extract_block_context(&original_valid_mask)?;
        let extracted_valid_mask_blocks = if extracted_valid_mask_blocks.dims().len() == 3 {
            extracted_valid_mask_blocks.transpose(1, 2)?
        } else {
            extracted_valid_mask_blocks
        };

        // Reshape if needed
        let extracted_valid_mask_blocks = if extracted_valid_mask_blocks.dims().len() == 4
            && extracted_valid_mask_blocks.dim(0)? == batch_size
            && extracted_valid_mask_blocks.dim(1)? == num_query_blocks
            && extracted_valid_mask_blocks.dim(2)? * extracted_valid_mask_blocks.dim(3)?
                == self.context_size
        {
            extracted_valid_mask_blocks.reshape((batch_size, num_query_blocks, self.context_size))?
        } else if extracted_valid_mask_blocks.dims().len() == 3 {
            // Already in the correct shape [batch_size, num_query_blocks, context_size]
            extracted_valid_mask_blocks
        } else {
            // If the shape doesn't match expected, try to handle common cases
            match extracted_valid_mask_blocks.dims() {
                &[b, n, _c] if b == batch_size && n == num_query_blocks => extracted_valid_mask_blocks,
                &[b, _c, n] if b == batch_size && n == num_query_blocks => extracted_valid_mask_blocks.transpose(1, 2)?,
                _ => extracted_valid_mask_blocks
            }
        };

        // Ensure the extracted mask has the correct context size
        let extracted_valid_mask_blocks = if extracted_valid_mask_blocks.dim(D::Minus1)? != self.context_size {
            // If context size doesn't match, we need to adjust it
            let current_context_size = extracted_valid_mask_blocks.dim(D::Minus1)?;
            if current_context_size < self.context_size {
                // Pad with zeros (invalid positions)
                let pad_amount = self.context_size - current_context_size;
                extracted_valid_mask_blocks.pad_with_zeros(D::Minus1, 0, pad_amount)?
            } else {
                // Truncate to match expected context size
                extracted_valid_mask_blocks.narrow(D::Minus1, 0, self.context_size)?
            }
        } else {
            extracted_valid_mask_blocks
        };

        // Expand dimensions for broadcasting
        // The key insight is that we need to match the attention tensor layout
        // which appears to be [batch, heads, num_query_blocks, chunk_size, context_size]
        
        // extracted_valid_mask_blocks: [batch_size, num_query_blocks, context_size]
        // Expand to: [batch_size, 1, num_query_blocks, 1, context_size]
        let condition_from_input_validity = extracted_valid_mask_blocks
            .unsqueeze(1)?    // [batch_size, 1, num_query_blocks, context_size]
            .unsqueeze(3)?;   // [batch_size, 1, num_query_blocks, 1, context_size]

        // local_causal_valid_mask: [chunk_size, context_size]
        // Expand to: [1, 1, 1, chunk_size, context_size]
        let condition_from_causality = self.local_causal_valid_mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;

        // Combine conditions
        let final_condition_for_where = condition_from_input_validity
            .broadcast_mul(&condition_from_causality)?
            .to_dtype(DType::U8)?;

        // Compute attention logits
        // Note: At this point query_blocks and key_blocks should have compatible shapes
        // query_blocks: [B, U, W, N, H] where U is num_query_blocks, W is chunk_size
        // key_blocks: [B, U, C, N, H] where C is context_size
        let logits = self.relative_position_embedding.forward(&query_blocks, &key_blocks)?;

        // Apply attention logit softcap
        let softcap_value = self.softcap.to_scalar::<f32>()? as f64;
        let logits = logits
            .affine(1.0 / softcap_value, 0.0)?
            .tanh()?
            .affine(softcap_value, 0.0)?;

        // Apply mask
        // Ensure mask has the same shape as logits by handling dimension mismatches
        let final_condition_for_where = if final_condition_for_where.dims() != logits.dims() {
            // The mask might have shape [1, 1, U_mask, W, C] while logits have [B, N, U, W, C]
            let logits_shape = logits.dims();
            let mask_shape = final_condition_for_where.dims().to_vec();
            
            let mut mask = final_condition_for_where;
            
            // First handle non-broadcast dimension mismatches (like U_mask != U)
            for (i, (&logit_dim, &mask_dim)) in logits_shape.iter().zip(mask_shape.iter()).enumerate() {
                if mask_dim != logit_dim && mask_dim != 1 {
                    // Handle dimension mismatch that isn't a broadcast case
                    if mask_dim > logit_dim {
                        // Truncate mask to match logits
                        mask = mask.narrow(i, 0, logit_dim)?;
                    } else {
                        bail!("Mask dimension {} has size {} which is smaller than logits size {} and cannot be broadcast", 
                              i, mask_dim, logit_dim);
                    }
                }
            }
            
            // Now handle broadcasting for dimensions that are 1
            if mask.dims() != logits.dims() {
                // Check if we can broadcast
                let can_broadcast = mask.dims().iter().zip(logits_shape.iter())
                    .all(|(&m, &l)| m == l || m == 1);
                
                if can_broadcast {
                    mask = mask.broadcast_as(logits_shape)?;
                } else {
                    bail!("Cannot broadcast mask shape {:?} to logits shape {:?}", 
                          mask.dims(), logits_shape);
                }
            }
            
            mask
        } else {
            final_condition_for_where
        };
        
        let invalid_value = Tensor::new(self.attention_invalid_logits_value as f32, logits.device())?
            .broadcast_as(logits.dims())?;
        let logits = final_condition_for_where.where_cond(
            &logits,
            &invalid_value,
        )?;

        let probabilities = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?
            .to_dtype(value_blocks.dtype())?;

        // Compute context vectors
        let (b_dim, n_dim, u_dim, w_dim, c_dim) = match probabilities.dims() {
            &[b, n, u, w, c] => (b, n, u, w, c),
            _ => bail!("Expected probabilities to have 5 dimensions"),
        };
        let h_dim = value_blocks.dim(D::Minus1)?;

        let prob_bun = probabilities
            .transpose(1, 2)?
            .transpose(0, 1)?
            .reshape((b_dim * u_dim * n_dim, w_dim, c_dim))?
            .contiguous()?;

        let v_bun = value_blocks
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((b_dim * u_dim * n_dim, c_dim, h_dim))?
            .contiguous()?;

        let result_bmm = prob_bun.matmul(&v_bun)?;
        let context_vectors = result_bmm
            .reshape((b_dim, u_dim, n_dim, w_dim, h_dim))?
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((batch_size, num_query_blocks * self.chunk_size, self.num_heads, self.head_dim))?;

        // Trim to original time dimension
        let context_vectors = context_vectors.narrow(1, 0, q_time)?;

        Ok(context_vectors)
    }
}

/// SSCP Convolution Block
pub struct Gemma3nAudioSSCPConvBlock {
    conv: Conv2d,
    norm: Gemma3nCumulativeGroupNorm,
    manual_padding: (usize, usize, usize, usize),
}

impl Gemma3nAudioSSCPConvBlock {
    pub fn new(
        idx: usize,
        input_freq_dim: usize,
        config: &Gemma3nAudioConfig,
        manual_padding: (usize, usize, usize, usize),
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channels = if idx == 0 {
            1
        } else {
            config.sscp_conv_channel_size[idx - 1]
        };
        let out_channels = config.sscp_conv_channel_size[idx];
        let kernel_h = config.sscp_conv_kernel_size[idx][0];
        let kernel_w = config.sscp_conv_kernel_size[idx][1];
        let stride_h = config.sscp_conv_stride_size[idx][0];
        let stride_w = config.sscp_conv_stride_size[idx][1];

        let conv = conv2d_no_bias(
            in_channels,
            out_channels,
kernel_h.min(kernel_w),
            Conv2dConfig {
                stride: stride_h.min(stride_w),
                padding: 0,
                dilation: 1,
                groups: 1,
            },
            vb.pp("conv"),
        )?;

        // Calculate output frequency dimension after convolution
        let f_in_padded = input_freq_dim + manual_padding.0 + manual_padding.1;
        let f_out_conv = (f_in_padded - kernel_w) / stride_w + 1;

        let norm = Gemma3nCumulativeGroupNorm::new(
            out_channels,
            vec![f_out_conv],
            config.sscp_conv_eps,
            true,
            false,
            vb.pp("norm"),
        )?;

        Ok(Self {
            conv,
            norm,
            manual_padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply manual padding
        let x_padded = x.pad_with_zeros(
            D::Minus1,
            self.manual_padding.0,
            self.manual_padding.1)?
        .pad_with_zeros(
            D::Minus2,
            self.manual_padding.2,
            self.manual_padding.3)?;

        // Apply Conv2d - expecting NCHW format
        let x_conv = self.conv.forward_t(&x_padded, false)?;

        // Reshape for normalization: [B, C_out, T_out, F_out] -> [B, T_out, F_out, C_out]
        let x_for_norm = x_conv.permute((0, 2, 3, 1))?;
        
        // Apply normalization
        let x_normed = self.norm.forward(&x_for_norm, None)?;

        // Reshape back to [B, C_out, T_out, F_out] and apply ReLU
        x_normed.permute((0, 3, 1, 2))?.relu()
    }
}

/// Sub-sample Convolution Projection
pub struct Gemma3nAudioSubSampleConvProjection {
    conv_0: Gemma3nAudioSSCPConvBlock,
    conv_1: Gemma3nAudioSSCPConvBlock,
    input_proj_linear: Linear,
    _input_proj_in_features: usize,
}

impl Gemma3nAudioSubSampleConvProjection {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut current_f_for_block_input = config.input_feat_size;
        let mut calculated_block_padding = Vec::new();
        let mut calculated_f_out_dims = Vec::new();

        for i in 0..2 {
            let kernel_h = config.sscp_conv_kernel_size[i][0];
            let kernel_w = config.sscp_conv_kernel_size[i][1];
            let _stride_h = config.sscp_conv_stride_size[i][0];
            let stride_w = config.sscp_conv_stride_size[i][1];

            // Padding for Time (reverse causal)
            let pad_t_top = 0;
            let pad_t_bottom = kernel_h - 1;

            // Frequency padding
            let pad_f_left = 1;
            let pad_f_right = 1;

            let manual_padding_tuple = (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom);
            calculated_block_padding.push(manual_padding_tuple);

            // Calculate output frequency dimension
            let f_in_padded = current_f_for_block_input + pad_f_left + pad_f_right;
            let f_out_after_conv = (f_in_padded - kernel_w) / stride_w + 1;
            calculated_f_out_dims.push(f_out_after_conv);
            current_f_for_block_input = f_out_after_conv;
        }

        let conv_0 = Gemma3nAudioSSCPConvBlock::new(
            0,
            config.input_feat_size,
            config,
            calculated_block_padding[0],
            vb.pp("conv_0"),
        )?;

        let conv_1 = Gemma3nAudioSSCPConvBlock::new(
            1,
            calculated_f_out_dims[0],
            config,
            calculated_block_padding[1],
            vb.pp("conv_1"),
        )?;

        let final_c_out = config.sscp_conv_channel_size[1];
        let final_f_out = calculated_f_out_dims[1];
        let input_proj_in_features = final_c_out * final_f_out;

        let input_proj_linear = linear_no_bias(
            input_proj_in_features,
            config.hidden_size,
            vb.pp("input_proj_linear"),
        )?;

        Ok(Self {
            conv_0,
            conv_1,
            input_proj_linear,
            _input_proj_in_features: input_proj_in_features,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, T, F_in]
        // Reshape to [B, 1, T, F_in]
        let x_reshaped = x.unsqueeze(1)?;

        let x = self.conv_0.forward(&x_reshaped)?;
        let x = self.conv_1.forward(&x)?;

        // x: [B, C_out, T_out, F_out]
        let (b, c_out, t_out, f_out) = match x.dims() {
            &[b, c, t, f] => (b, c, t, f),
            _ => bail!("Expected conv output to have 4 dimensions"),
        };

        // Permute to [B, T_out, F_out, C_out] then flatten F_out and C_out
        let x_transposed = x.transpose(1, 2)?.transpose(2, 3)?;
        let output_flattened = x_transposed.reshape((b, t_out, f_out * c_out))?;

        self.input_proj_linear.forward(&output_flattened)
    }
}

/// Conformer Attention Module
pub struct Gemma3nAudioConformerAttention {
    gradient_clipping: Tensor,
    pre_attn_norm: RmsNorm,
    attn: Gemma3nAudioAttention,
    post: Linear,
    post_norm: RmsNorm,
}

impl Gemma3nAudioConformerAttention {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let gradient_clipping = Tensor::new(config.gradient_clipping as f32, vb.device())?;
        let pre_attn_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("pre_attn_norm"))?;
        let attn = Gemma3nAudioAttention::new(config, vb.pp("attn"))?;
        let post = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("post"))?;
        let post_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("post_norm"))?;

        Ok(Self {
            gradient_clipping,
            pre_attn_norm,
            attn,
            post,
            post_norm,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let audio_encodings_input_to_attn = x;
        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let x = x.clamp(-clip_value, clip_value)?;
        let audio_encodings_norm = self.pre_attn_norm.forward(&x)?;

        // attn output: [B, T, NumHeads, HeadDim]
        let audio_encodings_attn_out = self.attn.forward(&audio_encodings_norm, mask)?;

        // Reshape to [B, T, NumHeads * HeadDim]
        let (b, t, num_heads, head_dim) = match audio_encodings_attn_out.dims() {
            &[b, t, n, h] => (b, t, n, h),
            _ => bail!("Expected attention output to have 4 dimensions"),
        };
        let audio_encodings_reshaped = audio_encodings_attn_out.reshape((b, t, num_heads * head_dim))?;

        let x = self.post.forward(&audio_encodings_reshaped)?;
        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let x = x.clamp(-clip_value, clip_value)?;
        
        audio_encodings_input_to_attn.broadcast_add(&self.post_norm.forward(&x)?)
    }
}

/// Conformer Feed-Forward Module
pub struct Gemma3nAudioConformerFeedForward {
    gradient_clipping: Tensor,
    pre_layer_norm: RmsNorm,
    ffw_layer_1: Linear,
    ffw_layer_2: Linear,
    post_layer_norm: RmsNorm,
    post_layer_scale: f64,
}

impl Gemma3nAudioConformerFeedForward {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let gradient_clipping = Tensor::new(config.gradient_clipping as f32, vb.device())?;
        let pre_layer_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("pre_layer_norm"))?;
        let ffw_layer_1 = linear_no_bias(config.hidden_size, config.hidden_size * 4, vb.pp("ffw_layer_1"))?;
        let ffw_layer_2 = linear_no_bias(config.hidden_size * 4, config.hidden_size, vb.pp("ffw_layer_2"))?;
        let post_layer_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("post_layer_norm"))?;
        let post_layer_scale = config.conf_residual_weight;

        Ok(Self {
            gradient_clipping,
            pre_layer_norm,
            ffw_layer_1,
            ffw_layer_2,
            post_layer_norm,
            post_layer_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let x = x.clamp(-clip_value, clip_value)?;
        let x = self.pre_layer_norm.forward(&x)?;
        let x = self.ffw_layer_1.forward(&x)?;
        let x = candle_nn::ops::silu(&x)?;
        let x = self.ffw_layer_2.forward(&x)?;
        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let x = x.clamp(-clip_value, clip_value)?;
        let x = self.post_layer_norm.forward(&x)?;
        
        let scale_tensor = Tensor::new(self.post_layer_scale as f32, x.device())?.to_dtype(x.dtype())?;
        residual.broadcast_add(&x.broadcast_mul(&scale_tensor)?)
    }
}

/// Lightweight 1D Convolution Module
pub struct Gemma3nAudioConformerLightConv1d {
    pre_layer_norm: RmsNorm,
    linear_start: Linear,
    depthwise_conv1d: Conv1d,
    gradient_clipping: Tensor,
    conv_norm: RmsNorm,
    linear_end: Linear,
    causal_padding: usize,
}

impl Gemma3nAudioConformerLightConv1d {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let pre_layer_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("pre_layer_norm"))?;
        let linear_start = linear_no_bias(config.hidden_size, config.hidden_size * 2, vb.pp("linear_start"))?;
        
        let depthwise_conv1d = conv1d_no_bias(
            config.hidden_size,
            config.hidden_size,
            config.conf_conv_kernel_size,
            Conv1dConfig {
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: config.hidden_size,
            },
            vb.pp("depthwise_conv1d"),
        )?;

        let gradient_clipping = Tensor::new(config.gradient_clipping as f32, vb.device())?;
        let conv_norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("conv_norm"))?;
        let linear_end = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("linear_end"))?;
        let causal_padding = config.conf_conv_kernel_size - 1;

        Ok(Self {
            pre_layer_norm,
            linear_start,
            depthwise_conv1d,
            gradient_clipping,
            conv_norm,
            linear_end,
            causal_padding,
        })
    }

    pub fn forward(&self, audio_encodings: &Tensor) -> Result<Tensor> {
        let audio_encodings_residual = audio_encodings;

        let audio_encodings = self.pre_layer_norm.forward(audio_encodings)?;
        let audio_encodings = self.linear_start.forward(&audio_encodings)?;
        // Implement GLU manually: split tensor in half and apply gating
        let chunks = audio_encodings.chunk(2, D::Minus1)?;
        let audio_encodings = chunks[0].broadcast_mul(&candle_nn::ops::sigmoid(&chunks[1])?)?;

        // Permute for Conv1d: [B, T, D] -> [B, D, T]
        let audio_encodings_transposed = audio_encodings.transpose(D::Minus1, D::Minus2)?;

        // Apply manual causal padding
        let audio_encodings_padded = audio_encodings_transposed.pad_with_zeros(
            D::Minus1,
            self.causal_padding,
            0)?;

        // Conv1d expects NCW format, apply directly
        let audio_encodings_conv = self.depthwise_conv1d.forward(&audio_encodings_padded)?;
        
        // Permute back: [B, D, T] -> [B, T, D]
        let audio_encodings = audio_encodings_conv.transpose(D::Minus2, D::Minus1)?;

        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let audio_encodings = audio_encodings.clamp(-clip_value, clip_value)?;
        let audio_encodings = self.conv_norm.forward(&audio_encodings)?;
        let audio_encodings = candle_nn::ops::silu(&audio_encodings)?;
        let audio_encodings = self.linear_end.forward(&audio_encodings)?;

        audio_encodings_residual.broadcast_add(&audio_encodings)
    }
}

/// Conformer Block
pub struct Gemma3nAudioConformerBlock {
    ffw_layer_start: Gemma3nAudioConformerFeedForward,
    attention: Gemma3nAudioConformerAttention,
    lconv1d: Gemma3nAudioConformerLightConv1d,
    ffw_layer_end: Gemma3nAudioConformerFeedForward,
    gradient_clipping: Tensor,
    norm: RmsNorm,
}

impl Gemma3nAudioConformerBlock {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ffw_layer_start = Gemma3nAudioConformerFeedForward::new(config, vb.pp("ffw_layer_start"))?;
        let attention = Gemma3nAudioConformerAttention::new(config, vb.pp("attention"))?;
        let lconv1d = Gemma3nAudioConformerLightConv1d::new(config, vb.pp("lconv1d"))?;
        let ffw_layer_end = Gemma3nAudioConformerFeedForward::new(config, vb.pp("ffw_layer_end"))?;
        let gradient_clipping = Tensor::new(config.gradient_clipping as f32, vb.device())?;
        let norm = RmsNorm::new_gemma_3n(config.hidden_size, config.rms_norm_eps, true, vb.pp("norm"))?;

        Ok(Self {
            ffw_layer_start,
            attention,
            lconv1d,
            ffw_layer_end,
            gradient_clipping,
            norm,
        })
    }

    pub fn forward(&self, audio_encodings: &Tensor, audio_mel_mask: &Tensor) -> Result<Tensor> {
        let audio_encodings = self.ffw_layer_start.forward(audio_encodings)?;
        let audio_encodings = self.attention.forward(&audio_encodings, audio_mel_mask)?;

        // Apply mask for lconv1d
        let validity_mask_for_lconv = audio_mel_mask.eq(0.0)?; // True for valid
        let audio_encodings_for_lconv_input = audio_encodings.broadcast_mul(
            &validity_mask_for_lconv
                .unsqueeze(D::Minus1)?
                .to_dtype(audio_encodings.dtype())?,
        )?;

        let audio_encodings = self.lconv1d.forward(&audio_encodings_for_lconv_input)?;
        let audio_encodings = self.ffw_layer_end.forward(&audio_encodings)?;
        let clip_value = self.gradient_clipping.to_scalar::<f32>()? as f64;
        let audio_encodings = audio_encodings.clamp(-clip_value, clip_value)?;

        self.norm.forward(&audio_encodings)
    }
}

/// Main Audio Model
pub struct AudioModel {
    _config: Gemma3nAudioConfig,
    subsample_conv_projection: Gemma3nAudioSubSampleConvProjection,
    conformer: Vec<Gemma3nAudioConformerBlock>,
}

impl AudioModel {
    pub fn new(config: &Gemma3nAudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let subsample_conv_projection = Gemma3nAudioSubSampleConvProjection::new(config, vb.pp("subsample_conv_projection"))?;
        
        let mut conformer = Vec::with_capacity(config.conf_num_hidden_layers);
        for i in 0..config.conf_num_hidden_layers {
            conformer.push(Gemma3nAudioConformerBlock::new(config, vb.pp(&format!("conformer.{}", i)))?);
        }

        Ok(Self {
            _config: config.clone(),
            subsample_conv_projection,
            conformer,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        // audio_mel: [B, T, F]
        let mut audio_encodings = self.subsample_conv_projection.forward(audio_mel)?;

        // Subsample the mask to match audio_encodings time dimension
        let t_sub = audio_encodings.dim(1)?;

        let mut time_stride_product = 1;
        for i in 0..self._config.sscp_conv_stride_size.len() {
            time_stride_product *= self._config.sscp_conv_stride_size[i][0];
        }

        // Create indices for gathering from the original mask
        // Use f32 for the affine operation, then convert to i64 for indexing
        let indices = Tensor::arange(0f32, t_sub as f32, audio_mel_mask.device())?
            .affine(time_stride_product as f64, 0.0)?
            .to_dtype(DType::I64)?;
        let max_idx = audio_mel_mask.dim(1)? as i64 - 1;
        let indices = indices.clamp(0i64, max_idx)?;

        // Expand indices for batch compatibility
        let indices = if audio_mel_mask.dims().len() > 1 && indices.dims().len() == 1 {
            indices
                .unsqueeze(0)?
                .broadcast_as((audio_mel_mask.dim(0)?, indices.dim(0)?))?
        } else if audio_mel_mask.dims().len() == indices.dims().len()
            && audio_mel_mask.dim(0)? == 1
            && indices.dim(0)? != 1
            && t_sub == indices.dim(0)?
        {
            indices.unsqueeze(0)?
        } else {
            indices
        };

        // Use index_select instead of gather for Metal compatibility
        let mut current_mask = if indices.dims().len() == 1 {
            // 1D indices case
            audio_mel_mask.index_select(&indices, 1)?
        } else {
            // 2D indices case - need to handle batch dimension
            let batch_size = audio_mel_mask.dim(0)?;
            let mut masks = Vec::new();
            for b in 0..batch_size {
                let batch_mask = audio_mel_mask.get(b)?;
                let batch_indices = if indices.dim(0)? > 1 {
                    indices.get(b)?
                } else {
                    indices.squeeze(0)?
                };
                let selected = batch_mask.index_select(&batch_indices, 0)?;
                masks.push(selected.unsqueeze(0)?);
            }
            Tensor::cat(&masks, 0)?
        };

        // Ensure mask length matches feature length
        if current_mask.dim(1)? != t_sub {
            if current_mask.dim(1)? > t_sub {
                current_mask = current_mask.narrow(1, 0, t_sub)?;
            } else {
                let padding_needed = t_sub - current_mask.dim(1)?;
                current_mask = current_mask.pad_with_zeros(1, 0, padding_needed)?;
            }
        }

        // Apply conformer blocks
        for block in &self.conformer {
            audio_encodings = block.forward(&audio_encodings, &current_mask)?;
        }

        // Apply reduction factor if specified
        if self._config.conf_reduction_factor > 1 {
            let stride = self._config.conf_reduction_factor;
            let indices = Tensor::arange(0f32, audio_encodings.dim(1)? as f32, audio_encodings.device())?
                .affine(stride as f64, 0.0)?
                .to_dtype(DType::I64)?;
            let max_idx = audio_encodings.dim(1)? as i64 - 1;
            let indices = indices
                .narrow(0, 0, (audio_encodings.dim(1)? / stride).min(indices.dim(0)?))?
                .clamp(0, max_idx)?;
            
            audio_encodings = audio_encodings.index_select(&indices, 1)?;
            current_mask = current_mask.index_select(&indices, 1)?;
        }

        // Final masking - mask is 1 for invalid positions, 0 for valid
        // We want to zero out invalid positions
        let valid_mask = current_mask.eq(0.0)?; // True for valid positions
        let zeros = Tensor::zeros_like(&audio_encodings)?;
        let audio_encodings = valid_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(audio_encodings.shape())?
            .where_cond(&audio_encodings, &zeros)?;

        Ok((audio_encodings, current_mask))
    }
}