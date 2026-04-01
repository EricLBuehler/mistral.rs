//! Gemma4 audio encoder.
//!
//! Gemma4's audio stack is similar to Gemma3n's, but it is not drop-in
//! compatible. In particular, the SSCP projection uses semicausal padding and
//! layer norm, the conformer blocks use standard RMSNorm without Gemma's +1
//! scale offset, the attention uses an extra `per_dim_key_scale`, and several
//! projections use optional clipping buffers from the checkpoint.

use candle_core::{bail, DType, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, ModuleT};
use mistralrs_quant::{Convolution, QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::layers::{conv1d_no_bias, conv2d_no_bias, layer_norm, RmsNorm};

use super::config::Gemma4AudioConfig;

pub struct Gemma4AudioRelativePositionEmbedding {
    num_heads: usize,
    head_dim: usize,
    pub(crate) pos_proj: Arc<dyn QuantMethod>,
    inv_timescales: Tensor,
    pos_indices: Tensor,
}

impl Gemma4AudioRelativePositionEmbedding {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = cfg.conf_num_attention_heads;
        let channels = cfg.hidden_size;
        let head_dim = channels / num_heads;
        let max_backward = cfg.conf_attention_context_left.saturating_sub(1);
        let max_forward = cfg.conf_attention_context_right;
        let num_timescales = channels / 2;

        let pos_proj = mistralrs_quant::linear_no_bias(
            channels,
            num_heads * head_dim,
            &None,
            vb.pp("relative_k_proj"),
        )?;

        // HF's _init_weights overwrites the initial ones buffer with the
        // standard sinusoidal timescale formula (min=1, max=10000).
        let min_timescale = 1.0_f64;
        let max_timescale = 10_000.0_f64;
        let log_timescale_increment =
            (max_timescale / min_timescale).ln() / num_timescales.saturating_sub(1).max(1) as f64;
        let inv_timescales = Tensor::from_vec(
            (0..num_timescales)
                .map(|i| (min_timescale * (-log_timescale_increment * i as f64).exp()) as f32)
                .collect::<Vec<_>>(),
            (1, 1, num_timescales),
            vb.device(),
        )?;
        let pos_values = (-(max_forward as i64)..=max_backward as i64)
            .rev()
            .collect::<Vec<_>>();
        let pos_indices =
            Tensor::from_vec(pos_values, (1, max_backward + max_forward + 1), vb.device())?;

        Ok(Self {
            num_heads,
            head_dim,
            pos_proj,
            inv_timescales,
            pos_indices,
        })
    }

    fn get_timing_signal_1d_pos(&self, position: &Tensor, dtype: DType) -> Result<Tensor> {
        let position = position.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
        let inv_timescales = self.inv_timescales.to_device(position.device())?;
        let scaled_time = position.broadcast_mul(&inv_timescales)?;
        let sin_emb = scaled_time.sin()?;
        let cos_emb = scaled_time.cos()?;
        Tensor::cat(&[sin_emb, cos_emb], D::Minus1)?.to_dtype(dtype)
    }

    #[allow(clippy::too_many_arguments)]
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
        let pad_amount_last_dim = (key_context_size + 1) - max_span_plus_1;
        let term_bd_padded =
            term_bd_before_shift.pad_with_zeros(D::Minus1, 0, pad_amount_last_dim)?;
        let term_bd_reshaped = term_bd_padded.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size * (key_context_size + 1),
        ))?;
        let term_bd_sliced =
            term_bd_reshaped.narrow(D::Minus1, 0, query_block_size * key_context_size)?;
        term_bd_sliced.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))
    }

    fn forward(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        let (batch_size, num_query_blocks, query_block_size, num_heads, head_dim) =
            match queries.dims() {
                &[b, u, w, n, h] => (b, u, w, n, h),
                _ => bail!("Expected queries to have 5 dimensions"),
            };

        let keys = if keys.dim(1)? != num_query_blocks {
            if keys.dim(1)? > num_query_blocks {
                keys.narrow(1, 0, num_query_blocks)?
            } else {
                bail!(
                    "Keys have fewer blocks than queries: {} < {}",
                    keys.dim(1)?,
                    num_query_blocks
                )
            }
        } else {
            keys.clone()
        };

        let key_context_size = keys.dim(2)?;
        let input_device = queries.device();
        let pos_indices = self.pos_indices.to_device(input_device)?;
        let max_span_plus_1 = pos_indices.dim(1)?;

        let sin_emb_timing_signal = self.get_timing_signal_1d_pos(&pos_indices, queries.dtype())?;
        let projected_sin_emb = self.pos_proj.forward_autocast(&sin_emb_timing_signal)?;
        let sin_emb = projected_sin_emb
            .reshape((1, max_span_plus_1, self.num_heads, self.head_dim))?
            .squeeze(0)?
            .to_dtype(queries.dtype())?;

        let queries_p = queries.transpose(1, 3)?.transpose(2, 3)?.contiguous()?;
        let keys_p_t = keys
            .transpose(1, 3)?
            .transpose(2, 3)?
            .transpose(3, 4)?
            .contiguous()?;

        let queries_3d = queries_p.reshape((
            batch_size * num_heads * num_query_blocks,
            query_block_size,
            head_dim,
        ))?;
        let keys_3d = keys_p_t.reshape((
            batch_size * num_heads * num_query_blocks,
            head_dim,
            key_context_size,
        ))?;
        let term_ac_3d = queries_3d.matmul(&keys_3d)?;
        let term_ac = term_ac_3d.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            key_context_size,
        ))?;

        let q_transposed = queries.transpose(1, 3)?.transpose(2, 3)?;
        let s_transposed = sin_emb.transpose(0, 2)?.transpose(0, 1)?;
        let q_reshaped = q_transposed.reshape((
            batch_size * num_heads,
            num_query_blocks * query_block_size,
            head_dim,
        ))?;
        let s_broadcast = s_transposed
            .unsqueeze(0)?
            .broadcast_as((batch_size, num_heads, head_dim, max_span_plus_1))?
            .reshape((batch_size * num_heads, head_dim, max_span_plus_1))?
            .contiguous()?;
        let term_bd_unshifted_matmul = q_reshaped.contiguous()?.matmul(&s_broadcast)?;
        let term_bd_unshifted = term_bd_unshifted_matmul.reshape((
            batch_size,
            num_heads,
            num_query_blocks,
            query_block_size,
            max_span_plus_1,
        ))?;
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

struct ClippableLinear {
    inner: Arc<dyn QuantMethod>,
    input_min: Option<f64>,
    input_max: Option<f64>,
    output_min: Option<f64>,
    output_max: Option<f64>,
}

impl ClippableLinear {
    fn new_no_bias(
        cfg: &Gemma4AudioConfig,
        in_features: usize,
        out_features: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_linear_prefix = vb.pp("linear").contains_tensor("weight");
        let linear_vb = if has_linear_prefix {
            vb.pp("linear")
        } else {
            vb.clone()
        };
        let inner = mistralrs_quant::linear_no_bias(in_features, out_features, &None, linear_vb)?;
        let (input_min, input_max, output_min, output_max) = if cfg.use_clipped_linears {
            (
                Self::load_clip_scalar(&vb, "input_min"),
                Self::load_clip_scalar(&vb, "input_max"),
                Self::load_clip_scalar(&vb, "output_min"),
                Self::load_clip_scalar(&vb, "output_max"),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            inner,
            input_min,
            input_max,
            output_min,
            output_max,
        })
    }

    fn load_clip_scalar(vb: &ShardedVarBuilder, name: &str) -> Option<f64> {
        vb.get((), name)
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_scalar::<f32>().ok())
            .map(f64::from)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        if let (Some(lo), Some(hi)) = (self.input_min, self.input_max) {
            x = x.clamp(lo, hi)?;
        }
        let mut out = self.inner.forward_autocast(&x)?;
        if let (Some(lo), Some(hi)) = (self.output_min, self.output_max) {
            out = out.clamp(lo, hi)?;
        }
        Ok(out)
    }
}

fn subsample_mask(mask: &Tensor, stride: usize, target_len: usize) -> Result<Tensor> {
    let indices = Tensor::arange(0f32, target_len as f32, mask.device())?
        .affine(stride as f64, 0.0)?
        .to_dtype(DType::I64)?;
    let max_idx = mask.dim(1)? as i64 - 1;
    let indices = indices.clamp(0i64, max_idx)?;

    if indices.dims().len() == 1 {
        if mask.dim(0)? == 1 {
            return mask.index_select(&indices, 1);
        }

        let batch_size = mask.dim(0)?;
        let mut out = Vec::with_capacity(batch_size);
        for batch_idx in 0..batch_size {
            out.push(
                mask.get(batch_idx)?
                    .index_select(&indices, 0)?
                    .unsqueeze(0)?,
            );
        }
        Tensor::cat(&out, 0)
    } else {
        bail!("Expected 1D indices for mask subsampling")
    }
}

struct Gemma4AudioSSCPConvBlock {
    conv: Conv2d,
    norm: LayerNorm,
    manual_padding: (usize, usize, usize, usize),
    time_stride: usize,
}

impl Gemma4AudioSSCPConvBlock {
    fn new(
        cfg: &Gemma4AudioConfig,
        idx: usize,
        input_freq_dim: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        if cfg.sscp_conv_norm_type != "layer_norm" {
            bail!(
                "Unsupported Gemma4 SSCP norm type `{}`; expected `layer_norm`.",
                cfg.sscp_conv_norm_type
            );
        }

        let in_channels = if idx == 0 {
            1
        } else {
            cfg.sscp_conv_channel_size[idx - 1]
        };
        let out_channels = cfg.sscp_conv_channel_size[idx];
        let kernel_t = cfg.sscp_conv_kernel_size[idx][0];
        let kernel_f = cfg.sscp_conv_kernel_size[idx][1];
        let stride_t = cfg.sscp_conv_stride_size[idx][0];
        let stride_f = cfg.sscp_conv_stride_size[idx][1];

        let (pad_t_top, pad_t_bottom) = if let (Some(top), Some(bottom)) =
            (cfg.sscp_conv_time_pad_top, cfg.sscp_conv_time_pad_bottom)
        {
            (top, bottom)
        } else if cfg.sscp_conv_padding_type == "semicausal" {
            let half = kernel_t / 2;
            (half, if cfg.streaming { 0 } else { half })
        } else {
            (0, if cfg.streaming { 0 } else { kernel_t - 1 })
        };
        let pad_f_left = 1;
        let pad_f_right = 1;

        assert_eq!(
            kernel_t, kernel_f,
            "Gemma4 SSCP conv2d requires square kernels (candle limitation), got ({kernel_t}, {kernel_f})"
        );
        assert_eq!(
            stride_t, stride_f,
            "Gemma4 SSCP conv2d requires square strides (candle limitation), got ({stride_t}, {stride_f})"
        );
        let conv = conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_t,
            Conv2dConfig {
                stride: stride_t,
                padding: 0,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("conv"),
        )?;
        let norm = layer_norm(
            out_channels,
            LayerNormConfig {
                eps: cfg.rms_norm_eps,
                affine: false,
                ..Default::default()
            },
            vb.pp("norm"),
        )?;

        let _f_in_padded = input_freq_dim + pad_f_left + pad_f_right;

        Ok(Self {
            conv,
            norm,
            manual_padding: (pad_f_left, pad_f_right, pad_t_top, pad_t_bottom),
            time_stride: stride_t,
        })
    }

    fn forward(
        &self,
        audio_encodings: &Tensor,
        audio_mel_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let valid_mask = audio_mel_mask
            .eq(0.0)?
            .unsqueeze(1)?
            .unsqueeze(D::Minus1)?
            .to_dtype(audio_encodings.dtype())?;
        let audio_encodings = audio_encodings.broadcast_mul(&valid_mask)?;

        let audio_encodings = audio_encodings
            .pad_with_zeros(D::Minus1, self.manual_padding.0, self.manual_padding.1)?
            .pad_with_zeros(D::Minus2, self.manual_padding.2, self.manual_padding.3)?;
        let audio_encodings_conv = self.conv.forward_t(&audio_encodings, false)?;
        let output_mask = subsample_mask(
            audio_mel_mask,
            self.time_stride,
            audio_encodings_conv.dim(2)?,
        )?;

        let x = audio_encodings_conv.permute((0, 2, 3, 1))?;
        let x = self.norm.forward(&x)?;
        let x = x.permute((0, 3, 1, 2))?.relu()?;
        Ok((x, output_mask))
    }
}

pub struct Gemma4AudioSubSampleConvProjection {
    conv_0: Gemma4AudioSSCPConvBlock,
    conv_1: Gemma4AudioSSCPConvBlock,
    input_proj_linear: Arc<dyn QuantMethod>,
}

impl Gemma4AudioSubSampleConvProjection {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let mut current_f_for_block_input = cfg.input_feat_size;
        let mut calculated_f_out_dims = Vec::new();

        for i in 0..2 {
            let kernel_w = cfg.sscp_conv_kernel_size[i][1];
            let stride_w = cfg.sscp_conv_stride_size[i][1];
            let f_in_padded = current_f_for_block_input + 2;
            let f_out_after_conv = (f_in_padded - kernel_w) / stride_w + 1;
            calculated_f_out_dims.push(f_out_after_conv);
            current_f_for_block_input = f_out_after_conv;
        }

        let conv_0 = Gemma4AudioSSCPConvBlock::new(cfg, 0, cfg.input_feat_size, vb.pp("layer0"))?;
        let conv_1 =
            Gemma4AudioSSCPConvBlock::new(cfg, 1, calculated_f_out_dims[0], vb.pp("layer1"))?;
        let final_c_out = cfg.sscp_conv_channel_size[1];
        let final_f_out = calculated_f_out_dims[1];
        let input_proj_in_features = final_c_out * final_f_out;
        let input_proj_linear = mistralrs_quant::linear_no_bias(
            input_proj_in_features,
            cfg.hidden_size,
            &None,
            vb.pp("input_proj_linear"),
        )?;

        Ok(Self {
            conv_0,
            conv_1,
            input_proj_linear,
        })
    }

    pub(crate) fn forward(
        &self,
        audio_mel: &Tensor,
        audio_mel_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let x = audio_mel.unsqueeze(1)?;
        let (x, mask) = self.conv_0.forward(&x, audio_mel_mask)?;
        let (x, mask) = self.conv_1.forward(&x, &mask)?;

        let (b, c_out, t_out, f_out) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .transpose(2, 3)?
            .reshape((b, t_out, f_out * c_out))?;
        Ok((self.input_proj_linear.forward_autocast(&x)?, mask))
    }
}

pub struct Gemma4AudioAttention {
    num_heads: usize,
    head_dim: usize,
    chunk_size: usize,
    max_future_horizon: usize,
    max_past_horizon: usize,
    context_size: usize,
    pub(crate) relative_position_embedding: Gemma4AudioRelativePositionEmbedding,
    _per_dim_scale: Tensor,
    q_proj: ClippableLinear,
    k_proj: ClippableLinear,
    v_proj: ClippableLinear,
    q_scale: f64,
    k_scale: f64,
    local_causal_valid_mask: Tensor,
    softcap: f64,
    invalid_logits_tensor: Tensor,
    per_dim_scale_softplus: Tensor,
}

impl Gemma4AudioAttention {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let num_heads = cfg.conf_num_attention_heads;
        let hidden_size = cfg.hidden_size;
        let head_dim = hidden_size / num_heads;
        let chunk_size = cfg.conf_attention_chunk_size;
        let max_future_horizon = cfg.conf_attention_context_right;
        let max_past_horizon = cfg.conf_attention_context_left.saturating_sub(1);
        let context_size = chunk_size + max_past_horizon + max_future_horizon;

        let relative_position_embedding =
            Gemma4AudioRelativePositionEmbedding::new(cfg, vb.clone())?;
        let per_dim_scale = vb.get(head_dim, "per_dim_scale")?;
        let q_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj =
            ClippableLinear::new_no_bias(cfg, hidden_size, num_heads * head_dim, vb.pp("v_proj"))?;

        let q_scale = (head_dim as f64).powf(-0.5) / 2.0_f64.ln();
        let k_scale = (1.0_f64 + std::f64::consts::E).ln() / 2.0_f64.ln();

        let lower_causal_mask = {
            let mut mask_vec = vec![0u8; chunk_size * context_size];
            for i in 0..chunk_size {
                for j in 0..context_size {
                    if j >= i {
                        mask_vec[i * context_size + j] = 1;
                    }
                }
            }
            Tensor::from_vec(mask_vec, (chunk_size, context_size), vb.device())?
                .to_dtype(DType::U8)?
        };
        let diag_offset = (max_past_horizon + max_future_horizon) as isize;
        let upper_causal_mask = {
            let mut mask_vec = vec![0u8; chunk_size * context_size];
            for i in 0..chunk_size {
                for j in 0..context_size {
                    if (j as isize) <= (i as isize + diag_offset) {
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

        let invalid_logits_tensor =
            Tensor::new(cfg.conf_attention_invalid_logits_value as f32, vb.device())?;
        let per_dim_scale_softplus = {
            let ones = Tensor::ones_like(&per_dim_scale)?.to_dtype(DType::F32)?;
            let exp_scale = per_dim_scale.to_dtype(DType::F32)?.exp()?;
            ones.broadcast_add(&exp_scale)?.log()?
        };

        Ok(Self {
            num_heads,
            head_dim,
            chunk_size,
            max_future_horizon,
            max_past_horizon,
            context_size,
            relative_position_embedding,
            _per_dim_scale: per_dim_scale,
            q_proj,
            k_proj,
            v_proj,
            q_scale,
            k_scale,
            local_causal_valid_mask,
            softcap: cfg.conf_attention_logit_cap,
            invalid_logits_tensor,
            per_dim_scale_softplus,
        })
    }

    fn pad_dim1(&self, x: &Tensor, left_pad: usize, right_pad: usize) -> Result<Tensor> {
        x.pad_with_zeros(1, left_pad, right_pad)
    }

    fn convert_to_block(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.dims();
        let (b, t) = (shape[0], shape[1]);
        let num_blocks = t.div_ceil(self.chunk_size);
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
        let time_dim = x.dim(1)?;
        let num_windows = (time_dim - frame_len) / frame_step + 1;

        let mut windows = Vec::with_capacity(num_windows);
        for i in 0..num_windows {
            let start_idx = i * frame_step;
            windows.push(x.narrow(1, start_idx, frame_len)?);
        }
        Tensor::stack(&windows, 1)
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let query_states = self.q_proj.forward(x)?.to_dtype(DType::F32)?;
        let key_states = self.k_proj.forward(x)?.to_dtype(DType::F32)?;
        let value_states = self.v_proj.forward(x)?.to_dtype(DType::F32)?;

        let (b, t) = match x.dims() {
            &[b, t, _] => (b, t),
            _ => bail!("Expected input to have 3 dimensions"),
        };

        let query_states = query_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let key_states = key_states.reshape((b, t, self.num_heads, self.head_dim))?;
        let value_states = value_states.reshape((b, t, self.num_heads, self.head_dim))?;

        let input_device = x.device();
        let per_dim_scale_softplus = self.per_dim_scale_softplus.to_device(input_device)?;
        let local_causal_valid_mask = self.local_causal_valid_mask.to_device(input_device)?;
        let invalid_logits_tensor = self.invalid_logits_tensor.to_device(input_device)?;

        let scale_shape = (1, 1, 1, self.head_dim);
        // HF: query_states = query_states * q_scale * softplus(per_dim_scale)
        let query_states = query_states.affine(self.q_scale, 0.0)?.broadcast_mul(
            &per_dim_scale_softplus
                .reshape(scale_shape)?
                .to_dtype(DType::F32)?,
        )?;
        // HF: key_states = key_states * k_scale
        let key_states = key_states.affine(self.k_scale, 0.0)?;

        let (batch_size, q_time) = (query_states.dim(0)?, query_states.dim(1)?);
        let query_blocks = self.convert_to_block(&query_states)?;
        let mut key_blocks = self.extract_block_context(&key_states)?;
        let mut value_blocks = self.extract_block_context(&value_states)?;
        let num_query_blocks = query_blocks.dim(1)?;

        if key_blocks.dim(2)? != self.context_size {
            let current_context = key_blocks.dim(2)?;
            if current_context < self.context_size {
                let pad_amount = self.context_size - current_context;
                key_blocks = key_blocks.pad_with_zeros(2, 0, pad_amount)?;
                value_blocks = value_blocks.pad_with_zeros(2, 0, pad_amount)?;
            } else {
                key_blocks = key_blocks.narrow(2, 0, self.context_size)?;
                value_blocks = value_blocks.narrow(2, 0, self.context_size)?;
            }
        }

        let num_key_blocks = key_blocks.dim(1)?;
        if num_query_blocks != num_key_blocks {
            if num_query_blocks < num_key_blocks {
                key_blocks = key_blocks.narrow(1, 0, num_query_blocks)?;
                value_blocks = value_blocks.narrow(1, 0, num_query_blocks)?;
            } else {
                bail!("Keys have fewer blocks than queries: {num_key_blocks} < {num_query_blocks}");
            }
        }

        let original_valid_mask = mask.eq(0.0)?.to_dtype(DType::U8)?;
        let extracted_valid_mask_blocks = self.extract_block_context(&original_valid_mask)?;
        let extracted_valid_mask_blocks = if extracted_valid_mask_blocks.dims().len() == 4
            && extracted_valid_mask_blocks.dim(0)? == batch_size
            && extracted_valid_mask_blocks.dim(1)? == num_query_blocks
            && extracted_valid_mask_blocks.dim(2)? * extracted_valid_mask_blocks.dim(3)?
                == self.context_size
        {
            extracted_valid_mask_blocks.reshape((
                batch_size,
                num_query_blocks,
                self.context_size,
            ))?
        } else if extracted_valid_mask_blocks.dims().len() == 3 {
            extracted_valid_mask_blocks
        } else {
            match *extracted_valid_mask_blocks.dims() {
                [b, n, _] if b == batch_size && n == num_query_blocks => {
                    extracted_valid_mask_blocks
                }
                [b, _, n] if b == batch_size && n == num_query_blocks => {
                    extracted_valid_mask_blocks.transpose(1, 2)?
                }
                _ => extracted_valid_mask_blocks,
            }
        };
        let extracted_valid_mask_blocks =
            if extracted_valid_mask_blocks.dim(D::Minus1)? != self.context_size {
                let current_context_size = extracted_valid_mask_blocks.dim(D::Minus1)?;
                if current_context_size < self.context_size {
                    extracted_valid_mask_blocks.pad_with_zeros(
                        D::Minus1,
                        0,
                        self.context_size - current_context_size,
                    )?
                } else {
                    extracted_valid_mask_blocks.narrow(D::Minus1, 0, self.context_size)?
                }
            } else {
                extracted_valid_mask_blocks
            };

        let condition_from_input_validity =
            extracted_valid_mask_blocks.unsqueeze(1)?.unsqueeze(3)?;
        let condition_from_causality = local_causal_valid_mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let final_condition_for_where = condition_from_input_validity
            .to_dtype(DType::U8)?
            .broadcast_mul(&condition_from_causality.to_dtype(DType::U8)?)?;

        let relative_logits = self
            .relative_position_embedding
            .forward(&query_blocks, &key_blocks)?;
        let logits = ((&relative_logits / self.softcap)?.tanh()? * self.softcap)?;

        let final_condition_for_where = if final_condition_for_where.dims() != logits.dims() {
            let logits_shape = logits.dims();
            let mask_shape = final_condition_for_where.dims().to_vec();
            let mut mask = final_condition_for_where;
            for (i, (&logit_dim, &mask_dim)) in
                logits_shape.iter().zip(mask_shape.iter()).enumerate()
            {
                if mask_dim != logit_dim && mask_dim != 1 {
                    if mask_dim > logit_dim {
                        mask = mask.narrow(i, 0, logit_dim)?;
                    } else {
                        bail!(
                            "Mask dimension {i} has size {mask_dim} which is smaller than logits size {logit_dim} and cannot be broadcast"
                        );
                    }
                }
            }
            if mask.dims() != logits.dims() {
                mask.broadcast_as(logits_shape)?
            } else {
                mask
            }
        } else {
            final_condition_for_where
        };

        let invalid_logits = invalid_logits_tensor.broadcast_as(logits.shape())?;
        let masked_logits = final_condition_for_where.where_cond(&logits, &invalid_logits)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&masked_logits.to_dtype(DType::F32)?)?
            .to_dtype(value_blocks.dtype())?;

        let (b_dim, n_dim, u_dim, w_dim, c_dim) = probabilities.dims5()?;
        let h_dim = value_blocks.dim(D::Minus1)?;
        let probabilities = probabilities.permute((0, 2, 1, 3, 4))?.reshape((
            b_dim * u_dim * n_dim,
            w_dim,
            c_dim,
        ))?;
        let value_blocks = value_blocks.permute((0, 1, 3, 2, 4))?.reshape((
            b_dim * u_dim * n_dim,
            c_dim,
            h_dim,
        ))?;
        let context_vectors = probabilities
            .matmul(&value_blocks)?
            .reshape((b_dim, u_dim, n_dim, w_dim, h_dim))?
            .permute((0, 1, 3, 2, 4))?
            .reshape((
                batch_size,
                num_query_blocks * self.chunk_size,
                self.num_heads,
                self.head_dim,
            ))?
            .narrow(1, 0, q_time)?;
        Ok(context_vectors)
    }
}

pub struct Gemma4AudioConformerAttention {
    pre_attn_norm: RmsNorm,
    attn: Gemma4AudioAttention,
    post: ClippableLinear,
    post_norm: RmsNorm,
    gradient_clipping: f64,
    hidden_size: usize,
}

impl Gemma4AudioConformerAttention {
    fn new(
        cfg: &Gemma4AudioConfig,
        attn_vb: ShardedVarBuilder,
        pre_attn_norm_vb: ShardedVarBuilder,
        post_norm_vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            pre_attn_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, pre_attn_norm_vb)?,
            attn: Gemma4AudioAttention::new(cfg, attn_vb.clone())?,
            post: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size,
                attn_vb.pp("post"),
            )?,
            post_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, post_norm_vb)?,
            gradient_clipping: cfg.gradient_clipping,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, x: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_attn_norm.forward(&x)?;
        let x = self.attn.forward(&x, mask)?;
        let (b, t, _, _) = x.dims4()?;
        let x = x.reshape((b, t, self.hidden_size))?;
        let x = self
            .post
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        residual.broadcast_add(&self.post_norm.forward(&x)?)
    }
}

pub struct Gemma4AudioConformerFeedForward {
    scale: f64,
    pre_layer_norm: RmsNorm,
    ffw_layer_1: ClippableLinear,
    ffw_layer_2: ClippableLinear,
    post_layer_norm: RmsNorm,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerFeedForward {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            scale: cfg.conf_residual_weight,
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            ffw_layer_1: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size * 4,
                vb.pp("ffw_layer_1"),
            )?,
            ffw_layer_2: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size * 4,
                cfg.hidden_size,
                vb.pp("ffw_layer_2"),
            )?,
            post_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_layer_norm"),
            )?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = x.clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.pre_layer_norm.forward(&x)?;
        let x = candle_nn::ops::silu(&self.ffw_layer_1.forward(&x)?)?;
        let x = self
            .ffw_layer_2
            .forward(&x)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let x = self.post_layer_norm.forward(&x)?;
        residual.broadcast_add(&(x * self.scale)?)
    }
}

pub struct Gemma4AudioConformerLightConv1d {
    pre_layer_norm: RmsNorm,
    depthwise_conv1d: Conv1d,
    conv_norm: RmsNorm,
    linear_start: ClippableLinear,
    linear_end: ClippableLinear,
    causal_padding: usize,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerLightConv1d {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            pre_layer_norm: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_layer_norm"),
            )?,
            linear_start: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size * 2,
                vb.pp("linear_start"),
            )?,
            depthwise_conv1d: conv1d_no_bias(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.conf_conv_kernel_size,
                candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: 0,
                    dilation: 1,
                    groups: cfg.hidden_size,
                    cudnn_fwd_algo: None,
                },
                vb.pp("depthwise_conv1d").set_dtype(DType::F32),
            )?,
            conv_norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("conv_norm"))?,
            linear_end: ClippableLinear::new_no_bias(
                cfg,
                cfg.hidden_size,
                cfg.hidden_size,
                vb.pp("linear_end"),
            )?,
            causal_padding: cfg.conf_conv_kernel_size - 1,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    fn forward(&self, audio_encodings: &Tensor) -> Result<Tensor> {
        let residual = audio_encodings;
        let audio_encodings = self.pre_layer_norm.forward(audio_encodings)?;
        let audio_encodings = self.linear_start.forward(&audio_encodings)?;
        let chunks = audio_encodings.chunk(2, D::Minus1)?;
        let audio_encodings = chunks[0].broadcast_mul(&candle_nn::ops::sigmoid(&chunks[1])?)?;
        let audio_encodings = audio_encodings.transpose(D::Minus1, D::Minus2)?;
        let audio_encodings = audio_encodings.pad_with_zeros(D::Minus1, self.causal_padding, 0)?;
        let audio_encodings = Convolution
            .forward_1d(
                &self.depthwise_conv1d,
                &audio_encodings.to_dtype(DType::F32)?,
            )?
            .to_dtype(audio_encodings.dtype())?
            .transpose(D::Minus2, D::Minus1)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        let audio_encodings = self.conv_norm.forward(&audio_encodings)?;
        let audio_encodings = candle_nn::ops::silu(&audio_encodings)?;
        let audio_encodings = self.linear_end.forward(&audio_encodings)?;
        residual.broadcast_add(&audio_encodings)
    }
}

pub struct Gemma4AudioConformerBlock {
    pub(crate) ffw_layer_start: Gemma4AudioConformerFeedForward,
    pub(crate) attention: Gemma4AudioConformerAttention,
    pub(crate) lconv1d: Gemma4AudioConformerLightConv1d,
    pub(crate) ffw_layer_end: Gemma4AudioConformerFeedForward,
    pub(crate) norm: RmsNorm,
    gradient_clipping: f64,
}

impl Gemma4AudioConformerBlock {
    fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            ffw_layer_start: Gemma4AudioConformerFeedForward::new(cfg, vb.pp("feed_forward1"))?,
            attention: Gemma4AudioConformerAttention::new(
                cfg,
                vb.pp("self_attn"),
                vb.pp("norm_pre_attn"),
                vb.pp("norm_post_attn"),
            )?,
            lconv1d: Gemma4AudioConformerLightConv1d::new(cfg, vb.pp("lconv1d"))?,
            ffw_layer_end: Gemma4AudioConformerFeedForward::new(cfg, vb.pp("feed_forward2"))?,
            norm: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm_out"))?,
            gradient_clipping: cfg.gradient_clipping,
        })
    }

    pub(crate) fn forward(
        &self,
        audio_encodings: &Tensor,
        audio_mel_mask: &Tensor,
    ) -> Result<Tensor> {
        let audio_encodings = self.ffw_layer_start.forward(audio_encodings)?;
        let audio_encodings = self.attention.forward(&audio_encodings, audio_mel_mask)?;
        let audio_encodings = self.lconv1d.forward(&audio_encodings)?;
        let audio_encodings = self
            .ffw_layer_end
            .forward(&audio_encodings)?
            .clamp(-self.gradient_clipping, self.gradient_clipping)?;
        self.norm.forward(&audio_encodings)
    }
}

pub struct AudioModel {
    pub(crate) subsample_conv_projection: Gemma4AudioSubSampleConvProjection,
    pub(crate) conformer: Vec<Gemma4AudioConformerBlock>,
    conf_reduction_factor: usize,
    pub(crate) output_proj: Option<Arc<dyn QuantMethod>>,
}

impl AudioModel {
    pub fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let subsample_conv_projection =
            Gemma4AudioSubSampleConvProjection::new(cfg, vb.pp("subsample_conv_projection"))?;
        let mut conformer = Vec::with_capacity(cfg.conf_num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.conf_num_hidden_layers {
            conformer.push(Gemma4AudioConformerBlock::new(cfg, vb_layers.pp(i))?);
        }
        let output_proj = if let Some(output_dim) = cfg.output_proj_dims {
            Some(mistralrs_quant::linear(
                cfg.hidden_size,
                output_dim,
                &None,
                vb.pp("output_proj"),
            )?)
        } else {
            None
        };

        Ok(Self {
            subsample_conv_projection,
            conformer,
            conf_reduction_factor: cfg.conf_reduction_factor,
            output_proj,
        })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mut audio_encodings, mut current_mask) = self
            .subsample_conv_projection
            .forward(audio_mel, audio_mel_mask)?;

        for block in &self.conformer {
            audio_encodings = block.forward(&audio_encodings, &current_mask)?;
        }

        if self.conf_reduction_factor > 1 {
            let stride = self.conf_reduction_factor;
            let reduced_len = audio_encodings.dim(1)?.div_ceil(stride);
            let indices = Tensor::arange(
                0f32,
                audio_encodings.dim(1)? as f32,
                audio_encodings.device(),
            )?
            .affine(stride as f64, 0.0)?
            .to_dtype(DType::I64)?;
            let max_idx = audio_encodings.dim(1)? as i64 - 1;
            let indices = indices
                .narrow(0, 0, reduced_len.min(indices.dim(0)?))?
                .clamp(0, max_idx)?;

            audio_encodings = audio_encodings.index_select(&indices, 1)?;
            current_mask = current_mask.index_select(&indices, 1)?;
        }

        if let Some(ref output_proj) = self.output_proj {
            audio_encodings = output_proj.forward_autocast(&audio_encodings)?;
        }

        let enc_len = audio_encodings.dim(1)?;
        let mask_len = current_mask.dim(1)?;
        if mask_len != enc_len {
            if enc_len < mask_len {
                current_mask = current_mask.narrow(1, 0, enc_len)?;
            } else {
                current_mask = current_mask.pad_with_zeros(1, 0, enc_len - mask_len)?;
            }
        }

        let valid_mask = current_mask.eq(0.0)?;
        let zeros = Tensor::zeros_like(&audio_encodings)?;
        let audio_encodings = valid_mask
            .unsqueeze(D::Minus1)?
            .broadcast_as(audio_encodings.shape())?
            .where_cond(&audio_encodings, &zeros)?;

        Ok((audio_encodings, current_mask))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }

    #[allow(dead_code)]
    pub fn get_isq_layers(&mut self) -> Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> {
        let mut tensors = Vec::new();
        for block in &mut self.conformer {
            tensors.push((&mut block.attention.attn.q_proj.inner, None));
            tensors.push((&mut block.attention.attn.k_proj.inner, None));
            tensors.push((&mut block.attention.attn.v_proj.inner, None));
            tensors.push((
                &mut block.attention.attn.relative_position_embedding.pos_proj,
                None,
            ));
            tensors.push((&mut block.attention.post.inner, None));
            tensors.push((&mut block.ffw_layer_start.ffw_layer_1.inner, None));
            tensors.push((&mut block.ffw_layer_start.ffw_layer_2.inner, None));
            tensors.push((&mut block.ffw_layer_end.ffw_layer_1.inner, None));
            tensors.push((&mut block.ffw_layer_end.ffw_layer_2.inner, None));
            tensors.push((&mut block.lconv1d.linear_start.inner, None));
            tensors.push((&mut block.lconv1d.linear_end.inner, None));
        }
        tensors.push((&mut self.subsample_conv_projection.input_proj_linear, None));
        if let Some(ref mut proj) = self.output_proj {
            tensors.push((proj, None));
        }
        tensors
    }
}
