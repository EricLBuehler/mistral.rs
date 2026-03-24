//! Gemma4 audio encoder.
//!
//! The Gemma4 audio encoder is architecturally identical to Gemma3n's conformer-based
//! audio encoder. We reuse the Gemma3n implementation directly by converting the
//! Gemma4AudioConfig into a Gemma3nAudioConfig.

use candle_core::{Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
use std::sync::Arc;

use crate::vision_models::gemma3n::{
    audio::AudioModel as Gemma3nAudioModel, config::Gemma3nAudioConfig,
};

use super::config::Gemma4AudioConfig;

/// Convert Gemma4AudioConfig to Gemma3nAudioConfig for reuse.
fn to_gemma3n_audio_config(cfg: &Gemma4AudioConfig) -> Gemma3nAudioConfig {
    Gemma3nAudioConfig {
        input_feat_size: cfg.input_feat_size,
        hidden_size: cfg.hidden_size,
        conf_attention_chunk_size: cfg.conf_attention_chunk_size,
        conf_attention_context_left: cfg.conf_attention_context_left,
        conf_attention_context_right: cfg.conf_attention_context_right,
        conf_attention_invalid_logits_value: cfg.conf_attention_invalid_logits_value,
        conf_attention_logit_cap: cfg.conf_attention_logit_cap,
        conf_num_attention_heads: cfg.conf_num_attention_heads,
        conf_num_hidden_layers: cfg.conf_num_hidden_layers,
        conf_conv_kernel_size: cfg.conf_conv_kernel_size,
        conf_reduction_factor: cfg.conf_reduction_factor,
        conf_residual_weight: cfg.conf_residual_weight,
        sscp_conv_channel_size: cfg.sscp_conv_channel_size.clone(),
        sscp_conv_kernel_size: cfg.sscp_conv_kernel_size.clone(),
        sscp_conv_stride_size: cfg.sscp_conv_stride_size.clone(),
        vocab_size: cfg.vocab_size,
        sscp_conv_group_norm_eps: cfg.sscp_conv_group_norm_eps,
        rms_norm_eps: cfg.rms_norm_eps,
        vocab_offset: cfg.vocab_offset,
    }
}

/// Gemma4 audio encoder wrapping the Gemma3n conformer-based implementation.
pub struct AudioModel {
    pub(crate) inner: Gemma3nAudioModel,
    pub(crate) output_proj: Option<Arc<dyn QuantMethod>>,
}

impl AudioModel {
    pub fn new(cfg: &Gemma4AudioConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let g3n_cfg = to_gemma3n_audio_config(cfg);
        let inner = Gemma3nAudioModel::new(&g3n_cfg, vb.clone())?;

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

        Ok(Self { inner, output_proj })
    }

    pub fn forward(&self, audio_mel: &Tensor, audio_mel_mask: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mut audio_encodings, mut current_mask) =
            self.inner.forward(audio_mel, audio_mel_mask)?;

        // Apply output projection if configured
        if let Some(ref proj) = self.output_proj {
            let original_dtype = audio_encodings.dtype();
            let mut xs = audio_encodings;
            if let Some(t) = proj.quantized_act_type() {
                xs = xs.to_dtype(t)?;
            }
            xs = crate::layers::MatMul.qmethod_matmul(&xs, &**proj)?;
            if proj.quantized_act_type().is_some() {
                xs = xs.to_dtype(original_dtype)?;
            }
            audio_encodings = xs;
        }

        // Ensure mask length matches encodings
        let enc_len = audio_encodings.dim(1)?;
        let mask_len = current_mask.dim(1)?;
        if mask_len != enc_len {
            if enc_len < mask_len {
                current_mask = current_mask.narrow(1, 0, enc_len)?;
            } else {
                current_mask = current_mask.pad_with_zeros(1, 0, enc_len - mask_len)?;
            }
        }

        Ok((audio_encodings, current_mask))
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.inner.residual_tensors()
    }

    pub fn get_isq_layers(&mut self) -> Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)> {
        let mut tensors = Vec::new();
        // Audio lives in the non-mapped submodel, so these do not correspond to
        // text repeating-layer indices for device placement.
        for block in &mut self.inner.conformer {
            tensors.push((&mut block.attention.attn.q_proj, None));
            tensors.push((&mut block.attention.attn.k_proj, None));
            tensors.push((&mut block.attention.attn.v_proj, None));
            tensors.push((
                &mut block.attention.attn.relative_position_embedding.pos_proj,
                None,
            ));
            tensors.push((&mut block.attention.post, None));
            tensors.push((&mut block.ffw_layer_start.ffw_layer_1, None));
            tensors.push((&mut block.ffw_layer_start.ffw_layer_2, None));
            tensors.push((&mut block.ffw_layer_end.ffw_layer_1, None));
            tensors.push((&mut block.ffw_layer_end.ffw_layer_2, None));
            tensors.push((&mut block.lconv1d.linear_start, None));
            tensors.push((&mut block.lconv1d.linear_end, None));
        }
        // SSCP input projection
        tensors.push((
            &mut self.inner.subsample_conv_projection.input_proj_linear,
            None,
        ));
        // Output projection
        if let Some(ref mut proj) = self.output_proj {
            tensors.push((proj, None));
        }
        tensors
    }
}
