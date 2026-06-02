use candle_core::{Result, Tensor};

use crate::attention::SdpaParams;

#[cfg(feature = "flash-attn")]
fn flash_attn_v2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let window_size_left = sdpa_params.sliding_window;
    let default_causal = seq_len > 1;
    let use_varlen = b_sz > 1 || seq_len != k.dim(1)?;

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some(cumulative_seqlens_q) =
                params.cumulative_seqlens_q.get(&q.device().location())
            {
                let k_meta = &params.logical_k;
                let cumulative_seqlens_k = &k_meta.cumulative_seqlens[&q.device().location()];

                let window_size_right = if params.causal { Some(0) } else { None };
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                if let Some(softcap) = sdpa_params.softcap {
                    return candle_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                        &q,
                        &k,
                        &v,
                        None,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                        softcap,
                    )?
                    .reshape(qshape);
                } else {
                    return candle_flash_attn::flash_attn_varlen_windowed(
                        &q,
                        &k,
                        &v,
                        cumulative_seqlens_q,
                        cumulative_seqlens_k,
                        params.max_q as usize,
                        k_meta.max as usize,
                        sdpa_params.softmax_scale,
                        window_size_left,
                        window_size_right,
                    )?
                    .reshape(qshape);
                }
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    let window_size_right = if causal { Some(0) } else { None };
    if let Some(softcap) = sdpa_params.softcap {
        candle_flash_attn::flash_attn_alibi_windowed_softcap(
            q,
            k,
            v,
            None,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            softcap,
        )
    } else {
        candle_flash_attn::flash_attn_windowed(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
        )
    }
}

#[cfg(feature = "flash-attn-v3")]
fn flash_attn_v3(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;
    let use_varlen = b_sz > 1 || seq_len != k.dim(1)?;

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some(cumulative_seqlens_q) =
                params.cumulative_seqlens_q.get(&q.device().location())
            {
                let k_meta = &params.logical_k;
                let cumulative_seqlens_k = &k_meta.cumulative_seqlens[&q.device().location()];
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                let window_size_left = sdpa_params.sliding_window;
                let window_size_right = if params.causal { Some(0) } else { None };

                return candle_flash_attn_v3::flash_attn_varlen_windowed(
                    &q,
                    &k,
                    &v,
                    cumulative_seqlens_q,
                    cumulative_seqlens_k,
                    params.max_q as usize,
                    k_meta.max as usize,
                    sdpa_params.softmax_scale,
                    window_size_left,
                    window_size_right,
                    true,
                )?
                .reshape(qshape);
            }
        }
    }

    let causal = flash_params.map_or(default_causal, |p| p.causal);
    candle_flash_attn_v3::flash_attn(q, k, v, sdpa_params.softmax_scale, causal, true)
}

#[cfg(feature = "flash-attn-v3")]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let q_dims = q.dims4()?;
    let head_dim = q_dims.3;
    if head_dim <= 512 {
        #[cfg(feature = "flash-attn")]
        {
            return flash_attn_v2(q, k, v, flash_params, sdpa_params);
        }
    }

    flash_attn_v3(q, k, v, flash_params, sdpa_params)
}

#[cfg(all(feature = "flash-attn", not(feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_v2(q, k, v, flash_params, sdpa_params)
}

#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with `--features flash-attn` or `--features flash-attn-v3`.")
}
