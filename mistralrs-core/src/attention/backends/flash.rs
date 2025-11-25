use candle_core::{Result, Tensor};

use crate::attention::SdpaParams;

#[cfg(feature = "flash-attn")]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let window_size_left = sdpa_params.sliding_window;
    let default_causal = seq_len > 1;

    if let Some(params) = flash_params {
        let window_size_right = if params.causal { Some(0) } else { None };
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let cumulative_seqlens_q = &params.cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &params.cumulative_seqlens_k[&q.device().location()];

        if let Some(softcap) = sdpa_params.softcap {
            candle_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                &q,
                &k,
                &v,
                None,
                cumulative_seqlens_q,
                cumulative_seqlens_k,
                params.max_q as usize,
                params.max_k as usize,
                sdpa_params.softmax_scale,
                window_size_left,
                window_size_right,
                softcap,
            )?
            .reshape(qshape)
        } else {
            candle_flash_attn::flash_attn_varlen_windowed(
                &q,
                &k,
                &v,
                cumulative_seqlens_q,
                cumulative_seqlens_k,
                params.max_q as usize,
                params.max_k as usize,
                sdpa_params.softmax_scale,
                window_size_left,
                window_size_right,
            )?
            .reshape(qshape)
        }
    } else {
        let window_size_right = if default_causal { Some(0) } else { None };
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
}

#[cfg(feature = "flash-attn-v3")]
pub(crate) fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&crate::pipeline::text_models_inputs_processor::FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (_b_sz, _n_attn_heads, seq_len, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;

    if let Some(params) = flash_params {
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let window_size_left = sdpa_params.sliding_window;
        let window_size_right = if params.causal { Some(0) } else { None };

        let cumulative_seqlens_q = &params.cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &params.cumulative_seqlens_k[&q.device().location()];

        candle_flash_attn_v3::flash_attn_varlen_windowed(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            params.max_q as usize,
            params.max_k as usize,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            true,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn_v3::flash_attn(q, k, v, sdpa_params.softmax_scale, default_causal, true)
    }
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
