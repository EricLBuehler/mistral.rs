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
    let causal = seq_len > 1;

    use crate::pipeline::text_models_inputs_processor::FlashParams;

    if let Some(FlashParams {
        max_q,
        max_k,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
    }) = flash_params
    {
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let window_size_left = sdpa_params.sliding_window;
        let window_size_right = if causal { Some(0) } else { None };

        let cumulative_seqlens_q = &cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &cumulative_seqlens_k[&q.device().location()];

        candle_flash_attn::flash_attn_varlen_windowed_softcap(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            *max_q as usize,
            *max_k as usize,
            sdpa_params.softmax_scale,
            sdpa_params.softcap,
            window_size_left,
            window_size_right,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn::flash_attn_softcap(
            q,
            k,
            v,
            sdpa_params.softmax_scale,
            sdpa_params.softcap,
            causal,
        )
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
    let causal = seq_len > 1;

    use crate::pipeline::text_models_inputs_processor::FlashParams;

    if let Some(FlashParams {
        max_q,
        max_k,
        cumulative_seqlens_q,
        cumulative_seqlens_k,
    }) = flash_params
    {
        let qshape = q.shape();
        let q = q.flatten_to(1)?;
        let k = k.flatten_to(1)?;
        let v = v.flatten_to(1)?;

        let window_size_left = sdpa_params.sliding_window;
        let window_size_right = if causal { Some(0) } else { None };

        let cumulative_seqlens_q = &cumulative_seqlens_q[&q.device().location()];
        let cumulative_seqlens_k = &cumulative_seqlens_k[&q.device().location()];

        candle_flash_attn_v3::flash_attn_varlen_windowed(
            &q,
            &k,
            &v,
            cumulative_seqlens_q,
            cumulative_seqlens_k,
            *max_q as usize,
            *max_k as usize,
            sdpa_params.softmax_scale,
            window_size_left,
            window_size_right,
            true,
        )?
        .reshape(qshape)
    } else {
        candle_flash_attn_v3::flash_attn(q, k, v, sdpa_params.softmax_scale, causal, true)
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
