use candle_core::{Result, Tensor};

#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
use crate::pipeline::text_models_inputs_processor::FlashKMeta;
use crate::{attention::SdpaParams, pipeline::text_models_inputs_processor::FlashParams};

#[cfg(any(feature = "flash-attn", feature = "flash-attn-v3"))]
fn varlen_metadata<'a>(
    q: &Tensor,
    params: &'a FlashParams,
) -> Result<Option<(&'a Tensor, &'a FlashKMeta, &'a Tensor)>> {
    let location = q.device().location();
    let Some(cumulative_seqlens_q) = params.cumulative_seqlens_q.get(&location) else {
        if params.packed {
            candle_core::bail!("packed prefill is missing query metadata for {location:?}");
        }
        return Ok(None);
    };
    let k_meta = &params.logical_k;
    let Some(cumulative_seqlens_k) = k_meta.cumulative_seqlens.get(&location) else {
        if params.packed {
            candle_core::bail!("packed prefill is missing key metadata for {location:?}");
        }
        return Ok(None);
    };
    Ok(Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)))
}

#[cfg(feature = "flash-attn")]
fn flash_attn_v2(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let window_size_left = sdpa_params.sliding_window;
    let default_causal = seq_len > 1;
    let use_varlen =
        b_sz > 1 || seq_len != k.dim(1)? || flash_params.is_some_and(|params| params.packed);

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)) =
                varlen_metadata(q, params)?
            {
                let window_size_right = if params.causal { Some(0) } else { None };
                let qshape = q.shape();
                let q = q.flatten_to(1)?;
                let k = k.flatten_to(1)?;
                let v = v.flatten_to(1)?;

                if let Some(softcap) = sdpa_params.softcap {
                    return mistralrs_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
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
                    return mistralrs_flash_attn::flash_attn_varlen_windowed(
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
        mistralrs_flash_attn::flash_attn_alibi_windowed_softcap(
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
        mistralrs_flash_attn::flash_attn_windowed(
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
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    let (b_sz, seq_len, _n_attn_heads, _head_dim) = q.dims4()?;
    let default_causal = seq_len > 1;
    let use_varlen =
        b_sz > 1 || seq_len != k.dim(1)? || flash_params.is_some_and(|params| params.packed);

    if use_varlen {
        if let Some(params) = flash_params {
            if let Some((cumulative_seqlens_q, k_meta, cumulative_seqlens_k)) =
                varlen_metadata(q, params)?
            {
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
    flash_params: Option<&FlashParams>,
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
    flash_params: Option<&FlashParams>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor> {
    flash_attn_v2(q, k, v, flash_params, sdpa_params)
}

#[cfg(not(any(feature = "flash-attn", feature = "flash-attn-v3")))]
pub(crate) fn flash_attn(
    _: &Tensor,
    _: &Tensor,
    _: &Tensor,
    _: Option<&FlashParams>,
    _: &SdpaParams,
) -> Result<Tensor> {
    unimplemented!("Compile with `--features flash-attn` or `--features flash-attn-v3`.")
}

#[cfg(all(test, any(feature = "flash-attn", feature = "flash-attn-v3")))]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn packed_varlen_metadata_fails_closed() {
        let q = Tensor::zeros((1, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        let mut params = FlashParams::empty(true);
        params.packed = true;

        let missing_query = varlen_metadata(&q, &params).unwrap_err();

        assert!(missing_query
            .to_string()
            .contains("packed prefill is missing query metadata"));

        params.cumulative_seqlens_q.insert(
            Device::Cpu.location(),
            Tensor::new(&[0u32, 1], &Device::Cpu).unwrap(),
        );
        let missing_key = varlen_metadata(&q, &params).unwrap_err();

        assert!(missing_key
            .to_string()
            .contains("packed prefill is missing key metadata"));
    }
}
