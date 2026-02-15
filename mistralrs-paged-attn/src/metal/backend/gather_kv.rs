use candle_core::{DType, IndexOp, Result, Storage, Tensor};

use crate::metal::kernels::{self, PagedAttentionDType};

pub fn gather_kv_cache(
    key_cache: &Tensor,   // [num_blocks, kv_heads, head_size/x, block_size, x]
    value_cache: &Tensor, // [num_blocks, kv_heads, head_size, block_size]
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    block_table: &Tensor, // [batch, max_blocks]
    cu_seq_lens: &Tensor, // [batch + 1]
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let cache_dtype = key_cache.dtype();
    if value_cache.dtype() != cache_dtype {
        candle_core::bail!(
            "gather_kv_cache expects matching cache dtypes, got {:?} and {:?}",
            cache_dtype,
            value_cache.dtype()
        );
    }

    let block_table = block_table.contiguous()?;
    let cu_seq_lens = cu_seq_lens.contiguous()?;

    if !matches!(block_table.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "gather_kv_cache expects i32/u32 block_table (got {:?})",
            block_table.dtype()
        );
    }
    if !matches!(cu_seq_lens.dtype(), DType::I32 | DType::U32) {
        candle_core::bail!(
            "gather_kv_cache expects i32/u32 cu_seq_lens (got {:?})",
            cu_seq_lens.dtype()
        );
    }

    let cache_ty = match cache_dtype {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        DType::F8E4M3 => PagedAttentionDType::F8E4M3,
        other => candle_core::bail!("unsupported cache dtype {other:?}"),
    };
    let out_ty = match out_dtype {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        other => candle_core::bail!("unsupported output dtype {other:?}"),
    };

    // Extract dimensions
    let k_dims = key_cache.dims5()?;
    let num_kv_heads = k_dims.1;
    let head_size_over_x = k_dims.2;
    let block_size = k_dims.3;
    let x = k_dims.4;
    let head_size = head_size_over_x * x;

    // num_tokens = cu_seq_lens[-1], num_seqs = len(cu_seq_lens) - 1
    let cu_seq_lens_len = cu_seq_lens.dims1()?;
    let num_seqs = cu_seq_lens_len - 1;
    let num_tokens = if cu_seq_lens.dtype() == DType::I32 {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<i32>()? as usize
    } else {
        cu_seq_lens.i(cu_seq_lens_len - 1)?.to_scalar::<u32>()? as usize
    };

    if num_tokens == 0 {
        let k_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, key_cache.device())?;
        let v_out = Tensor::zeros((0, num_kv_heads, head_size), out_dtype, key_cache.device())?;
        return Ok((k_out, v_out));
    }

    let k_out = Tensor::zeros(
        (num_tokens, num_kv_heads, head_size),
        out_dtype,
        key_cache.device(),
    )?;
    let v_out = Tensor::zeros(
        (num_tokens, num_kv_heads, head_size),
        out_dtype,
        key_cache.device(),
    )?;

    // Scope all storage guards so they drop before we return k_out/v_out.
    {
        // Extract storage
        let (kc_s, kc_l) = key_cache.storage_and_layout();
        let kc = match &*kc_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("key_cache must be a metal tensor"),
        };
        let (vc_s, vc_l) = value_cache.storage_and_layout();
        let vc = match &*vc_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("value_cache must be a metal tensor"),
        };
        let (ko_s, ko_l) = k_out.storage_and_layout();
        let ko = match &*ko_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("k_out must be a metal tensor"),
        };
        let (vo_s, vo_l) = v_out.storage_and_layout();
        let vo = match &*vo_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("v_out must be a metal tensor"),
        };
        let (bt_s, bt_l) = block_table.storage_and_layout();
        let bt = match &*bt_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("block_table must be a metal tensor"),
        };
        let (cu_s, cu_l) = cu_seq_lens.storage_and_layout();
        let cu = match &*cu_s {
            Storage::Metal(s) => s,
            _ => candle_core::bail!("cu_seq_lens must be a metal tensor"),
        };

        // Scale buffers - guards must live as long as k_v_scale
        let ks_guard;
        let vs_guard;
        let k_v_scale = if let (Some(ks), Some(vs)) = (k_scale, v_scale) {
            ks_guard = ks.storage_and_layout();
            let ks = match &*ks_guard.0 {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("k_scale must be a metal tensor"),
            };
            vs_guard = vs.storage_and_layout();
            let vs = match &*vs_guard.0 {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("v_scale must be a metal tensor"),
            };
            Some((ks.buffer(), vs.buffer()))
        } else {
            None
        };

        let (_, block_table_stride) = bt_l.shape().dims2()?;

        let dev = key_cache.device().as_metal_device()?;
        let encoder = dev.command_encoder()?;
        encoder.set_label("gather-kv-cache");

        kernels::call_gather_kv_cache(
            dev.device(),
            &encoder,
            &kernels::Kernels::new(),
            cache_ty,
            out_ty,
            kc.buffer(),
            kc_l.start_offset() * cache_dtype.size_in_bytes(),
            vc.buffer(),
            vc_l.start_offset() * cache_dtype.size_in_bytes(),
            ko.buffer(),
            ko_l.start_offset() * out_dtype.size_in_bytes(),
            vo.buffer(),
            vo_l.start_offset() * out_dtype.size_in_bytes(),
            k_v_scale,
            bt.buffer(),
            bt_l.start_offset() * block_table.dtype().size_in_bytes(),
            cu.buffer(),
            cu_l.start_offset() * cu_seq_lens.dtype().size_in_bytes(),
            num_tokens as i32,
            num_seqs as i32,
            block_size as i32,
            block_table_stride as i32,
            num_kv_heads as i32,
            head_size as i32,
            x as i32,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    Ok((k_out, v_out))
}
