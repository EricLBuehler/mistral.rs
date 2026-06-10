#![cfg(feature = "cuda")]

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn fa_acausal(q: &Tensor, k: &Tensor, v: &Tensor, softmax_scale: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

fn fa_acausal_softcap(q: &Tensor, k: &Tensor, v: &Tensor, softcap: f32) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    // let att = (q.matmul(&k.t()?)? * softmax_scale as f64)?;
    let att = q.matmul(&k.t()?)?;
    let att = (softcap as f64 * ((att / softcap as f64)?.tanh())?)?;
    let att = candle_nn::ops::softmax(&att, D::Minus1)?;
    // Convert to contiguous as matmul doesn't support strided vs for now.
    let output = att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?;
    Ok(output)
}

fn fa_windowed_mm_prefix(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window: usize,
    ranges: &[(usize, usize)],
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let (seq_len, n_heads, _) = q.dims3()?;
    let (_, n_kv_heads, _) = k.dims3()?;
    let groups = n_heads / n_kv_heads;
    let mut heads = Vec::with_capacity(n_heads);
    for head in 0..n_heads {
        let kv_head = head / groups;
        let q_h = q.i((.., head, ..))?.contiguous()?;
        let k_h = k.i((.., kv_head, ..))?.contiguous()?;
        let v_h = v.i((.., kv_head, ..))?.contiguous()?;
        let mut mask = Vec::with_capacity(seq_len * seq_len);
        for q_idx in 0..seq_len {
            for k_idx in 0..seq_len {
                let mm_prefix = ranges.iter().any(|&(start, end)| {
                    q_idx >= start && q_idx < end && k_idx >= start && k_idx < end
                });
                let future = k_idx > q_idx;
                let too_old = q_idx >= window && k_idx <= q_idx - window;
                mask.push(if (future || too_old) && !mm_prefix {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        let mask = Tensor::from_vec(mask, (seq_len, seq_len), q.device())?;
        let att = ((q_h.matmul(&k_h.t()?)? * softmax_scale as f64)? + mask)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        heads.push(att.matmul(&v_h.contiguous()?)?);
    }
    Ok(Tensor::stack(&heads, 1)?.to_dtype(in_dtype)?)
}

#[test]
fn flash_attn_acausal() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let ys1 = fa_acausal(&q, &k, &v, 0.5)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        mistralrs_flash_attn::flash_attn(&q, &k, &v, 0.5, false)?.transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys1, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );

    assert_eq!(ys2.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys2, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-5);
    Ok(())
}

#[test]
fn flash_attn_varlen_paged_mm_prefix_windowed() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let seq_len: usize = 296;
    let n_heads: usize = 8;
    let n_kv_heads: usize = 2;
    let head_dim: usize = 256;
    let block_size: usize = 32;
    let num_blocks = seq_len.div_ceil(block_size);
    let padded_len = num_blocks * block_size;
    let elem_count = seq_len * n_heads * head_dim;
    let q_data = (0..elem_count)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect::<Vec<_>>();
    let q =
        Tensor::from_vec(q_data, (seq_len, n_heads, head_dim), &device)?.to_dtype(DType::BF16)?;
    let kv_elem_count = seq_len * n_kv_heads * head_dim;
    let kv_data = (0..kv_elem_count)
        .map(|i| ((i % 193) as f32 - 96.0) / 96.0)
        .collect::<Vec<_>>();
    let kv = Tensor::from_vec(kv_data, (seq_len, n_kv_heads, head_dim), &device)?
        .to_dtype(DType::BF16)?;
    let k_seq = (&kv / 4.)?;
    let v_seq = (&kv / 5.)?;
    let pad = Tensor::zeros(
        (padded_len - seq_len, n_kv_heads, head_dim),
        DType::BF16,
        &device,
    )?;
    let k_paged =
        Tensor::cat(&[&k_seq, &pad], 0)?.reshape((num_blocks, block_size, n_kv_heads, head_dim))?;
    let v_paged =
        Tensor::cat(&[&v_seq, &pad], 0)?.reshape((num_blocks, block_size, n_kv_heads, head_dim))?;
    let seqlens = Tensor::new(&[0u32, seq_len as u32], &device)?;
    let block_table = Tensor::new((0..num_blocks as u32).collect::<Vec<_>>(), &device)?
        .reshape((1, num_blocks))?;
    let mm_prefix_ranges = Tensor::new(&[6i32, 262i32], &device)?.reshape((1, 1, 2))?;

    let ys_ref =
        fa_windowed_mm_prefix(&q, &k_seq, &v_seq, 1.0, 1024, &[(6, 262)])?.to_dtype(DType::F32)?;
    let ys_causal_ref =
        fa_windowed_mm_prefix(&q, &k_seq, &v_seq, 1.0, 1024, &[])?.to_dtype(DType::F32)?;
    let ys_causal = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
        &q,
        &k_paged,
        &v_paged,
        &seqlens,
        &seqlens,
        &block_table,
        None,
        seq_len,
        seq_len,
        1.0,
        Some(1024),
        Some(0),
        block_size,
        None,
    )?
    .to_dtype(DType::F32)?;
    let causal_diff = ys_causal_ref
        .sub(&ys_causal)?
        .abs()?
        .flatten_all()?
        .max(0)?;
    let causal_diff = causal_diff.to_vec0::<f32>()?.abs();
    assert!(causal_diff < 0.125, "causal max diff {causal_diff}");
    let ys = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
        &q,
        &k_paged,
        &v_paged,
        &seqlens,
        &seqlens,
        &block_table,
        Some(&mm_prefix_ranges),
        seq_len,
        seq_len,
        1.0,
        Some(1024),
        Some(0),
        block_size,
        None,
    )?
    .to_dtype(DType::F32)?;

    let diff = ys_ref.sub(&ys)?.abs()?.flatten_all()?.max(0)?;
    let diff = diff.to_vec0::<f32>()?.abs();
    assert!(diff < 0.125, "max diff {diff}");
    Ok(())
}

#[test]
fn flash_attn_varlen_paged_shuffled_block_table_hd256() -> Result<()> {
    paged_shuffled_block_table(256)
}

#[test]
fn flash_attn_varlen_paged_shuffled_block_table_hd512() -> Result<()> {
    paged_shuffled_block_table(512)
}

fn paged_shuffled_block_table(head_dim: usize) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let seq_len: usize = 296;
    let n_heads: usize = 8;
    let n_kv_heads: usize = 2;
    let block_size: usize = 32;
    let num_blocks = seq_len.div_ceil(block_size);
    // Scatter logical blocks across a larger pool in non-identity order so the kernel
    // must honor the block table; an identity table hides missing paged addressing.
    let pool_blocks = num_blocks + 7;
    let perm = (0..num_blocks)
        .map(|i| (i * 5 + 3) % pool_blocks)
        .collect::<Vec<_>>();
    let elem_count = seq_len * n_heads * head_dim;
    let q_data = (0..elem_count)
        .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
        .collect::<Vec<_>>();
    let q =
        Tensor::from_vec(q_data, (seq_len, n_heads, head_dim), &device)?.to_dtype(DType::BF16)?;
    let kv_elem_count = seq_len * n_kv_heads * head_dim;
    let kv_data = (0..kv_elem_count)
        .map(|i| ((i % 193) as f32 - 96.0) / 96.0)
        .collect::<Vec<_>>();
    let kv = Tensor::from_vec(kv_data, (seq_len, n_kv_heads, head_dim), &device)?
        .to_dtype(DType::BF16)?;
    let k_seq = (&kv / 4.)?;
    let v_seq = (&kv / 5.)?;

    let block_elems = block_size * n_kv_heads * head_dim;
    let scatter = |seq: &Tensor| -> Result<Tensor> {
        let data = seq.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut pool = vec![0f32; pool_blocks * block_elems];
        for (logical, &physical) in perm.iter().enumerate() {
            let src = logical * block_elems;
            let len = block_elems.min(data.len() - src);
            pool[physical * block_elems..physical * block_elems + len]
                .copy_from_slice(&data[src..src + len]);
        }
        Ok(Tensor::from_vec(
            pool,
            (pool_blocks, block_size, n_kv_heads, head_dim),
            &device,
        )?
        .to_dtype(DType::BF16)?)
    };
    let k_paged = scatter(&k_seq)?;
    let v_paged = scatter(&v_seq)?;

    let seqlens = Tensor::new(&[0u32, seq_len as u32], &device)?;
    let block_table = Tensor::new(perm.iter().map(|&b| b as u32).collect::<Vec<_>>(), &device)?
        .reshape((1, num_blocks))?;

    let ys_ref = fa_windowed_mm_prefix(&q, &k_seq, &v_seq, 1.0, 1024, &[])?.to_dtype(DType::F32)?;
    let ys = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
        &q,
        &k_paged,
        &v_paged,
        &seqlens,
        &seqlens,
        &block_table,
        None,
        seq_len,
        seq_len,
        1.0,
        Some(1024),
        Some(0),
        block_size,
        None,
    )?
    .to_dtype(DType::F32)?;

    let diff = ys_ref.sub(&ys)?.abs()?.flatten_all()?.max(0)?;
    let diff = diff.to_vec0::<f32>()?.abs();
    assert!(diff < 0.125, "max diff {diff}");
    Ok(())
}

#[test]
fn flash_attn_acausal_softcap() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 3 * 5 * 8, &device)?
        .to_dtype(DType::F16)?
        .reshape((1, 3, 5, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;
    let softcap = 5.0f32;

    let ys1 = fa_acausal_softcap(&q, &k, &v, softcap)?;
    let ys1 = ys1.i(0)?.to_dtype(DType::F32)?;
    let ys2 = {
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        mistralrs_flash_attn::flash_attn_alibi_windowed_softcap(
            &q, &k, &v, None,    //  alibi_slopes //
            1.0,     // softmax //
            None,    // window_size_left //
            None,    // window_size_right //
            softcap, // softcap //
        )?
        .transpose(1, 2)?
    };
    let ys2 = ys2.i(0)?.to_dtype(DType::F32)?;
    let diff = ys1.sub(&ys2)?.abs()?.flatten_all()?.max(0)?;

    assert_eq!(ys1.dims(), &[3, 5, 8]);
    assert_eq!(ys2.dims(), &[3, 5, 8]);
    assert!(diff.to_vec0::<f32>()?.abs() < 1e-3);
    Ok(())
}

#[test]
fn flash_attn_varlen() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let q = Tensor::arange(0u32, 48, &device)?
        .to_dtype(DType::F16)?
        .reshape((3, 2, 8))?;
    let k = (&q / 40.)?;
    let v = (&q / 50.)?;
    let q = (&q / 30.)?;

    let seqlens_q = Tensor::new(&[0u32, 2u32], &device)?;
    let seqlens_k = Tensor::new(&[0u32, 2u32], &device)?;

    let ys = {
        let q = q.transpose(0, 1)?;
        let k = k.transpose(0, 1)?;
        let v = v.transpose(0, 1)?;
        mistralrs_flash_attn::flash_attn_varlen(
            &q, &k, &v, &seqlens_q, &seqlens_k, 32, 32, 0.5, false,
        )?
        .transpose(0, 1)?
    };
    let ys = ys.to_dtype(DType::F32)?;

    assert_eq!(ys.dims(), &[3, 2, 8]);
    assert_eq!(
        to_vec3_round(ys, 4)?,
        &[
            [
                [0.0837, 0.1038, 0.1238, 0.1438, 0.1637, 0.1837, 0.2037, 0.2238],
                [0.0922, 0.1122, 0.1322, 0.1522, 0.1721, 0.1921, 0.2122, 0.2322]
            ],
            [
                [0.4204, 0.4404, 0.4604, 0.4805, 0.5005, 0.5205, 0.5405, 0.5605],
                [0.428, 0.448, 0.468, 0.488, 0.5083, 0.5283, 0.5483, 0.5684]
            ],
            [
                [0.7554, 0.7754, 0.7954, 0.8154, 0.8354, 0.8555, 0.8755, 0.8955],
                [0.7622, 0.7822, 0.8022, 0.8223, 0.8423, 0.8623, 0.8823, 0.9023]
            ]
        ]
    );
    Ok(())
}

fn fa_ref_prefix(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    window: Option<usize>,
) -> Result<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let (q_len, n_heads, _) = q.dims3()?;
    let (kv_len, n_kv_heads, _) = k.dims3()?;
    let groups = n_heads / n_kv_heads;
    let prefix = kv_len - q_len;
    let mut heads = Vec::with_capacity(n_heads);
    for head in 0..n_heads {
        let kv_head = head / groups;
        let q_h = q.i((.., head, ..))?.contiguous()?;
        let k_h = k.i((.., kv_head, ..))?.contiguous()?;
        let v_h = v.i((.., kv_head, ..))?.contiguous()?;
        let mut mask = Vec::with_capacity(q_len * kv_len);
        for q_idx in 0..q_len {
            let pos = prefix + q_idx;
            for k_idx in 0..kv_len {
                let future = k_idx > pos;
                // flash-attn window_size_left=w allows [pos - w, pos]
                let too_old = window.is_some_and(|w| pos >= w && k_idx < pos - w);
                mask.push(if future || too_old {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        let mask = Tensor::from_vec(mask, (q_len, kv_len), q.device())?;
        let att = ((q_h.matmul(&k_h.t()?)? * softmax_scale as f64)? + mask)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        heads.push(att.matmul(&v_h.contiguous()?)?);
    }
    Ok(Tensor::stack(&heads, 1)?.to_dtype(in_dtype)?)
}

struct PagedSeq {
    q_len: usize,
    kv_len: usize,
}

fn paged_prefix_case(head_dim: usize, seqs: &[PagedSeq], window: Option<usize>) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let n_heads: usize = 8;
    let n_kv_heads: usize = 2;
    let block_size: usize = 32;
    let block_elems = block_size * n_kv_heads * head_dim;
    let blocks_per_seq: Vec<usize> = seqs.iter().map(|s| s.kv_len.div_ceil(block_size)).collect();
    let total_blocks: usize = blocks_per_seq.iter().sum();
    let pool_blocks = total_blocks + 5;
    // Non-identity, collision-free physical placement covering all sequences.
    let perm: Vec<usize> = (0..total_blocks).map(|i| pool_blocks - 1 - i).collect();

    let mut q_parts = Vec::new();
    let mut k_parts = Vec::new();
    let mut v_parts = Vec::new();
    let mut refs = Vec::new();
    let mut pool_k = vec![0f32; pool_blocks * block_elems];
    let mut pool_v = vec![0f32; pool_blocks * block_elems];
    let mut block_rows = Vec::new();
    let mut cu_q = vec![0u32];
    let mut cu_k = vec![0u32];
    let mut next_logical = 0usize;
    for (seq_idx, seq) in seqs.iter().enumerate() {
        let q_elems = seq.q_len * n_heads * head_dim;
        let q_data: Vec<f32> = (0..q_elems)
            .map(|i| (((i + seq_idx * 7919) % 251) as f32 - 125.0) / 125.0)
            .collect();
        let q = Tensor::from_vec(q_data, (seq.q_len, n_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let kv_elems = seq.kv_len * n_kv_heads * head_dim;
        let kv_data: Vec<f32> = (0..kv_elems)
            .map(|i| (((i + seq_idx * 104729) % 193) as f32 - 96.0) / 96.0)
            .collect();
        let kv = Tensor::from_vec(kv_data, (seq.kv_len, n_kv_heads, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = (&kv / 4.)?;
        let v = (&kv / 5.)?;
        refs.push(fa_ref_prefix(&q, &k, &v, 1.0, window)?.to_dtype(DType::F32)?);

        let k_data = k.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let v_data = v.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut row = Vec::new();
        for logical in 0..blocks_per_seq[seq_idx] {
            let physical = perm[next_logical + logical];
            row.push(physical as u32);
            let src = logical * block_elems;
            let len = block_elems.min(k_data.len() - src);
            pool_k[physical * block_elems..physical * block_elems + len]
                .copy_from_slice(&k_data[src..src + len]);
            pool_v[physical * block_elems..physical * block_elems + len]
                .copy_from_slice(&v_data[src..src + len]);
        }
        next_logical += blocks_per_seq[seq_idx];
        block_rows.push(row);
        cu_q.push(cu_q.last().unwrap() + seq.q_len as u32);
        cu_k.push(cu_k.last().unwrap() + seq.kv_len as u32);
        q_parts.push(q);
        k_parts.push(k);
        v_parts.push(v);
    }

    let q_all = Tensor::cat(&q_parts, 0)?;
    let k_pool = Tensor::from_vec(
        pool_k,
        (pool_blocks, block_size, n_kv_heads, head_dim),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let v_pool = Tensor::from_vec(
        pool_v,
        (pool_blocks, block_size, n_kv_heads, head_dim),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let max_blocks = blocks_per_seq.iter().copied().max().unwrap();
    let mut table_flat = Vec::new();
    for row in &block_rows {
        table_flat.extend(row.iter().copied());
        table_flat.extend(std::iter::repeat_n(0u32, max_blocks - row.len()));
    }
    let block_table = Tensor::from_vec(table_flat, (seqs.len(), max_blocks), &device)?;
    let cu_q_t = Tensor::new(&cu_q[..], &device)?;
    let cu_k_t = Tensor::new(&cu_k[..], &device)?;
    let max_q = seqs.iter().map(|s| s.q_len).max().unwrap();
    let max_k = seqs.iter().map(|s| s.kv_len).max().unwrap();

    let ys = mistralrs_flash_attn::flash_attn_varlen_paged_windowed(
        &q_all,
        &k_pool,
        &v_pool,
        &cu_q_t,
        &cu_k_t,
        &block_table,
        None,
        max_q,
        max_k,
        1.0,
        window,
        Some(0),
        block_size,
        None,
    )?
    .to_dtype(DType::F32)?;

    let ys_ref = Tensor::cat(&refs, 0)?;
    let diff = ys_ref.sub(&ys)?.abs()?.flatten_all()?.max(0)?;
    let diff = diff.to_vec0::<f32>()?.abs();
    assert!(diff < 0.125, "head_dim={head_dim} max diff {diff}");
    Ok(())
}

#[test]
fn flash_attn_varlen_paged_cached_context_windowed_hd256() -> Result<()> {
    // Chunked-prefill shape: 64 new queries over 296 total kv, binding sliding window.
    paged_prefix_case(
        256,
        &[PagedSeq {
            q_len: 64,
            kv_len: 296,
        }],
        Some(128),
    )
}

#[test]
fn flash_attn_varlen_paged_cached_context_hd512() -> Result<()> {
    paged_prefix_case(
        512,
        &[PagedSeq {
            q_len: 64,
            kv_len: 296,
        }],
        None,
    )
}

#[test]
fn flash_attn_varlen_paged_unequal_batch_hd256() -> Result<()> {
    paged_prefix_case(
        256,
        &[
            PagedSeq {
                q_len: 37,
                kv_len: 37,
            },
            PagedSeq {
                q_len: 64,
                kv_len: 200,
            },
        ],
        None,
    )
}

#[test]
fn flash_attn_varlen_paged_unequal_batch_hd512() -> Result<()> {
    paged_prefix_case(
        512,
        &[
            PagedSeq {
                q_len: 37,
                kv_len: 37,
            },
            PagedSeq {
                q_len: 64,
                kv_len: 200,
            },
        ],
        None,
    )
}

#[test]
fn flash_attn_varlen_paged_padded_head_dim_hd96() -> Result<()> {
    // 96 pads up to the 128-wide kernel bucket (Is_even_K = false path).
    paged_prefix_case(
        96,
        &[PagedSeq {
            q_len: 64,
            kv_len: 296,
        }],
        Some(128),
    )
}

#[test]
fn flash_attn_varlen_paged_padded_head_dim_hd160() -> Result<()> {
    // 160 pads up to the 256-wide kernel bucket.
    paged_prefix_case(
        160,
        &[
            PagedSeq {
                q_len: 37,
                kv_len: 37,
            },
            PagedSeq {
                q_len: 64,
                kv_len: 200,
            },
        ],
        None,
    )
}
