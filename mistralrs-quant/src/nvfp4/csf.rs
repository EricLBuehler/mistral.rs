use candle_core::{DType, Device, Result, Tensor};
use float8::F8E4M3;

use crate::{Shard, ShardedVarBuilder};

const SLAB_ROWS: usize = 16;
const CODES_PER_BYTE: usize = 2;
const CODE_BITS: usize = 4;
const CODE_MASK: u8 = 0x0f;
const MAX_BASE: u8 = u8::MAX - CODE_MASK;
const EXCEPTION_POSITION_BITS: usize = 24;
const EXCEPTION_POSITION_MASK: u32 = (1 << EXCEPTION_POSITION_BITS) - 1;
const FIXED_SUFFIX: &str = "nvfp4_csf_fixed";
const EXCEPTIONS_SUFFIX: &str = "nvfp4_csf_exceptions";

pub(super) fn contains(vb: &ShardedVarBuilder, name: &str) -> bool {
    let vb = vb.pp(name);
    vb.contains_tensor(FIXED_SUFFIX) || vb.contains_tensor(EXCEPTIONS_SUFFIX)
}

pub(super) fn load(
    vb: &ShardedVarBuilder,
    name: &str,
    shape: &[usize],
    shard: Shard,
) -> Result<Tensor> {
    let cols = shape[shape.len() - 1];
    let rows = shape[..shape.len() - 1].iter().product::<usize>();
    if !rows.is_multiple_of(SLAB_ROWS) || !cols.is_multiple_of(CODES_PER_BYTE) {
        candle_core::bail!(
            "NVFP4 CSF scales require rows divisible by 16 and even columns, got {shape:?}"
        );
    }
    let cpu_vb = vb.clone().set_device(Device::Cpu).pp(name);
    let fixed = cpu_vb.get_with_hints_dtype(
        (rows / SLAB_ROWS, SLAB_ROWS * (1 + cols / CODES_PER_BYTE)),
        FIXED_SUFFIX,
        Shard::default(),
        DType::U8,
    )?;
    let exceptions = cpu_vb.get_unchecked_dtype(EXCEPTIONS_SUFFIX, DType::U32)?;
    let scales = Tensor::from_vec(
        decode(
            &fixed.flatten_all()?.to_vec1::<u8>()?,
            &exceptions.flatten_all()?.to_vec1::<u32>()?,
            cols,
        )?,
        shape,
        &Device::Cpu,
    )?;
    let scales = match crate::shard_range(shard, shape)? {
        Some((axis, offset, len)) => scales.narrow(axis, offset, len)?,
        None => scales,
    };
    crate::utils::contiguous_fp8(&scales)?.to_device(vb.device())
}

// byte-window4-fixed-stream-u24-exceptions/1: https://github.com/vllm-project/vllm/pull/60646
fn decode(fixed: &[u8], exceptions: &[u32], cols: usize) -> Result<Vec<F8E4M3>> {
    let packed_cols = cols / CODES_PER_BYTE;
    let slab_bytes = SLAB_ROWS * (1 + packed_cols);
    let rows = fixed.len() / slab_bytes * SLAB_ROWS;
    let mut scales = Vec::with_capacity(rows * cols);
    for slab in fixed.chunks_exact(slab_bytes) {
        for (row, &base) in slab[..SLAB_ROWS].iter().enumerate() {
            if base > MAX_BASE {
                candle_core::bail!("NVFP4 CSF row base {base} exceeds {MAX_BASE}");
            }
            let start = SLAB_ROWS + row * packed_cols;
            for &codes in &slab[start..start + packed_cols] {
                scales.push(F8E4M3::from_bits(base + (codes & CODE_MASK)));
                scales.push(F8E4M3::from_bits(base + (codes >> CODE_BITS)));
            }
        }
    }
    let mut previous = None;
    for &exception in exceptions {
        let position = (exception & EXCEPTION_POSITION_MASK) as usize;
        if previous.is_some_and(|previous| position <= previous) {
            candle_core::bail!("NVFP4 CSF exception positions must be strictly increasing");
        }
        let Some(scale) = scales.get_mut(position) else {
            candle_core::bail!("NVFP4 CSF exception position {position} exceeds the scale matrix");
        };
        *scale = F8E4M3::from_bits((exception >> EXCEPTION_POSITION_BITS) as u8);
        previous = Some(position);
    }
    Ok(scales)
}

#[cfg(test)]
#[path = "csf_tests.rs"]
mod tests;
