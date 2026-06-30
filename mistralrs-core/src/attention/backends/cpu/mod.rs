#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod elem;
mod full;
mod mask;
#[cfg(target_arch = "aarch64")]
mod neon;
mod prefetch;
mod single_q;
#[cfg(test)]
mod tests;
mod threading;

use candle_core::{Context, DType, Result, Storage, Tensor, WithDType};
use std::iter::Sum;

use crate::attention::SdpaParams;

use elem::ElemOps;
use mask::MaskInfo;

const SINGLE_Q_STACK_DV: usize = 256;

enum MaskData<'a> {
    Borrowed(&'a [f32]),
    Owned(Vec<f32>),
}

impl<'a> MaskData<'a> {
    fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed(data) => data,
            Self::Owned(data) => data,
        }
    }
}

#[derive(Clone, Copy)]
struct TensorView<'a, T> {
    data: &'a [T],
    dims: [usize; 4],
    stride: [usize; 4],
}

impl<'a, T> TensorView<'a, T> {
    fn new(data: &'a [T], tensor: &Tensor) -> Result<Self> {
        let dims = tensor.dims4()?;
        let stride = tensor.stride();
        if stride.len() != 4 {
            candle_core::bail!("Expected rank-4 CPU attention tensor.");
        }
        Ok(Self {
            data,
            dims: [dims.0, dims.1, dims.2, dims.3],
            stride: [stride[0], stride[1], stride[2], stride[3]],
        })
    }
}

struct CpuAttnCtx<'a, T> {
    q: TensorView<'a, T>,
    k: TensorView<'a, T>,
    v: TensorView<'a, T>,
    mask: Option<MaskInfo<'a>>,
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
}

fn cpu_mask_data(storage: &Storage, start_offset: usize, dtype: DType) -> Result<MaskData<'_>> {
    let Storage::Cpu(cpu) = storage else {
        candle_core::bail!("Expected CPU storage for mask");
    };

    match dtype {
        DType::F32 => {
            let data = cpu
                .as_slice::<f32>()
                .context("Expected f32 CPU storage for mask")?;
            Ok(MaskData::Borrowed(&data[start_offset..]))
        }
        DType::F16 => {
            let data = cpu
                .as_slice::<half::f16>()
                .context("Expected f16 CPU storage for mask")?;
            Ok(MaskData::Owned(
                data[start_offset..].iter().map(|v| v.to_f32()).collect(),
            ))
        }
        DType::BF16 => {
            let data = cpu
                .as_slice::<half::bf16>()
                .context("Expected bf16 CPU storage for mask")?;
            Ok(MaskData::Owned(
                data[start_offset..].iter().map(|v| v.to_f32()).collect(),
            ))
        }
        _ => candle_core::bail!("Unsupported CPU attention mask dtype {dtype:?}"),
    }
}

pub(in crate::attention) fn run_flash_attn_cpu<T>(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
) -> Result<Tensor>
where
    T: WithDType + Sum + num_traits::real::Real + ElemOps + Send + Sync + 'static,
{
    let (q_guard, q_layout) = q.storage_and_layout();
    let q_data: &[T] = if let Storage::Cpu(cpu) = &*q_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for q")?;
        &data[q_layout.start_offset()..]
    } else {
        candle_core::bail!("Expected CPU storage for q");
    };

    let (k_guard, k_layout) = k.storage_and_layout();
    let k_data: &[T] = if let Storage::Cpu(cpu) = &*k_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for k")?;
        &data[k_layout.start_offset()..]
    } else {
        candle_core::bail!("Expected CPU storage for k");
    };

    let (v_guard, v_layout) = v.storage_and_layout();
    let v_data: &[T] = if let Storage::Cpu(cpu) = &*v_guard {
        let data = cpu.as_slice::<T>().context("Expected CPU storage for v")?;
        &data[v_layout.start_offset()..]
    } else {
        candle_core::bail!("Expected CPU storage for v");
    };

    let mut mask_guard = None;
    let mut mask_dtype = None;
    let mut mask_start_offset = 0;
    let mut mask_dims = None;
    if let Some(mask_tensor) = mask {
        mask_dims = Some(mask_tensor.shape().dims().to_vec());
        let (guard, layout) = mask_tensor.storage_and_layout();
        mask_dtype = Some(mask_tensor.dtype());
        mask_start_offset = layout.start_offset();
        mask_guard = Some(guard);
    }
    let mask_data = match (mask_guard.as_deref(), mask_dtype) {
        (Some(storage), Some(dtype)) => Some(cpu_mask_data(storage, mask_start_offset, dtype)?),
        _ => None,
    };

    let q = TensorView::new(q_data, q)?;
    let k = TensorView::new(k_data, k)?;
    let v = TensorView::new(v_data, v)?;
    let mask = match (mask_data.as_ref(), mask_dims.as_deref()) {
        (Some(data), Some(dims)) => {
            Some(MaskInfo::new(data.as_slice(), dims, q.dims[0], q.dims[2]))
        }
        _ => None,
    };
    let ctx = CpuAttnCtx {
        q,
        k,
        v,
        mask,
        scale: sdpa_params.softmax_scale,
        max_bias: 0.0,
        logit_softcap: sdpa_params.softcap.unwrap_or(0.0),
    };

    if ctx.q.dims[1] == 1 {
        single_q::run(&ctx)
    } else {
        full::run(&ctx)
    }
}
