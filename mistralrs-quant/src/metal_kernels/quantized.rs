use candle_core::DType;
use metal::{Buffer, ComputeCommandEncoderRef, Device, MTLSize};

use crate::set_params;

use super::{utils::EncoderProvider, Kernels, MetalKernelError};

#[allow(clippy::too_many_arguments)]
fn qmm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_shape: &[usize],
    x_stride: &[usize],
    w: &Buffer,
    w_shape: &[usize],
    w_stride: &[usize],
    scales: &Buffer,
    s_stride: &[usize],
    biases: &Buffer,
    b_stride: &[usize],
    out: &Buffer,
    out_shape: &[usize],
    transpose: bool,
    bits: usize,
    group_size: usize,
    (m, n, k): (usize, usize, usize),
) -> Result<(), MetalKernelError> {
    let type_string = match ty {
        DType::F32 => "float",
        DType::BF16 => "bfloat16_t",
        DType::F16 => "float16_t",
        other => {
            return Err(MetalKernelError::DTypeMismatch {
                expected: vec![DType::F32, DType::F16, DType::BF16],
                got: other,
            })
        }
    };

    let b = out_shape.iter().product::<usize>() / m / n;

    let wm = 2;
    let wn = 2;
    let bm = 32;
    let bn = 32;

    let group_dims = MTLSize {
        width: 32,
        height: wn,
        depth: wm,
    };
    let grid_dims = MTLSize {
        width: n.div_ceil(bn) as u64,
        height: m.div_ceil(bm) as u64,
        depth: b as u64,
    };

    let aligned = n % 32 == 0;
    let batched = b > 1;
    let name = format!(
        "{}_{type_string}_gs_{group_size}_b_{bits}_{}_{}",
        if transpose { "qmm_t" } else { "qmm_n" },
        if transpose {
            if aligned {
                "alN_true"
            } else {
                "alN_false"
            }
        } else {
            ""
        },
        if batched { "batch_1" } else { "batch_0" }
    );

    let pipeline = kernels.load_pipeline(device, &name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (w, scales, biases, x, out, k as i32, n as i32, m as i32)
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_afq_qmm_new(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    ty: DType,
    x: &Buffer,
    x_shape: &[usize],
    x_stride: &[usize],
    w: &Buffer,
    w_shape: &[usize],
    w_stride: &[usize],
    scales: &Buffer,
    s_stride: &[usize],
    biases: &Buffer,
    b_stride: &[usize],
    out: &Buffer,
    out_shape: &[usize],
    transpose: bool,
    bits: usize,
    group_size: usize,
    x_row_contiguous: bool,
) -> Result<(), MetalKernelError> {
    let non_batched = w_shape.len() == 2 && x_row_contiguous;
    let k = x_shape[x_shape.len() - 1];
    let m = if non_batched {
        x_shape.iter().product::<usize>() / k
    } else {
        x_shape[x_shape.len() - 2]
    };
    let n = out_shape[out_shape.len() - 1];

    let vector_limit = if transpose { 18 } else { 4 };

    // It is a matrix matrix product
    if m >= vector_limit {
        todo!("qmm");
        return Ok(());
    }

    // It is a qmv with a small inner dimension so route to qmv_quad kernel
    if transpose && (k == 128 || k == 64) && bits.is_power_of_two() {
        todo!("qmv_quad");
        return Ok(());
    }

    // Run of the mill qmv
    if transpose {
        todo!("qmv");
        return Ok(());
    }

    // Run of the mill qvm
    if k < 1024 {
        todo!("qvm");
        return Ok(());
    }

    todo!("qvm splitk");
    return Ok(());
}
