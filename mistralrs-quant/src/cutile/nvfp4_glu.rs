#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::cutile_compiler;

    const BLOCK: i32 = 16;
    const FP4_MAX: f32 = 6.0;
    const FP4_MIN: f32 = -6.0;
    const FP8_MAX: f32 = 448.0;
    const SILU: i32 = 0;
    const RELU: i32 = 2;
    const LOG2_E: f32 = std::f32::consts::LOG2_E;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn quantize_bf16<
        const BM: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const ACTIVATION: i32,
    >(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        gate: *mut bf16,
        value: *mut bf16,
        rows: i32,
        columns: i32,
        scale_stride: i32,
        gate_stride: i64,
        value_stride: i64,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            let row: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
            let column: Tile<i32, { [BK] }> =
                iota(const_shape![BK]) + broadcast_scalar(kg * BK, const_shape![BK]);
            let row64: Tile<i64, { [BM] }> = exti(row);
            let column64: Tile<i64, { [BK] }> = exti(column);
            let gate_row = row64 * broadcast_scalar(gate_stride, const_shape![BM]);
            let value_row = row64 * broadcast_scalar(value_stride, const_shape![BM]);
            let gate_offsets = gate_row
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                + column64
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let value_offsets = value_row
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                + column64
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                & lt_tile(column, broadcast_scalar(columns, const_shape![BK]))
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let gp: PointerTile<*mut bf16, { [] }> = pointer_to_tile(gate);
            let gp: PointerTile<*mut bf16, { [1, 1] }> = gp.reshape(const_shape![1, 1]);
            let gp: PointerTile<*mut bf16, { [BM, BK] }> = gp.broadcast(const_shape![BM, BK]);
            let vp: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value);
            let vp: PointerTile<*mut bf16, { [1, 1] }> = vp.reshape(const_shape![1, 1]);
            let vp: PointerTile<*mut bf16, { [BM, BK] }> = vp.broadcast(const_shape![BM, BK]);
            let gp: PointerTile<*mut bf16, { [BM, BK] }> = gp.offset_tile(gate_offsets);
            let vp: PointerTile<*mut bf16, { [BM, BK] }> = vp.offset_tile(value_offsets);
            let (gt, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                gp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            );
            let (vt, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                vp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            );
            let zeros: Tile<f32, { [BM, BK] }> = constant(0.0f32, const_shape![BM, BK]);
            let gt: Tile<f32, { [BM, BK] }> = convert_tile(gt);
            let gt = select(mask, gt, zeros);
            let vt: Tile<f32, { [BM, BK] }> = convert_tile(vt);
            let vt = select(mask, vt, zeros);
            let activated = if ACTIVATION == RELU {
                max_tile(gt, zeros)
            } else {
                let log2e: Tile<f32, { [BM, BK] }> = constant(LOG2_E, const_shape![BM, BK]);
                let exponent = mulf(negf(gt), log2e, rounding::NearestEven, ftz::Enabled);
                let exponent = exp2(exponent, ftz::Enabled);
                let ones: Tile<f32, { [BM, BK] }> = constant(1.0f32, const_shape![BM, BK]);
                let denominator = addf(ones, exponent, rounding::NearestEven, ftz::Enabled);
                if ACTIVATION == SILU {
                    divf(gt, denominator, rounding::Approx, ftz::Enabled)
                } else {
                    divf(ones, denominator, rounding::Approx, ftz::Enabled)
                }
            };
            // Both casts preserve the existing activation-then-product rounding boundaries.
            let activated: Tile<bf16, { [BM, BK] }> = convert_tile(activated);
            let vt: Tile<bf16, { [BM, BK] }> = convert_tile(vt);
            let product: Tile<bf16, { [BM, BK] }> = activated * vt;
            let xf: Tile<f32, { [BM, BK] }> = convert_tile(product);
            let xf = xf
                / global
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![BM, BK]);
            let blocks: Tile<f32, { [BM, SK, BLOCK] }> = xf.reshape(const_shape![BM, SK, BLOCK]);
            let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
            let maxima = maxima.reshape(const_shape![BM, SK]);
            let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
            let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
            let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
            let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
            let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
            let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
            let raw_scale = min_tile(maxima / fp4_max, fp8_max);
            let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
            let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
            let denominator = select(eq_tile(rounded, zeros), ones, rounded);
            let denominator = denominator
                .reshape(const_shape![BM, SK, 1])
                .broadcast(const_shape![BM, SK, BLOCK]);
            let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
            let normalized = max_tile(min_tile(normalized, upper), lower);
            let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
            let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
            q.store(packed, idx);
            let col: Tile<i32, { [SK] }> =
                iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
            let stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(scale_stride, const_shape![BM]));
            let col_offset: Tile<i64, { [SK] }> = exti(col);
            let offset = (row64 * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                + col_offset
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
            let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> = base.reshape(const_shape![1, 1]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                base.broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
            store_ptr_tko(
                base,
                sx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn quantize_f16<
        const BM: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const ACTIVATION: i32,
    >(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        gate: *mut f16,
        value: *mut f16,
        rows: i32,
        columns: i32,
        scale_stride: i32,
        gate_stride: i64,
        value_stride: i64,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            let row: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
            let column: Tile<i32, { [BK] }> =
                iota(const_shape![BK]) + broadcast_scalar(kg * BK, const_shape![BK]);
            let row64: Tile<i64, { [BM] }> = exti(row);
            let column64: Tile<i64, { [BK] }> = exti(column);
            let gate_row = row64 * broadcast_scalar(gate_stride, const_shape![BM]);
            let value_row = row64 * broadcast_scalar(value_stride, const_shape![BM]);
            let gate_offsets = gate_row
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                + column64
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let value_offsets = value_row
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                + column64
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BK])
                & lt_tile(column, broadcast_scalar(columns, const_shape![BK]))
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![BM, BK]);
            let gp: PointerTile<*mut f16, { [] }> = pointer_to_tile(gate);
            let gp: PointerTile<*mut f16, { [1, 1] }> = gp.reshape(const_shape![1, 1]);
            let gp: PointerTile<*mut f16, { [BM, BK] }> = gp.broadcast(const_shape![BM, BK]);
            let vp: PointerTile<*mut f16, { [] }> = pointer_to_tile(value);
            let vp: PointerTile<*mut f16, { [1, 1] }> = vp.reshape(const_shape![1, 1]);
            let vp: PointerTile<*mut f16, { [BM, BK] }> = vp.broadcast(const_shape![BM, BK]);
            let gp: PointerTile<*mut f16, { [BM, BK] }> = gp.offset_tile(gate_offsets);
            let vp: PointerTile<*mut f16, { [BM, BK] }> = vp.offset_tile(value_offsets);
            let (gt, _): (Tile<f16, { [BM, BK] }>, Token) = load_ptr_tko(
                gp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            );
            let (vt, _): (Tile<f16, { [BM, BK] }>, Token) = load_ptr_tko(
                vp,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                None,
                Latency::<0>,
            );
            let zeros: Tile<f32, { [BM, BK] }> = constant(0.0f32, const_shape![BM, BK]);
            let gt: Tile<f32, { [BM, BK] }> = convert_tile(gt);
            let gt = select(mask, gt, zeros);
            let vt: Tile<f32, { [BM, BK] }> = convert_tile(vt);
            let vt = select(mask, vt, zeros);
            let activated = if ACTIVATION == RELU {
                max_tile(gt, zeros)
            } else {
                let log2e: Tile<f32, { [BM, BK] }> = constant(LOG2_E, const_shape![BM, BK]);
                let exponent = mulf(negf(gt), log2e, rounding::NearestEven, ftz::Enabled);
                let exponent = exp2(exponent, ftz::Enabled);
                let ones: Tile<f32, { [BM, BK] }> = constant(1.0f32, const_shape![BM, BK]);
                let denominator = addf(ones, exponent, rounding::NearestEven, ftz::Enabled);
                if ACTIVATION == SILU {
                    divf(gt, denominator, rounding::Approx, ftz::Enabled)
                } else {
                    divf(ones, denominator, rounding::Approx, ftz::Enabled)
                }
            };
            // Both casts preserve the existing activation-then-product rounding boundaries.
            let activated: Tile<f16, { [BM, BK] }> = convert_tile(activated);
            let vt: Tile<f16, { [BM, BK] }> = convert_tile(vt);
            let product: Tile<f16, { [BM, BK] }> = activated * vt;
            let xf: Tile<f32, { [BM, BK] }> = convert_tile(product);
            let xf = xf
                / global
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![BM, BK]);
            let blocks: Tile<f32, { [BM, SK, BLOCK] }> = xf.reshape(const_shape![BM, SK, BLOCK]);
            let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
            let maxima = maxima.reshape(const_shape![BM, SK]);
            let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
            let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
            let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
            let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
            let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
            let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
            let raw_scale = min_tile(maxima / fp4_max, fp8_max);
            let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
            let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
            let denominator = select(eq_tile(rounded, zeros), ones, rounded);
            let denominator = denominator
                .reshape(const_shape![BM, SK, 1])
                .broadcast(const_shape![BM, SK, BLOCK]);
            let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
            let normalized = max_tile(min_tile(normalized, upper), lower);
            let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
            let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
            q.store(packed, idx);
            let col: Tile<i32, { [SK] }> =
                iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
            let stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(scale_stride, const_shape![BM]));
            let col_offset: Tile<i64, { [SK] }> = exti(col);
            let offset = (row64 * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                + col_offset
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
            let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> = base.reshape(const_shape![1, 1]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                base.broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
            store_ptr_tko(
                base,
                sx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }
}

use std::sync::Arc;
#[cfg(test)]
use std::sync::Mutex;

use candle_core::{CudaStorage, DType, Device, Layout, Result, Shape, Storage, Tensor};
use cutile::core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::{contains_cuda_function, CompileOptions, TileKernel};
use float8::F8E4M3;
use half::{bf16, f16};

use super::nvfp4::nvfp4_supported;
use super::{catch_cutile_panic, context, device_compute_capability, device_multiprocessor_count};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};
use crate::GluActivationType;

const BLOCK_SIZE: usize = 16;
const QUANT_ROWS: usize = 4;
const QUANT_K: usize = 256;
const BLOCKS_PER_SM: usize = 2;
const TUNED_COMPUTE_CAPABILITY: (i32, i32) = (12, 1);
const TUNED_OCCUPANCY: i32 = 2;
const MAX_DIMENSION: usize = i32::MAX as usize;
const WARMUP_ROWS: [usize; 5] = [1, 2, 4, 8, 16];
const WARMUP_ACTIVATIONS: [GluActivationType; 3] = [
    GluActivationType::Silu,
    GluActivationType::Relu,
    GluActivationType::Sigmoid,
];

pub(crate) struct GluQuantArgs<'a> {
    pub gate: &'a Tensor,
    pub value: &'a Tensor,
    pub activation_global_scale: &'a Tensor,
    pub activation: GluActivationType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixLayout {
    rows: usize,
    columns: usize,
    row_stride: usize,
}

fn matrix_layout(layout: &Layout) -> Option<MatrixLayout> {
    let dims = layout.dims();
    let (&columns, batch) = dims.split_last()?;
    let strides = layout.stride();
    if *strides.last()? != 1 {
        return None;
    }
    let rows = batch.iter().try_fold(1usize, |a, &b| a.checked_mul(b))?;
    let row_stride = if batch.is_empty() {
        columns
    } else {
        strides[strides.len() - 2]
    };
    if row_stride < columns {
        return None;
    }
    for axis in 0..dims.len().saturating_sub(2) {
        if strides[axis] != dims[axis + 1].checked_mul(strides[axis + 1])? {
            return None;
        }
    }
    if rows == 0 || columns == 0 || rows > MAX_DIMENSION || columns > MAX_DIMENSION {
        return None;
    }
    let final_offset = (rows - 1)
        .checked_mul(row_stride)?
        .checked_add(columns - 1)?;
    i64::try_from(final_offset).ok()?;
    i64::try_from(row_stride).ok()?;
    Some(MatrixLayout {
        rows,
        columns,
        row_stride,
    })
}

pub(crate) fn launch(
    args: GluQuantArgs<'_>,
    compile_only: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    if !matches!(
        args.activation,
        GluActivationType::Silu | GluActivationType::Relu | GluActivationType::Sigmoid
    ) || !matches!(args.gate.dtype(), DType::BF16 | DType::F16)
        || args.gate.dtype() != args.value.dtype()
        || args.gate.shape() != args.value.shape()
        || !args.gate.device().same_device(args.value.device())
    {
        return Ok(None);
    }
    let Device::Cuda(dev) = args.gate.device() else {
        return Ok(None);
    };
    if !nvfp4_supported(dev) {
        return Ok(None);
    }
    if args.activation_global_scale.dtype() != DType::F32
        || args.activation_global_scale.elem_count() != 1
        || !args
            .gate
            .device()
            .same_device(args.activation_global_scale.device())
    {
        candle_core::bail!("NVFP4 GLU global scale must be an F32 scalar on the input device")
    }
    let (gate_storage, gate_layout) = args.gate.storage_and_layout();
    let (value_storage, value_layout) = args.value.storage_and_layout();
    let (global_storage, global_layout) = args.activation_global_scale.storage_and_layout();
    let (Some(gate_matrix), Some(value_matrix)) =
        (matrix_layout(gate_layout), matrix_layout(value_layout))
    else {
        return Ok(None);
    };
    let MatrixLayout {
        rows,
        columns,
        row_stride: gate_stride,
    } = gate_matrix;
    if !columns.is_multiple_of(BLOCK_SIZE) {
        return Ok(None);
    }
    let (Storage::Cuda(gate_cuda), Storage::Cuda(value_cuda), Storage::Cuda(global_cuda)) =
        (&*gate_storage, &*value_storage, &*global_storage)
    else {
        candle_core::bail!("NVFP4 GLU inputs must use CUDA storage")
    };
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let compile_options = if device_compute_capability(dev) == TUNED_COMPUTE_CAPABILITY {
        CompileOptions::default().occupancy(TUNED_OCCUPANCY)
    } else {
        CompileOptions::default()
    };
    let (global_address, _global_guard) = slice_ptr_on_stream(
        global_cuda.as_cuda_slice::<f32>()?,
        global_layout.start_offset(),
        &stream,
    );
    let global = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            global_address as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    });
    let packed_columns = columns / 2;
    let scale_columns = columns / BLOCK_SIZE;
    let mut packed = unsafe { dev.alloc::<u8>(rows * packed_columns)? };
    let mut scales = unsafe { dev.alloc::<F8E4M3>(rows * scale_columns)? };
    let (packed_address, packed_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
    let (scale_address, scale_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
    let blocks = (BLOCKS_PER_SM * device_multiprocessor_count(dev))
        .min(rows.div_ceil(QUANT_ROWS) * columns.div_ceil(QUANT_K)) as u32;
    let generics = vec![
        QUANT_ROWS.to_string(),
        QUANT_K.to_string(),
        (QUANT_K / 2).to_string(),
        (QUANT_K / BLOCK_SIZE).to_string(),
        (args.activation as i32).to_string(),
    ];
    macro_rules! run {
        ($dtype:ty, $kernel:path) => {{
            let (gate_address, _gate_guard) = slice_ptr_on_stream(
                gate_cuda.as_cuda_slice::<$dtype>()?,
                gate_layout.start_offset(),
                &stream,
            );
            let (value_address, _value_guard) = slice_ptr_on_stream(
                value_cuda.as_cuda_slice::<$dtype>()?,
                value_layout.start_offset(),
                &stream,
            );
            let make_launcher = || {
                let output = unsafe {
                    cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                        packed_address as CUdeviceptr,
                        ordinal,
                        vec![rows as i32, packed_columns as i32],
                        vec![packed_columns as i32, 1],
                    )
                };
                let output = output
                    .partition([QUANT_ROWS, QUANT_K / 2])
                    .map([1, 1], blocks);
                unsafe {
                    $kernel(
                        output,
                        DevicePointer::<f8e4m3fn>::from_cu_deviceptr(scale_address as CUdeviceptr),
                        DevicePointer::<$dtype>::from_cu_deviceptr(gate_address as CUdeviceptr),
                        DevicePointer::<$dtype>::from_cu_deviceptr(value_address as CUdeviceptr),
                        rows as i32,
                        columns as i32,
                        scale_columns as i32,
                        gate_stride as i64,
                        value_matrix.row_stride as i64,
                        global.clone(),
                    )
                }
                .generics(generics.clone())
                .compile_options(compile_options.clone())
            };
            if compile_only {
                catch_cutile_panic("NVFP4 GLU compile", || {
                    make_launcher().compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::msg(format!(
                            "cuTile NVFP4 GLU compile failed: {error:?}"
                        ))
                    })
                })?;
            } else {
                let key = catch_cutile_panic("NVFP4 GLU specialization", || {
                    make_launcher()
                        .l1_cache_key_on(&cutile_stream)
                        .map_err(|error| {
                            candle_core::Error::msg(format!(
                                "cuTile NVFP4 GLU specialization failed: {error:?}"
                            ))
                        })
                })?;
                // Never compile a missing specialization during inference or graph capture.
                if !contains_cuda_function(&key) {
                    return Ok(None);
                }
                catch_cutile_panic("NVFP4 GLU launch", || unsafe {
                    make_launcher().async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::msg(format!(
                            "cuTile NVFP4 GLU launch failed: {error:?}"
                        ))
                    })
                })?;
            }
        }};
    }
    match args.gate.dtype() {
        DType::BF16 => run!(bf16, kernels::quantize_bf16),
        DType::F16 => run!(f16, kernels::quantize_f16),
        _ => unreachable!(),
    }
    drop(packed_guard);
    drop(scale_guard);
    let packed = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(packed, dev.clone())),
        Shape::from_dims(&[rows, packed_columns]),
    ));
    let scales = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, dev.clone())),
        Shape::from_dims(&[rows, scale_columns]),
    ));
    Ok(Some((packed, scales)))
}

pub(super) fn warm_common(
    activation_global_scale: &Tensor,
    dtype: DType,
    columns: usize,
) -> Result<()> {
    for rows in WARMUP_ROWS {
        let packed = Tensor::zeros((rows, columns * 2), dtype, activation_global_scale.device())?;
        let gate = packed.narrow(1, 0, columns)?;
        let value = packed.narrow(1, columns, columns)?;
        for activation in WARMUP_ACTIVATIONS {
            launch(
                GluQuantArgs {
                    gate: &gate,
                    value: &value,
                    activation_global_scale,
                    activation,
                },
                true,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

    const CASES: [(usize, usize, usize, usize); 6] = [
        (1, 16, 0, 0),
        (3, 80, 3, 1),
        (4, 272, 5, 2),
        (5, 4112, 0, 0),
        (17, 80, 9, 4),
        (129, 272, 0, 0),
    ];

    fn fixture(
        rows: usize,
        columns: usize,
        padding: usize,
        offset: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let stride = columns * 2 + padding;
        let values: Vec<f32> = (0..rows * stride)
            .map(|index| ((index * 37 + 11) % 257) as f32 / 17.0 - 7.5)
            .collect();
        let source = Tensor::from_vec(values, (rows, stride), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok((
            source.narrow(1, offset, columns)?,
            source.narrow(1, offset + columns, columns)?,
        ))
    }

    fn assert_quantized_equal(
        actual: &(Tensor, Tensor),
        expected: &(Tensor, Tensor),
    ) -> Result<()> {
        let packed_actual = actual
            .0
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let packed_expected = expected
            .0
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<u8>()?;
        let scale_actual = actual
            .1
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let scale_expected = expected
            .1
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let packed_mismatches = packed_actual
            .iter()
            .zip(&packed_expected)
            .filter(|(a, b)| a != b)
            .count();
        let scale_mismatches = scale_actual
            .iter()
            .zip(&scale_expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(packed_actual.len(), packed_expected.len());
        assert_eq!(scale_actual.len(), scale_expected.len());
        assert_eq!(
            (packed_mismatches, scale_mismatches),
            (0, 0),
            "packed/scales mismatches"
        );
        Ok(())
    }

    fn compare(
        gate: &Tensor,
        value: &Tensor,
        global: &Tensor,
        activation: GluActivationType,
    ) -> Result<()> {
        let args = || GluQuantArgs {
            gate,
            value,
            activation_global_scale: global,
            activation,
        };
        launch(args(), true)?.expect("fixture must support explicit warmup");
        let actual = launch(args(), false)?.expect("explicitly warmed fixture must run");
        let expected_glu = crate::fused_glu(gate, value, activation)?;
        let columns = *gate.dims().last().unwrap();
        let expected_glu = expected_glu.reshape((gate.elem_count() / columns, columns))?;
        let expected = crate::cutile::cutile_nvfp4_quantize(&expected_glu, global)?;
        assert_quantized_equal(&actual, &expected)
    }

    #[test]
    fn row_dense_layout_accepts_packed_offsets_and_64_bit_spans() {
        let layout = Layout::new(Shape::from_dims(&[3, 5, 80]), vec![815, 163, 1], 1);
        assert_eq!(
            matrix_layout(&layout),
            Some(MatrixLayout {
                rows: 15,
                columns: 80,
                row_stride: 163
            })
        );
        let large_stride = i32::MAX as usize + 17;
        let layout = Layout::new(Shape::from_dims(&[3, 16]), vec![large_stride, 1], 7);
        assert_eq!(matrix_layout(&layout).unwrap().row_stride, large_stride);
        let transposed = Layout::new(Shape::from_dims(&[80, 3]), vec![1, 80], 0);
        assert_eq!(matrix_layout(&transposed), None);
        let non_dense_batch = Layout::new(Shape::from_dims(&[3, 5, 80]), vec![999, 163, 1], 1);
        assert_eq!(matrix_layout(&non_dense_batch), None);
    }

    #[test]
    fn fused_quantization_matches_separate_glu_for_tail_and_strided_inputs() -> Result<()> {
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16] {
            for (rows, columns, padding, offset) in CASES {
                let (gate, value) = fixture(rows, columns, padding, offset, dtype, &device)?;
                for activation in WARMUP_ACTIVATIONS {
                    for scalar in [0.375f32, 2.0] {
                        let global = Tensor::from_vec(vec![scalar], 1, &device)?;
                        compare(&gate, &value, &global, activation)?;
                        if padding != 0 {
                            compare(&gate, &value.contiguous()?, &global, activation)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn finite_16_bit_gate_patterns_match_separate_nonlinear_glu() -> Result<()> {
        const SIDE: usize = 256;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        let global = Tensor::from_vec(vec![2.0f32], 1, &device)?;
        for dtype in [DType::BF16, DType::F16] {
            let values: Vec<f32> = (0..=u16::MAX)
                .map(|bits| {
                    let value = match dtype {
                        DType::BF16 => bf16::from_bits(bits).to_f32(),
                        DType::F16 => f16::from_bits(bits).to_f32(),
                        _ => unreachable!(),
                    };
                    if value.is_finite() {
                        value
                    } else {
                        0.0
                    }
                })
                .collect();
            let gate = Tensor::from_vec(values, (SIDE, SIDE), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?;
            let value = Tensor::ones((SIDE, SIDE), dtype, &device)?;
            for activation in [GluActivationType::Silu, GluActivationType::Sigmoid] {
                compare(&gate, &value, &global, activation)?;
            }
        }
        Ok(())
    }

    #[test]
    fn common_warmup_covers_packed_tail_graph_replay() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;

        const ROWS: usize = 33;
        const COLUMNS: usize = 272;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let data: Vec<f32> = (0..ROWS * COLUMNS * 2)
            .map(|index| ((index * 37 + 11) % 257) as f32 / 17.0 - 7.5)
            .collect();
        let source = Tensor::from_vec(data, (ROWS, COLUMNS * 2), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_device(&device)?;
        let alternate = source.neg()?;
        let packed_input = Tensor::zeros((ROWS, COLUMNS * 2), DType::BF16, &device)?;
        let gate = packed_input.narrow(1, 0, COLUMNS)?;
        let value = packed_input.narrow(1, COLUMNS, COLUMNS)?;
        let global = Tensor::from_vec(vec![0.375f32], 1, &device)?;
        let activation = GluActivationType::Silu;
        let reference = |input: &Tensor| -> Result<(Tensor, Tensor)> {
            let gate = input.narrow(1, 0, COLUMNS)?;
            let value = input.narrow(1, COLUMNS, COLUMNS)?;
            let intermediate = crate::fused_glu(&gate, &value, activation)?;
            crate::cutile::cutile_nvfp4_quantize(&intermediate, &global)
        };
        let expected = reference(&source)?;
        let expected_alternate = reference(&alternate)?;
        warm_common(&global, DType::BF16, COLUMNS)?;
        let warmed_count = cutile::tile_kernel::jit_compile_count();
        let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
        packed_input.slice_set(&source, 0, 0)?;
        device.synchronize()?;
        let tracking = stream.context().is_event_tracking();
        if tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = launch(
            GluQuantArgs {
                gate: &gate,
                value: &value,
                activation_global_scale: &global,
                activation,
            },
            false,
        );
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = captured?.expect("common warmup must cover the packed tail layout");
        assert_eq!(cutile::tile_kernel::jit_compile_count(), warmed_count);
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("NVFP4 GLU capture produced no graph"))?;
        for alternate_input in [false, true, false] {
            packed_input.slice_set(if alternate_input { &alternate } else { &source }, 0, 0)?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            device.synchronize()?;
            assert_quantized_equal(
                &output,
                if alternate_input {
                    &expected_alternate
                } else {
                    &expected
                },
            )?;
        }
        assert_eq!(cutile::tile_kernel::jit_compile_count(), warmed_count);
        drop(output);
        device.synchronize()?;
        drop(graph);
        packed_input.slice_set(&source, 0, 0)?;
        device.synchronize()?;
        Ok(())
    }

    #[test]
    fn unregistered_stride_metadata_falls_back_until_explicitly_warmed() -> Result<()> {
        const COLUMNS: usize = 368;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        let (gate, value) = fixture(3, COLUMNS, 3, 1, DType::BF16, &device)?;
        let global = Tensor::from_vec(vec![0.0f32, 2.0], 2, &device)?.narrow(0, 1, 1)?;
        assert_eq!(global.layout().start_offset(), 1);
        warm_common(&global, DType::BF16, COLUMNS)?;
        let args = || GluQuantArgs {
            gate: &gate,
            value: &value,
            activation_global_scale: &global,
            activation: GluActivationType::Silu,
        };
        let warmed_count = cutile::tile_kernel::jit_compile_count();
        assert!(launch(args(), false)?.is_none());
        assert_eq!(cutile::tile_kernel::jit_compile_count(), warmed_count);
        compare(&gate, &value, &global, GluActivationType::Silu)?;
        Ok(())
    }

    #[test]
    fn packed_projection_hook_preserves_shape_bias_and_calibration() -> Result<()> {
        use crate::nvfp4::{Nvfp4Layer, Nvfp4LayerParts};
        use crate::{Nvfp4ActivationMode, QuantMethod, QuantizedActivation};

        const ROWS: usize = 6;
        const COLUMNS: usize = 80;
        const OUTPUT: usize = 7;
        const GLOBAL: f32 = 0.375;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16] {
            let weights = Tensor::from_vec(
                vec![0x42u8; OUTPUT * COLUMNS / 2],
                (OUTPUT, COLUMNS / 2),
                &device,
            )?;
            let scales = Tensor::ones((OUTPUT, COLUMNS / BLOCK_SIZE), DType::F8E4M3, &Device::Cpu)?
                .to_device(&device)?;
            let globals = Tensor::from_vec(vec![0.75f32; OUTPUT], OUTPUT, &device)?;
            let bias = Tensor::from_vec(vec![0.125f32; OUTPUT], OUTPUT, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?;
            let global = Tensor::new(GLOBAL, &device)?;
            let layer = Nvfp4Layer::from_parts(Nvfp4LayerParts {
                weight: weights.clone(),
                scales: scales.clone(),
                global_scales: globals.clone(),
                input_scale: Some(global.clone()),
                activation: Nvfp4ActivationMode::DynamicBlock,
                bias: Some(bias.clone()),
                dtype,
            })?;
            crate::cutile::warmup_moe_kernels(&device)?;
            let data: Vec<f32> = (0..ROWS * COLUMNS * 2)
                .map(|index| ((index * 37 + 11) % 257) as f32 / 17.0 - 7.5)
                .collect();
            let input = Tensor::from_vec(data, (2, 3, COLUMNS * 2), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?;
            let gate = input.narrow(2, 0, COLUMNS)?;
            let value = input.narrow(2, COLUMNS, COLUMNS)?;
            let activation = GluActivationType::Relu;
            let intermediate = crate::fused_glu(&gate, &value, activation)?;
            let expected = layer.forward(&intermediate)?;
            let actual = layer
                .try_forward_fused_split_glu(&input, COLUMNS, activation)?
                .expect("registered A4 layer should use the packed hook");
            assert_eq!(actual.dims(), &[2, 3, OUTPUT]);
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            let separate_gate = gate.contiguous()?;
            let separate_value = value.contiguous()?;
            let compiled = cutile::tile_kernel::jit_compile_count();
            let quantized = layer
                .try_quantize_glu(&separate_gate, &separate_value, activation)?
                .expect("registered layer should quantize separate GLU sources");
            assert_eq!(quantized.global_scale(), Some(GLOBAL));
            assert_eq!(quantized.source_shape(), gate.dims());
            let separate_output = crate::try_forward_fused_quantized_glu(
                &separate_gate,
                &separate_value,
                &layer,
                activation,
            )?
            .expect("separate GLU helper should preserve NVFP4 capability");
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            assert_eq!(
                separate_output
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            assert!(layer
                .try_quantize_glu(&separate_gate, &separate_value, GluActivationType::Gelu)?
                .is_none());
            assert!(layer
                .try_quantize_glu(
                    &separate_gate,
                    &separate_value.narrow(2, 0, COLUMNS - 1)?,
                    activation
                )?
                .is_none());
            let one_row = input.reshape((ROWS, COLUMNS * 2))?.narrow(0, 0, 1)?;
            assert!(layer
                .try_forward_fused_split_glu(&one_row, COLUMNS, activation)?
                .is_none());
            assert!(layer
                .try_forward_fused_split_glu(&input, COLUMNS - 1, activation)?
                .is_none());
            assert!(layer
                .try_forward_fused_split_glu(&input, COLUMNS, GluActivationType::Gelu)?
                .is_none());
            assert!(layer
                .try_forward_fused_split_glu(&input.to_dtype(DType::F32)?, COLUMNS, activation)?
                .is_none());
            let separate = crate::cutile::cutile_nvfp4_quantize(
                &intermediate.reshape((ROWS, COLUMNS))?,
                &global,
            )?;
            let wrong_global =
                QuantizedActivation::new_nvfp4(separate.0, separate.1, &gate, GLOBAL * 2.0)?;
            assert!(layer.forward_quantized(&wrong_global).is_err());
            let a16 = Nvfp4Layer::from_parts(Nvfp4LayerParts {
                weight: weights,
                scales,
                global_scales: globals,
                input_scale: None,
                activation: Nvfp4ActivationMode::None,
                bias: Some(bias),
                dtype,
            })?;
            assert!(a16
                .try_forward_fused_split_glu(&input, COLUMNS, activation)?
                .is_none());
            assert!(a16
                .try_quantize_glu(&separate_gate, &separate_value, activation)?
                .is_none());
        }
        Ok(())
    }

    #[test]
    fn common_packed_warmup_covers_contiguous_inputs_without_compiling() -> Result<()> {
        const ROWS: usize = 33;
        const COLUMNS: usize = 272;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        for dtype in [DType::BF16, DType::F16] {
            let (gate, value) = fixture(ROWS, COLUMNS, 0, 0, dtype, &device)?;
            let gate = gate.contiguous()?;
            let value = value.contiguous()?;
            let global = Tensor::new(2.0f32, &device)?;
            warm_common(&global, dtype, COLUMNS)?;
            let compiled = cutile::tile_kernel::jit_compile_count();
            let actual = launch(
                GluQuantArgs {
                    gate: &gate,
                    value: &value,
                    activation_global_scale: &global,
                    activation: GluActivationType::Relu,
                },
                false,
            )?
            .expect("common packed keys must cover contiguous aligned sources");
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            let intermediate = crate::fused_glu(&gate, &value, GluActivationType::Relu)?;
            let expected = crate::cutile::cutile_nvfp4_quantize(&intermediate, &global)?;
            assert_quantized_equal(&actual, &expected)?;
        }
        Ok(())
    }

    #[test]
    fn separate_projection_hook_supports_graph_replay_without_compiling() -> Result<()> {
        use crate::nvfp4::{Nvfp4Layer, Nvfp4LayerParts};
        use crate::{Nvfp4ActivationMode, QuantMethod};
        use candle_core::cuda::cudarc::driver::sys;

        const ROWS: usize = 33;
        const COLUMNS: usize = 272;
        const OUTPUT: usize = 7;
        const GLOBAL: f32 = 0.375;
        let _gpu_guard = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda(0)?;
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        for dtype in [DType::BF16, DType::F16] {
            let layer = Nvfp4Layer::from_parts(Nvfp4LayerParts {
                weight: Tensor::from_vec(
                    vec![0x42u8; OUTPUT * COLUMNS / 2],
                    (OUTPUT, COLUMNS / 2),
                    &device,
                )?,
                scales: Tensor::ones((OUTPUT, COLUMNS / BLOCK_SIZE), DType::F8E4M3, &Device::Cpu)?
                    .to_device(&device)?,
                global_scales: Tensor::from_vec(vec![0.75f32; OUTPUT], OUTPUT, &device)?,
                input_scale: Some(Tensor::new(GLOBAL, &device)?),
                activation: Nvfp4ActivationMode::DynamicBlock,
                bias: Some(
                    Tensor::from_vec(vec![0.125f32; OUTPUT], OUTPUT, &Device::Cpu)?
                        .to_dtype(dtype)?
                        .to_device(&device)?,
                ),
                dtype,
            })?;
            crate::cutile::warmup_moe_kernels(&device)?;
            let (gate_source, value_source) = fixture(ROWS, COLUMNS, 0, 0, dtype, &device)?;
            let gate_source = gate_source.contiguous()?;
            let value_source = value_source.contiguous()?;
            let alternate = gate_source.neg()?;
            let activation = GluActivationType::Silu;
            let reference = |gate: &Tensor| -> Result<Vec<f32>> {
                layer
                    .forward(&crate::fused_glu(gate, &value_source, activation)?)?
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            };
            let expected = reference(&gate_source)?;
            let expected_alternate = reference(&alternate)?;
            let gate = Tensor::zeros((ROWS, COLUMNS), dtype, &device)?;
            let value = Tensor::zeros((ROWS, COLUMNS), dtype, &device)?;
            gate.slice_set(&gate_source, 0, 0)?;
            value.slice_set(&value_source, 0, 0)?;
            let compiled = cutile::tile_kernel::jit_compile_count();
            let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
            crate::try_forward_fused_quantized_glu(&gate, &value, &layer, activation)?
                .expect("warmup must cover the separate-source projection hook");
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            device.synchronize()?;
            let tracking = stream.context().is_event_tracking();
            if tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            if let Err(error) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                return Err(candle_core::Error::msg(error.to_string()));
            }
            let captured =
                crate::try_forward_fused_quantized_glu(&gate, &value, &layer, activation);
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = captured?.expect("warmup must cover the separate-source projection hook");
            let graph = graph
                .map_err(|error| candle_core::Error::msg(error.to_string()))?
                .ok_or_else(|| candle_core::Error::msg("separate GLU capture returned no graph"))?;
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            for alternate_input in [false, true, false] {
                gate.slice_set(
                    if alternate_input {
                        &alternate
                    } else {
                        &gate_source
                    },
                    0,
                    0,
                )?;
                graph
                    .launch()
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
                device.synchronize()?;
                let actual = output
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = if alternate_input {
                    &expected_alternate
                } else {
                    &expected
                };
                assert!(actual.iter().chain(expected).all(|value| value.is_finite()));
                assert_eq!(&actual, expected);
            }
            assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
            device.synchronize()?;
            drop(graph);
            drop(output);
            gate.slice_set(&gate_source, 0, 0)?;
            value.slice_set(&value_source, 0, 0)?;
            device.synchronize()?;
        }
        Ok(())
    }
}
