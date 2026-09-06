#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

use std::sync::Arc;

use candle_core::{CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::{bf16, f16};

use super::nvfp4::Nvfp4GemmArgs;
use super::{catch_cutile_panic, context};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const BLOCK_SIZE: usize = 16;
const SMALL_MATRIX_COLUMNS: usize = 32;
const SMALL_MATRIX_K: usize = 512;
const LARGE_MATRIX_COLUMNS: usize = 16;
const LARGE_MATRIX_K: usize = 1024;
const LARGE_MATRIX_MIN_ELEMENTS: usize = 32 * 1024 * 1024;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    use cutile::cutile_compiler;

    const BLOCK: i32 = 16;
    const FP4_MAX: f32 = 6.0;
    const FP4_MIN: f32 = -6.0;
    const FP8_MAX: f32 = 448.0;

    #[cutile::entry(unchecked_accesses = false)]
    fn gemv_bf16<
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const ROUTED: bool,
    >(
        mut y: MappedPartitionMut<bf16, { [1, BN] }, { [1, 1] }>,
        x: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        wg: &Tensor<f32, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
        ids: &Tensor<u32, { [-1] }>,
        x_stride: i32,
    ) {
        let px = x.partition(const_shape![1, BK]);
        let pw = w.partition(const_shape![1, BN, PK]);
        let pws = ws.partition(const_shape![1, BN, SK]);
        let pwg = wg.partition(const_shape![1, BN]);
        let k = num_tiles(&px, 1);
        for out_idx in y.iter_indices() {
            let (row, col) = out_idx.components();
            let expert: i32 = if ROUTED {
                let expert: Tile<u32, { [1] }> = load_tile(ids, const_shape![1], [row]);
                let expert: Tile<i32, { [1] }> = bitcast(expert);
                tile_to_scalar(expert.reshape(const_shape![]))
            } else {
                0i32
            };
            let global: Tile<f32, { [1] }> = if A4 {
                load_tile(ag, const_shape![1], [expert])
            } else {
                let one: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
                one
            };
            let mut acc: Tile<f32, { [BN, BK] }> = constant(0.0f32, const_shape![BN, BK]);
            for ki in 0..k {
                let xt: Tile<bf16, { [1, BK] }> = px.load([row / x_stride, ki]);
                let xf: Tile<f32, { [1, BK] }> = convert_tile(xt);
                let xf: Tile<f32, { [1, BK] }> = if A4 {
                    let xf = xf
                        / global
                            .reshape(const_shape![1, 1])
                            .broadcast(const_shape![1, BK]);
                    let xb: Tile<f32, { [SK, BLOCK] }> = xf.reshape(const_shape![SK, BLOCK]);
                    let maxima: Tile<f32, { [SK] }> = reduce_max(absf(xb), 1);
                    let maxima = maxima.reshape(const_shape![SK]);
                    let six: Tile<f32, { [SK] }> = constant(FP4_MAX, const_shape![SK]);
                    let limit: Tile<f32, { [SK] }> = constant(FP8_MAX, const_shape![SK]);
                    let raw = min_tile(maxima / six, limit);
                    let sx: Tile<f8e4m3fn, { [SK] }> = convert_tile(raw);
                    let sx: Tile<f32, { [SK] }> = convert_tile(sx);
                    let zero: Tile<f32, { [SK] }> = constant(0.0f32, const_shape![SK]);
                    let one: Tile<f32, { [SK] }> = constant(1.0f32, const_shape![SK]);
                    let denominator = select(eq_tile(sx, zero), one, sx)
                        .reshape(const_shape![SK, 1])
                        .broadcast(const_shape![SK, BLOCK]);
                    let normalized = xb / denominator;
                    let upper: Tile<f32, { [SK, BLOCK] }> =
                        constant(FP4_MAX, const_shape![SK, BLOCK]);
                    let lower: Tile<f32, { [SK, BLOCK] }> =
                        constant(FP4_MIN, const_shape![SK, BLOCK]);
                    let normalized = max_tile(min_tile(normalized, upper), lower);
                    let xq: Tile<f4e2m1fn, { [SK, BLOCK] }> = convert_tile(normalized);
                    let xq: Tile<f32, { [SK, BLOCK] }> = convert_tile(xq);
                    (xq * sx
                        .reshape(const_shape![SK, 1])
                        .broadcast(const_shape![SK, BLOCK]))
                    .reshape(const_shape![1, BK])
                } else {
                    xf
                };
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> =
                    pw.load([expert, col, ki]).reshape(const_shape![BN, PK]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> =
                    pws.load([expert, col, ki]).reshape(const_shape![BN, SK]);
                let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                let sw = sw
                    .reshape(const_shape![BN, SK, 1])
                    .broadcast(const_shape![BN, SK, BLOCK])
                    .reshape(const_shape![BN, BK]);
                let wf = wf * sw;
                let wf: Tile<f32, { [BN, BK] }> = if A4 {
                    wf
                } else {
                    let rounded: Tile<bf16, { [BN, BK] }> = convert_tile(wf);
                    let rounded: Tile<f32, { [BN, BK] }> = convert_tile(rounded);
                    rounded
                };
                acc = acc + xf.broadcast(const_shape![BN, BK]) * wf;
            }
            let sum: Tile<f32, { [BN] }> = reduce_sum(acc, 1);
            let sum = sum.reshape(const_shape![BN]);
            let gw: Tile<f32, { [BN] }> = pwg.load([expert, col]).reshape(const_shape![BN]);
            let out = sum * gw * global.broadcast(const_shape![BN]);
            let out: Tile<bf16, { [BN] }> = convert_tile(out);
            y.store(out.reshape(const_shape![1, BN]), out_idx);
        }
    }
    #[cutile::entry(unchecked_accesses = false)]
    fn gemv_f16<
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const ROUTED: bool,
    >(
        mut y: MappedPartitionMut<f16, { [1, BN] }, { [1, 1] }>,
        x: &Tensor<f16, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        wg: &Tensor<f32, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
        ids: &Tensor<u32, { [-1] }>,
        x_stride: i32,
    ) {
        let px = x.partition(const_shape![1, BK]);
        let pw = w.partition(const_shape![1, BN, PK]);
        let pws = ws.partition(const_shape![1, BN, SK]);
        let pwg = wg.partition(const_shape![1, BN]);
        let k = num_tiles(&px, 1);
        for out_idx in y.iter_indices() {
            let (row, col) = out_idx.components();
            let expert: i32 = if ROUTED {
                let expert: Tile<u32, { [1] }> = load_tile(ids, const_shape![1], [row]);
                let expert: Tile<i32, { [1] }> = bitcast(expert);
                tile_to_scalar(expert.reshape(const_shape![]))
            } else {
                0i32
            };
            let global: Tile<f32, { [1] }> = if A4 {
                load_tile(ag, const_shape![1], [expert])
            } else {
                let one: Tile<f32, { [1] }> = constant(1.0f32, const_shape![1]);
                one
            };
            let mut acc: Tile<f32, { [BN, BK] }> = constant(0.0f32, const_shape![BN, BK]);
            for ki in 0..k {
                let xt: Tile<f16, { [1, BK] }> = px.load([row / x_stride, ki]);
                let xf: Tile<f32, { [1, BK] }> = convert_tile(xt);
                let xf: Tile<f32, { [1, BK] }> = if A4 {
                    let xf = xf
                        / global
                            .reshape(const_shape![1, 1])
                            .broadcast(const_shape![1, BK]);
                    let xb: Tile<f32, { [SK, BLOCK] }> = xf.reshape(const_shape![SK, BLOCK]);
                    let maxima: Tile<f32, { [SK] }> = reduce_max(absf(xb), 1);
                    let maxima = maxima.reshape(const_shape![SK]);
                    let six: Tile<f32, { [SK] }> = constant(FP4_MAX, const_shape![SK]);
                    let limit: Tile<f32, { [SK] }> = constant(FP8_MAX, const_shape![SK]);
                    let raw = min_tile(maxima / six, limit);
                    let sx: Tile<f8e4m3fn, { [SK] }> = convert_tile(raw);
                    let sx: Tile<f32, { [SK] }> = convert_tile(sx);
                    let zero: Tile<f32, { [SK] }> = constant(0.0f32, const_shape![SK]);
                    let one: Tile<f32, { [SK] }> = constant(1.0f32, const_shape![SK]);
                    let denominator = select(eq_tile(sx, zero), one, sx)
                        .reshape(const_shape![SK, 1])
                        .broadcast(const_shape![SK, BLOCK]);
                    let normalized = xb / denominator;
                    let upper: Tile<f32, { [SK, BLOCK] }> =
                        constant(FP4_MAX, const_shape![SK, BLOCK]);
                    let lower: Tile<f32, { [SK, BLOCK] }> =
                        constant(FP4_MIN, const_shape![SK, BLOCK]);
                    let normalized = max_tile(min_tile(normalized, upper), lower);
                    let xq: Tile<f4e2m1fn, { [SK, BLOCK] }> = convert_tile(normalized);
                    let xq: Tile<f32, { [SK, BLOCK] }> = convert_tile(xq);
                    (xq * sx
                        .reshape(const_shape![SK, 1])
                        .broadcast(const_shape![SK, BLOCK]))
                    .reshape(const_shape![1, BK])
                } else {
                    xf
                };
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> =
                    pw.load([expert, col, ki]).reshape(const_shape![BN, PK]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> =
                    pws.load([expert, col, ki]).reshape(const_shape![BN, SK]);
                let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                let sw = sw
                    .reshape(const_shape![BN, SK, 1])
                    .broadcast(const_shape![BN, SK, BLOCK])
                    .reshape(const_shape![BN, BK]);
                let wf = wf * sw;
                let wf: Tile<f32, { [BN, BK] }> = if A4 {
                    wf
                } else {
                    let rounded: Tile<f16, { [BN, BK] }> = convert_tile(wf);
                    let rounded: Tile<f32, { [BN, BK] }> = convert_tile(rounded);
                    rounded
                };
                acc = acc + xf.broadcast(const_shape![BN, BK]) * wf;
            }
            let sum: Tile<f32, { [BN] }> = reduce_sum(acc, 1);
            let sum = sum.reshape(const_shape![BN]);
            let gw: Tile<f32, { [BN] }> = pwg.load([expert, col]).reshape(const_shape![BN]);
            let out = sum * gw * global.broadcast(const_shape![BN]);
            let out: Tile<f16, { [BN] }> = convert_tile(out);
            y.store(out.reshape(const_shape![1, BN]), out_idx);
        }
    }
}

pub(super) fn launch(
    x: &Tensor,
    indices: Option<&Tensor>,
    args: Nvfp4GemmArgs<'_>,
    compile_only: bool,
) -> Result<Tensor> {
    let (experts, n, packed_k) = if indices.is_some() {
        args.weights.dims3()?
    } else {
        let (n, packed_k) = args.weights.dims2()?;
        (1, n, packed_k)
    };
    let (input_rows, rows, k, x_stride) = if let Some(indices) = indices {
        let (tokens, topk) = indices.dims2()?;
        let (input_rows, k, stride) = match x.dims() {
            &[m, k] if m == tokens => (m, k, topk),
            &[m, 1, k] if m == tokens => (m, k, topk),
            &[m, routes, k] if m == tokens && routes == topk => (m * routes, k, 1),
            dims => candle_core::bail!(
                "cuTile NVFP4 gather activation shape {dims:?} does not match indices {:?}",
                indices.dims()
            ),
        };
        if indices.dtype() != DType::U32 || !indices.device().same_device(x.device()) {
            candle_core::bail!("cuTile NVFP4 gather indices must be U32 on the activation device")
        }
        (input_rows, tokens * topk, k, stride)
    } else {
        let (rows, k) = x.dims2()?;
        (rows, rows, k, 1)
    };
    let scale_shape = [experts, n, k / BLOCK_SIZE];
    let global_shape = [experts, n];
    let Device::Cuda(dev) = x.device() else {
        candle_core::bail!("cuTile NVFP4 GEMV requires CUDA tensors")
    };
    let x = x.contiguous()?.reshape((input_rows, k))?;
    let weights = args.weights.contiguous()?;
    let scales = args.weight_scales.contiguous()?;
    let weight_global = args.weight_global_scale.contiguous()?;
    let activation_global = args
        .activation_global_scale
        .unwrap_or(&weight_global)
        .contiguous()?;
    let (tile_columns, tile_k) = if n * k >= LARGE_MATRIX_MIN_ELEMENTS {
        (LARGE_MATRIX_COLUMNS, LARGE_MATRIX_K)
    } else {
        (SMALL_MATRIX_COLUMNS, SMALL_MATRIX_K)
    };
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = weights.storage_and_layout();
    let (s_storage, s_layout) = scales.storage_and_layout();
    let (wg_storage, wg_layout) = weight_global.storage_and_layout();
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (
        Storage::Cuda(x_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(s_cuda),
        Storage::Cuda(wg_cuda),
        Storage::Cuda(ag_cuda),
    ) = (
        &*x_storage,
        &*w_storage,
        &*s_storage,
        &*wg_storage,
        &*ag_storage,
    )
    else {
        candle_core::bail!("cuTile NVFP4 operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (wg_addr, _wg_guard) = slice_ptr_on_stream(
        wg_cuda.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let w = unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32, packed_k as i32],
            vec![(n * packed_k) as i32, packed_k as i32, 1],
        )
    };
    let s = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            scale_shape.iter().map(|&dim| dim as i32).collect(),
            vec![(n * k / BLOCK_SIZE) as i32, (k / BLOCK_SIZE) as i32, 1],
        )
    };
    let wg = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            wg_addr as CUdeviceptr,
            ordinal,
            global_shape.iter().map(|&dim| dim as i32).collect(),
            vec![n as i32, 1],
        )
    };
    let ag = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32],
            vec![1],
        )
    };
    let tiles = rows * n.div_ceil(tile_columns);
    let tile_blocks = tiles as u32;
    let generics = vec![
        tile_columns.to_string(),
        tile_k.to_string(),
        (tile_k / 2).to_string(),
        (tile_k / BLOCK_SIZE).to_string(),
        args.activation_global_scale.is_some().to_string(),
        indices.is_some().to_string(),
    ];
    let cutile_stream = context::stream(dev);

    macro_rules! run {
        ($dtype:ty, $kernel:path) => {{
            let (x_addr, _x_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<$dtype>(rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let x = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    x_addr as CUdeviceptr,
                    ordinal,
                    vec![input_rows as i32, k as i32],
                    vec![k as i32, 1],
                )
            };
            let y = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    vec![rows as i32, n as i32],
                    vec![n as i32, 1],
                )
            };
            let mapped = y.partition([1, tile_columns]).map([1, 1], tile_blocks);
            macro_rules! dispatch {
                ($launcher:expr) => {{
                    let launcher = $launcher.generics(generics);
                    if compile_only {
                        catch_cutile_panic("NVFP4 compile", || {
                            launcher.compile_on(&cutile_stream).map_err(|error| {
                                candle_core::Error::Msg(format!(
                                    "cuTile NVFP4 compile failed: {error:?}"
                                ))
                            })
                        })?;
                    } else {
                        catch_cutile_panic("NVFP4 launch", || unsafe {
                            launcher.async_on(&cutile_stream).map_err(|error| {
                                candle_core::Error::Msg(format!(
                                    "cuTile NVFP4 launch failed: {error:?}"
                                ))
                            })
                        })?;
                    }
                }};
            }
            if let Some(indices) = indices {
                let indices = indices.contiguous()?;
                let (id_storage, id_layout) = indices.storage_and_layout();
                let Storage::Cuda(id_cuda) = &*id_storage else {
                    unreachable!()
                };
                let (id_addr, _id_guard) = slice_ptr_on_stream(
                    id_cuda.as_cuda_slice::<u32>()?,
                    id_layout.start_offset(),
                    &stream,
                );
                let ids = unsafe {
                    cutile::tensor::Tensor::<u32>::borrow_raw_parts(
                        id_addr as CUdeviceptr,
                        ordinal,
                        vec![rows as i32],
                        vec![1],
                    )
                };
                dispatch!($kernel(
                    mapped,
                    Arc::new(x),
                    Arc::new(w),
                    Arc::new(s),
                    Arc::new(wg),
                    Arc::new(ag),
                    Arc::new(ids),
                    x_stride as i32
                ));
            } else {
                let ids = unsafe {
                    cutile::tensor::Tensor::<u32>::borrow_raw_parts(
                        wg_addr as CUdeviceptr,
                        ordinal,
                        vec![1],
                        vec![1],
                    )
                };
                dispatch!($kernel(
                    mapped,
                    Arc::new(x),
                    Arc::new(w),
                    Arc::new(s),
                    Arc::new(wg),
                    Arc::new(ag),
                    Arc::new(ids),
                    1i32
                ));
            }
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[rows, n]),
            ))
        }};
    }
    let output = match x.dtype() {
        DType::BF16 => run!(bf16, kernels::gemv_bf16),
        DType::F16 => run!(f16, kernels::gemv_f16),
        _ => unreachable!(),
    };
    if compile_only {
        return Ok(output);
    }
    if let Some(indices) = indices {
        let (tokens, topk) = indices.dims2()?;
        output.reshape((tokens, topk, n))
    } else {
        Ok(output)
    }
}
