//! Persistent blockwise FP8 W8A8 GEMM: 128x128 weight scales, 1x128 activation scales, BF16 output.

use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{CudaDevice, CudaStorage, DType, Result, Shape, Storage, Tensor};
use cutile::core::f8e4m3fn;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::bf16;

use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, device_multiprocessor_count, jit_available};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub const FP8_GEMM_BLOCK_ROWS: usize = 128;
const BLOCK_COLS: usize = 128;
const GROUP_SIZE: usize = 128;
const MAP_GROUP_M: usize = 8;
const MAP_GROUP_N: usize = 1;
const TILE_BLOCKS_PER_SM: usize = 2;

// Bounded partitions are deprecated in cutile 0.3 but plain indexed loads compile to checked loads
// that run 3x slower on sm_121.
#[cutile::module]
mod kernels {
    #![allow(deprecated)]
    use cutile::core::*;
    use cutile::cutile_compiler;

    // BN stays at one weight-scale column and BK at one scale group, so every k step is scaled by a
    // single (row, group) product before it joins the running accumulator.
    #[allow(deprecated)]
    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_blockwise_gemm<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        xs: &Tensor<f32, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = num_tiles(&y, 0);
        let n = num_tiles(&y, 1);
        let k = Dim::new(x.shape()[1] / BK);
        let px = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let pw = w.partition(const_shape![BN, BK]).with_bounds((n, k));
        let pxs = xs.partition(const_shape![1, BM]).with_bounds((k, m));
        let pws = ws.partition(const_shape![1, 1]).with_bounds((n, k));
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in k {
                let xt: Tile<f8e4m3fn, { [BM, BK] }> = px.load(coord((bid_m, kg)));
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = pw.load(coord((bid_n, kg)));
                let wtt: Tile<f8e4m3fn, { [BK, BN] }> = permute(wt, transpose);
                let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let part: Tile<f32, { [BM, BN] }> = mmaf(xt, wtt, zero);
                let sx: Tile<f32, { [1, BM] }> = pxs.load(coord((kg, bid_m)));
                let sxc: Tile<f32, { [BM, 1] }> = sx.reshape(const_shape![BM, 1]);
                let sw: Tile<f32, { [1, 1] }> = pws.load(coord((bid_n, kg)));
                let scale: Tile<f32, { [BM, BN] }> =
                    sxc.broadcast(const_shape![BM, BN]) * sw.broadcast(const_shape![BM, BN]);
                acc = acc + part * scale;
            }
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
            y.store(out, out_idx);
        }
    }
}

pub struct Fp8GemmKernel;

pub(super) static FP8_GEMM: Fp8GemmKernel = Fp8GemmKernel;

static SHAPES: OnceLock<Mutex<Vec<(usize, usize)>>> = OnceLock::new();

pub fn register_fp8_gemm_shape(output_features: usize, input_features: usize) {
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if !shapes.contains(&(output_features, input_features)) {
        shapes.push((output_features, input_features));
    }
}

pub fn fp8_gemm_supported(dev: &CudaDevice, output_features: usize, input_features: usize) -> bool {
    jit_available(dev)
        && output_features.is_multiple_of(BLOCK_COLS)
        && input_features.is_multiple_of(GROUP_SIZE)
}

struct GemmOperands<'a> {
    activation: &'a Tensor,
    activation_scales: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
}

/// `activation` [rows, K] E4M3 whose storage holds `padded` rows, `activation_scales` [K/128, padded]
/// F32 with `padded` a multiple of [`FP8_GEMM_BLOCK_ROWS`] and rows <= padded, `weight` [N, K] E4M3,
/// `weight_scales` [N/128, K/128] F32 -> [padded, N] BF16; rows past `rows` are unspecified.
pub fn cutile_fp8_gemm(
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let operands = GemmOperands {
        activation,
        activation_scales,
        weight,
        weight_scales,
    };
    launch(&operands, dev, false)
}

/// Rows an activation view's storage can supply past its start offset.
fn activation_storage_rows(activation: &Tensor) -> Result<usize> {
    let (_, k) = activation.dims2()?;
    let (storage, layout) = activation.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        candle_core::bail!("cuTile FP8 GEMM operands must be CUDA tensors")
    };
    Ok((cuda.as_cuda_slice::<F8E4M3>()?.len() - layout.start_offset()) / k)
}

fn launch(operands: &GemmOperands<'_>, dev: &CudaDevice, compile_only: bool) -> Result<Tensor> {
    let (logical_rows, k) = operands.activation.dims2()?;
    let (groups_dim, rows) = operands.activation_scales.dims2()?;
    let (n, weight_k) = operands.weight.dims2()?;
    let groups = k / GROUP_SIZE;
    if weight_k != k
        || !rows.is_multiple_of(FP8_GEMM_BLOCK_ROWS)
        || !n.is_multiple_of(BLOCK_COLS)
        || !k.is_multiple_of(GROUP_SIZE)
        || logical_rows == 0
        || logical_rows > rows
        || !operands.activation.is_contiguous()
    {
        candle_core::bail!(
            "cuTile FP8 GEMM got unsupported shape rows={logical_rows} padded={rows} n={n} k={k}"
        )
    }
    if operands.activation.dtype() != DType::F8E4M3
        || operands.weight.dtype() != DType::F8E4M3
        || operands.activation_scales.dtype() != DType::F32
        || operands.weight_scales.dtype() != DType::F32
    {
        candle_core::bail!("cuTile FP8 GEMM needs E4M3 operands with F32 scales")
    }
    if groups_dim != groups || operands.weight_scales.dims2()? != (n / BLOCK_COLS, groups) {
        candle_core::bail!("cuTile FP8 GEMM scale shapes do not match the operands")
    }
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let mut out = unsafe { dev.alloc::<bf16>(rows * n)? };
    // producers pad their FP8 rows to the scale stride; anything else gets copied into a padded buffer
    let padded_copy;
    let activation = if activation_storage_rows(operands.activation)? >= rows {
        operands.activation
    } else {
        padded_copy = Tensor::zeros((rows, k), DType::F8E4M3, operands.activation.device())?;
        padded_copy.slice_set(operands.activation, 0, 0)?;
        &padded_copy
    };
    let (a_storage, a_layout) = activation.storage_and_layout();
    let (as_storage, as_layout) = operands.activation_scales.storage_and_layout();
    let (w_storage, w_layout) = operands.weight.storage_and_layout();
    let (ws_storage, ws_layout) = operands.weight_scales.storage_and_layout();
    let (
        Storage::Cuda(a_cuda),
        Storage::Cuda(as_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(ws_cuda),
    ) = (&*a_storage, &*as_storage, &*w_storage, &*ws_storage)
    else {
        candle_core::bail!("cuTile FP8 GEMM operands must be CUDA tensors")
    };
    let (a_addr, _a_guard) = slice_ptr_on_stream(
        a_cuda.as_cuda_slice::<F8E4M3>()?,
        a_layout.start_offset(),
        &stream,
    );
    let (as_addr, _as_guard) = slice_ptr_on_stream(
        as_cuda.as_cuda_slice::<f32>()?,
        as_layout.start_offset(),
        &stream,
    );
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<F8E4M3>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (ws_addr, _ws_guard) = slice_ptr_on_stream(
        ws_cuda.as_cuda_slice::<f32>()?,
        ws_layout.start_offset(),
        &stream,
    );
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let dims = |r: usize, c: usize| (vec![r as i32, c as i32], vec![c as i32, 1]);
    // SAFETY: every borrowed buffer is a live candle allocation that outlives the launch on the
    // same stream, and the output is written only by this kernel.
    let (y, x, w, xs, ws) = unsafe {
        let (shape, strides) = dims(rows, n);
        let y = cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
            out_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(rows, k);
        let x = cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            a_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(n, k);
        let w = cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(groups, rows);
        let xs = cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            as_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(n / BLOCK_COLS, groups);
        let ws = cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ws_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        (y, x, w, xs, ws)
    };
    let tiles = (rows / FP8_GEMM_BLOCK_ROWS) * (n / BLOCK_COLS);
    let tile_blocks =
        (TILE_BLOCKS_PER_SM * device_multiprocessor_count(dev)).clamp(1, tiles) as u32;
    let mapped = y
        .partition([FP8_GEMM_BLOCK_ROWS, BLOCK_COLS])
        .map([MAP_GROUP_M, MAP_GROUP_N], tile_blocks);
    let generics = vec![
        FP8_GEMM_BLOCK_ROWS.to_string(),
        BLOCK_COLS.to_string(),
        GROUP_SIZE.to_string(),
        MAP_GROUP_M.to_string(),
        MAP_GROUP_N.to_string(),
    ];
    let cutile_stream = context::stream(dev);
    let launcher =
        kernels::fp8_blockwise_gemm(mapped, Arc::new(x), Arc::new(w), Arc::new(xs), Arc::new(ws))
            .generics(generics);
    if compile_only {
        catch_cutile_panic("FP8 GEMM compile", || {
            launcher
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile fp8 gemm compile: {e:?}")))
        })?;
    } else {
        catch_cutile_panic("FP8 GEMM launch", || unsafe {
            launcher
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile fp8 gemm launch: {e:?}")))
        })?;
    }
    drop(out_guard);
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[rows, n]),
    )))
}

impl CutileKernel for Fp8GemmKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        let shapes = SHAPES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        if shapes.is_empty() {
            return Ok(());
        }
        tracing::info!("Warming {} cuTile FP8 GEMM kernels.", shapes.len());
        let device = candle_core::Device::Cuda(dev.clone());
        for (n, k) in shapes {
            let rows = FP8_GEMM_BLOCK_ROWS;
            let activation = Tensor::zeros((rows, k), DType::F8E4M3, &device)?;
            let activation_scales = Tensor::zeros((k / GROUP_SIZE, rows), DType::F32, &device)?;
            let weight = Tensor::zeros((n, k), DType::F8E4M3, &device)?;
            let weight_scales =
                Tensor::zeros((n / BLOCK_COLS, k / GROUP_SIZE), DType::F32, &device)?;
            let operands = GemmOperands {
                activation: &activation,
                activation_scales: &activation_scales,
                weight: &weight,
                weight_scales: &weight_scales,
            };
            if let Err(err) = launch(&operands, dev, true) {
                tracing::warn!("cuTile FP8 GEMM warmup failed (n={n} k={k}): {err}");
            }
        }
        Ok(())
    }
}

#[cfg(all(test, has_blockwise_fp8_kernels))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{cutile_fp8_gemm, FP8_GEMM_BLOCK_ROWS};
    use crate::blockwise_fp8::{mma, ops};

    const GROUP_SIZE: usize = 128;

    fn patterned(len: usize, seed: usize, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7919 + seed * 104729) % 2001) as f32 / 1000.0 - 1.0)
            .map(|value| value * amplitude + offset)
            .collect()
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fp8_gemm_reads_row_views_over_padded_storage() -> Result<()> {
        const N: usize = 256;
        const K: usize = 512;
        const ROWS: usize = 200;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let weight =
            Tensor::from_vec(patterned(N * K, 5, 2.0, 0.1), (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![GROUP_SIZE, GROUP_SIZE])?;
        let x = Tensor::from_vec(patterned(ROWS * K, 9, 3.0, -0.2), (ROWS, K), &dev)?
            .to_dtype(DType::BF16)?;
        let padded_rows = ROWS.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
        let (quantized, scales) = mma::quantize_activation_padded(&x, padded_rows)?;
        let full = cutile_fp8_gemm(&quantized, &scales, &weight, &weight_scales, cuda)?;
        let view = quantized.narrow(0, 0, ROWS)?;
        let viewed = cutile_fp8_gemm(&view, &scales, &weight, &weight_scales, cuda)?;
        assert_eq!(full.dims(), &[padded_rows, N]);
        assert_eq!(viewed.dims(), &[padded_rows, N]);
        let difference = full
            .narrow(0, 0, ROWS)?
            .to_dtype(DType::F32)?
            .sub(&viewed.narrow(0, 0, ROWS)?.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(difference, 0.0);
        // unpadded storage takes the copy fallback and must agree too
        let (unpadded, _) = mma::quantize_activation_padded(&x, ROWS)?;
        let copied = cutile_fp8_gemm(&unpadded, &scales, &weight, &weight_scales, cuda)?;
        let difference = full
            .narrow(0, 0, ROWS)?
            .to_dtype(DType::F32)?
            .sub(&copied.narrow(0, 0, ROWS)?.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(difference, 0.0);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fp8_gemm_matches_dequantized_reference() -> Result<()> {
        const N: usize = 256;
        const K: usize = 1024;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let weight =
            Tensor::from_vec(patterned(N * K, 3, 2.0, 0.1), (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![GROUP_SIZE, GROUP_SIZE])?;
        let weight_ref = ops::fp8_blockwise_dequantize(
            &weight,
            &weight_scales,
            vec![GROUP_SIZE, GROUP_SIZE],
            DType::F32,
        )?
        .to_device(&Device::Cpu)?;
        for rows in [33usize, 128, 300] {
            let padded_rows = rows.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
            let x = Tensor::from_vec(patterned(rows * K, rows, 3.0, -0.2), (rows, K), &dev)?
                .to_dtype(DType::BF16)?;
            let (quantized, scales) = mma::quantize_activation_padded(&x, padded_rows)?;
            let output = cutile_fp8_gemm(&quantized, &scales, &weight, &weight_scales, cuda)?
                .narrow(0, 0, rows)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;
            let cpu = Device::Cpu;
            let values = quantized
                .narrow(0, 0, rows)?
                .to_device(&cpu)?
                .to_dtype(DType::F32)?;
            let row_scales = scales
                .to_device(&cpu)?
                .t()?
                .narrow(0, 0, rows)?
                .contiguous()?;
            let dequantized = values
                .reshape((rows, K / GROUP_SIZE, GROUP_SIZE))?
                .broadcast_mul(&row_scales.reshape((rows, K / GROUP_SIZE, 1))?)?
                .reshape((rows, K))?;
            let reference = dequantized.matmul(&weight_ref.t()?)?;
            let max_error = output
                .sub(&reference)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max_error <= 1.0e-2 * max_reference,
                "rows={rows}: max error {max_error} vs reference {max_reference}"
            );
        }
        Ok(())
    }
}
