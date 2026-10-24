use candle_core::{DType, MetalDevice};
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ConstantValues, Device, Value as ConstantValue,
};
use objc2_metal::MTLSize;

use super::{
    utils::{BufferOffset, EncoderProvider, Output},
    Kernels, MetalKernelError,
};
use crate::set_params;

const SCALE_MODE_INDEX: usize = 500;
const ALIGNED_INDEX: usize = 501;
const BLOCK_SIZE: u32 = 128;
const GENERIC: u16 = 0;
const BLOCK: u16 = 1;
const TENSOR: u16 = 2;
const CHANNEL: u16 = 3;
const GEMV_MAX_ROWS: u32 = 4;
const GEMV_SIMDGROUPS: usize = 8;
const GEMV_OUTPUTS_PER_SIMD: usize = 4;
const GEMM_SIMDGROUPS: usize = 4;
const SIMD_WIDTH: usize = 32;
const GEMM_TILE: usize = 32;
const GEMM_SMALL_TILE: usize = 8;
const GEMM_MEDIUM_TILE: usize = 16;
const GEMM_LARGE_TILE: usize = 64;
const GEMV_K_CHUNK: u32 = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Fp8LinearParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub block_y: u32,
    pub block_x: u32,
    pub scale_stride: u32,
}

pub(crate) struct Fp8LinearDispatch<'a> {
    pub dtype: DType,
    pub params: Fp8LinearParams,
    pub input: BufferOffset<'a>,
    pub weight: BufferOffset<'a>,
    pub scales: BufferOffset<'a>,
    pub output: &'a Buffer,
}

fn fp8_linear_type_name(dtype: DType) -> Result<&'static str, MetalKernelError> {
    match dtype {
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        DType::F32 => Ok("f32"),
        got => Err(MetalKernelError::DTypeMismatch {
            expected: vec![DType::F16, DType::BF16, DType::F32],
            got,
        }),
    }
}

fn fp8_linear_constants(params: &Fp8LinearParams, aligned: bool) -> ConstantValues {
    let mode = if params.block_x == BLOCK_SIZE && params.block_y == BLOCK_SIZE {
        BLOCK
    } else if params.block_x == params.k && params.scale_stride == 1 {
        if params.block_y == params.n {
            TENSOR
        } else if params.block_y == 1 {
            CHANNEL
        } else {
            GENERIC
        }
    } else {
        GENERIC
    };
    ConstantValues::new(vec![
        (SCALE_MODE_INDEX, ConstantValue::U16(mode)),
        (ALIGNED_INDEX, ConstantValue::Bool(aligned)),
    ])
}

pub(crate) fn prepare_fp8_linear(
    device: &MetalDevice,
    dtype: DType,
    params: Fp8LinearParams,
) -> Result<(), MetalKernelError> {
    let ty = fp8_linear_type_name(dtype)?;
    for aligned in [false, true] {
        for rows in [1, 2, 4] {
            Kernels::global().load_pipeline_with_constants(
                device.device(),
                format!("fp8_gemv_{ty}_r{rows}"),
                Some(fp8_linear_constants(&params, aligned)),
            )?;
        }
        for tile in [8, 16, 32, 64] {
            Kernels::global().load_pipeline_with_constants(
                device.device(),
                format!("fp8_gemm_{ty}_{tile}"),
                Some(fp8_linear_constants(&params, aligned)),
            )?;
        }
    }
    Ok(())
}

pub(crate) fn call_fp8_linear(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dispatch: Fp8LinearDispatch<'_>,
) -> Result<(), MetalKernelError> {
    let ty = fp8_linear_type_name(dispatch.dtype)?;
    let params = dispatch.params;
    let (name, grid, simdgroups, aligned) = if params.m <= GEMV_MAX_ROWS {
        let rows = params.m.next_power_of_two();
        (
            format!("fp8_gemv_{ty}_r{rows}"),
            MTLSize {
                width: (params.n as usize).div_ceil(GEMV_SIMDGROUPS * GEMV_OUTPUTS_PER_SIMD),
                height: 1,
                depth: 1,
            },
            GEMV_SIMDGROUPS,
            params.m == rows
                && (params.n as usize).is_multiple_of(GEMV_SIMDGROUPS * GEMV_OUTPUTS_PER_SIMD)
                && params.k.is_multiple_of(GEMV_K_CHUNK),
        )
    } else {
        let (tile_m, tile_n) = if params.m as usize <= GEMM_SMALL_TILE {
            (GEMM_SMALL_TILE, GEMM_TILE)
        } else if params.m as usize <= GEMM_MEDIUM_TILE {
            (GEMM_MEDIUM_TILE, GEMM_TILE)
        } else if params.m as usize > GEMM_TILE {
            (GEMM_LARGE_TILE, GEMM_TILE)
        } else {
            (GEMM_TILE, GEMM_TILE)
        };
        (
            format!("fp8_gemm_{ty}_{tile_m}"),
            MTLSize {
                width: (params.n as usize).div_ceil(tile_n),
                height: (params.m as usize).div_ceil(tile_m),
                depth: 1,
            },
            GEMM_SIMDGROUPS,
            (params.n as usize).is_multiple_of(tile_n)
                && (params.k as usize).is_multiple_of(GEMM_TILE),
        )
    };
    let pipeline = kernels.load_pipeline_with_constants(
        device,
        name,
        Some(fp8_linear_constants(&params, aligned)),
    )?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            &dispatch.input,
            &dispatch.weight,
            &dispatch.scales,
            Output::new(dispatch.output)
        )
    );
    encoder.set_bytes(4, &params);
    encoder.dispatch_thread_groups(
        grid,
        MTLSize {
            width: SIMD_WIDTH * simdgroups,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
