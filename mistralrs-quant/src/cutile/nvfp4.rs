use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use candle_core::{CudaDevice, DType, Device, DeviceLocation, Result, Tensor};

use super::warmup::CutileKernel;
use super::{device_compute_major, jit_available};

const BLOCK_SIZE: usize = 16;
const MIN_CUDA_VERSION: u32 = 1303;
const MIN_TILEIRAS_VERSION: (u32, u32) = (13, 3);
const GEMV_MAX_ROWS: usize = 1;
const SMALL_MATMUL_MIN_COLUMNS: usize = 4096;
const SMALL_MATMUL_MAX_K: usize = 8192;
const SMALL_MATMUL_MAX_ELEMENTS: usize = 32 * 1024 * 1024;
const GROUPED_MIN_TOKENS: usize = 4;
const GROUPED_MIN_ROUTES_PER_EXPERT: usize = 2;
const MAX_KERNEL_DIMENSION: usize = i32::MAX as usize;
// cuTile specializes dynamic dimensions by powers of two capped at 16.
const MAX_DIMENSION_DIVISOR: usize = 16;
const DIMENSION_DIVISORS: [usize; 5] = [1, 2, 4, 8, 16];
const LARGE_DENSE_WARMUP_ROWS: [usize; 5] = [129, 130, 132, 136, 128];
const PREFILL_WARMUP_ROWS: [usize; 5] = [17, 18, 20, 24, 32];
const ROUTED_GEMV_WARMUP_ROWS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(Clone, Copy)]
pub struct Nvfp4GemmArgs<'a> {
    pub weights: &'a Tensor,
    pub weight_scales: &'a Tensor,
    pub weight_global_scale: &'a Tensor,
    pub activation_global_scale: Option<&'a Tensor>,
}

pub fn nvfp4_supported(dev: &CudaDevice) -> bool {
    device_compute_major(dev) >= 10
        && super::build_cuda_version_code().is_some_and(|version| version >= MIN_CUDA_VERSION)
        && super::tileiras_capabilities()
            .is_some_and(|capabilities| capabilities.version >= MIN_TILEIRAS_VERSION)
        && jit_available(dev)
}

pub fn cutile_nvfp4(x: &Tensor, args: Nvfp4GemmArgs<'_>) -> Result<Tensor> {
    launch(x, None, args, false)
}

pub fn cutile_nvfp4_gather(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
) -> Result<Tensor> {
    launch(x, Some(indices), args, false)
}

fn validate_kernel_dimensions(dimensions: &[usize]) -> Result<()> {
    if dimensions
        .iter()
        .any(|&dimension| dimension > MAX_KERNEL_DIMENSION)
    {
        candle_core::bail!(
            "cuTile NVFP4 dimensions and tensor strides must fit signed 32-bit indexing"
        )
    }
    Ok(())
}

fn launch(
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
    let (rows, k) = if let Some(indices) = indices {
        let (tokens, topk) = indices.dims2()?;
        let k = match x.dims() {
            &[m, k] if m == tokens => k,
            &[m, 1, k] if m == tokens => k,
            &[m, routes, k] if m == tokens && routes == topk => k,
            dims => candle_core::bail!(
                "cuTile NVFP4 gather activation shape {dims:?} does not match indices {:?}",
                indices.dims()
            ),
        };
        if indices.dtype() != DType::U32 || !indices.device().same_device(x.device()) {
            candle_core::bail!("cuTile NVFP4 gather indices must be U32 on the activation device")
        }
        (tokens * topk, k)
    } else {
        let (rows, k) = x.dims2()?;
        (rows, k)
    };
    if rows == 0
        || experts == 0
        || n == 0
        || k == 0
        || !k.is_multiple_of(BLOCK_SIZE)
        || packed_k * 2 != k
    {
        candle_core::bail!(
            "cuTile NVFP4 got unsupported shape rows={rows} n={n} k={k} packed_k={packed_k}"
        )
    }
    validate_kernel_dimensions(&[experts, n, k, rows, n * packed_k])?;
    let scale_shape = if indices.is_some() {
        vec![experts, n, k / BLOCK_SIZE]
    } else {
        vec![n, k / BLOCK_SIZE]
    };
    let global_shape = if indices.is_some() {
        vec![experts, n]
    } else {
        vec![n]
    };
    if !matches!(x.dtype(), DType::BF16 | DType::F16)
        || args.weights.dtype() != DType::U8
        || args.weight_scales.dtype() != DType::F8E4M3
        || args.weight_global_scale.dtype() != DType::F32
        || args.weight_scales.dims() != scale_shape
        || args.weight_global_scale.dims() != global_shape
    {
        candle_core::bail!("cuTile NVFP4 requires A16 inputs, U8 packed weights, E4M3 block scales, and F32 global scales with matching dimensions")
    }
    if args
        .activation_global_scale
        .is_some_and(|scale| scale.dtype() != DType::F32 || scale.elem_count() != experts)
    {
        candle_core::bail!(
            "cuTile NVFP4 activation global scale must contain one F32 value per expert"
        )
    }
    for tensor in [
        Some(args.weights),
        Some(args.weight_scales),
        Some(args.weight_global_scale),
        args.activation_global_scale,
    ]
    .into_iter()
    .flatten()
    {
        if !x.device().same_device(tensor.device()) {
            candle_core::bail!("cuTile NVFP4 operands must be on the same device")
        }
    }
    let Device::Cuda(dev) = x.device() else {
        candle_core::bail!("cuTile NVFP4 requires CUDA tensors")
    };
    if !nvfp4_supported(dev) {
        candle_core::bail!(
            "cuTile NVFP4 requires Blackwell or newer with CUDA 13.3 and a compatible tileiras"
        )
    }
    if let Some(indices) = indices {
        if indices.dim(0)? < grouped_min_tokens(experts, indices.dim(1)?) {
            return super::nvfp4_gemv::launch(x, Some(indices), args, compile_only);
        }
        validate_kernel_dimensions(&[crate::moe::cuda::moe_align_em(
            indices.dim(0)?,
            indices.dim(1)?,
            experts,
            super::nvfp4_matmul::ROUTED_ROWS,
        )])?;
    } else if rows <= GEMV_MAX_ROWS
        && !(args.activation_global_scale.is_some()
            && n >= SMALL_MATMUL_MIN_COLUMNS
            && k < SMALL_MATMUL_MAX_K
            && n * k <= SMALL_MATMUL_MAX_ELEMENTS)
    {
        return super::nvfp4_gemv::launch(x, indices, args, compile_only);
    }
    let x = x.contiguous()?;
    let weights = args.weights.contiguous()?;
    let weight_scales = args.weight_scales.contiguous()?;
    let weight_global_scale = args.weight_global_scale.contiguous()?;
    let activation_global_scale = args
        .activation_global_scale
        .map(Tensor::contiguous)
        .transpose()?;
    let args = Nvfp4GemmArgs {
        weights: &weights,
        weight_scales: &weight_scales,
        weight_global_scale: &weight_global_scale,
        activation_global_scale: activation_global_scale.as_ref(),
    };
    if let Some(indices) = indices {
        super::nvfp4_matmul::launch_gather(&x, indices, args, compile_only)
    } else {
        super::nvfp4_matmul::launch(&x, args, compile_only)
    }
}

struct RegisteredShape {
    weights: Tensor,
    weight_scales: Tensor,
    weight_global_scale: Tensor,
    activation_global_scale: Option<Tensor>,
    activation_dtype: DType,
}

static SHAPES: OnceLock<Mutex<Vec<RegisteredShape>>> = OnceLock::new();

#[derive(Clone, Copy, PartialEq, Eq)]
struct RegisteredRouting {
    device: DeviceLocation,
    experts: usize,
    topk: usize,
}

static ROUTING: OnceLock<Mutex<Vec<RegisteredRouting>>> = OnceLock::new();

/// Register routing before loading expert weights so warmup covers the model's top-k.
pub fn register_nvfp4_routing(device: &Device, experts: usize, topk: usize) -> Result<()> {
    if experts == 0 || topk == 0 || topk > experts {
        candle_core::bail!("NVFP4 routing requires 1 <= topk <= experts")
    }
    if !device.is_cuda() {
        return Ok(());
    }
    let entry = RegisteredRouting {
        device: device.location(),
        experts,
        topk,
    };
    let mut routing = ROUTING
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if !routing.contains(&entry) {
        routing.push(entry);
        super::warmup::mark_dirty();
    }
    Ok(())
}

pub fn register_nvfp4_shape(args: Nvfp4GemmArgs<'_>, activation_dtype: DType) {
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if shapes.iter().any(|shape| {
        shape.weights.dims() == args.weights.dims()
            && shape.weights.device().same_device(args.weights.device())
            && shape.activation_dtype == activation_dtype
            && shape.activation_global_scale.is_some() == args.activation_global_scale.is_some()
    }) {
        return;
    }
    shapes.push(RegisteredShape {
        weights: args.weights.clone(),
        weight_scales: args.weight_scales.clone(),
        weight_global_scale: args.weight_global_scale.clone(),
        activation_global_scale: args.activation_global_scale.cloned(),
        activation_dtype,
    });
    super::warmup::mark_dirty();
}

pub struct Nvfp4Kernel;
pub(super) static NVFP4: Nvfp4Kernel = Nvfp4Kernel;

fn dimension_divisor(value: usize) -> usize {
    (1usize << value.trailing_zeros()).min(MAX_DIMENSION_DIVISOR)
}

fn grouped_min_tokens(experts: usize, topk: usize) -> usize {
    (experts * GROUPED_MIN_ROUTES_PER_EXPERT)
        .div_ceil(topk)
        .max(GROUPED_MIN_TOKENS)
}

fn dense_warmup_rows() -> Vec<usize> {
    let mut rows: Vec<_> = DIMENSION_DIVISORS
        .into_iter()
        .filter(|&rows| rows <= GEMV_MAX_ROWS)
        .chain(DIMENSION_DIVISORS.map(|divisor| {
            let rows = (GEMV_MAX_ROWS + 1).div_ceil(divisor) * divisor;
            if dimension_divisor(rows) == divisor {
                rows
            } else {
                rows + divisor
            }
        }))
        .chain(PREFILL_WARMUP_ROWS)
        .chain(LARGE_DENSE_WARMUP_ROWS)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn grouped_warmup_rows(experts: usize, topk: usize) -> Vec<usize> {
    let block = super::nvfp4_matmul::ROUTED_ROWS;
    let period = block * MAX_DIMENSION_DIVISOR;
    let start = grouped_min_tokens(experts, topk);
    let mut seen = HashSet::new();
    (start..start + period)
        .filter(|&tokens| {
            let routes = tokens * topk;
            let em = crate::moe::cuda::moe_align_em(tokens, topk, experts, block);
            seen.insert((
                dimension_divisor(routes),
                dimension_divisor(em),
                dimension_divisor(em.div_ceil(block)),
            ))
        })
        .collect()
}

impl RegisteredShape {
    fn args(&self) -> Nvfp4GemmArgs<'_> {
        Nvfp4GemmArgs {
            weights: &self.weights,
            weight_scales: &self.weight_scales,
            weight_global_scale: &self.weight_global_scale,
            activation_global_scale: self.activation_global_scale.as_ref(),
        }
    }

    fn warm(&self) -> Result<()> {
        let device = self.weights.device();
        let k = self.weights.dim(self.weights.rank() - 1)? * 2;
        if self.weights.rank() == 2 {
            for rows in dense_warmup_rows() {
                let x = Tensor::zeros((rows, k), self.activation_dtype, device)?;
                launch(&x, None, self.args(), true)?;
            }
            return Ok(());
        }
        let experts = self.weights.dim(0)?;
        let mut topks: Vec<_> = ROUTING
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .iter()
            .filter(|entry| entry.device == device.location() && entry.experts == experts)
            .map(|entry| entry.topk)
            .collect();
        if topks.is_empty() {
            topks.push(1);
        }
        for topk in topks {
            for rows in ROUTED_GEMV_WARMUP_ROWS
                .into_iter()
                .chain(grouped_warmup_rows(experts, topk))
            {
                let indices = Tensor::zeros((rows, topk), DType::U32, device)?;
                let x = Tensor::zeros((rows, k), self.activation_dtype, device)?;
                launch(&x, Some(&indices), self.args(), true)?;
                if topk > 1 {
                    let x = Tensor::zeros((rows, topk, k), self.activation_dtype, device)?;
                    launch(&x, Some(&indices), self.args(), true)?;
                }
            }
        }
        Ok(())
    }
}

impl CutileKernel for Nvfp4Kernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        let device = Device::Cuda(dev.clone());
        let shapes = {
            let mut registered = SHAPES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap();
            let (shapes, other_devices): (Vec<_>, Vec<_>) = std::mem::take(&mut *registered)
                .into_iter()
                .partition(|shape| shape.weights.device().same_device(&device));
            *registered = other_devices;
            shapes
        };
        if let Err(error) = shapes.iter().try_for_each(RegisteredShape::warm) {
            SHAPES.get().unwrap().lock().unwrap().extend(shapes);
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COVERAGE_TOKENS: usize = 8192;
    const SMALL_MATMUL_ROWS: usize = 16;
    const WIDE_MATMUL_ROWS: usize = 128;

    #[test]
    fn kernel_metadata_rejects_overflow_before_launch() -> Result<()> {
        validate_kernel_dimensions(&[1, BLOCK_SIZE, MAX_KERNEL_DIMENSION])?;
        assert!(validate_kernel_dimensions(&[MAX_KERNEL_DIMENSION + 1]).is_err());
        Ok(())
    }

    #[test]
    fn warmup_covers_batch_and_routing_specializations() {
        let dense_key = |rows| {
            let path = if rows <= GEMV_MAX_ROWS {
                0
            } else if rows <= SMALL_MATMUL_ROWS {
                1
            } else if rows < WIDE_MATMUL_ROWS {
                2
            } else {
                3
            };
            (path, dimension_divisor(rows))
        };
        let dense_keys: HashSet<_> = dense_warmup_rows().into_iter().map(dense_key).collect();
        for rows in 1..=COVERAGE_TOKENS {
            assert!(dense_keys.contains(&dense_key(rows)), "dense rows={rows}");
        }
        for experts in [3, 128, 256] {
            for topk in [1, 2, 8, 32].into_iter().filter(|&topk| topk <= experts) {
                let grouped_key = |tokens| {
                    let block = super::super::nvfp4_matmul::ROUTED_ROWS;
                    let routes = tokens * topk;
                    let em = crate::moe::cuda::moe_align_em(tokens, topk, experts, block);
                    (
                        dimension_divisor(routes),
                        dimension_divisor(em),
                        dimension_divisor(em.div_ceil(block)),
                    )
                };
                let grouped_keys: HashSet<_> = grouped_warmup_rows(experts, topk)
                    .into_iter()
                    .map(grouped_key)
                    .collect();
                let gemv_keys: HashSet<_> = ROUTED_GEMV_WARMUP_ROWS
                    .into_iter()
                    .filter(|&tokens| tokens < grouped_min_tokens(experts, topk))
                    .map(|tokens| (dimension_divisor(tokens), dimension_divisor(tokens * topk)))
                    .collect();
                for tokens in 1..grouped_min_tokens(experts, topk) {
                    assert!(
                        gemv_keys.contains(&(
                            dimension_divisor(tokens),
                            dimension_divisor(tokens * topk)
                        )),
                        "routed GEMV tokens={tokens} experts={experts} topk={topk}"
                    );
                }
                for tokens in grouped_min_tokens(experts, topk)..=COVERAGE_TOKENS {
                    assert!(
                        grouped_keys.contains(&grouped_key(tokens)),
                        "grouped tokens={tokens} experts={experts} topk={topk}"
                    );
                }
            }
        }
    }
}
