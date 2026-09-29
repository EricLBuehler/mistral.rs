use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use candle_core::{CudaDevice, DType, Device, DeviceLocation, Result, Storage, Tensor};
use cutile::cutile_compiler::specialization::DivHint;

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
const DEFAULT_DENSE_WARMUP_ROWS: [usize; 5] = [33, 34, 36, 40, 48];
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

pub fn cutile_nvfp4_quantize(x: &Tensor, global_scale: &Tensor) -> Result<(Tensor, Tensor)> {
    let (rows, k) = x.dims2()?;
    if rows == 0
        || k == 0
        || !k.is_multiple_of(BLOCK_SIZE)
        || !matches!(x.dtype(), DType::BF16 | DType::F16)
        || global_scale.dtype() != DType::F32
        || global_scale.elem_count() != 1
        || !x.device().same_device(global_scale.device())
    {
        candle_core::bail!("invalid NVFP4 activation quantization shape, dtype, or global scale");
    }
    validate_kernel_dimensions(&[rows, k])?;
    if !nvfp4_supported(x.device().as_cuda_device()?) {
        candle_core::bail!("NVFP4 quantization requires supported Blackwell CUDA and cuTile");
    }
    super::nvfp4_matmul::quantize(&x.contiguous()?, &global_scale.contiguous()?, false)
}

pub fn cutile_nvfp4_prequantized(
    packed: &Tensor,
    scales: &Tensor,
    dtype: DType,
    args: Nvfp4GemmArgs<'_>,
) -> Result<Tensor> {
    let (rows, packed_k) = packed.dims2()?;
    let (n, weight_k) = args.weights.dims2()?;
    let k = packed_k * 2;
    if rows == 0
        || n == 0
        || k == 0
        || !k.is_multiple_of(BLOCK_SIZE)
        || packed_k != weight_k
        || packed.dtype() != DType::U8
        || scales.dtype() != DType::F8E4M3
        || scales.dims() != [rows, k / BLOCK_SIZE]
        || !matches!(dtype, DType::BF16 | DType::F16)
        || args.weights.dtype() != DType::U8
        || args.weight_scales.dtype() != DType::F8E4M3
        || args.weight_scales.dims() != [n, k / BLOCK_SIZE]
        || args.weight_global_scale.dtype() != DType::F32
        || args.weight_global_scale.dims() != [n]
        || args
            .activation_global_scale
            .is_none_or(|global| global.dtype() != DType::F32 || global.elem_count() != 1)
    {
        candle_core::bail!("invalid prequantized NVFP4 matmul shape, dtype, or global scale");
    }
    for tensor in [
        scales,
        args.weights,
        args.weight_scales,
        args.weight_global_scale,
        args.activation_global_scale.unwrap(),
    ] {
        if !tensor.device().same_device(packed.device()) {
            candle_core::bail!("prequantized NVFP4 operands must be on the same device");
        }
    }
    validate_kernel_dimensions(&[rows, n, k, n * packed_k])?;
    if !nvfp4_supported(packed.device().as_cuda_device()?) {
        candle_core::bail!("NVFP4 matmul requires supported Blackwell CUDA and cuTile");
    }
    let packed = packed.contiguous()?;
    let scales = crate::utils::contiguous_fp8(scales)?;
    let weights = args.weights.contiguous()?;
    let weight_scales = crate::utils::contiguous_fp8(args.weight_scales)?;
    let weight_global_scale = args.weight_global_scale.contiguous()?;
    let activation_global_scale = args.activation_global_scale.unwrap().contiguous()?;
    super::nvfp4_matmul::launch_prequantized(
        &packed,
        &scales,
        dtype,
        Nvfp4GemmArgs {
            weights: &weights,
            weight_scales: &weight_scales,
            weight_global_scale: &weight_global_scale,
            activation_global_scale: Some(&activation_global_scale),
        },
        false,
    )
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
    let weight_scales = crate::utils::contiguous_fp8(args.weight_scales)?;
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
    pointer_hints: [Option<DivHint>; 4],
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

fn tensor_pointer_hint<T>(tensor: &Tensor) -> Result<DivHint>
where
    T: candle_core::cuda::CudaDType + candle_core::cuda::cudarc::driver::DeviceRepr,
{
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("NVFP4 warmup registration requires CUDA tensors");
    };
    let stream = storage.device.cuda_stream();
    let (pointer, _guard) = crate::utils::slice_ptr_on_stream(
        storage.as_cuda_slice::<T>()?,
        layout.start_offset(),
        &stream,
    );
    Ok(DivHint::from_ptr(pointer))
}

pub fn register_nvfp4_shape(args: Nvfp4GemmArgs<'_>, activation_dtype: DType) -> Result<()> {
    let pointer_hints = [
        Some(tensor_pointer_hint::<u8>(args.weights)?),
        Some(tensor_pointer_hint::<float8::F8E4M3>(args.weight_scales)?),
        Some(tensor_pointer_hint::<f32>(args.weight_global_scale)?),
        args.activation_global_scale
            .map(tensor_pointer_hint::<f32>)
            .transpose()?,
    ];
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if shapes.iter().any(|shape| {
        shape.weights.dims() == args.weights.dims()
            && shape.weights.device().same_device(args.weights.device())
            && shape.activation_dtype == activation_dtype
            && shape.activation_global_scale.is_some() == args.activation_global_scale.is_some()
            && shape.pointer_hints == pointer_hints
    }) {
        return Ok(());
    }
    shapes.push(RegisteredShape {
        weights: args.weights.clone(),
        weight_scales: args.weight_scales.clone(),
        weight_global_scale: args.weight_global_scale.clone(),
        activation_global_scale: args.activation_global_scale.cloned(),
        activation_dtype,
        pointer_hints,
    });
    super::warmup::mark_dirty();
    Ok(())
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
        .chain(DEFAULT_DENSE_WARMUP_ROWS)
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
            if let Some(global) = &self.activation_global_scale {
                super::nvfp4_glu::warm_common(global, self.activation_dtype, k)?;
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
    const TEST_L2_BYTES: usize = 24 * 1024 * 1024;
    const TEST_DENSE_SHAPES: [(usize, usize); 5] = [
        (73, 144),
        (4096, 4096),
        (11009, 4112),
        (4097, 8208),
        (4096, 14336),
    ];

    #[test]
    fn kernel_metadata_rejects_overflow_before_launch() -> Result<()> {
        validate_kernel_dimensions(&[1, BLOCK_SIZE, MAX_KERNEL_DIMENSION])?;
        assert!(validate_kernel_dimensions(&[MAX_KERNEL_DIMENSION + 1]).is_err());
        Ok(())
    }

    #[test]
    fn dense_grouping_respects_device_and_weight_capacity() {
        use super::super::nvfp4_matmul::{dense_geometry, MatmulDevice};

        const ROWS: usize = 128;
        const WIDTH: usize = 8192;
        const WEIGHT_BYTES: usize = WIDTH * WIDTH / 2 + WIDTH * WIDTH / BLOCK_SIZE;
        let device = MatmulDevice {
            compute_major: 12,
            l2_bytes: WEIGHT_BYTES - 1,
        };
        assert_eq!(
            dense_geometry(ROWS, WIDTH, WIDTH, true, device),
            (128, 64, 8)
        );
        assert_eq!(
            dense_geometry(ROWS, WIDTH, WIDTH, false, device),
            (128, 64, 1)
        );
        for l2_bytes in [0, WEIGHT_BYTES, WEIGHT_BYTES + 1] {
            assert_eq!(
                dense_geometry(
                    ROWS,
                    WIDTH,
                    WIDTH,
                    true,
                    MatmulDevice { l2_bytes, ..device }
                ),
                (128, 64, 1)
            );
        }
        assert_eq!(
            dense_geometry(
                ROWS,
                WIDTH,
                WIDTH,
                true,
                MatmulDevice {
                    compute_major: 10,
                    ..device
                }
            ),
            (128, 64, 1)
        );
        assert_eq!(
            dense_geometry(ROWS - 1, WIDTH, WIDTH, true, device),
            (64, 128, 1)
        );
        assert_eq!(
            dense_geometry(ROWS, WIDTH * 8, WIDTH / 4, true, device),
            (64, 128, 1)
        );
        assert_eq!(
            dense_geometry(ROWS, WIDTH / 4, WIDTH * 8, true, device),
            (64, 128, 1)
        );
    }

    #[test]
    fn medium_dense_geometry_preserves_untuned_paths() {
        use super::super::nvfp4_matmul::{dense_geometry, dense_worker_warps, MatmulDevice};

        const N: usize = 1024;
        const K: usize = 4096;
        const MEDIUM_ROWS: [usize; 7] = [17, 18, 20, 23, 24, 31, 32];
        let device = MatmulDevice {
            compute_major: 12,
            l2_bytes: TEST_L2_BYTES,
        };
        for rows in MEDIUM_ROWS {
            assert_eq!(dense_geometry(rows, N, K, true, device), (32, 64, 1));
            assert_eq!(dense_worker_warps(rows, N, K, true, device), Some(16));
            for (n, k, a4, compute_major) in [
                (N - 1, K, true, 12),
                (N, K - BLOCK_SIZE, true, 12),
                (N, K, false, 12),
                (N, K, true, 10),
                (N, K, true, 11),
                (N, K, true, 13),
            ] {
                let device = MatmulDevice {
                    compute_major,
                    ..device
                };
                assert_eq!(dense_geometry(rows, n, k, a4, device), (64, 128, 1));
                assert_eq!(dense_worker_warps(rows, n, k, a4, device), None);
            }
        }
        assert_eq!(dense_geometry(16, N, K, true, device), (16, 64, 1));
        assert_eq!(dense_worker_warps(16, N, K, true, device), None);
        for rows in [33, 34, 36, 40, 48, 64, 127] {
            assert_eq!(dense_geometry(rows, N, K, true, device), (64, 128, 1));
            assert_eq!(dense_worker_warps(rows, N, K, true, device), None);
        }
        assert_eq!(dense_worker_warps(128, N, K, true, device), Some(16));
    }

    #[test]
    fn warmup_covers_batch_and_routing_specializations() {
        use super::super::nvfp4_matmul::{dense_geometry, dense_worker_warps, MatmulDevice};

        for compute_major in [10, 12] {
            for l2_bytes in [0, TEST_L2_BYTES, TEST_L2_BYTES * 4] {
                let device = MatmulDevice {
                    compute_major,
                    l2_bytes,
                };
                for a4 in [false, true] {
                    for (n, k) in TEST_DENSE_SHAPES {
                        let dense_key = |rows| {
                            (
                                rows <= GEMV_MAX_ROWS,
                                dense_geometry(rows, n, k, a4, device),
                                dense_worker_warps(rows, n, k, a4, device),
                                dimension_divisor(rows),
                            )
                        };
                        let dense_keys: HashSet<_> =
                            dense_warmup_rows().into_iter().map(dense_key).collect();
                        for rows in 1..=COVERAGE_TOKENS {
                            assert!(dense_keys.contains(&dense_key(rows)), "dense rows={rows} n={n} k={k} a4={a4} major={compute_major} l2={l2_bytes}");
                        }
                    }
                }
            }
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
