use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use candle_core::{CudaDevice, DType, Device, Result, Storage, Tensor};
use cutile::cutile_compiler::specialization::DivHint;

use super::warmup::{CutileKernel, WarmupProgress};

const BLOCK_SIZE: usize = 16;
const GEMV_MAX_ROWS: usize = 1;
const SMALL_MATMUL_MIN_COLUMNS: usize = 4096;
const SMALL_MATMUL_MAX_K: usize = 8192;
const SMALL_MATMUL_MAX_ELEMENTS: usize = 32 * 1024 * 1024;
const GROUPED_MIN_TOKENS: usize = 4;
const GROUPED_MIN_ROUTES_PER_EXPERT: usize = 2;
const SHARED_MIN_PROJECTIONS: usize = 2;
const SHARED_MIN_TOKENS: usize = 2;
const SHARED_MIN_K_TILES: usize = 4;
const SHARED_MIN_OUTPUT_WAVES: usize = 4;
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

/// Owned [`Nvfp4GemmArgs`], for state that outlives the borrowed layer.
#[derive(Clone)]
pub(super) struct Nvfp4Weights {
    pub(super) weights: Tensor,
    pub(super) weight_scales: Tensor,
    pub(super) weight_global_scale: Tensor,
    pub(super) activation_global_scale: Option<Tensor>,
}

impl Nvfp4Weights {
    pub(super) fn new(args: Nvfp4GemmArgs<'_>) -> Self {
        Self {
            weights: args.weights.clone(),
            weight_scales: args.weight_scales.clone(),
            weight_global_scale: args.weight_global_scale.clone(),
            activation_global_scale: args.activation_global_scale.cloned(),
        }
    }

    pub(super) fn args(&self) -> Nvfp4GemmArgs<'_> {
        Nvfp4GemmArgs {
            weights: &self.weights,
            weight_scales: &self.weight_scales,
            weight_global_scale: &self.weight_global_scale,
            activation_global_scale: self.activation_global_scale.as_ref(),
        }
    }
}

pub fn nvfp4_supported(dev: &CudaDevice) -> bool {
    super::block_scaled_mma_supported(dev)
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

pub(crate) fn cutile_nvfp4_gather_calibrated(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
    uniform_input_scale: bool,
) -> Result<Tensor> {
    launch_calibrated(x, Some(indices), args, false, uniform_input_scale)
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
    launch_calibrated(x, indices, args, compile_only, false)
}

fn validate_input(
    x: &Tensor,
    indices: Option<&Tensor>,
    args: Nvfp4GemmArgs<'_>,
) -> Result<(usize, usize, usize, usize)> {
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
            "cuTile NVFP4 requires Blackwell or newer with CUDA 13.3 or newer and a compatible tileiras"
        )
    }
    Ok((experts, n, k, rows))
}

pub(crate) fn try_cutile_nvfp4_gather_shared(
    x: &Tensor,
    indices: &Tensor,
    projections: &[Nvfp4GemmArgs<'_>],
) -> Result<Option<Vec<Tensor>>> {
    let first = projections[0];
    let shape = validate_input(x, Some(indices), first)?;
    let (experts, n, k, _) = shape;
    let (tokens, topk) = indices.dims2()?;
    let dev = x.device().as_cuda_device()?;
    let a4 = first.activation_global_scale.is_some();
    let key = super::nvfp4_tune::RoutedKey::new(dev, experts, n, k, topk, a4, x.dtype());
    let geometry = super::nvfp4_tune::routed_config(key, tokens);
    if tokens < grouped_min_tokens(experts, topk)
        && (projections.len() < SHARED_MIN_PROJECTIONS
            || first.activation_global_scale.is_none()
            || tokens
                < shared_grouped_min_tokens(
                    n,
                    k,
                    topk,
                    super::device_multiprocessor_count(dev),
                    geometry,
                ))
    {
        return Ok(None);
    }
    for &projection in projections {
        if validate_input(x, Some(indices), projection)? != shape
            || projection.activation_global_scale.is_some()
                != first.activation_global_scale.is_some()
            || !projection.weights.is_contiguous()
            || !projection.weight_scales.is_contiguous()
            || !projection.weight_global_scale.is_contiguous()
        {
            return Ok(None);
        }
    }
    validate_kernel_dimensions(&[crate::moe::cuda::moe_align_em(
        tokens,
        topk,
        experts,
        geometry.rows,
    )])?;
    let prepared = super::nvfp4_matmul::prepare_gather(x, indices, first, true, false)?;
    projections
        .iter()
        .map(|&projection| {
            super::nvfp4_matmul::launch_gather_prepared(&prepared, projection, false)
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

fn launch_calibrated(
    x: &Tensor,
    indices: Option<&Tensor>,
    args: Nvfp4GemmArgs<'_>,
    compile_only: bool,
    uniform_input_scale: bool,
) -> Result<Tensor> {
    let (experts, n, k, rows) = validate_input(x, indices, args)?;
    if let Some(indices) = indices {
        if indices.dim(0)? < grouped_min_tokens(experts, indices.dim(1)?) {
            return super::nvfp4_gemv::launch(x, Some(indices), args, compile_only);
        }
        let (tokens, topk) = indices.dims2()?;
        let key = super::nvfp4_tune::RoutedKey::new(
            x.device().as_cuda_device()?,
            experts,
            n,
            k,
            topk,
            args.activation_global_scale.is_some(),
            x.dtype(),
        );
        validate_kernel_dimensions(&[crate::moe::cuda::moe_align_em(
            tokens,
            topk,
            experts,
            super::nvfp4_tune::routed_config(key, tokens).rows,
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
        super::nvfp4_matmul::launch_gather(&x, indices, args, uniform_input_scale, compile_only)
    } else {
        super::nvfp4_matmul::launch(&x, args, compile_only)
    }
}

struct RegisteredShape {
    layer: Nvfp4Weights,
    activation_dtype: DType,
    pointer_hints: [Option<DivHint>; 4],
}

static SHAPES: OnceLock<Mutex<Vec<RegisteredShape>>> = OnceLock::new();

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
        let layer = shape.layer.args();
        layer.weights.dims() == args.weights.dims()
            && layer.weights.device().same_device(args.weights.device())
            && shape.activation_dtype == activation_dtype
            && layer.activation_global_scale.is_some() == args.activation_global_scale.is_some()
            && shape.pointer_hints == pointer_hints
    }) {
        return Ok(());
    }
    shapes.push(RegisteredShape {
        layer: Nvfp4Weights::new(args),
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

pub(crate) fn grouped_min_tokens(experts: usize, topk: usize) -> usize {
    (experts * GROUPED_MIN_ROUTES_PER_EXPERT)
        .div_ceil(topk)
        .max(GROUPED_MIN_TOKENS)
}

fn shared_grouped_min_tokens(
    n: usize,
    k: usize,
    topk: usize,
    multiprocessors: usize,
    geometry: super::nvfp4_matmul::RoutedGeometry,
) -> usize {
    if n >= k || k < SHARED_MIN_K_TILES * geometry.k {
        return usize::MAX;
    }
    SHARED_MIN_TOKENS.max(multiprocessors.div_ceil(topk)).max(
        (SHARED_MIN_OUTPUT_WAVES * multiprocessors).div_ceil(topk * n.div_ceil(geometry.columns)),
    )
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

fn grouped_warmup_rows(
    experts: usize,
    topk: usize,
    geometry: impl Fn(usize) -> super::nvfp4_matmul::RoutedGeometry,
) -> Vec<usize> {
    grouped_warmup_rows_from(experts, topk, grouped_min_tokens(experts, topk), geometry)
}

fn grouped_warmup_rows_from(
    experts: usize,
    topk: usize,
    start: usize,
    geometry: impl Fn(usize) -> super::nvfp4_matmul::RoutedGeometry,
) -> Vec<usize> {
    let period = super::nvfp4_matmul::MAX_ROUTED_ROWS * MAX_DIMENSION_DIVISOR;
    let mut seen = HashSet::new();
    [
        start,
        start.max(super::nvfp4_matmul::ROUTED_PREFILL_MIN_TOKENS),
        start.max(super::nvfp4_matmul::ROUTED_LARGE_PREFILL_MIN_TOKENS),
    ]
    .into_iter()
    .flat_map(|start| start..start + period)
    .filter(|&tokens| {
        let cfg = geometry(tokens);
        let routes = tokens * topk;
        let em = crate::moe::cuda::moe_align_em(tokens, topk, experts, cfg.rows);
        seen.insert((
            cfg,
            dimension_divisor(routes),
            dimension_divisor(em),
            dimension_divisor(em.div_ceil(cfg.rows)),
        ))
    })
    .collect()
}

impl RegisteredShape {
    fn warm(&self) -> Result<()> {
        let args = self.layer.args();
        let device = args.weights.device();
        let k = args.weights.dim(args.weights.rank() - 1)? * 2;
        if args.weights.rank() == 2 {
            super::nvfp4_tune::tune_dense(args, self.activation_dtype)?;
            let mut rows = dense_warmup_rows();
            rows.extend(super::nvfp4_tune::dense_warm_rows());
            rows.sort_unstable();
            rows.dedup();
            for rows in rows {
                let x = Tensor::zeros((rows, k), self.activation_dtype, device)?;
                launch(&x, None, args, true)?;
            }
            if let Some(global) = args.activation_global_scale {
                super::nvfp4_glu::warm_common(global, self.activation_dtype, k)?;
            }
            return Ok(());
        }
        let experts = args.weights.dim(0)?;
        let n = args.weights.dim(1)?;
        let a4 = args.activation_global_scale.is_some();
        let topks = super::warmup::registered_moe_topks(device, experts);
        for topk in topks {
            let key = super::nvfp4_tune::RoutedKey::new(
                device.as_cuda_device()?,
                experts,
                n,
                k,
                topk,
                a4,
                self.activation_dtype,
            );
            super::nvfp4_tune::tune_routed(args, key)?;
            let geometry = |tokens| super::nvfp4_tune::routed_config(key, tokens);
            let grouped_start = grouped_min_tokens(experts, topk);
            let shared_start = if args.activation_global_scale.is_some() {
                shared_grouped_min_tokens(
                    n,
                    k,
                    topk,
                    super::device_multiprocessor_count(device.as_cuda_device()?),
                    geometry(1),
                )
            } else {
                usize::MAX
            };
            for rows in ROUTED_GEMV_WARMUP_ROWS
                .into_iter()
                .chain(grouped_warmup_rows(experts, topk, geometry))
                .chain(
                    grouped_warmup_rows_from(
                        experts,
                        topk,
                        grouped_start.min(shared_start),
                        geometry,
                    )
                    .into_iter()
                    .filter(|&rows| rows < grouped_start),
                )
            {
                let indices = Tensor::zeros((rows, topk), DType::U32, device)?;
                let x = Tensor::zeros((rows, k), self.activation_dtype, device)?;
                launch(&x, Some(&indices), args, true)?;
                if rows >= shared_start && rows < grouped_start {
                    super::nvfp4_matmul::launch_gather(&x, &indices, args, true, true)?;
                }
                if topk > 1 {
                    let x = Tensor::zeros((rows, topk, k), self.activation_dtype, device)?;
                    launch(&x, Some(&indices), args, true)?;
                    if rows >= shared_start && rows < grouped_start {
                        super::nvfp4_matmul::launch_gather(&x, &indices, args, true, true)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl CutileKernel for Nvfp4Kernel {
    fn warm(&self, dev: &CudaDevice, silent: bool) -> Result<()> {
        let device = Device::Cuda(dev.clone());
        let shapes = {
            let mut registered = SHAPES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap();
            let (shapes, other_devices): (Vec<_>, Vec<_>) = std::mem::take(&mut *registered)
                .into_iter()
                .partition(|shape| shape.layer.weights.device().same_device(&device));
            *registered = other_devices;
            shapes
        };
        if shapes.is_empty() {
            return Ok(());
        }
        let progress = WarmupProgress::new("NVFP4 shapes", shapes.len(), silent);
        if let Err(error) = shapes.iter().try_for_each(|shape| {
            progress.run(
                format!(
                    "{:?} {:?}",
                    shape.activation_dtype,
                    shape.layer.weights.dims()
                ),
                || shape.warm(),
            )
        }) {
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
                dense_geometry(ROWS, WIDTH, WIDTH, true, MatmulDevice { l2_bytes }),
                (128, 64, 1)
            );
        }
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
            l2_bytes: TEST_L2_BYTES,
        };
        for rows in MEDIUM_ROWS {
            assert_eq!(dense_geometry(rows, N, K, true, device), (32, 64, 1));
            assert_eq!(dense_worker_warps(rows, N, K, true), Some(16));
            for (n, k, a4) in [(N - 1, K, true), (N, K - BLOCK_SIZE, true), (N, K, false)] {
                assert_eq!(dense_geometry(rows, n, k, a4, device), (64, 128, 1));
                assert_eq!(dense_worker_warps(rows, n, k, a4), None);
            }
        }
        assert_eq!(dense_geometry(16, N, K, true, device), (16, 64, 1));
        assert_eq!(dense_worker_warps(16, N, K, true), None);
        for rows in [33, 34, 36, 40, 48, 64, 127] {
            assert_eq!(dense_geometry(rows, N, K, true, device), (64, 128, 1));
            assert_eq!(dense_worker_warps(rows, N, K, true), None);
        }
        assert_eq!(dense_worker_warps(128, N, K, true), Some(16));
    }

    #[test]
    fn warmup_covers_batch_and_routing_specializations() {
        use super::super::nvfp4_matmul::{dense_geometry, dense_worker_warps, MatmulDevice};

        for l2_bytes in [0, TEST_L2_BYTES, TEST_L2_BYTES * 4] {
            let device = MatmulDevice { l2_bytes };
            for a4 in [false, true] {
                for (n, k) in TEST_DENSE_SHAPES {
                    let dense_key = |rows| {
                        (
                            rows <= GEMV_MAX_ROWS,
                            dense_geometry(rows, n, k, a4, device),
                            dense_worker_warps(rows, n, k, a4),
                            dimension_divisor(rows),
                        )
                    };
                    let dense_keys: HashSet<_> =
                        dense_warmup_rows().into_iter().map(dense_key).collect();
                    for rows in 1..=COVERAGE_TOKENS {
                        assert!(
                            dense_keys.contains(&dense_key(rows)),
                            "dense rows={rows} n={n} k={k} a4={a4} l2={l2_bytes}"
                        );
                    }
                }
            }
        }
        for (n, k) in [(640, 2560), (2560, 640), (73, 144)] {
            for a4 in [false, true] {
                let geometry =
                    |tokens| super::super::nvfp4_matmul::routed_geometry(tokens, n, k, a4);
                for experts in [3, 128, 256, 512] {
                    for topk in [1, 2, 8, 10, 32]
                        .into_iter()
                        .filter(|&topk| topk <= experts)
                    {
                        let grouped_key = |tokens| {
                            let cfg = geometry(tokens);
                            let routes = tokens * topk;
                            let em =
                                crate::moe::cuda::moe_align_em(tokens, topk, experts, cfg.rows);
                            (
                                cfg,
                                dimension_divisor(routes),
                                dimension_divisor(em),
                                dimension_divisor(em.div_ceil(cfg.rows)),
                            )
                        };
                        let grouped_keys: HashSet<_> = grouped_warmup_rows(experts, topk, geometry)
                            .into_iter()
                            .map(grouped_key)
                            .collect();
                        let gemv_keys: HashSet<_> = ROUTED_GEMV_WARMUP_ROWS
                            .into_iter()
                            .filter(|&tokens| tokens < grouped_min_tokens(experts, topk))
                            .map(|tokens| {
                                (dimension_divisor(tokens), dimension_divisor(tokens * topk))
                            })
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
                            assert!(grouped_keys.contains(&grouped_key(tokens)), "grouped tokens={tokens} experts={experts} topk={topk} n={n} k={k} a4={a4}");
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "nvfp4_routed_tests.rs"]
mod routed_tests;
