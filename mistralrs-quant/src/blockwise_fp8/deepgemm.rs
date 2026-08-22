use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::{CStr, CString},
    os::unix::{ffi::OsStrExt, fs::MetadataExt, fs::PermissionsExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

use candle_core::{DType, Result, Tensor};
use float8::F8E4M3;

use super::{
    ffi,
    ops::{fp8_tensor_aligned, fp8_workspace, is_sm90, FP8_BLOCK_SIZE},
};

const DENSE_SERVING_MAX_M: usize = 16;
const SERVING_M_VALUES: [usize; 19] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128,
];
const JIT_SOURCE_HASH: &str = env!("MISTRALRS_DEEPGEMM_SOURCE_HASH");

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct PreparedShape {
    device: candle_core::cuda::DeviceId,
    n: usize,
    k: usize,
}

type Preparation = std::result::Result<Arc<Prepared>, String>;
type PreparationSlot = Arc<OnceLock<Preparation>>;

static PREPARED_SHAPES: OnceLock<Mutex<HashMap<PreparedShape, PreparationSlot>>> = OnceLock::new();

#[derive(Debug)]
pub(super) struct Prepared {
    device: candle_core::cuda::DeviceId,
    n: usize,
    k: usize,
    plans: [ffi::DeepGemmPrepared; SERVING_M_VALUES.len()],
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
const DEEPGEMM_JIT_HEADERS: &[(&str, &[u8])] = &[
    (
        "deep_gemm/fp8_gemm_impl.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/fp8_gemm_impl.cuh"),
    ),
    (
        "deep_gemm/mma_utils.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/mma_utils.cuh"),
    ),
    (
        "deep_gemm/nvrtc_cutlass.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/nvrtc_cutlass.cuh"),
    ),
    (
        "deep_gemm/scheduler.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/scheduler.cuh"),
    ),
    (
        "deep_gemm/tma_utils.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/tma_utils.cuh"),
    ),
    (
        "deep_gemm/utils.cuh",
        include_bytes!("../../third_party/deepgemm_sm90/include/deep_gemm/utils.cuh"),
    ),
];

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
const DEEPGEMM_JIT_LEGAL_FILES: &[(&str, &[u8])] = &[
    (
        "NOTICE",
        include_bytes!("../../third_party/deepgemm_sm90/NOTICE"),
    ),
    (
        "LICENSE-APACHE",
        include_bytes!("../../third_party/deepgemm_sm90/LICENSE-APACHE"),
    ),
    (
        "LICENSE-MIT",
        include_bytes!("../../third_party/deepgemm_sm90/LICENSE-MIT"),
    ),
];

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn deepgemm_cache_root() -> PathBuf {
    if let Some(path) = std::env::var_os("MISTRALRS_DEEPGEMM_CACHE_DIR") {
        return PathBuf::from(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("mistralrs/deepgemm-sm90");
    }
    if let Some(path) = std::env::var_os("HOME") {
        return PathBuf::from(path).join(".cache/mistralrs/deepgemm-sm90");
    }
    let uid = unsafe { libc::geteuid() };
    std::env::temp_dir().join(format!("mistralrs-deepgemm-sm90-{uid}"))
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn ensure_private_deepgemm_cache_root(root: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(root) {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(root)?;
        }
        Err(error) => return Err(error),
    }
    let metadata = std::fs::symlink_metadata(root)?;
    if !metadata.is_dir() || metadata.uid() != unsafe { libc::geteuid() } {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "DeepGEMM cache root must be an owner-controlled directory",
        ));
    }
    if metadata.permissions().mode() & 0o777 != 0o700 {
        std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn install_deepgemm_file(root: &Path, name: &str, contents: &[u8]) -> std::io::Result<()> {
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    let target = root.join(name);
    if std::fs::read(&target).is_ok_and(|existing| existing == contents) {
        return Ok(());
    }
    let parent = target
        .parent()
        .ok_or_else(|| std::io::Error::other("DeepGEMM header path has no parent"))?;
    std::fs::create_dir_all(parent)?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = target.with_extension(format!("tmp.{}.{}", std::process::id(), sequence));
    std::fs::write(&temp, contents)?;
    match std::fs::rename(&temp, &target) {
        Ok(()) => Ok(()),
        Err(_) if std::fs::read(&target).is_ok_and(|existing| existing == contents) => {
            let _ = std::fs::remove_file(temp);
            Ok(())
        }
        Err(error) => {
            let _ = std::fs::remove_file(temp);
            Err(error)
        }
    }
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn deepgemm_include_dir() -> Result<&'static Path> {
    static INCLUDE_DIR: OnceLock<std::result::Result<PathBuf, String>> = OnceLock::new();

    let result = INCLUDE_DIR.get_or_init(|| {
        let cache_root = deepgemm_cache_root();
        ensure_private_deepgemm_cache_root(&cache_root)
            .map_err(|error| format!("failed to secure cache root: {error}"))?;
        let source_dir = cache_root.join(format!("source-0x{JIT_SOURCE_HASH}"));
        for (name, contents) in DEEPGEMM_JIT_LEGAL_FILES {
            install_deepgemm_file(&source_dir, name, contents)
                .map_err(|error| format!("failed to install {name}: {error}"))?;
        }
        let include_dir = source_dir.join("include");
        for (name, contents) in DEEPGEMM_JIT_HEADERS {
            install_deepgemm_file(&include_dir, name, contents)
                .map_err(|error| format!("failed to install {name}: {error}"))?;
        }
        Ok(include_dir)
    });
    match result {
        Ok(path) => Ok(path.as_path()),
        Err(error) => candle_core::bail!("DeepGEMM JIT include bundle is unavailable: {error}"),
    }
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn deepgemm_status_message(status: i32) -> Cow<'static, str> {
    unsafe {
        let detail = ffi::mistralrs_deepgemm_sm90_last_error();
        if !detail.is_null() {
            let detail = CStr::from_ptr(detail).to_string_lossy();
            if !detail.is_empty() {
                return Cow::Owned(detail.into_owned());
            }
        }
        let message = ffi::mistralrs_deepgemm_sm90_error_string(status);
        if message.is_null() {
            Cow::Borrowed("unknown error")
        } else {
            Cow::Owned(CStr::from_ptr(message).to_string_lossy().into_owned())
        }
    }
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn check_deepgemm_status(operation: &str, status: i32) -> Result<()> {
    if status == ffi::DEEPGEMM_SUCCESS {
        return Ok(());
    }
    let message = deepgemm_status_message(status);
    candle_core::bail!("{operation} failed: {message} (DeepGEMM status {status})")
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
pub(super) fn supported(
    weight: &Tensor,
    weight_scales: &Tensor,
    weight_block_size: &[usize],
) -> bool {
    use candle_core::Device;

    if !ffi::HAVE_DEEPGEMM_FP8_SM90_PROVIDER
        || weight_block_size != [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE]
        || weight.dtype() != DType::F8E4M3
        || weight_scales.dtype() != DType::F32
        || !weight.is_contiguous()
        || !weight_scales.is_contiguous()
        || !weight.device().same_device(weight_scales.device())
        || !fp8_tensor_aligned(weight)
        || !fp8_tensor_aligned(weight_scales)
    {
        return false;
    }
    let [n, k] = weight.dims() else {
        return false;
    };
    if *n == 0
        || *k == 0
        || n % FP8_BLOCK_SIZE != 0
        || k % FP8_BLOCK_SIZE != 0
        || weight_scales.dims() != [n / FP8_BLOCK_SIZE, k / FP8_BLOCK_SIZE]
    {
        return false;
    }
    let Device::Cuda(dev) = weight.device() else {
        return false;
    };
    is_sm90(dev)
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
fn deepgemm_plan(m: usize, n: usize, k: usize) -> Result<ffi::DeepGemmPlan> {
    let m = u32::try_from(m)
        .map_err(|_| candle_core::Error::msg("DeepGEMM M dimension exceeds u32"))?;
    let n = u32::try_from(n)
        .map_err(|_| candle_core::Error::msg("DeepGEMM N dimension exceeds u32"))?;
    let k = u32::try_from(k)
        .map_err(|_| candle_core::Error::msg("DeepGEMM K dimension exceeds u32"))?;
    let mut plan = std::mem::MaybeUninit::uninit();
    let status = unsafe { ffi::mistralrs_deepgemm_sm90_plan(m, n, k, plan.as_mut_ptr()) };
    check_deepgemm_status("DeepGEMM shape planning", status)?;
    Ok(unsafe { plan.assume_init() })
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
pub(super) fn prepare(
    weight: &Tensor,
    weight_scales: &Tensor,
    weight_block_size: &[usize],
) -> Result<Arc<Prepared>> {
    use candle_core::Device;

    if !supported(weight, weight_scales, weight_block_size) {
        candle_core::bail!("DeepGEMM does not support this blockwise FP8 weight layout")
    }
    let [n, k] = weight.dims() else {
        unreachable!()
    };
    let Device::Cuda(dev) = weight.device() else {
        unreachable!()
    };
    let prepared_key = PreparedShape {
        device: dev.id(),
        n: *n,
        k: *k,
    };
    let preparation = {
        let prepared_shapes = PREPARED_SHAPES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut prepared_shapes = prepared_shapes.lock().unwrap();
        prepared_shapes
            .entry(prepared_key)
            .or_insert_with(|| Arc::new(OnceLock::new()))
            .clone()
    };
    let preparation = preparation.get_or_init(|| {
        let prepare_inner = || -> Result<Prepared> {
            dev.cuda_stream()
                .context()
                .bind_to_thread()
                .map_err(|error| {
                    candle_core::Error::msg(format!("CUDA context binding failed: {error}"))
                })?;
            let stream = dev.cuda_stream().cu_stream() as *mut core::ffi::c_void;
            let mut include_dir = None;
            let mut plans = Vec::with_capacity(SERVING_M_VALUES.len());
            for m in SERVING_M_VALUES {
                let plan = deepgemm_plan(m, *n, *k)?;
                let mut prepared = std::mem::MaybeUninit::uninit();
                let mut status = unsafe {
                    ffi::mistralrs_deepgemm_sm90_prepare(
                        &plan,
                        std::ptr::null(),
                        stream,
                        prepared.as_mut_ptr(),
                    )
                };
                if status == ffi::DEEPGEMM_UNAVAILABLE {
                    if include_dir.is_none() {
                        include_dir = Some(
                            CString::new(deepgemm_include_dir()?.as_os_str().as_bytes()).map_err(
                                |_| {
                                    candle_core::Error::msg(
                                        "DeepGEMM include path contains a null byte",
                                    )
                                },
                            )?,
                        );
                    }
                    status = unsafe {
                        ffi::mistralrs_deepgemm_sm90_prepare(
                            &plan,
                            include_dir.as_ref().unwrap().as_ptr(),
                            stream,
                            prepared.as_mut_ptr(),
                        )
                    };
                }
                check_deepgemm_status(&format!("DeepGEMM M={m} kernel preparation"), status)?;
                plans.push(unsafe { prepared.assume_init() });
            }
            Ok(Prepared {
                device: dev.id(),
                n: *n,
                k: *k,
                plans: plans.try_into().unwrap(),
            })
        };
        prepare_inner()
            .map(Arc::new)
            .map_err(|error| error.to_string())
    });
    match preparation {
        Ok(prepared) => Ok(prepared.clone()),
        Err(error) => Err(candle_core::Error::msg(error.clone())),
    }
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
pub(super) fn serving_supported(input: &Tensor) -> bool {
    input.dims().len() == 2 && serving_shape_supported(input.dtype(), input.dims()[0])
}

pub(super) fn serving_shape_supported(dtype: DType, rows: usize) -> bool {
    dtype == DType::BF16 && serving_plan_index(rows).is_some()
}

fn serving_plan_index(rows: usize) -> Option<usize> {
    if (1..=DENSE_SERVING_MAX_M).contains(&rows) {
        Some(rows - 1)
    } else {
        SERVING_M_VALUES.binary_search(&rows).ok()
    }
}

#[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
pub(super) fn matmul(
    prepared: &Prepared,
    input: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Shape, Storage};
    use half::bf16;

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    if !serving_supported(input) {
        candle_core::bail!(
            "DeepGEMM serving requires BF16 rows in {:?}",
            SERVING_M_VALUES
        )
    }
    let input = input.contiguous()?;
    let input = if fp8_tensor_aligned(&input) {
        input
    } else {
        input.copy()?
    };
    let (m, k) = input.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if weight.dtype() != DType::F8E4M3
        || weight_scales.dtype() != DType::F32
        || !weight.is_contiguous()
        || !weight_scales.is_contiguous()
        || !weight.device().same_device(weight_scales.device())
        || weight_scales.dims() != [n / FP8_BLOCK_SIZE, k / FP8_BLOCK_SIZE]
    {
        candle_core::bail!("DeepGEMM weight tensors do not match the prepared FP8 layout")
    }
    if weight_k != k {
        candle_core::bail!(
            "DeepGEMM input K dimension {k} does not match weight K dimension {weight_k}"
        )
    }
    if !input.device().same_device(weight.device()) {
        candle_core::bail!("DeepGEMM operands must be on the same CUDA device")
    }
    let Device::Cuda(dev) = input.device() else {
        candle_core::bail!("DeepGEMM requires CUDA tensors")
    };
    if prepared.device != dev.id() || prepared.n != n || prepared.k != k {
        candle_core::bail!("DeepGEMM prepared state does not match the CUDA device or weight shape")
    }
    let plan = &prepared.plans[serving_plan_index(m).unwrap()];
    dev.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(|error| {
            candle_core::Error::msg(format!("CUDA context binding failed: {error}"))
        })?;
    let stream = dev.cuda_stream();
    let workspace = fp8_workspace(dev, plan.plan.workspace_bytes, "DeepGEMM")?
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM returned an empty workspace plan"))?;
    let mut workspace = workspace.lock().unwrap();
    let (workspace_ptr, workspace_guard) =
        slice_ptr_mut_on_stream(&mut workspace.slice, 0, &stream);

    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("DeepGEMM output shape overflows usize"))?;
    let mut output = unsafe { dev.alloc::<bf16>(output_len)? };
    let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);

    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        unreachable!()
    };
    let (input_ptr, input_guard) = slice_ptr_on_stream(
        input_storage.as_cuda_slice::<bf16>()?,
        input_layout.start_offset(),
        &stream,
    );
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Cuda(weight_storage) = &*weight_storage else {
        unreachable!()
    };
    let (weight_ptr, weight_guard) = slice_ptr_on_stream(
        weight_storage.as_cuda_slice::<F8E4M3>()?,
        weight_layout.start_offset(),
        &stream,
    );
    let (scale_storage, scale_layout) = weight_scales.storage_and_layout();
    let Storage::Cuda(scale_storage) = &*scale_storage else {
        unreachable!()
    };
    let (scale_ptr, scale_guard) = slice_ptr_on_stream(
        scale_storage.as_cuda_slice::<f32>()?,
        scale_layout.start_offset(),
        &stream,
    );

    let status = unsafe {
        ffi::mistralrs_deepgemm_sm90_gemm(
            plan,
            input_ptr as *const core::ffi::c_void,
            weight_ptr as *const core::ffi::c_void,
            scale_ptr as *const f32,
            output_ptr as *mut core::ffi::c_void,
            workspace_ptr as *mut core::ffi::c_void,
            plan.plan.workspace_bytes,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    check_deepgemm_status("DeepGEMM blockwise FP8 GEMM", status)?;
    drop((
        input_guard,
        weight_guard,
        scale_guard,
        output_guard,
        workspace_guard,
    ));
    drop(workspace);

    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        Shape::from_dims(&[m, n]),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_decode_route_covers_decode_and_verification_buckets() {
        assert!(serving_shape_supported(DType::BF16, 1));
        assert!(serving_shape_supported(DType::BF16, DENSE_SERVING_MAX_M));
        assert!(serving_shape_supported(DType::BF16, 32));
        assert!(serving_shape_supported(DType::BF16, 64));
        assert!(serving_shape_supported(DType::BF16, 128));
        assert!(!serving_shape_supported(DType::BF16, 0));
        assert!(!serving_shape_supported(
            DType::BF16,
            DENSE_SERVING_MAX_M + 1
        ));
        assert!(!serving_shape_supported(DType::BF16, 256));
        assert!(!serving_shape_supported(DType::F16, 1));
    }
}
