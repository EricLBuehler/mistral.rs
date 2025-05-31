use candle_core::cuda::cudarc::cublaslt::result::set_matrix_layout_attribute;
use candle_core::cuda::cudarc::cublaslt::{result, result::CublasError, sys};
use candle_core::cuda::cudarc::driver::sys::{CUdevice_attribute, CUdeviceptr, CUstream};
use candle_core::cuda::cudarc::driver::{
    CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
};
use candle_core::cuda::CudaDType;
use candle_core::DType;
use core::ffi::c_int;
use core::mem;
use float8::F8E4M3;
use half::{bf16, f16};
use std::sync::Arc;

/// Wrapper around [sys::cublasLtHandle_t]
///
/// 1. Create with [CudaBlasLT::new()]
/// 2. Execute matmul kernel with matmul. f32 is supported. f16 and bf16 are supported
///    if feature `half` is activated
///
/// Note: This maintains a instance of [`Arc<CudaDevice>`], so will prevent the device
/// from being dropped. Kernels will be launched on the device device default stream.
#[derive(Debug)]
pub struct CudaBlasLT {
    handle: sys::cublasLtHandle_t,
    workspace: Workspace,
    device: Arc<CudaDevice>,
}

unsafe impl Send for CudaBlasLT {}

unsafe impl Sync for CudaBlasLT {}

impl CudaBlasLT {
    /// Creates a new cublasLt handle.
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, CublasError> {
        let handle = result::create_handle()?;
        let workspace = Workspace::new(device.clone()).unwrap();

        Ok(Self {
            handle,
            workspace,
            device,
        })
    }
}

impl Drop for CudaBlasLT {
    fn drop(&mut self) {
        let handle = mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::destroy_handle(handle) }.unwrap();
        }
    }
}

/// User owned CublasLt workspace buffer.
/// The workspace is initialised following the Nvidia recommendations:
///
/// 1. NVIDIA Hopper Architecture: 32 MiB
/// 2. Other: 4 MiB
#[derive(Debug, Clone)]
pub struct Workspace {
    pub(crate) buffer: CudaSlice<u8>,
    pub(crate) size: usize,
}

impl Workspace {
    /// Creates a CublasLt workspace buffer on the provided device
    pub fn new(device: Arc<CudaDevice>) -> Result<Self, DriverError> {
        device.bind_to_thread()?;

        let major =
            device.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let workspace_size = if major >= 9 { 33_554_432 } else { 4_194_304 };

        let buffer = unsafe { device.alloc::<u8>(workspace_size)? };
        Ok(Self {
            buffer,
            size: workspace_size,
        })
    }
}

/// Available activation for kernel fusing in matmul
#[derive(Debug, Clone)]
pub enum Activation {
    Relu,
    Gelu,
}

/// MatrixLayout helper type
struct MatrixLayout {
    handle: sys::cublasLtMatrixLayout_t,
}

impl MatrixLayout {
    fn new(
        matrix_type: sys::cudaDataType,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> Result<Self, CublasError> {
        let handle = result::create_matrix_layout(matrix_type, rows, cols, ld)?;
        Ok(Self { handle })
    }

    fn set_batch(&self, size: c_int, stride: i64) -> Result<(), CublasError> {
        unsafe {
            // Set batch size
            set_matrix_layout_attribute(
                self.handle,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&size) as *const _ as *const _,
                mem::size_of::<c_int>(),
            )?;
            // Set batch stride
            set_matrix_layout_attribute(
                self.handle,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                (&stride) as *const _ as *const _,
                mem::size_of::<i64>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatrixLayout {
    fn drop(&mut self) {
        // panic on failure
        unsafe {
            result::destroy_matrix_layout(self.handle).expect("Unable to destroy matrix layout")
        }
    }
}

enum Matrix {
    A,
    B,
    C,
    D,
}

/// MatmulDesc helper type
struct MatmulDesc {
    handle: sys::cublasLtMatmulDesc_t,
}

impl MatmulDesc {
    fn new(
        compute_type: sys::cublasComputeType_t,
        scale_type: sys::cudaDataType,
    ) -> Result<Self, CublasError> {
        let handle = result::create_matmul_desc(compute_type, scale_type)?;
        Ok(Self { handle })
    }

    fn set_transpose(&self, transpose: bool, matrix: Matrix) -> Result<(), CublasError> {
        // Set transpose
        // 1 == T, 0 == N
        let transpose = transpose as i32;
        let attr = match matrix {
            Matrix::A => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            Matrix::B => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            Matrix::C => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSC,
            Matrix::D => unreachable!(),
        };

        unsafe {
            result::set_matmul_desc_attribute(
                self.handle,
                attr,
                (&transpose) as *const _ as *const _,
                mem::size_of::<u32>(),
            )?;
        }
        Ok(())
    }

    fn set_scale_ptr(&self, device_ptr: &CUdeviceptr, matrix: Matrix) -> Result<(), CublasError> {
        let attr = match matrix {
            Matrix::A => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            Matrix::B => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            Matrix::C => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
            Matrix::D => sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        };

        unsafe {
            result::set_matmul_desc_attribute(
                self.handle,
                attr,
                device_ptr as *const CUdeviceptr as *const _,
                mem::size_of::<CUdeviceptr>(),
            )?;
        }
        Ok(())
    }

    // Epilogue system can be leveraged to fuse add and activation operations
    fn set_epilogue(
        &self,
        act: Option<&Activation>,
        bias_ptr: Option<&CUdeviceptr>,
        stride_bias: Option<i64>,
    ) -> Result<(), CublasError> {
        let epilogue = if let Some(bias_ptr) = bias_ptr {
            let epilogue = act
                .map(|act| match act {
                    // Act + bias
                    Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU_BIAS,
                    Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
                })
                // Only bias
                .unwrap_or(sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS);

            // Set bias CUdeviceptr in matmul_desc
            unsafe {
                result::set_matmul_desc_attribute(
                    self.handle,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    bias_ptr as *const CUdeviceptr as *const _,
                    mem::size_of::<CUdeviceptr>(),
                )?;
            }

            if let Some(stride_bias) = stride_bias {
                // Set bias batch stride
                unsafe {
                    result::set_matmul_desc_attribute(
                        self.handle,
                        sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
                        (&stride_bias) as *const _ as *const _,
                        mem::size_of::<i64>(),
                    )?;
                }
            }
            epilogue
        } else if let Some(act) = act {
            // Only Act
            match act {
                Activation::Relu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_RELU,
                Activation::Gelu => sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU,
            }
        } else {
            // No epilogue
            sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT
        };

        // Set epilogue
        unsafe {
            result::set_matmul_desc_attribute(
                self.handle,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&epilogue) as *const _ as *const _,
                mem::size_of::<sys::cublasLtMatmulDescAttributes_t>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulDesc {
    fn drop(&mut self) {
        unsafe { result::destroy_matmul_desc(self.handle).expect("Unable to destroy matmul desc") }
    }
}

/// MatmulPref helper type
struct MatmulPref {
    handle: sys::cublasLtMatmulPreference_t,
}

impl MatmulPref {
    fn new() -> Result<Self, CublasError> {
        let handle = result::create_matmul_pref()?;
        Ok(Self { handle })
    }

    fn set_workspace_size(&self, size: usize) -> Result<(), CublasError> {
        unsafe {
            // Set workspace size
            result::set_matmul_pref_attribute(
                self.handle,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&size) as *const _ as *const _,
                mem::size_of::<usize>(),
            )?;
        }
        Ok(())
    }
}

impl Drop for MatmulPref {
    fn drop(&mut self) {
        unsafe { result::destroy_matmul_pref(self.handle).expect("Unable to destroy matmul pref") }
    }
}

/// [Matmul] super-trait
pub trait MatmulShared {
    /// Returns a reference to the underlying cublasLt handle.
    fn handle(&self) -> &sys::cublasLtHandle_t;

    /// Returns a reference to the underlying cublasLt workspace
    fn workspace(&self) -> &Workspace;

    /// Returns a reference to the underlying stream
    fn stream(&self) -> &CUstream;
}

/// Configuration for [Matmul]
#[derive(Debug, Copy, Clone)]
pub struct MatmulConfig {
    pub transa: bool,
    pub transb: bool,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f32,
    pub lda: i64,
    pub ldb: i64,
    pub beta: f32,
    pub ldc: i64,
    pub stride_a: Option<i64>,
    pub stride_b: Option<i64>,
    pub stride_c: Option<i64>,
    pub stride_bias: Option<i64>,
    pub batch_size: Option<c_int>,
}

pub enum OutSlice<A: DevicePtrMut<F8E4M3>, B: DevicePtrMut<bf16>> {
    F8(A),
    BF16(B),
}

pub enum CublasLTInternalDType {
    F32,
    BF16,
    F16,
    F8E4M3,
}

pub trait CublasLTDType: CudaDType + DeviceRepr {
    const T: CublasLTInternalDType;
}

impl CublasLTDType for f32 {
    const T: CublasLTInternalDType = CublasLTInternalDType::F32;
}

impl CublasLTDType for f16 {
    const T: CublasLTInternalDType = CublasLTInternalDType::F16;
}

impl CublasLTDType for bf16 {
    const T: CublasLTInternalDType = CublasLTInternalDType::BF16;
}

impl CublasLTDType for F8E4M3 {
    const T: CublasLTInternalDType = CublasLTInternalDType::F8E4M3;
}

/// Matrix matrix multiplication with elements of type `T`.
pub trait Matmul<T: CublasLTDType>: MatmulShared {
    /// Underlying CUDA Type for `T`
    fn matrix_type() -> sys::cudaDataType;

    /// Underlying CUDA Compute Type for `T`
    fn compute_type() -> sys::cublasComputeType_t;

    /// Matrix matrix multiplication. See
    /// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
    ///
    /// https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
    /// There are a few requirements:
    /// - Compute type must be f32  (upheld)
    /// - `transa && !transb` (upheld)
    /// - Scale type must be  (upheld)
    /// - A and B must be f8e4m3, but C must be bf16  (upheld)
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    #[allow(clippy::too_many_arguments)]
    unsafe fn matmul_fp8_like<
        I: DevicePtr<T>,
        C: DevicePtr<bf16>,
        OA: DevicePtrMut<F8E4M3>,
        OB: DevicePtrMut<bf16>,
        S: DevicePtr<f32>,
        B: DevicePtr<bf16>,
    >(
        &self,
        cfg: MatmulConfig,
        a: &I,
        b: &I,
        scale_a: &S,
        scale_b: &S,
        scale_d: &S,
        c: &C,
        out: &mut OutSlice<OA, OB>,
        // amax_d: &mut A,
        bias: Option<&B>,
        act: Option<&Activation>,
    ) -> Result<(), CublasError> {
        let (a_rows, a_cols) = (cfg.k, cfg.m);
        let (b_rows, b_cols) = (cfg.k, cfg.n);
        assert!(cfg.transa);
        assert!(!cfg.transb);

        // Matmul description
        let matmul_desc = MatmulDesc::new(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;

        // Set transa
        matmul_desc.set_transpose(cfg.transa, Matrix::A)?;
        // Set transb
        matmul_desc.set_transpose(cfg.transb, Matrix::B)?;

        // Creates matrix layouts
        let a_layout = MatrixLayout::new(Self::matrix_type(), a_rows, a_cols, cfg.lda)?;
        if let (Some(batch_size), Some(stride_a)) = (cfg.batch_size, cfg.stride_a) {
            a_layout.set_batch(batch_size, stride_a)?;
        }

        let b_layout = MatrixLayout::new(Self::matrix_type(), b_rows, b_cols, cfg.ldb)?;
        if let (Some(batch_size), Some(stride_b)) = (cfg.batch_size, cfg.stride_b) {
            b_layout.set_batch(batch_size, stride_b)?;
        }

        let c_layout = MatrixLayout::new(sys::cudaDataType_t::CUDA_R_16BF, cfg.m, cfg.n, cfg.ldc)?;
        if let (Some(batch_size), Some(stride_c)) = (cfg.batch_size, cfg.stride_c) {
            c_layout.set_batch(batch_size, stride_c)?;
        }

        let out_ty = match &out {
            OutSlice::F8(_) => Self::matrix_type(),
            OutSlice::BF16(_) => sys::cudaDataType_t::CUDA_R_16BF,
        };
        let d_layout = MatrixLayout::new(out_ty, cfg.m, cfg.n, cfg.ldc)?;
        if let (Some(batch_size), Some(stride_c)) = (cfg.batch_size, cfg.stride_c) {
            d_layout.set_batch(batch_size, stride_c)?;
        }

        // Set scale factors
        matmul_desc.set_scale_ptr(scale_a.device_ptr(), Matrix::A)?;
        matmul_desc.set_scale_ptr(scale_b.device_ptr(), Matrix::B)?;
        matmul_desc.set_scale_ptr(scale_d.device_ptr(), Matrix::D)?;

        // Pass amaxd ptr
        // unsafe {
        //     result::set_matmul_desc_attribute(
        //         matmul_desc.handle,
        //         sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
        //         amax_d.device_ptr_mut() as *const CUdeviceptr as *const _,
        //         mem::size_of::<CUdeviceptr>(),
        //     )
        //     ?;
        // }

        // Epilogue system can be leveraged to fuse add and activation operations
        matmul_desc.set_epilogue(act, bias.map(|b| b.device_ptr()), cfg.stride_bias)?;

        // Create matmul heuristic search preferences
        let matmul_pref = MatmulPref::new()?;

        // Set workspace size
        matmul_pref.set_workspace_size(self.workspace().size)?;

        // Get heuristic given Config, bias, act and workspace size
        let heuristic = result::get_matmul_algo_heuristic(
            *self.handle(),
            matmul_desc.handle,
            a_layout.handle,
            b_layout.handle,
            c_layout.handle,
            d_layout.handle,
            matmul_pref.handle,
        )?;

        let out_ptr = match out {
            OutSlice::BF16(s) => s.device_ptr_mut(),
            OutSlice::F8(s) => s.device_ptr_mut(),
        };

        // Launch matmul kernel
        result::matmul(
            *self.handle(),
            matmul_desc.handle,
            (&cfg.alpha) as *const _ as *const _,
            (&cfg.beta) as *const _ as *const _,
            *a.device_ptr() as *const _,
            a_layout.handle,
            *b.device_ptr() as *const _,
            b_layout.handle,
            *c.device_ptr() as *const _,
            c_layout.handle,
            *out_ptr as *mut _,
            d_layout.handle,
            (&heuristic.algo) as *const _,
            *self.workspace().buffer.device_ptr() as *const CUdeviceptr as *mut _,
            self.workspace().size,
            *self.stream() as *mut _,
        )
    }

    /// Matrix matrix multiplication. See
    /// [nvidia docs](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul)
    ///
    /// # Safety
    /// This is unsafe because improper arguments may lead to invalid
    /// memory accesses.
    unsafe fn matmul<I: DevicePtr<T>, O: DevicePtrMut<T>>(
        &self,
        cfg: MatmulConfig,
        a: &I,
        b: &I,
        c: &mut O,
        bias: Option<&I>,
        act: Option<&Activation>,
    ) -> Result<(), CublasError> {
        let (a_rows, a_cols) = if cfg.transa {
            (cfg.k, cfg.m)
        } else {
            (cfg.m, cfg.k)
        };
        let (b_rows, b_cols) = if cfg.transb {
            (cfg.n, cfg.k)
        } else {
            (cfg.k, cfg.n)
        };

        // Creates matrix layouts
        let a_layout = MatrixLayout::new(Self::matrix_type(), a_rows, a_cols, cfg.lda)?;
        if let (Some(batch_size), Some(stride_a)) = (cfg.batch_size, cfg.stride_a) {
            a_layout.set_batch(batch_size, stride_a)?;
        }

        let b_layout = MatrixLayout::new(Self::matrix_type(), b_rows, b_cols, cfg.ldb)?;
        if let (Some(batch_size), Some(stride_b)) = (cfg.batch_size, cfg.stride_b) {
            b_layout.set_batch(batch_size, stride_b)?;
        }

        let c_layout = MatrixLayout::new(Self::matrix_type(), cfg.m, cfg.n, cfg.ldc)?;
        if let (Some(batch_size), Some(stride_c)) = (cfg.batch_size, cfg.stride_c) {
            c_layout.set_batch(batch_size, stride_c)?;
        }

        // Matmul description
        let matmul_desc = MatmulDesc::new(Self::compute_type(), sys::cudaDataType_t::CUDA_R_32F)?;

        // Set transa
        matmul_desc.set_transpose(cfg.transa, Matrix::A)?;
        // Set transb
        matmul_desc.set_transpose(cfg.transb, Matrix::B)?;

        // Epilogue system can be leveraged to fuse add and activation operations
        matmul_desc.set_epilogue(act, bias.map(|b| b.device_ptr()), cfg.stride_bias)?;

        // Create matmul heuristic search preferences
        let matmul_pref = MatmulPref::new()?;

        // Set workspace size
        matmul_pref.set_workspace_size(self.workspace().size)?;

        // Get heuristic given Config, bias, act and workspace size
        let heuristic = result::get_matmul_algo_heuristic(
            *self.handle(),
            matmul_desc.handle,
            a_layout.handle,
            b_layout.handle,
            c_layout.handle,
            c_layout.handle,
            matmul_pref.handle,
        )?;

        // Launch matmul kernel
        result::matmul(
            *self.handle(),
            matmul_desc.handle,
            (&cfg.alpha) as *const _ as *const _,
            (&cfg.beta) as *const _ as *const _,
            *a.device_ptr() as *const _,
            a_layout.handle,
            *b.device_ptr() as *const _,
            b_layout.handle,
            *c.device_ptr_mut() as *const _,
            c_layout.handle,
            *c.device_ptr_mut() as *mut _,
            c_layout.handle,
            (&heuristic.algo) as *const _,
            *self.workspace().buffer.device_ptr() as *const CUdeviceptr as *mut _,
            self.workspace().size,
            *self.stream() as *mut _,
        )
    }
}

impl MatmulShared for CudaBlasLT {
    fn handle(&self) -> &sys::cublasLtHandle_t {
        &self.handle
    }

    fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    fn stream(&self) -> &CUstream {
        self.device.cu_stream()
    }
}

impl<T: CublasLTDType> Matmul<T> for CudaBlasLT {
    fn matrix_type() -> sys::cudaDataType {
        match T::T {
            CublasLTInternalDType::F32 => sys::cudaDataType_t::CUDA_R_32F,
            CublasLTInternalDType::BF16 => sys::cudaDataType_t::CUDA_R_16BF,
            CublasLTInternalDType::F16 => sys::cudaDataType_t::CUDA_R_16F,
            CublasLTInternalDType::F8E4M3 => sys::cudaDataType_t::CUDA_R_8F_E4M3,
        }
    }

    fn compute_type() -> sys::cublasComputeType_t {
        match T::T {
            CublasLTInternalDType::F32 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            CublasLTInternalDType::BF16
            | CublasLTInternalDType::F16
            | CublasLTInternalDType::F8E4M3 => sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        }
    }
}
