use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceRepr};
use candle_core::cuda::CudaDType;
use float8::F8E4M3;
use std::ffi::c_int;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Storage, Tensor, WithDType};
use half::{bf16, f16};
use std::sync::Arc;

use crate::cublaslt::matmul::{F8Scale, MatmulShared};

use super::matmul::{Activation, CublasLTDType, CudaBlasLT, Matmul, MatmulConfig};

#[derive(Debug, Clone)]
pub struct CublasLt(Arc<CudaBlasLT>);

impl CublasLt {
    pub fn new(device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("`device` must be a `cuda` device"),
        };

        let inner = CudaBlasLT::new(dev.cuda_stream()).unwrap();

        Ok(Self(Arc::new(inner)))
    }
}

pub struct CublasLTBatchMatmulF8Scalar {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
    // Dequantize
    pub a_scale: Tensor,
    pub b_scale: Tensor,
    // Quantize
    pub d_scale: Tensor,
}

impl CublasLTBatchMatmulF8Scalar {
    pub fn fwd_f8e4m3_scalar(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: Option<&candle_core::CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = a.device();

        // Validate input tensor dimensions
        let (batch_size, m, k) = a_l.shape().dims3()?;
        let (b_0, n, b_2) = b_l.shape().dims3()?;

        // Check matrix dimensions compatibility for GEMM (A: m×k, B: k×n -> C: m×n)
        if b_2 != k {
            candle_core::bail!(
                "Matrix dimension mismatch: A has k={} but B has k={} (TN layout required)",
                k,
                b_2
            );
        }

        // Validate batch dimensions consistency
        if b_0 != batch_size {
            candle_core::bail!(
                "Batch size mismatch: A has batch_size={} but B has batch_size={}",
                batch_size,
                b_0
            )
        }

        // Validate dimension sizes for tensor core requirements
        if m == 0 || n == 0 || k == 0 {
            candle_core::bail!(
                "Invalid matrix dimensions: m={}, n={}, k={} (all must be > 0)",
                m,
                n,
                k
            );
        }

        // Validate scaling tensor requirements for scalar scaling mode
        if !self.a_scale.dims().is_empty() || self.a_scale.dtype() != DType::F32 {
            candle_core::bail!(
                "`a_scale` must be a f32 scalar. Got dims: {:?}, dtype: {:?}",
                self.a_scale.dims(),
                self.a_scale.dtype()
            );
        }
        if !self.b_scale.dims().is_empty() || self.b_scale.dtype() != DType::F32 {
            candle_core::bail!(
                "`b_scale` must be a f32 scalar. Got dims: {:?}, dtype: {:?}",
                self.b_scale.dims(),
                self.b_scale.dtype()
            );
        }
        if !self.d_scale.dims().is_empty() || self.d_scale.dtype() != DType::F32 {
            candle_core::bail!(
                "`d_scale` must be a f32 scalar. Got dims: {:?}, dtype: {:?}",
                self.d_scale.dims(),
                self.d_scale.dtype()
            );
        }
        let (a_s, _) = self.a_scale.storage_and_layout();
        let (b_s, _) = self.b_scale.storage_and_layout();
        let (d_s, _) = self.d_scale.storage_and_layout();

        let a_scale = match &*a_s {
            Storage::Cuda(scale) => scale.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("`a_scale` must be a cuda tensor"),
        };
        let b_scale = match &*b_s {
            Storage::Cuda(scale) => scale.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("`b_scale` must be a cuda tensor"),
        };
        let d_scale = match &*d_s {
            Storage::Cuda(scale) => scale.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("`d_scale` must be a cuda tensor"),
        };

        // Set leading dimensions - must be at least as large as the corresponding dimension
        let lda = k.max(1); // Leading dimension of A (k for TN layout)
        let ldb = k.max(1); // Leading dimension of B (k for TN layout)
        let ldc = m.max(1); // Leading dimension of C (m for row-major output)

        // Validate leading dimensions
        if lda == 0 || ldb == 0 || ldc == 0 {
            candle_core::bail!(
                "Leading dimensions must be positive: lda={}, ldb={}, ldc={}",
                lda,
                ldb,
                ldc
            );
        }

        let out_shape = Shape::from((batch_size, n, m));

        let a = a.as_cuda_slice::<F8E4M3>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<F8E4M3>()?.slice(b_l.start_offset()..);

        let (bias, bias_stride) = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            // Validate bias data type
            if bias.dtype() != candle_core::DType::BF16 {
                candle_core::bail!(
                    "Bias must be BF16 dtype for FP8 operations. Got: {:?}",
                    bias.dtype()
                );
            }

            if bias_l.dims().len() == 1 {
                // 1D bias case - must match output dimension m
                let bias_dim = bias_l.shape().dims1()?;
                if bias_dim != m {
                    candle_core::bail!(
                        "Bias dimension mismatch: expected {} (output dimension m), got {}",
                        m,
                        bias_dim
                    );
                }
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    None,
                )
            } else if bias_l.dims().len() == 2 {
                // 2D bias case - must match batch_size x m
                let (bias_batch, bias_m) = bias_l.shape().dims2()?;
                if bias_m != m {
                    candle_core::bail!("Bias dimension mismatch: expected m={}, got {}", m, bias_m);
                }
                if bias_batch != batch_size {
                    candle_core::bail!(
                        "Bias batch size mismatch: expected {}, got {}",
                        batch_size,
                        bias_batch
                    );
                }
                let bias_stride = bias_l.stride()[0] as i64;
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    Some(bias_stride),
                )
            } else {
                candle_core::bail!("Bias must be 1D or 2D tensor, got {}D", bias_l.dims().len());
            }
        } else {
            (None, None)
        };

        let (c, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();

            // Validate C tensor data type
            if c.dtype() != DType::BF16 {
                candle_core::bail!(
                    "C tensor must be BF16 dtype for FP8 operations. Got: {:?}",
                    c.dtype()
                );
            }

            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("`c` must be a cuda tensor"),
            };

            // Validate C tensor is contiguous
            match c_l.contiguous_offsets() {
                Some((o1, o2)) => {
                    if o1 != 0 {
                        candle_core::bail!(
                            "`c` tensor must start at offset 0 for cuBLASLt. Got offset: {}",
                            o1
                        );
                    }
                    if o2 != out_shape.elem_count() {
                        candle_core::bail!(
                            "`c` tensor size mismatch: expected {} elements, got {}",
                            out_shape.elem_count(),
                            o2
                        )
                    }
                }
                None => candle_core::bail!("`c` tensor must be contiguous for cuBLASLt operations"),
            };

            // Validate C tensor shape matches output shape
            let c_shape = c_l.shape().dims3()?;
            if c_shape != (batch_size, n, m) {
                candle_core::bail!(
                    "`c` shape mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                    batch_size,
                    n,
                    m,
                    c_shape.0,
                    c_shape.1,
                    c_shape.2
                );
            }

            // Validate stride
            let expected_stride = (n * m) as i64;
            let actual_stride = c_l.stride()[0] as i64;
            if actual_stride < expected_stride {
                candle_core::bail!(
                    "C tensor stride {} is less than minimum required {}",
                    actual_stride,
                    expected_stride
                );
            }

            (c.clone(), actual_stride)
        } else {
            // Allocate out tensor
            (
                unsafe { dev.alloc::<bf16>(out_shape.elem_count())? },
                (n * m) as i64,
            )
        };
        let (mut out, _stride_c_out) = (
            unsafe { dev.alloc::<bf16>(out_shape.elem_count())? },
            (n * m) as i64,
        );

        // Check alignment requirements for FP8 tensor core usage
        // FP8 requires 16-byte alignment for dimensions and 256-byte for pointers
        let dimension_cases = [
            ("k", k * std::mem::size_of::<F8E4M3>()),
            ("m", m * std::mem::size_of::<F8E4M3>()),
            ("n", n * std::mem::size_of::<F8E4M3>()),
            ("lda", lda * std::mem::size_of::<F8E4M3>()),
            ("ldb", ldb * std::mem::size_of::<F8E4M3>()),
            ("ldc", ldc * std::mem::size_of::<F8E4M3>()),
        ];

        for (name, value) in dimension_cases {
            if value % 16 != 0 {
                candle_core::bail!("F8 cuBLASlt dimension {} has byte size {} which is not 16-byte aligned. All dimensions must be divisible by 16 bytes for tensor core usage.", name, value);
            }
        }

        // Check pointer alignment (256-byte requirement for FP8)
        let pointer_cases = [
            ("a", a.device_ptr(self.cublaslt.stream()).0 as usize),
            ("b", b.device_ptr(self.cublaslt.stream()).0 as usize),
            ("c", c.device_ptr(self.cublaslt.stream()).0 as usize),
            (
                "a_scale",
                a_scale.device_ptr(self.cublaslt.stream()).0 as usize,
            ),
            (
                "b_scale",
                b_scale.device_ptr(self.cublaslt.stream()).0 as usize,
            ),
            (
                "d_scale",
                d_scale.device_ptr(self.cublaslt.stream()).0 as usize,
            ),
        ];

        for (name, ptr) in pointer_cases {
            if ptr % 256 != 0 {
                candle_core::bail!("F8 cuBLASlt pointer {} has address {:#x} which is not 256-byte aligned. FP8 operations require 256-byte alignment.", name, ptr);
            }
        }

        // Validate strides for batch operations
        let stride_a = a_l.stride()[0] as i64;
        let stride_b = b_l.stride()[0] as i64;

        // Strides must be at least the size of one matrix
        let min_stride_a = (m * k) as i64;
        let min_stride_b = (n * k) as i64;
        let min_stride_c = (n * m) as i64;

        if stride_a < min_stride_a {
            candle_core::bail!(
                "Stride A ({}) is less than minimum required {} for matrix of size {}x{}",
                stride_a,
                min_stride_a,
                m,
                k
            );
        }
        if stride_b < min_stride_b {
            candle_core::bail!(
                "Stride B ({}) is less than minimum required {} for matrix of size {}x{}",
                stride_b,
                min_stride_b,
                n,
                k
            );
        }
        if stride_c < min_stride_c {
            candle_core::bail!(
                "Stride C ({}) is less than minimum required {} for matrix of size {}x{}",
                stride_c,
                min_stride_c,
                n,
                m
            );
        }

        let config = MatmulConfig {
            transa: true,
            transb: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha: self.alpha.unwrap_or(1.0),
            lda: lda as i64,
            ldb: ldb as i64,
            beta: self.beta.unwrap_or(0.0),
            ldc: ldc as i64,
            stride_a: Some(stride_a),
            stride_b: Some(stride_b),
            stride_c: Some(stride_c),
            stride_bias: bias_stride,
            batch_size: Some(c_int::try_from(batch_size)?),
        };

        // let mut amaxd = unsafe { dev.alloc_zeros::<f32>(1).w()? };

        let scale = F8Scale::Scalar {
            scale_a: a_scale,
            scale_b: b_scale,
            scale_d: d_scale,
        };

        unsafe {
            self.cublaslt
                .matmul_fp8_like(
                    config,
                    &a,
                    &b,
                    scale,
                    &c,
                    &mut out,
                    // &mut amaxd,
                    bias.as_ref(),
                    self.act.as_ref(),
                )
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

/// Fused batch matmul + add + Relu/Gelu activation using CublasLt for F8 dtypes.
///
/// # Arguments
///
/// * `a` - Input tensor of size BxMxK
/// * `b` - Input tensor of size BxNxK
/// * `dequant_a_scale` - F32 scalar tensor, used to `a` the out tensor.
/// * `dequant_b_scale` - F32 scalar tensor, used to `b` the out tensor.
/// * `quantize_scale` - F32 scalar tensor, used to requantize.
/// * `out` - Optional Output tensor of size BxNxK.
///   If set and beta != 0, will be added to the end result of A*B before `act`
/// * `alpha` - Optional scaling factor for A*B
/// * `beta` - Optional scaling factor for C
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
#[allow(clippy::too_many_arguments)]
pub fn fused_batch_matmul_f8(
    a: &Tensor,
    b: &Tensor,
    dequant_a_scale: &Tensor,
    dequant_b_scale: &Tensor,
    quantize_scale: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmulF8Scalar {
        act,
        cublaslt: cublaslt.0,
        c: out.cloned(),
        alpha,
        beta,
        a_scale: dequant_a_scale.clone(),
        b_scale: dequant_b_scale.clone(),
        d_scale: quantize_scale.clone(),
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

impl candle_core::CustomOp2 for CublasLTBatchMatmulF8Scalar {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-f8"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul-f8")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F8E4M3 => self.fwd_f8e4m3_scalar(a, a_l, b, b_l, None, None),
            dt => {
                candle_core::bail!("cublaslt-batch-matmul is only supported for f8e4m3 ({dt:?})")
            }
        }
    }
}

impl candle_core::CustomOp3 for CublasLTBatchMatmulF8Scalar {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-add-f8"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul-add-f8")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: &candle_core::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F8E4M3 => {
                self.fwd_f8e4m3_scalar(a, a_l, b, b_l, Some(bias), Some(bias_l))
            }
            dt => candle_core::bail!(
                "cublaslt-batch-matmul-add is only supported for f8e4m3 ({dt:?})"
            ),
        }
    }
}

pub struct CublasLTBatchMatmulF8Blockwise {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
    // Dequantize
    pub a_scale: Tensor,
    pub b_scale: Tensor,
    pub block_size: Vec<usize>,
}

impl CublasLTBatchMatmulF8Blockwise {
    pub fn fwd_f8e4m3_block(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: Option<&candle_core::CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = a.device();

        // Assume TN
        let (batch_size, m, k) = a_l.shape().dims3()?;
        let (b_0, n, b_2) = b_l.shape().dims3()?;

        if b_2 != k {
            candle_core::bail!("This layer only supports TN layout");
        }

        if b_0 != batch_size {
            candle_core::bail!("`b` must have the same batch size as `a`")
        }

        if self.block_size != vec![128, 128] {
            candle_core::bail!("Expected block size to be 128x128.");
        }
        let a_scale_shape_case = self.a_scale.dim(0)? * self.block_size[0] == a_l.dim(1)?
            && self.a_scale.dim(1)? * self.block_size[1] == a_l.dim(2)?;
        if !a_scale_shape_case
            || self.a_scale.dtype() != DType::F32
            || !self.a_scale.is_contiguous()
        {
            candle_core::bail!("`a_scale` must be a f32 contiguous blockwise tensor. Expected dims: ({}, {}), got: ({}, {}), a_l dims: {:?}", 
                a_l.dim(1)? / self.block_size[0], a_l.dim(2)? / self.block_size[1],
                self.a_scale.dim(0)?, self.a_scale.dim(1)?, a_l.dims());
        }
        let b_scale_shape_case = self.b_scale.dim(0)? * self.block_size[0] == b_l.dim(1)?
            && self.b_scale.dim(1)? * self.block_size[1] == b_l.dim(2)?;
        if !b_scale_shape_case
            || self.b_scale.dtype() != DType::F32
            || !self.b_scale.is_contiguous()
        {
            candle_core::bail!("`b_scale` must be a f32 contiguous blockwise tensor.");
        }
        let (a_s, _) = self.a_scale.storage_and_layout();
        let (b_s, _) = self.b_scale.storage_and_layout();

        let a_scale = match &*a_s {
            Storage::Cuda(scale) => scale.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("`a_scale` must be a cuda tensor"),
        };
        let b_scale = match &*b_s {
            Storage::Cuda(scale) => scale.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("`b_scale` must be a cuda tensor"),
        };

        // Set leading dimensions - must be at least as large as the corresponding dimension
        let lda = k.max(1); // Leading dimension of A (k for TN layout)
        let ldb = k.max(1); // Leading dimension of B (k for TN layout)
        let ldc = m.max(1); // Leading dimension of C (m for row-major output)

        // Validate leading dimensions
        if lda == 0 || ldb == 0 || ldc == 0 {
            candle_core::bail!(
                "Leading dimensions must be positive: lda={}, ldb={}, ldc={}",
                lda,
                ldb,
                ldc
            );
        }

        let out_shape = Shape::from((batch_size, n, m));

        let a = a.as_cuda_slice::<F8E4M3>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<F8E4M3>()?.slice(b_l.start_offset()..);

        let (bias, bias_stride) = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            // Validate bias data type
            if bias.dtype() != candle_core::DType::BF16 {
                candle_core::bail!(
                    "Bias must be BF16 dtype for FP8 operations. Got: {:?}",
                    bias.dtype()
                );
            }

            if bias_l.dims().len() == 1 {
                // 1D bias case - must match output dimension m
                let bias_dim = bias_l.shape().dims1()?;
                if bias_dim != m {
                    candle_core::bail!(
                        "Bias dimension mismatch: expected {} (output dimension m), got {}",
                        m,
                        bias_dim
                    );
                }
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    None,
                )
            } else if bias_l.dims().len() == 2 {
                // 2D bias case - must match batch_size x m
                let (bias_batch, bias_m) = bias_l.shape().dims2()?;
                if bias_m != m {
                    candle_core::bail!("Bias dimension mismatch: expected m={}, got {}", m, bias_m);
                }
                if bias_batch != batch_size {
                    candle_core::bail!(
                        "Bias batch size mismatch: expected {}, got {}",
                        batch_size,
                        bias_batch
                    );
                }
                let bias_stride = bias_l.stride()[0] as i64;
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    Some(bias_stride),
                )
            } else {
                candle_core::bail!("Bias must be 1D or 2D tensor, got {}D", bias_l.dims().len());
            }
        } else {
            (None, None)
        };

        let (c, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();

            // Validate C tensor data type
            if c.dtype() != DType::BF16 {
                candle_core::bail!(
                    "C tensor must be BF16 dtype for FP8 operations. Got: {:?}",
                    c.dtype()
                );
            }

            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("`c` must be a cuda tensor"),
            };

            // Validate C tensor is contiguous
            match c_l.contiguous_offsets() {
                Some((o1, o2)) => {
                    if o1 != 0 {
                        candle_core::bail!(
                            "`c` tensor must start at offset 0 for cuBLASLt. Got offset: {}",
                            o1
                        );
                    }
                    if o2 != out_shape.elem_count() {
                        candle_core::bail!(
                            "`c` tensor size mismatch: expected {} elements, got {}",
                            out_shape.elem_count(),
                            o2
                        )
                    }
                }
                None => candle_core::bail!("`c` tensor must be contiguous for cuBLASLt operations"),
            };

            // Validate C tensor shape matches output shape
            let c_shape = c_l.shape().dims3()?;
            if c_shape != (batch_size, n, m) {
                candle_core::bail!(
                    "`c` shape mismatch: expected ({}, {}, {}), got ({}, {}, {})",
                    batch_size,
                    n,
                    m,
                    c_shape.0,
                    c_shape.1,
                    c_shape.2
                );
            }

            // Validate stride
            let expected_stride = (n * m) as i64;
            let actual_stride = c_l.stride()[0] as i64;
            if actual_stride < expected_stride {
                candle_core::bail!(
                    "C tensor stride {} is less than minimum required {}",
                    actual_stride,
                    expected_stride
                );
            }

            (c.clone(), actual_stride)
        } else {
            // Allocate out tensor
            (
                unsafe { dev.alloc::<bf16>(out_shape.elem_count())? },
                (n * m) as i64,
            )
        };
        let (mut out, _stride_c_out) = (
            unsafe { dev.alloc::<bf16>(out_shape.elem_count())? },
            (n * m) as i64,
        );

        let cases = [
            k * std::mem::size_of::<F8E4M3>(),
            k * std::mem::size_of::<F8E4M3>(),
            m * std::mem::size_of::<F8E4M3>(),   // C type size
            lda * std::mem::size_of::<F8E4M3>(), // A type size
            ldb * std::mem::size_of::<F8E4M3>(), // B type size
            ldc * std::mem::size_of::<F8E4M3>(), // C type size
            a.device_ptr(self.cublaslt.stream()).0 as usize,
            b.device_ptr(self.cublaslt.stream()).0 as usize,
            c.device_ptr(self.cublaslt.stream()).0 as usize,
            a_scale.device_ptr(self.cublaslt.stream()).0 as usize,
            b_scale.device_ptr(self.cublaslt.stream()).0 as usize,
        ];

        for case in cases {
            if case % 16 != 0 {
                candle_core::bail!("F8 cuBLASlt matmul must match all cases described here: https://docs.nvidia.com/cuda/cublas/#tensor-core-usage");
            }
        }

        let config = MatmulConfig {
            transa: true,
            transb: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha: self.alpha.unwrap_or(1.0),
            lda: lda as i64,
            ldb: ldb as i64,
            beta: self.beta.unwrap_or(0.0),
            ldc: ldc as i64,
            stride_a: Some(a_l.stride()[0] as i64),
            stride_b: Some(b_l.stride()[0] as i64),
            stride_c: Some(stride_c as i64),
            stride_bias: bias_stride,
            batch_size: Some(c_int::try_from(batch_size)?),
        };

        // let mut amaxd = unsafe { dev.alloc_zeros::<f32>(1).w()? };

        let scale = F8Scale::Block {
            scale_a: a_scale,
            scale_b: b_scale,
        };

        unsafe {
            self.cublaslt
                .matmul_fp8_like(
                    config,
                    &a,
                    &b,
                    scale,
                    &c,
                    &mut out,
                    // &mut amaxd,
                    bias.as_ref(),
                    self.act.as_ref(),
                )
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

/// Fused batch matmul + add + Relu/Gelu activation using CublasLt for F8 dtypes.
///
/// # Arguments
///
/// * `a` - Input tensor of size BxMxK
/// * `b` - Input tensor of size BxNxK
/// * `dequant_a_scale` - F32 blockwise tensor, used to `a` the out tensor.
/// * `dequant_b_scale` - F32 blockwise tensor, used to `b` the out tensor.
/// * `out` - Optional Output tensor of size BxNxK.
///   If set and beta != 0, will be added to the end result of A*B before `act`
/// * `alpha` - Optional scaling factor for A*B
/// * `beta` - Optional scaling factor for C
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
#[allow(clippy::too_many_arguments)]
pub fn fused_batch_matmul_f8_blockwise(
    a: &Tensor,
    b: &Tensor,
    dequant_a_scale: &Tensor,
    dequant_b_scale: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    block_size: Vec<usize>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmulF8Blockwise {
        act,
        cublaslt: cublaslt.0,
        c: out.cloned(),
        alpha,
        beta,
        a_scale: dequant_a_scale.clone(),
        b_scale: dequant_b_scale.clone(),
        block_size,
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

impl candle_core::CustomOp2 for CublasLTBatchMatmulF8Blockwise {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-f8-block"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul-f8-block")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F8E4M3 => self.fwd_f8e4m3_block(a, a_l, b, b_l, None, None),
            dt => {
                candle_core::bail!(
                    "cublaslt-batch-matmul-f8-block is only supported for f8e4m3 ({dt:?})"
                )
            }
        }
    }
}

impl candle_core::CustomOp3 for CublasLTBatchMatmulF8Blockwise {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-add-f8-block"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul-add-f8-block")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: &candle_core::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F8E4M3 => {
                self.fwd_f8e4m3_block(a, a_l, b, b_l, Some(bias), Some(bias_l))
            }
            dt => candle_core::bail!(
                "cublaslt-batch-matmul-add-f8-block is only supported for f8e4m3 ({dt:?})"
            ),
        }
    }
}

pub struct CublasLTBatchMatmul {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CublasLTBatchMatmul {
    pub fn fwd<T: CublasLTDType>(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: Option<&candle_core::CudaStorage>,
        bias_l: Option<&Layout>,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = a.device();

        // Assume TN
        let (batch_size, m, k) = a_l.shape().dims3()?;
        let (b_0, n, b_2) = b_l.shape().dims3()?;

        if b_2 != k {
            candle_core::bail!("This layer only supports TN layout");
        }

        if b_0 != batch_size {
            candle_core::bail!("`b` must have the same batch size as `a`")
        }

        // Set leading dimensions - must be at least as large as the corresponding dimension
        let lda = k.max(1); // Leading dimension of A (k for TN layout)
        let ldb = k.max(1); // Leading dimension of B (k for TN layout)
        let ldc = m.max(1); // Leading dimension of C (m for row-major output)

        // Validate leading dimensions
        if lda == 0 || ldb == 0 || ldc == 0 {
            candle_core::bail!(
                "Leading dimensions must be positive: lda={}, ldb={}, ldc={}",
                lda,
                ldb,
                ldc
            );
        }

        let out_shape = Shape::from((batch_size, n, m));

        let a = a.as_cuda_slice::<T>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<T>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<T>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let (mut out, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
                _ => candle_core::bail!("`c` must be a cuda tensor"),
            };
            match c_l.contiguous_offsets() {
                Some((o1, o2)) => {
                    if o1 != 0 {
                        candle_core::bail!("`c` start offset must be 0");
                    }
                    if o2 != out_shape.elem_count() {
                        candle_core::bail!("`c` end offset must be {}", out_shape.elem_count())
                    }
                }
                None => candle_core::bail!("`c` has to be contiguous"),
            };

            if c_l.shape().dims3()? != (batch_size, n, m) {
                candle_core::bail!("`c` does not have the correct shape");
            }

            // Set beta to 0.0 if it is not set
            (c.clone(), c_l.stride()[0])
        } else {
            // Allocate out tensor
            (unsafe { dev.alloc::<T>(out_shape.elem_count())? }, (n * m))
        };

        let config = MatmulConfig {
            transa: true,
            transb: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha: self.alpha.unwrap_or(1.0),
            lda: lda as i64,
            ldb: ldb as i64,
            beta: self.beta.unwrap_or(0.0),
            ldc: ldc as i64,
            stride_a: Some(a_l.stride()[0] as i64),
            stride_b: Some(b_l.stride()[0] as i64),
            stride_c: Some(stride_c as i64),
            stride_bias: None,
            batch_size: Some(c_int::try_from(batch_size)?),
        };

        unsafe {
            self.cublaslt
                .matmul(config, &a, &b, &mut out, bias.as_ref(), self.act.as_ref())
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle_core::CustomOp2 for CublasLTBatchMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F16 => self.fwd::<f16>(a, a_l, b, b_l, None, None),
            candle_core::DType::BF16 => self.fwd::<bf16>(a, a_l, b, b_l, None, None),
            candle_core::DType::F32 => self.fwd::<f32>(a, a_l, b, b_l, None, None),
            dt => {
                candle_core::bail!(
                    "cublaslt-batch-matmul is only supported for f16/bf16/f32 ({dt:?})"
                )
            }
        }
    }
}

impl candle_core::CustomOp3 for CublasLTBatchMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-batch-matmul-add"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-batch-matmul-add")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
        bias: &candle_core::CudaStorage,
        bias_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F16 => self.fwd::<f16>(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::BF16 => self.fwd::<bf16>(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::F32 => self.fwd::<f32>(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            dt => candle_core::bail!(
                "cublaslt-batch-matmul-add is only supported for f16/bf16/f32 ({dt:?})"
            ),
        }
    }
}

/// Fused batch matmul + add + Relu/Gelu activation using CublasLt
///
/// # Arguments
///
/// * `a` - Input tensor of size BxMxK
/// * `b` - Input tensor of size BxNxK
/// * `out` - Optional Output tensor of size BxNxK.
///   If set and beta != 0, will be added to the end result of A*B before `act`
/// * `alpha` - Optional scaling factor for A*B
/// * `beta` - Optional scaling factor for C
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
#[allow(clippy::too_many_arguments)]
pub fn fused_batch_matmul(
    a: &Tensor,
    b: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmul {
        act,
        cublaslt: cublaslt.0,
        c: out.cloned(),
        alpha,
        beta,
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}
