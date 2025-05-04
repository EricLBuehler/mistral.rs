use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceRepr};
use candle_core::cuda::CudaDType;
use float8::F8E4M3;
use std::ffi::c_int;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, DType, Device, Layout, Result, Shape, Storage, Tensor, WithDType};
use half::{bf16, f16};
use std::sync::Arc;

use super::matmul::{Activation, CublasLTDType, CudaBlasLT, Matmul, MatmulConfig, OutSlice};
use super::F8MatmulOutType;

#[derive(Debug, Clone)]
pub struct CublasLt(Arc<CudaBlasLT>);

impl CublasLt {
    pub fn new(device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("`device` must be a `cuda` device"),
        };

        let inner = CudaBlasLT::new(dev.cuda_device()).unwrap();

        Ok(Self(Arc::new(inner)))
    }
}

pub struct CublasLTBatchMatmulF8 {
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
    pub out_dtype: F8MatmulOutType,
}

impl CublasLTBatchMatmulF8 {
    pub fn fwd_f8e4m3(
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

        if !self.a_scale.dims().is_empty() || self.a_scale.dtype() != DType::F32 {
            candle_core::bail!("`a_scale` must be a f32 scalar.");
        }
        if !self.b_scale.dims().is_empty() || self.b_scale.dtype() != DType::F32 {
            candle_core::bail!("`b_scale` must be a f32 scalar.");
        }
        if !self.d_scale.dims().is_empty() || self.d_scale.dtype() != DType::F32 {
            candle_core::bail!("`d_scale` must be a f32 scalar.");
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

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((batch_size, n, m));

        let a = a.as_cuda_slice::<F8E4M3>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<F8E4M3>()?.slice(b_l.start_offset()..);

        let (bias, bias_stride) = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.dims().len() == 1 {
                if bias_l.shape().dims1()? != m {
                    candle_core::bail!("Bias does not have the correct shape");
                }
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    None,
                )
            } else {
                if bias_l.shape().dims2()?.1 != m {
                    candle_core::bail!("Bias does not have the correct shape");
                }
                if bias_l.shape().dims2()?.0 != batch_size {
                    candle_core::bail!("Bias batch size must match batch size of `a`");
                }
                let bias_stride = bias_l.stride()[0] as i64;
                (
                    Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..)),
                    Some(bias_stride),
                )
            }
        } else {
            (None, None)
        };

        let (c, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<bf16>()?,
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
            (
                unsafe { dev.alloc::<bf16>(out_shape.elem_count()).w()? },
                (n * m),
            )
        };
        let (mut out, stride_c) = match self.out_dtype {
            F8MatmulOutType::BF16 => (
                OutSlice::BF16(unsafe { dev.alloc::<bf16>(out_shape.elem_count()).w()? }),
                (n * m),
            ),
            F8MatmulOutType::F8 => (
                OutSlice::F8(unsafe { dev.alloc::<F8E4M3>(out_shape.elem_count()).w()? }),
                (n * m),
            ),
        };

        let cases = [
            k * std::mem::size_of::<F8E4M3>(),
            k * std::mem::size_of::<F8E4M3>(),
            m * std::mem::size_of::<F8E4M3>(),   // C type size
            lda * std::mem::size_of::<F8E4M3>(), // A type size
            ldb * std::mem::size_of::<F8E4M3>(), // B type size
            ldc * std::mem::size_of::<F8E4M3>(), // C type size
            *a.device_ptr() as usize,
            *b.device_ptr() as usize,
            *c.device_ptr() as usize,
            *a_scale.device_ptr() as usize,
            *b_scale.device_ptr() as usize,
            *d_scale.device_ptr() as usize,
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

        unsafe {
            self.cublaslt
                .matmul_fp8_like(
                    config,
                    &a,
                    &b,
                    a_scale,
                    b_scale,
                    d_scale,
                    &c,
                    &mut out,
                    // &mut amaxd,
                    bias.as_ref(),
                    self.act.as_ref(),
                )
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = match out {
            OutSlice::BF16(s) => candle_core::CudaStorage::wrap_cuda_slice(s, dev.clone()),
            OutSlice::F8(s) => candle_core::CudaStorage::wrap_cuda_slice(s, dev.clone()),
        };

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
///           If set and beta != 0, will be added to the end result of A*B before `act`
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
    out_dtype: F8MatmulOutType,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmulF8 {
        act,
        cublaslt: cublaslt.0,
        c: out.cloned(),
        alpha,
        beta,
        a_scale: dequant_a_scale.clone(),
        b_scale: dequant_b_scale.clone(),
        d_scale: quantize_scale.clone(),
        out_dtype,
    };

    if let Some(bias) = bias {
        a.apply_op3(b, bias, op)
    } else {
        a.apply_op2(b, op)
    }
}

impl candle_core::CustomOp2 for CublasLTBatchMatmulF8 {
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
            candle_core::DType::F8E4M3 => self.fwd_f8e4m3(a, a_l, b, b_l, None, None),
            dt => {
                candle_core::bail!("cublaslt-batch-matmul is only supported for f8e4m3 ({dt:?})")
            }
        }
    }
}

impl candle_core::CustomOp3 for CublasLTBatchMatmulF8 {
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
            candle_core::DType::F8E4M3 => self.fwd_f8e4m3(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            dt => candle_core::bail!(
                "cublaslt-batch-matmul-add is only supported for f8e4m3 ({dt:?})"
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

        let lda = k;
        let ldb = k;
        let ldc = m;

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
            (
                unsafe { dev.alloc::<T>(out_shape.elem_count()).w()? },
                (n * m),
            )
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
///           If set and beta != 0, will be added to the end result of A*B before `act`
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
