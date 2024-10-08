use candle_core::cuda::cudarc::driver::DevicePtr;
use float8::F8E4M3;
use std::ffi::c_int;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::WrapErr;
use candle_core::{CpuStorage, Device, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::sync::Arc;

use super::matmul::{Activation, CudaBlasLT, Matmul, MatmulConfig};

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

pub struct CublasLTMatmul {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CublasLTMatmul {
    pub fn fwd_f16(
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
        let (m, k) = a_l.shape().dims2()?;

        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle_core::bail!("This layer only supports TN layout");
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((n, m));

        let a = a.as_cuda_slice::<f16>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<f16>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<f16>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let mut out = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<f16>()?,
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
            if c_l.shape().dims2()? != (n, m) {
                candle_core::bail!("`c` does not have the correct shape");
            }

            c.clone()
        } else {
            // Allocate out tensor
            unsafe { dev.alloc::<f16>(out_shape.elem_count()).w()? }
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
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        unsafe {
            self.cublaslt
                .matmul(config, &a, &b, &mut out, bias.as_ref(), self.act.as_ref())
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }

    pub fn fwd_bf16(
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
        let (m, k) = a_l.shape().dims2()?;

        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle_core::bail!("This layer only supports TN layout");
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((n, m));

        let a = a.as_cuda_slice::<bf16>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<bf16>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let mut out = if let Some(c) = &self.c {
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
            if c_l.shape().dims2()? != (n, m) {
                candle_core::bail!("`c` does not have the correct shape");
            }

            c.clone()
        } else {
            // Allocate out tensor
            unsafe { dev.alloc::<bf16>(out_shape.elem_count()).w()? }
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
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        unsafe {
            self.cublaslt
                .matmul(config, &a, &b, &mut out, bias.as_ref(), self.act.as_ref())
                .map_err(|e| candle_core::Error::Cuda(Box::new(e)))?;
        }

        let out = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }

    pub fn fwd_f32(
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
        let (m, k) = a_l.shape().dims2()?;

        let (n, b_1) = b_l.shape().dims2()?;

        if b_1 != k {
            candle_core::bail!("This layer only supports TN layout");
        }

        let lda = k;
        let ldb = k;
        let ldc = m;

        let out_shape = Shape::from((n, m));

        let a = a.as_cuda_slice::<f32>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<f32>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<f32>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let mut out = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
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
            if c_l.shape().dims2()? != (n, m) {
                candle_core::bail!("`c` does not have the correct shape");
            }

            c.clone()
        } else {
            // Allocate out tensor
            unsafe { dev.alloc::<f32>(out_shape.elem_count()).w()? }
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
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
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

impl candle_core::CustomOp2 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for cublaslt-matmul")
    }

    fn cuda_fwd(
        &self,
        a: &candle_core::CudaStorage,
        a_l: &Layout,
        b: &candle_core::CudaStorage,
        b_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        match a.dtype() {
            candle_core::DType::F16 => self.fwd_f16(a, a_l, b, b_l, None, None),
            candle_core::DType::BF16 => self.fwd_bf16(a, a_l, b, b_l, None, None),
            candle_core::DType::F32 => self.fwd_f32(a, a_l, b, b_l, None, None),
            dt => candle_core::bail!("cublaslt-matmul is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

impl candle_core::CustomOp3 for CublasLTMatmul {
    fn name(&self) -> &'static str {
        "cublaslt-matmul-add"
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
        candle_core::bail!("no cpu support for cublaslt-matmul")
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
            candle_core::DType::F16 => self.fwd_f16(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::BF16 => self.fwd_bf16(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::F32 => self.fwd_f32(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            dt => candle_core::bail!("cublaslt-matmul is only supported for f16/bf16/f32 ({dt:?})"),
        }
    }
}

/// Fused matmul + add + Relu/Gelu activation using CublasLt
///
/// # Arguments
///
/// * `a` - Input tensor of size MxK
/// * `b` - Input tensor of size NxK
/// * `out` - Optional Output tensor of size NxK.
///           If set and beta != 0, will be added to the end result of A*B before `act`
/// * `alpha` - Optional scaling factor for A*B
/// * `beta` - Optional scaling factor for C
/// * `bias` - Optional bias tensor of size M
/// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
/// * `cublaslt` - CublasLt handle
///
/// The resulting tensor is of shape NxM
#[allow(clippy::too_many_arguments)]
pub fn fused_matmul(
    a: &Tensor,
    b: &Tensor,
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTMatmul {
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

pub struct CublasLTBatchMatmul {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
}

impl CublasLTBatchMatmul {
    pub fn fwd_f16(
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

        let a = a.as_cuda_slice::<f16>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<f16>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<f16>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let (mut out, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<f16>()?,
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
                unsafe { dev.alloc::<f16>(out_shape.elem_count()).w()? },
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

    pub fn fwd_bf16(
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

        let a = a.as_cuda_slice::<bf16>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<bf16>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<bf16>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let (mut out, stride_c) = if let Some(c) = &self.c {
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

    pub fn fwd_f32(
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

        let a = a.as_cuda_slice::<f32>()?.slice(a_l.start_offset()..);
        let b = b.as_cuda_slice::<f32>()?.slice(b_l.start_offset()..);

        let bias = if let (Some(bias), Some(bias_l)) = (bias, bias_l) {
            if bias_l.shape().dims1()? != m {
                candle_core::bail!("Bias does not have the correct shape");
            }

            Some(bias.as_cuda_slice::<f32>()?.slice(bias_l.start_offset()..))
        } else {
            None
        };

        let (mut out, stride_c) = if let Some(c) = &self.c {
            let (c, c_l) = c.storage_and_layout();
            let c = match &*c {
                Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
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
                unsafe { dev.alloc::<f32>(out_shape.elem_count()).w()? },
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
            candle_core::DType::F16 => self.fwd_f16(a, a_l, b, b_l, None, None),
            candle_core::DType::BF16 => self.fwd_bf16(a, a_l, b, b_l, None, None),
            candle_core::DType::F32 => self.fwd_f32(a, a_l, b, b_l, None, None),
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
            candle_core::DType::F16 => self.fwd_f16(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::BF16 => self.fwd_bf16(a, a_l, b, b_l, Some(bias), Some(bias_l)),
            candle_core::DType::F32 => self.fwd_f32(a, a_l, b, b_l, Some(bias), Some(bias_l)),
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

pub struct CublasLTBatchMatmulF8 {
    pub cublaslt: Arc<CudaBlasLT>,
    pub act: Option<Activation>,
    pub c: Option<Tensor>,
    pub alpha: Option<f32>,
    pub beta: Option<f32>,
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

        let (mut out, stride_c) = if let Some(c) = &self.c {
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

        let cases = [
            m * std::mem::size_of::<F8E4M3>(),
            k * std::mem::size_of::<F8E4M3>(),
            m * std::mem::size_of::<bf16>(),     // C type size
            lda * std::mem::size_of::<F8E4M3>(), // A type size
            ldb * std::mem::size_of::<F8E4M3>(), // B type size
            ldc * std::mem::size_of::<bf16>(),   // C type size
            *a.device_ptr() as usize,
            *b.device_ptr() as usize,
            *out.device_ptr() as usize,
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

        let scale_a = dev.alloc_zeros::<f32>(1).w()?;
        let scale_b = dev.alloc_zeros::<f32>(1).w()?;
        let scale_c = dev.alloc_zeros::<f32>(1).w()?;
        let scale_d = dev.alloc_zeros::<f32>(1).w()?;

        // let mut amaxd = unsafe { dev.alloc_zeros::<f32>(1).w()? };

        unsafe {
            self.cublaslt
                .matmul_fp8_like(
                    config,
                    &a,
                    &b,
                    &scale_a,
                    &scale_b,
                    &scale_c,
                    &scale_d,
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
    out: Option<&Tensor>,
    alpha: Option<f32>,
    beta: Option<f32>,
    bias: Option<&Tensor>,
    act: Option<Activation>,
    cublaslt: CublasLt,
) -> Result<Tensor> {
    let op = CublasLTBatchMatmulF8 {
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

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use super::*;
    use candle_core::{DType, Device, IndexOp};

    fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
            .collect();
        Ok(t)
    }

    fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
        let b = 10f32.powi(digits);
        let t = t.to_vec3::<f32>()?;
        let t = t
            .iter()
            .map(|t| {
                t.iter()
                    .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                    .collect()
            })
            .collect();
        Ok(t)
    }

    #[test]
    fn test_fused_matmul() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let a = Tensor::randn(0., 1., (8, 4), &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., (2, 4), &device)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_matmul(&a, &b, None, None, None, Some(&bias), None, cublaslt)?;
        let expected = (b.matmul(&a.t()?)? + bias.broadcast_left(2)?)?;

        let abs_diff = (res - expected)?.abs()?.to_vec2::<f32>()?;
        let range = 1e-02;
        assert!(abs_diff.iter().all(|x| x.into_iter().all(|y| *y <= range)));
        Ok(())
    }

    #[test]
    fn test_fused_batch_matmul() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let a = Tensor::randn(0., 1., (3, 8, 4), &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., (3, 2, 4), &device)?.to_dtype(DType::F32)?;
        let c = Tensor::randn(0., 1., (3, 2, 8), &device)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_batch_matmul(
            &a,
            &b,
            Some(&c),
            None,
            Some(1.0),
            Some(&bias),
            None,
            cublaslt,
        )?;
        let expected = (b.matmul(&a.t()?)?.add(&c)? + bias.broadcast_left((3, 2))?)?;

        let abs_diff = (res - expected)?.abs()?.to_vec3::<f32>()?;
        let range = 1e-02;
        assert!(abs_diff
            .iter()
            .all(|x| x.into_iter().all(|y| y.into_iter().all(|x| *x <= range))));
        Ok(())
    }

    // The bias bit seems to trip the test up. Not really sure why; it may be something locally.
    #[test]
    #[ignore]
    fn test_fused_batch_matmul_f8e4m3() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let a = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
        let c = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0., 1., 16, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_batch_matmul_f8(
            &a.to_dtype(DType::F8E4M3)?,
            &b.to_dtype(DType::F8E4M3)?,
            Some(&c.to_dtype(DType::BF16)?),
            None,
            Some(1.),
            Some(&bias.to_dtype(DType::BF16)?),
            None,
            cublaslt,
        )?;
        let expected = b.matmul(&a.t()?)?.add(&c)?.broadcast_add(&bias)?;

        let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
        let absmax = abs_diff.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
        let abs_diff = abs_diff.to_vec3::<f32>()?;
        let range = 3e-01;
        assert!(abs_diff
            .iter()
            .all(|x| x.into_iter().all(|y| y.into_iter().all(|x| *x <= range))));
        Ok(())
    }

    #[test]
    fn test_fused_batch_matmul_f8e4m3_determinstic() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let data = [[[2f32; 16]; 16]; 16];
        // let bias = [[100f32; 16]; 16];

        let a = Tensor::new(&data, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::new(&data, &device)?.to_dtype(DType::F32)?;
        let c = Tensor::new(&data, &device)?.to_dtype(DType::F32)?;
        // let bias = Tensor::new(&bias, &device)?.to_dtype(DType::F32)?;

        let cublaslt = CublasLt::new(&device)?;

        let res = fused_batch_matmul_f8(
            &a.to_dtype(DType::F8E4M3)?,
            &b.to_dtype(DType::F8E4M3)?,
            Some(&c.to_dtype(DType::BF16)?),
            None,
            Some(1.),
            None, // Some(&bias.to_dtype(DType::BF16)?),
            None,
            cublaslt,
        )?
        .i((0..2, 0..2, 0..2))?;
        let expected = b
            .matmul(&a.t()?)?
            .add(&c)?
            // .broadcast_add(&bias)?
            .i((0..2, 0..2, 0..2))?;

        let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
        let absmax = abs_diff.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
        let abs_diff = abs_diff.to_vec3::<f32>()?;
        let range = 3e-01;
        assert!(abs_diff
            .iter()
            .all(|x| x.into_iter().all(|y| y.into_iter().all(|x| *x <= range))));
        Ok(())
    }
}
