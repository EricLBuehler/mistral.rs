use super::marlin_ffi::{
    awq_marlin_repack, gptq_marlin_repack, marlin_awq_4bit_bf16, marlin_awq_4bit_f16,
    marlin_gptq_4bit_bf16, marlin_gptq_4bit_f16, HAVE_MARLIN_KERNELS,
};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use half::{bf16, f16};

struct MarlinMatMul {
    workspace: Tensor,
    qzeros: Option<Tensor>,
    bits: i32,
    is_awq: bool,
}

impl MarlinMatMul {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use std::ffi::c_void;
        let dev = qweight.device();
        let x_shape = x_l.dims();
        let weight_shape = qweight_l.dims();
        // let zero_shape = self.qzeros.shape().dims();
        let scale_shape = scale_l.dims();

        let pack_factor: usize = 32 / self.bits as usize;
        let size_m = x_shape[0] * x_shape[1];
        let size_k = weight_shape[0] * pack_factor * 2; //marlin format
        let size_n = weight_shape[1] / 2; //marlin format

        let mut out_shape: Vec<usize> = x_shape.to_vec();
        out_shape[x_shape.len() - 1] = size_n;
        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let input = x.as_cuda_slice::<T>()?;
        let qw = qweight.as_cuda_slice::<i32>()?;
        let qs = scale.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let input = input.slice(x_l.start_offset()..);
        let qw = qw.slice(qweight_l.start_offset()..);
        let qs = qs.slice(scale_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let in_ptr = *input.device_ptr() as *const core::ffi::c_void;
        let qw_ptr = *qw.device_ptr() as *const core::ffi::c_void;
        let qs_ptr = *qs.device_ptr() as *const core::ffi::c_void;
        let workspace_ptr = {
            let (workspace, workspace_l) = self.workspace.storage_and_layout();
            let workspace = match &*workspace {
                Storage::Cuda(p) => p,
                _ => candle::bail!("workspace must be a cuda tensor"),
            };
            let workspace_ = workspace.as_cuda_slice::<u32>()?;
            let workspace_ = workspace_.slice(workspace_l.start_offset()..);
            *workspace_.device_ptr() as *const core::ffi::c_void
        };

        let qzeros_ptr = if self.qzeros.is_some() {
            let (qzeros, qzeros_l) = self.qzeros.as_ref().unwrap().storage_and_layout();
            let qzeros = match &*qzeros {
                Storage::Cuda(p) => p,
                _ => candle::bail!("qzeros must be a cuda tensor"),
            };
            let qzeros_ = qzeros.as_cuda_slice::<i32>()?;
            let qzeros_ = qzeros_.slice(qzeros_l.start_offset()..);
            *qzeros_.device_ptr() as *const c_void
        } else {
            std::ptr::null() as *const c_void
        };

        let groupsize: i32 = if scale_shape[0] == 1 {
            -1i32
        } else {
            (size_k / scale_shape[0]) as i32
        };
        if !HAVE_MARLIN_KERNELS {
            candle_core::bail!(
                "Marlin INT4xF16 matmul kernels were not compiled, please raise an issue."
            )
        }
        if x.dtype() == DType::F16 {
            unsafe {
                if self.is_awq {
                    marlin_awq_4bit_f16(
                        in_ptr,
                        qw_ptr as *const i32,
                        qs_ptr,
                        qzeros_ptr,
                        out_ptr,
                        size_m as i32,
                        size_k as i32,
                        size_n as i32,
                        workspace_ptr,
                        groupsize,
                        *dev.cu_stream() as i64,
                    );
                } else {
                    marlin_gptq_4bit_f16(
                        in_ptr,
                        qw_ptr as *const i32,
                        qs_ptr,
                        qzeros_ptr,
                        out_ptr,
                        size_m as i32,
                        size_k as i32,
                        size_n as i32,
                        workspace_ptr,
                        groupsize,
                        *dev.cu_stream() as i64,
                    );
                }
            }
        } else if x.dtype() == DType::BF16 {
            unsafe {
                if self.is_awq {
                    marlin_awq_4bit_bf16(
                        in_ptr,
                        qw_ptr as *const i32,
                        qs_ptr,
                        qzeros_ptr,
                        out_ptr,
                        size_m as i32,
                        size_k as i32,
                        size_n as i32,
                        workspace_ptr,
                        groupsize,
                        *dev.cu_stream() as i64,
                    );
                } else {
                    marlin_gptq_4bit_bf16(
                        in_ptr,
                        qw_ptr as *const i32,
                        qs_ptr,
                        qzeros_ptr,
                        out_ptr,
                        size_m as i32,
                        size_k as i32,
                        size_n as i32,
                        workspace_ptr,
                        groupsize,
                        *dev.cu_stream() as i64,
                    );
                }
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp3 for MarlinMatMul {
    fn name(&self) -> &'static str {
        "MarlinMatMul"
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
        candle::bail!("no cpu support for MarlinMatMul")
    }

    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.cuda_fwd_t::<f16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(x, x_l, qweight, qweight_l, scale, scale_l),
            dt => candle::bail!("MarlinMatMul is only supported for f16 and bf16 ({dt:?})"),
        }
    }
}

pub fn marlin_matmul(
    x: &Tensor,
    qweight: &Tensor,
    scale: &Tensor,
    qzeros: &Option<Tensor>,
    workspace: &Tensor,
    bits: i32,
    is_awq: bool,
) -> Result<Tensor> {
    let op = MarlinMatMul {
        qzeros: qzeros.to_owned(),
        workspace: workspace.to_owned(),
        bits,
        is_awq,
    };
    x.apply_op3(qweight, scale, op)
}

struct MarlinRepack {
    perm: Option<Tensor>,
    k: i32,
    bits: i32,
    is_awq: bool, //awq or gptq
}

impl MarlinRepack {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        qweight: &CudaStorage,
        qweight_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use std::ffi::c_void;
        let dev = qweight.device();
        let q_shape = qweight_l.dims();
        let mut out_shape: Vec<usize> = q_shape.to_vec();
        let pack_factor = (32 / self.bits) as usize;
        if self.is_awq {
            out_shape[0] = out_shape[0] / pack_factor / 2;
            out_shape[1] = out_shape[1] * pack_factor * 2;
        } else {
            out_shape[0] /= 2;
            out_shape[1] *= 2;
        }

        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let q = qweight.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let q = q.slice(qweight_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let q_ptr = *q.device_ptr() as *const core::ffi::c_void;

        let perm_ptr = if self.perm.is_some() {
            let (perm_, perm_l) = self.perm.as_ref().unwrap().storage_and_layout();
            let perm_ = match &*perm_ {
                Storage::Cuda(p) => p,
                _ => candle::bail!("perm must be a cuda tensor"),
            };
            let perm_ = perm_.as_cuda_slice::<u32>()?;
            let perm_ = perm_.slice(perm_l.start_offset()..);
            *perm_.device_ptr() as *const c_void
        } else {
            std::ptr::null() as *const c_void
        };

        if HAVE_MARLIN_KERNELS {
            unsafe {
                if self.is_awq {
                    awq_marlin_repack(
                        q_ptr,
                        perm_ptr,
                        out_ptr,
                        q_shape[0] as i32,
                        q_shape[1] as i32,
                        self.bits,
                        *dev.cu_stream() as i64,
                    )
                } else {
                    gptq_marlin_repack(
                        q_ptr,
                        perm_ptr,
                        out_ptr,
                        self.k,
                        q_shape[1] as i32,
                        self.bits,
                        *dev.cu_stream() as i64,
                    )
                }
            }
        } else {
            candle_core::bail!("Not compiled with marlin kernels, but attempted to use one. Please raise an issue.");
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp1 for MarlinRepack {
    fn name(&self) -> &'static str {
        "MarlinRepack"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for MarlinRepack")
    }

    fn cuda_fwd(&self, qweight: &CudaStorage, qweight_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match qweight.dtype() {
            DType::U32 => self.cuda_fwd_t::<u32>(qweight, qweight_l),
            DType::I32 => self.cuda_fwd_t::<i32>(qweight, qweight_l),
            dt => candle::bail!("MarlinRepack is only supported for i32/u32 weight ({dt:?})"),
        }
    }
}

pub fn marlin_weight_repack(
    qweight: &Tensor,
    perm: &Option<Tensor>,
    size_k: usize,
    bits: i32,
    is_awq: bool,
) -> Result<Tensor> {
    let op = MarlinRepack {
        bits,
        perm: perm.to_owned(),
        k: size_k as i32,
        is_awq,
    };
    qweight.apply_op1(op)
}
