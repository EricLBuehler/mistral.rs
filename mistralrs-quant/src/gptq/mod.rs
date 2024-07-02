use candle_core::{
    cuda::{
        cudarc::{cublas::sys::cublasHandle_t, driver::DevicePtr},
        CudaStorageSlice, WrapErr,
    },
    from_storage_no_op, CudaDevice, CudaStorage, Device, Result, Shape, Storage, Tensor, WithDType,
    D,
};
use cudarc::driver::CudaSlice;
use ffi::gemm_half_q_half_cuda;
use half::f16;

mod ffi;

pub struct GptQMatMul {
    q_weight: Tensor,
    gptq_qzeros: Tensor,
    qptq_scales: Tensor,
    g_idx: Tensor,
    temp_dq: Tensor,
    bit: i32,
}

fn get_cuda_slice<T: WithDType>(x: &Tensor) -> *const T {
    match &*x.storage_and_layout().0 {
        Storage::Cuda(a_storage) => a_storage
            .as_cuda_slice::<T>()
            .expect("DType is not T")
            .device_ptr() as *const T,
        _ => panic!("Expected CUDA storage."),
    }
}

fn get_cuda_slice_mut<T: WithDType>(x: &mut Tensor) -> *mut T {
    match &*x.storage_and_layout().0 {
        Storage::Cuda(a_storage) => a_storage
            .as_cuda_slice::<T>()
            .expect("DType is not T")
            .device_ptr() as *mut T,
        _ => panic!("Expected CUDA storage."),
    }
}

fn get_cuda_device(x: &Tensor) -> &CudaDevice {
    match x.device() {
        Device::Cuda(dev) => dev,
        _ => panic!("Expected CUDA device"),
    }
}

impl GptQMatMul {
    fn forward(
        &mut self,
        a: &Tensor,
        (m, n, k): (i32, i32, i32),
        groups: i32,
        use_exllama: bool,
        gemm_handle: *const cublasHandle_t,
    ) -> Result<Tensor> {
        // https://github.com/vllm-project/vllm/blob/ba991d5c84adbc0685075af88333c688ddb06011/vllm/model_executor/layers/quantization/gptq.py#L200
        let out_shape = Shape::from_dims(
            &[
                &a.dims()[..a.dims().len() - 1],
                &[self.q_weight.dim(D::Minus1)?],
            ]
            .concat(),
        );
        let a = a.reshape(((), a.dim(D::Minus1)?))?;

        let a_ptr = get_cuda_slice::<f16>(&a);
        let b_q_weight = get_cuda_slice::<u32>(&self.q_weight);
        let b_gptq_qzeros = get_cuda_slice::<u32>(&self.gptq_qzeros);
        let b_q_scales = get_cuda_slice::<f16>(&self.q_weight);
        let b_g_idx = get_cuda_slice::<i64>(&self.g_idx) as *const i32;

        let dev = get_cuda_device(&a);

        let c_shape = Shape::from_dims(&[a.dims()[0] * self.q_weight.dims()[1]]);
        let c = unsafe { dev.alloc::<f16>(c_shape.elem_count()).w()? };

        let c_ptr = c.device_ptr() as *mut f16;

        let temp_dq = unsafe {
            dev.alloc::<f16>((self.q_weight.dims()[0] / 32 * self.bit) * self.q_weight.dims()[1])
                .w()?
        };

        let temp_dq_ptr = temp_dq.device_ptr() as *mut f16;
        unsafe {
            gemm_half_q_half_cuda(
                gemm_handle,
                a_ptr,
                b_q_weight,
                b_gptq_qzeros,
                b_gptq_scales,
                b_g_idx,
                c_ptr,
                temp_dq_ptr,
                m,
                n,
                k,
                groups,
                use_exllama,
                bit,
            )
        };

        let storage = CudaStorage {
            slice: CudaStorageSlice::F16(c),
            device: dev.clone(),
        };
        let storage = Storage::Cuda(storage);
        let res = from_storage_no_op(storage, c_shape, false);

        res.reshape(out_shape)
    }
}
