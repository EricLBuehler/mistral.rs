use candle_core::{backend::BackendStorage, CpuStorage, CustomOp3, DType, Error, Layout, Result, Shape};

pub struct GatherMm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub right_sorted: bool,
}

impl GatherMm {
    pub fn new(m: usize, n: usize, k: usize, right_sorted: bool) -> Self {
        Self { m, n, k, right_sorted }
    }
}

impl CustomOp3 for GatherMm {
    fn name(&self) -> &'static str {
        "gather_mm"
    }

    fn cpu_fwd(
        &self,
        _a: &CpuStorage,
        a_l: &Layout,
        _b: &CpuStorage,
        b_l: &Layout,
        _indices: &CpuStorage,
        _indices_l: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        // Get the actual data arrays
        let (_a_shape, _a_strides) = (a_l.shape(), a_l.stride());
        let (_b_shape, _b_strides) = (b_l.shape(), b_l.stride());
        
        // For CPU implementation, we would need to implement the gather MM logic
        // This is a placeholder that returns an error
        Err(Error::Msg("gather_mm CPU implementation not yet available".to_string()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _a: &candle_core::CudaStorage,
        _a_l: &Layout,
        _b: &candle_core::CudaStorage,
        _b_l: &Layout,
        _indices: &candle_core::CudaStorage,
        _indices_l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        Err(Error::Msg("gather_mm CUDA implementation not yet available".to_string()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        a: &candle_core::MetalStorage,
        a_l: &Layout,
        b: &candle_core::MetalStorage,
        b_l: &Layout,
        indices: &candle_core::MetalStorage,
        indices_l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use crate::metal_kernels::{gather_mm, Kernels, EncoderProvider};
        
        let device = a.device();
        let a_dtype = a.dtype();
        let b_dtype = b.dtype();
        let indices_dtype = indices.dtype();
        
        if a_dtype != b_dtype {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: a_dtype,
                rhs: b_dtype,
                op: "gather_mm",
            }.bt());
        }
        
        if indices_dtype != DType::U32 {
            return Err(Error::Msg(format!(
                "gather_mm: indices must be U32, got {:?}",
                indices_dtype
            )));
        }
        
        let a_shape = a_l.shape();
        let b_shape = b_l.shape();
        let indices_shape = indices_l.shape();
        
        // Calculate output shape
        let batch_dims = a_shape.dims()[..a_shape.dims().len() - 2].to_vec();
        let mut out_shape_dims = batch_dims;
        out_shape_dims.push(self.m);
        out_shape_dims.push(self.n);
        let out_shape = Shape::from_dims(&out_shape_dims);
        
        // Allocate output buffer
        let out_buffer = device.new_buffer(
            out_shape.elem_count(),
            a_dtype,
            "gather_mm_output",
        )?;
        
        // Get the kernel loader
        let kernels = Kernels::new();
        
        // Launch kernel based on whether it's a gather_mm_rhs or full gather_mm
        if self.m == 1 && self.right_sorted {
            // Use gather_mm_rhs kernel
            struct EncoderWrapper<'a> {
                encoder: candle_core::MetalCommand,
                device: &'a candle_core::MetalDevice,
            }
            
            impl<'a> EncoderProvider for EncoderWrapper<'a> {
                fn encoder(&self) -> candle_core::MetalCommand {
                    self.encoder.clone()
                }
            }
            
            let encoder = device.encoder();
            let wrapper = EncoderWrapper { encoder: encoder.clone(), device };
            
            gather_mm::call_gather_mm_rhs(
                device.device(),
                wrapper,
                &kernels,
                a_dtype,
                a_shape,
                a_l,
                a.buffer(),
                0,
                b_shape,
                b_l,
                b.buffer(),
                0,
                indices.buffer(),
                0,
                &out_buffer,
                0,
            ).map_err(|e| Error::Msg(format!("Metal kernel error: {:?}", e)))?;
            encoder.commit();
        } else {
            return Err(Error::Msg("gather_mm: full gather MM not yet implemented for Metal".to_string()));
        }
        
        let out_storage = candle_core::MetalStorage::new(
            out_buffer,
            device.clone(),
            out_shape.elem_count(),
            a_dtype,
        );
        Ok((out_storage, out_shape))
    }
}