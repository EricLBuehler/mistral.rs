use candle_core::{DType, Device, Result, Tensor}; // Shape removed from here
use std::any::Any;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
// ptr and c_void removed from here

use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
};

// CUDA specific imports and modules are gated
#[cfg(feature = "cuda")]
use crate::{
    cublaslt_ffi, kernels_ffi, ShardedVarBuilder // ShardedVarBuilder moved here
};
#[cfg(feature = "cuda")]
use crate::cublaslt_ffi::*;

#[cfg(feature = "cuda")]
use candle_core::cuda::{CudaDevice, CudaStorage, WrapErr, ffi as cuda_ffi_sys};
#[cfg(feature = "cuda")]
use std::ptr; // Added here
#[cfg(feature = "cuda")]
use std::ffi::c_void; // Added here
#[cfg(feature = "cuda")]
use candle_core::Shape; // Added here


/// Linear layer implementation for loading pre-quantized FP8 models,
/// specifically targeting a hybrid E4M3 weight / E5M2 activation format
/// compatible with CUTLASS/cuBLASLt for high-performance GEMM.
///
/// Weights are expected to be E4M3.
/// Weight scales are per-channel, stored in BF16 or FP16.
/// Activations are dynamically quantized to E5M2 per-token at runtime.
/// GEMM accumulation and layer output are in BF16 or FP16.
#[derive(Debug)]
pub struct CutlassHybridFP8Linear {
    weights_e4m3: Tensor,
    weight_scales: Tensor,
    #[allow(dead_code)] // Used in cuda-gated forward pass and constructor
    bias: Option<Tensor>,
    #[allow(dead_code)] // Used in cuda-gated forward pass and constructor
    in_features: usize,
    out_features: usize,
    activation_dtype: DType,
}

#[cfg(feature = "cuda")]
fn check_cublas_status(status: cublaslt_ffi::cublasStatus_t, operation_name: &str) -> Result<()> {
    if status != 0 { // 0 is CUBLAS_STATUS_SUCCESS
        candle_core::bail!("cuBLASLt operation '{}' failed with status: {}", operation_name, status);
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn to_cuda_datatype(dtype: DType) -> Result<cublaslt_ffi::cudaDataType_t> {
    match dtype {
        DType::F16 => Ok(cublaslt_ffi::cudaDataType_t::CUDA_R_16F),
        DType::BF16 => Ok(cublaslt_ffi::cudaDataType_t::CUDA_R_16BF),
        DType::F32 => Ok(cublaslt_ffi::cudaDataType_t::CUDA_R_32F),
        DType::U8 => Ok(cublaslt_ffi::cudaDataType_t::CUDA_R_8F_E5M2),
        DType::F8E4M3 => Ok(cublaslt_ffi::cudaDataType_t::CUDA_R_8F_E4M3),
        _ => candle_core::bail!("Unsupported DType {:?} for CUDA DataType for cuBLASLt", dtype),
    }
}


impl CutlassHybridFP8Linear {
    pub fn new(
        weights_e4m3: Tensor,
        weight_scales: Tensor,
        bias: Option<Tensor>,
        in_features: usize,
        out_features: usize,
        activation_dtype: DType,
    ) -> Result<Self> {
        if weights_e4m3.dtype() != DType::F8E4M3 {
            candle_core::bail!("CutlassHybridFP8Linear weights must be F8E4M3, got {:?}", weights_e4m3.dtype());
        }
        if weight_scales.dtype() != DType::BF16 && weight_scales.dtype() != DType::F16 {
            candle_core::bail!("CutlassHybridFP8Linear weight_scales must be BF16 or F16, got {:?}", weight_scales.dtype());
        }
        if let Some(b) = &bias {
            if b.dtype() != DType::BF16 && b.dtype() != DType::F16 {
                candle_core::bail!("CutlassHybridFP8Linear bias must be BF16 or F16, got {:?}", b.dtype());
            }
        }
        if activation_dtype != DType::BF16 && activation_dtype != DType::F16 {
            candle_core::bail!("CutlassHybridFP8Linear activation_dtype must be BF16 or F16, got {:?}", activation_dtype);
        }
        Ok(Self {
            weights_e4m3,
            weight_scales,
            bias,
            in_features,
            out_features,
            activation_dtype,
        })
    }
}

impl QuantMethod for CutlassHybridFP8Linear {
    fn new(config: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match config {
            QuantMethodConfig::CutlassFP8PTQ {
                weights_e4m3,
                weight_scales,
                bias,
                in_features,
                out_features,
                activation_dtype,
            } => Self::new(weights_e4m3, weight_scales, bias, in_features, out_features, activation_dtype),
            _ => candle_core::bail!("Invalid QuantMethodConfig for CutlassHybridFP8Linear. Use specific constructor for pre-quantized weights."),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        let weights_act_dtype = self.weights_e4m3.to_dtype(self.activation_dtype)?;
        let scales_expanded = self.weight_scales.reshape((self.out_features, 1))?;
        weights_act_dtype.broadcast_mul(&scales_expanded)
    }

    #[cfg(not(feature = "cuda"))]
    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        candle_core::bail!("CutlassHybridFP8Linear::forward requires CUDA feature to be enabled.")
    }

    #[cfg(feature = "cuda")]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if !x.device().is_cuda() {
            candle_core::bail!("Input tensor must be on a CUDA device for CutlassHybridFP8Linear");
        }
        if x.dtype() != self.activation_dtype {
            candle_core::bail!(
                "Input tensor dtype {:?} does not match layer activation_dtype {:?}",
                x.dtype(),
                self.activation_dtype
            );
        }

        let device = x.device().as_cuda_device()?;
        let stream = device.stream();

        let x_shape = x.shape();
        let x_dims = x_shape.dims();
        let num_tokens = x_dims[..x_dims.len()-1].iter().product();
        let token_size = self.in_features;

        let quantized_act_e5m2 = Tensor::zeros((num_tokens, token_size), DType::U8, x.device())?;
        let act_scales = Tensor::zeros((num_tokens, 1), DType::F16, x.device())?;

        let x_storage = x.storage().as_cuda_storage()?;
        let q_act_storage = quantized_act_e5m2.storage().as_cuda_storage()?;
        let act_scales_storage = act_scales.storage().as_cuda_storage()?;

        let x_ptr = match x.dtype() {
            DType::F16 => x_storage.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
            DType::BF16 => x_storage.as_cuda_slice::<half::bf16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
            _ => unreachable!(),
        };
        let q_act_ptr = q_act_storage.as_cuda_slice::<u8>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
        let act_scales_ptr = act_scales_storage.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
        let stream_raw = stream.as_raw() as *mut _;

        unsafe {
            match x.dtype() {
                DType::F16 => {
                    kernels_ffi::quantize_fp16_to_e5m2_per_token_kernel(
                        x_ptr, q_act_ptr, act_scales_ptr,
                        num_tokens as i32, token_size as i32, stream_raw,
                    );
                }
                DType::BF16 => {
                    kernels_ffi::quantize_bf16_to_e5m2_fp16scales_kernel(
                        x_ptr, q_act_ptr, act_scales_ptr,
                        num_tokens as i32, token_size as i32, stream_raw,
                    );
                }
                _ => candle_core::bail!("Unsupported activation dtype for quantization kernel: {:?}", x.dtype()),
            }
        }

        let quantized_act_e5m2_flat = quantized_act_e5m2.reshape((num_tokens, token_size))?;

        let mut handle: cublaslt_ffi::cublasLtHandle_t = ptr::null_mut();
        let mut matmul_desc: cublaslt_ffi::cublasLtMatmulDesc_t = ptr::null_mut();
        let mut Adesc: cublaslt_ffi::cublasLtMatrixLayout_t = ptr::null_mut();
        let mut Bdesc: cublaslt_ffi::cublasLtMatrixLayout_t = ptr::null_mut();
        let mut Ddesc: cublaslt_ffi::cublasLtMatrixLayout_t = ptr::null_mut();

        let res: Result<Tensor> = unsafe {
            check_cublas_status(cublaslt_ffi::cublasLtCreate(&mut handle), "cublasLtCreate")?;

            let compute_type = cublaslt_ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F;
            let scale_type_cuda = cublaslt_ffi::cudaDataType_t::CUDA_R_32F;

            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescCreate(&mut matmul_desc, compute_type, scale_type_cuda), "cublasLtMatmulDescCreate")?;

            let op_n = cublaslt_ffi::cublasOperation_t::CUBLAS_OP_N;
            let op_t = cublaslt_ffi::cublasOperation_t::CUBLAS_OP_T;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &op_n as *const _ as *const c_void, std::mem::size_of_val(&op_n)), "Set TRANSA")?;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &op_t as *const _ as *const c_void, std::mem::size_of_val(&op_t)), "Set TRANSB")?;

            let ascale_mode = cublaslt_ffi::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &ascale_mode as *const _ as *const c_void, std::mem::size_of_val(&ascale_mode)), "Set A_SCALE_MODE")?;
            let act_scales_storage_ptr_ffi = act_scales.storage().as_cuda_storage()?.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
            let act_scales_ptr_void = act_scales_storage_ptr_ffi as *const c_void;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &act_scales_ptr_void as *const _ as *const c_void, std::mem::size_of::<*const c_void>()), "Set A_SCALE_POINTER")?;

            let bscale_mode = cublaslt_ffi::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &bscale_mode as *const _ as *const c_void, std::mem::size_of_val(&bscale_mode)), "Set B_SCALE_MODE")?;
            let ws_dtype = self.weight_scales.dtype();
            let weight_scales_storage = self.weight_scales.storage().as_cuda_storage()?;
            let weight_scales_ptr_raw = match ws_dtype {
                DType::F16 => weight_scales_storage.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
                DType::BF16 => weight_scales_storage.as_cuda_slice::<half::bf16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
                _ => panic!("Unsupported weight_scales dtype")
            };
            let weight_scales_ptr_void = weight_scales_ptr_raw as *const c_void;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &weight_scales_ptr_void as *const _ as *const c_void, std::mem::size_of::<*const c_void>()), "Set B_SCALE_POINTER")?;

            if self.bias.is_some() {
                let epilogue = cublaslt_ffi::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
                check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue as *const _ as *const c_void, std::mem::size_of_val(&epilogue)), "Set EPILOGUE_BIAS")?;
                let bias_s = self.bias.as_ref().unwrap().storage().as_cuda_storage()?;
                let bias_ptr_raw = match self.bias.as_ref().unwrap().dtype() {
                     DType::F16 => bias_s.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
                     DType::BF16 => bias_s.as_cuda_slice::<half::bf16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr,
                     _ => panic!("Unsupported bias dtype")
                };
                let bias_ptr_void = bias_ptr_raw as *const c_void;
                check_cublas_status(cublaslt_ffi::cublasLtMatmulDescSetAttribute(matmul_desc, cublaslt_ffi::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr_void as *const _ as *const c_void, std::mem::size_of::<*const c_void>()), "Set BIAS_POINTER")?;
            }

            let m = num_tokens as u64;
            let k = self.in_features as u64;
            let n = self.out_features as u64;

            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutCreate(&mut Adesc, cublaslt_ffi::cudaDataType_t::CUDA_R_8F_E5M2, m, k, k as i64), "Create Adesc")?;
            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutCreate(&mut Bdesc, cublaslt_ffi::cudaDataType_t::CUDA_R_8F_E4M3, self.out_features as u64, self.in_features as u64, self.in_features as i64), "Create Bdesc")?;

            let out_cuda_dtype = to_cuda_datatype(self.activation_dtype)?;
            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutCreate(&mut Ddesc, out_cuda_dtype, m, n, n as i64), "Create Ddesc")?;

            let output_shape_vec: Vec<usize> = x_dims[..x_dims.len()-1].iter().cloned().chain(std::iter::once(self.out_features)).collect();
            let output_shape_final = Shape::from_dims(&output_shape_vec); // Requires Shape in scope
            let d_tensor = Tensor::zeros(&output_shape_final, self.activation_dtype, x.device())?;

            let d_storage = d_tensor.storage().as_cuda_storage()?;
            let d_ptr = match self.activation_dtype {
                DType::F16 => d_storage.as_cuda_slice::<half::f16>()?.as_mut_ptr() as cuda_ffi_sys::CUdeviceptr,
                DType::BF16 => d_storage.as_cuda_slice::<half::bf16>()?.as_mut_ptr() as cuda_ffi_sys::CUdeviceptr,
                _ => panic!("Unsupported output DType"),
            };

            let alpha_val: f32 = 1.0;
            let beta_val: f32 = 0.0;

            let workspace_size: usize = 1024 * 1024 * 32;
            let workspace_storage = device.alloc_zeros::<u8>(workspace_size).w?;
            let workspace_ptr = workspace_storage.as_cuda_slice::<u8>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;

            let q_act_flat_storage = quantized_act_e5m2_flat.storage().as_cuda_storage()?;
            let q_act_flat_ptr = q_act_flat_storage.as_cuda_slice::<u8>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
            let weights_storage = self.weights_e4m3.storage().as_cuda_storage()?;
            let weights_ptr = weights_storage.as_cuda_slice::<u8>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;

            check_cublas_status(cublaslt_ffi::cublasLtMatmul(
                handle, matmul_desc,
                &alpha_val as *const _ as *const c_void,
                q_act_flat_ptr, Adesc,
                weights_ptr, Bdesc,
                &beta_val as *const _ as *const c_void,
                0 as cuda_ffi_sys::CUdeviceptr, ptr::null_mut(),
                d_ptr, Ddesc,
                ptr::null(),
                workspace_ptr, workspace_size,
                stream_raw,
            ), "cublasLtMatmul")?;

            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutDestroy(Adesc), "Destroy Adesc")?;
            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutDestroy(Bdesc), "Destroy Bdesc")?;
            check_cublas_status(cublaslt_ffi::cublasLtMatrixLayoutDestroy(Ddesc), "Destroy Ddesc")?;
            check_cublas_status(cublaslt_ffi::cublasLtMatmulDescDestroy(matmul_desc), "Destroy MatmulDesc")?;
            check_cublas_status(cublaslt_ffi::cublasLtDestroy(handle), "Destroy Handle")?;

            Ok(d_tensor)
        };
        res
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("LoRA/delta weight addition not supported for CutlassHybridFP8Linear yet.")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F8E4M3, self.weights_e4m3.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("In-situ quantization (ISQ) is not applicable to pre-quantized CutlassHybridFP8Linear layers.")
    }
}

impl QuantizedSerde for CutlassHybridFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "cutlass-hybrid-fp8-linear"
    }
}

/// Constructor for `CutlassHybridFP8Linear` when loading from `safetensors`.
///
/// This function is typically called from `mistralrs-quant/src/lib.rs`'s `linear_b`
/// function when a `config.json` specifies a compatible FP8 quantization method.
///
/// # Arguments
///
/// * `in_features`: Number of input features.
/// * `out_features`: Number of output features.
/// * `bias`: Whether the layer includes a bias.
/// * `activation_dtype_str`: An `Option<String>` specifying the expected activation
///   data type for the layer's interface (e.g., "bf16", "f16"). This also
///   determines the precision for scales, bias, and GEMM accumulation. Defaults to BF16.
/// * `vb`: The `ShardedVarBuilder` used to load tensors. It expects:
///     - `model.layer.weight`: Tensor with E4M3 data type.
///     - `model.layer.weight_scale`: Tensor with `activation_dtype` (BF16/FP16), shape `(out_features,)`.
///     - `model.layer.bias` (optional): Tensor with `activation_dtype` (BF16/FP16), shape `(out_features,)`.
#[cfg(feature = "cuda")]
pub fn cutlass_hybrid_fp8_linear_b(
    in_features: usize,
    out_features: usize,
    bias: bool,
    activation_dtype_str: Option<String>,
    vb: crate::ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let activation_dtype = match activation_dtype_str.as_deref() {
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        Some(other) => candle_core::bail!("Unsupported activation_dtype string '{}' for CutlassHybridFP8. Expected 'bf16' or 'f16'.", other),
        None => {
            let vb_dtype = vb.dtype();
            if vb_dtype == DType::BF16 || vb_dtype == DType::F16 {
                vb_dtype
            } else {
                DType::BF16
            }
        }
    };

    let weights_e4m3 = vb.get_with_hints_dtype(
        (out_features, in_features),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;

    let weight_scales = vb.get_with_hints_dtype(
        (out_features,),
        "weight_scale",
        Default::default(),
        activation_dtype,
    )?;

    let bias_tensor = if bias {
        Some(vb.get_with_hints_dtype(
            (out_features,),
            "bias",
            Default::default(),
            activation_dtype,
        )?)
    } else {
        None
    };

    let layer = CutlassHybridFP8Linear::new(
        weights_e4m3,
        weight_scales,
        bias_tensor,
        in_features,
        out_features,
        activation_dtype,
    )?;
    Ok(Arc::new(layer))
}

#[cfg(not(feature = "cuda"))]
pub fn cutlass_hybrid_fp8_linear_b(
    _in_features: usize,
    _out_features: usize,
    _bias: bool,
    _activation_dtype_str: Option<String>,
    _vb: crate::ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    unreachable!("CutlassHybridFP8Linear loading is only available with CUDA feature")
}


#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor, Result, Error}; // Var removed
    use crate::{ShardedVarBuilder};
    use crate::safetensors::MmapedSafetensors;
    use std::collections::HashMap;
    use safetensors::serialize;
    use tempfile::NamedTempFile;
    use std::fs::File;
    use std::io::Write;
    use half::f16;

    fn create_dummy_safetensors_file(
        tensors: &HashMap<String, Tensor>,
        path: &std::path::Path,
    ) -> Result<()> {
        let metadata: Option<HashMap<String, String>> = None;
        let serialized_bytes = serialize(tensors, &metadata)
            .map_err(|e| Error::Msg(format!("Failed to serialize tensors: {}", e)))?;
        let mut file = File::create(path)
            .map_err(|e| Error::Msg(format!("Failed to create safetensors file: {}", e)))?;
        file.write_all(&serialized_bytes)
            .map_err(|e| Error::Msg(format!("Failed to write to safetensors file: {}", e)))?;
        Ok(())
    }

    #[test]
    fn test_cutlass_fp8_load_and_dequantize() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let in_f = 4;
        let out_f = 2;

        let weights_data_u8 = vec![1u8; out_f * in_f];
        let weights_e4m3_tensor = Tensor::from_slice(&weights_data_u8, (out_f, in_f), &Device::Cpu)?
            .to_dtype(DType::U8)?
            .to_device(&device)?
            .to_dtype(DType::F8E4M3)?;

        let weight_scales_data_f16: Vec<f16> = (0..out_f).map(|i| f16::from_f32(0.5 + i as f32 * 0.1)).collect();
        let weight_scales_tensor = Tensor::from_slice(&weight_scales_data_f16, (out_f,), &device)?;

        let bias_data_f16: Vec<f16> = (0..out_f).map(|i| f16::from_f32(0.1 + i as f32 * 0.05)).collect();
        let bias_tensor = Tensor::from_slice(&bias_data_f16, (out_f,), &device)?;

        let mut tensors_map = HashMap::new();
        tensors_map.insert("model.layer.weight".to_string(), weights_e4m3_tensor.clone());
        tensors_map.insert("model.layer.weight_scale".to_string(), weight_scales_tensor.clone());
        tensors_map.insert("model.layer.bias".to_string(), bias_tensor.clone());

        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path();
        create_dummy_safetensors_file(&tensors_map, temp_path)?;

        let safetensors_files = unsafe { MmapedSafetensors::multi(&[temp_path])? };
        let vb = ShardedVarBuilder::from_single(safetensors_files, DType::F16, &device)
            .pp("model.layer");

        let layer_arc = cutlass_hybrid_fp8_linear_b(
            in_f,
            out_f,
            true,
            Some("f16".to_string()),
            vb
        )?;

        let layer = layer_arc.as_any().downcast_ref::<CutlassHybridFP8Linear>().unwrap();

        assert_eq!(layer.in_features, in_f);
        assert_eq!(layer.out_features, out_f);
        assert_eq!(layer.activation_dtype, DType::F16);

        assert_eq!(layer.weights_e4m3.shape().dims2()?, (out_f, in_f));
        assert_eq!(layer.weights_e4m3.dtype(), DType::F8E4M3);

        assert_eq!(layer.weight_scales.to_vec1::<f16>()?, weight_scales_data_f16);

        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().to_vec1::<f16>()?, bias_data_f16);

        let dequant_w = layer.dequantize_w()?;
        assert_eq!(dequant_w.shape().dims2()?, (out_f, in_f));
        assert_eq!(dequant_w.dtype(), DType::F16);

        Ok(())
    }

    #[test]
    fn test_activation_quant_kernel_ffi_fp16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let num_tokens = 2;
        let token_size = 256;

        let input_data_f16: Vec<f16> = (0..(num_tokens * token_size))
            .map(|i| f16::from_f32( ((i % token_size) as f32 % 5.0) - 2.5) )
            .collect();
        let input_tensor = Tensor::from_slice(&input_data_f16, (num_tokens, token_size), &device)?;

        let quantized_act_e5m2 = Tensor::zeros((num_tokens, token_size), DType::U8, &device)?;
        let act_scales = Tensor::zeros((num_tokens, 1), DType::F16, &device)?;

        let stream = device.stream();

        let x_storage = input_tensor.storage().as_cuda_storage()?;
        let x_ptr = x_storage.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
        let q_act_storage = quantized_act_e5m2.storage().as_cuda_storage()?;
        let q_act_ptr = q_act_storage.as_cuda_slice::<u8>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;
        let act_scales_storage = act_scales.storage().as_cuda_storage()?;
        let act_scales_ptr = act_scales_storage.as_cuda_slice::<half::f16>()?.as_ptr() as cuda_ffi_sys::CUdeviceptr;

        unsafe {
             kernels_ffi::quantize_fp16_to_e5m2_per_token_kernel(
                 x_ptr,
                 q_act_ptr,
                 act_scales_ptr,
                 num_tokens as i32,
                 token_size as i32,
                 stream.as_raw() as *mut _,
             );
        }
        device.synchronize()?;

        let scales_vec = act_scales.to_vec2::<f16>()?;
        let first_token_data_host: Vec<f16> = input_data_f16.iter().take(token_size).cloned().collect();
        let abs_max_first_token = first_token_data_host.iter().map(|&x| x.to_f32().abs()).fold(0.0f32, f32::max);

        let expected_scale_val = if abs_max_first_token == 0.0 { 1.0 } else { 57344.0 / abs_max_first_token };
        assert!((scales_vec[0][0].to_f32() - expected_scale_val).abs() < expected_scale_val * 0.01, "Scale for token 0 mismatch: got {}, expected {}", scales_vec[0][0], expected_scale_val);

        Ok(())
    }

    #[test]
    fn test_forward_pass_executes() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let batch_size = 1;
        let in_f = 256;
        let out_f = 128;

        let weights_e4m3 = Tensor::randn(0f32, 1.0f32, (out_f, in_f), &device)?.to_dtype(DType::F8E4M3)?;
        let weight_scales = Tensor::ones((out_f,1), DType::F16, &device)?;
        let bias = Some(Tensor::zeros(out_f, DType::F16, &device)?);

        let layer = CutlassHybridFP8Linear::new(
            weights_e4m3,
            weight_scales,
            bias,
            in_f,
            out_f,
            DType::F16
        )?;

        let input_data = Tensor::randn(0f32, 1f32, (batch_size, in_f), &device)?.to_dtype(DType::F16)?;

        let output = layer.forward(&input_data)?;

        assert_eq!(output.shape().dims2()?, (batch_size, out_f));
        assert_eq!(output.dtype(), DType::F16);

        Ok(())
    }
}
