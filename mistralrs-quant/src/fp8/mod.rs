use std::sync::{atomic::AtomicUsize, Arc};

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};
use quantize::QuantizationResult;
use safetensors::tensor::Dtype;

mod quantize;

use crate::uqff::{UqffHeaderMatch, UqffLayerHeaderView};
use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, CUBLASLT_CONTROLLER},
    utils::{dtype_to_uqff_code, uqff_code_to_dtype},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
    Shard, UqffReader, UqffTensor,
};

#[derive(Debug)]
pub struct FP8Linear {
    lin: Linear,
    dequant_w_scale: Tensor,
    dequant_x_scale: Tensor,
    quant_scale: Tensor,
    /// Quantized type
    dtype: DType,
}

impl FP8Linear {
    pub(crate) fn inspect_uqff_header(layer: &UqffLayerHeaderView<'_>) -> Option<UqffHeaderMatch> {
        const WEIGHT_SUFFIXES: &[&str] = &[
            "weight",
            "weight.format",
            "weight.dequant_w_scale",
            "weight.dequant_x_scale",
            "weight.quant_scale",
            "weight.dtype",
        ];
        if layer.exact_weight_suffixes(WEIGHT_SUFFIXES)
            && layer.scalar("weight.format", Dtype::U8)
            && layer.scalar("weight.dtype", Dtype::U32)
        {
            Some(UqffHeaderMatch {
                serde_type: QuantizedSerdeType::Fp8,
            })
        } else {
            None
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        _tensors: &[UqffTensor],
        _prefix: &str,
    ) -> Result<String> {
        Ok("fp8".to_string())
    }

    pub fn from_parts(
        weight: Tensor,
        bias: Option<Tensor>,
        dequant_w_scale: Tensor,
        dequant_x_scale: Tensor,
        quant_scale: Tensor,
        dtype: DType,
    ) -> Self {
        Self {
            lin: Linear::new(weight, bias),
            dequant_w_scale,
            dequant_x_scale,
            quant_scale,
            dtype,
        }
    }

    fn from_uqff(reader: &UqffReader, key: &str, device: &Device, shard: Shard) -> Result<Self> {
        let dims = reader.tensor_dims(&format!("{key}.weight"))?;
        let range = crate::uqff::shard_range(shard, &dims)?;
        let weight = reader.load_tensor_sharded(&format!("{key}.weight"), device, range)?;
        let dequant_w_scale =
            reader.load_tensor(&format!("{key}.weight.dequant_w_scale"), device)?;
        let dequant_x_scale =
            reader.load_tensor(&format!("{key}.weight.dequant_x_scale"), device)?;
        let quant_scale = reader.load_tensor(&format!("{key}.weight.quant_scale"), device)?;
        let dtype = uqff_code_to_dtype(reader.load_u32_scalar(&format!("{key}.weight.dtype"))?)?;
        let bias = reader.load_bias(key, device, range, dims.len())?;
        Ok(Self::from_parts(
            weight,
            bias,
            dequant_w_scale,
            dequant_x_scale,
            quant_scale,
            dtype,
        ))
    }

    fn gather_quantized_rows(&self, ids: &Tensor) -> Result<Tensor> {
        let weight = self.lin.weight();
        let ids = ids.flatten_all()?;
        if !weight.device().is_metal() {
            return weight.index_select(&ids.to_device(weight.device())?, 0);
        }

        let ids = ids
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;

        let row_count = weight.dim(0)?;
        let mut rows = Vec::with_capacity(ids.len());
        for id in ids {
            let id = id as usize;
            if id >= row_count {
                candle_core::bail!("embedding index {id} is out of bounds for {row_count} rows");
            }
            let row = weight.narrow(0, id, 1)?.force_contiguous()?;
            rows.push(crate::scalar_fp8::ops::fp8_to_dtype(&row, DType::F32)?);
        }
        let rows = rows.iter().collect::<Vec<_>>();
        Tensor::cat(&rows, 0)
    }
}

impl QuantMethod for FP8Linear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
            QuantMethodConfig::FP8 { lin, dtype } => {
                let QuantizationResult {
                    qw,
                    quantize_scale,
                    dequantize_scale,
                } = Self::quantize(lin.weight(), dtype)?;
                Ok(Self {
                    lin: Linear::new(qw, lin.bias().cloned()),
                    dequant_x_scale: dequantize_scale.clone(), // This is probably wrong!
                    dequant_w_scale: dequantize_scale,
                    quant_scale: quantize_scale,
                    dtype,
                })
            }
        }
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        Ok(self.dequantize(DType::F32)?.weight().clone())
    }

    fn embedding_forward_raw(&self, ids: &Tensor) -> Result<Tensor> {
        let mut output_shape = ids.dims().to_vec();
        output_shape.push(self.lin.weight().dim(D::Minus1)?);
        if ids.elem_count() == 0 && self.lin.weight().device().is_metal() {
            let row = self.lin.weight().narrow(0, 0, 1)?.force_contiguous()?;
            let row = crate::scalar_fp8::ops::fp8_to_dtype(&row, DType::F32)?
                .broadcast_mul(&self.dequant_w_scale)?;
            return row.narrow(0, 0, 0)?.reshape(output_shape);
        }
        self.gather_quantized_rows(ids)?
            .to_dtype(DType::F32)?
            .broadcast_mul(&self.dequant_w_scale)?
            .reshape(output_shape)
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        // Batch matrix multiplication
        maybe_init_cublas_lt_wrapper(x.device().clone());

        match CUBLASLT_CONTROLLER.get_for_device(x.device()) {
            Some(handle) => {
                let n_dims = x.dims().len();
                if n_dims < 3 {
                    candle_core::bail!(
                        "FP8Linear `matmul` via cuBLASlt expects `x` to have at least 3 dimensions"
                    );
                }
                // Set up target shape
                let mut tgt_shape = x.dims().to_vec();
                *tgt_shape.last_mut().unwrap() = self.lin.weight().dim(0)?;

                // Flatten for correct dims
                let mut x = x.flatten_to(D::Minus(3))?;

                // Prepare the b tensor. If it is not quantized, quantize it
                let mut dequant_x_scale = self.dequant_x_scale.clone();
                if !matches!(x.dtype(), DType::F8E4M3) {
                    let QuantizationResult {
                        qw,
                        quantize_scale: _,
                        dequantize_scale,
                    } = Self::quantize(&x, DType::F8E4M3)?;
                    x = qw;
                    dequant_x_scale = dequantize_scale;
                }

                // Handle bias
                let beta = match self.lin.bias().is_some() {
                    true => Some(1.0),
                    false => None,
                };

                // Naming
                let a = self.lin.weight().unsqueeze(0)?;
                let b = x;

                handle
                    .batch_matmul_f8(
                        &a,
                        &b,
                        &self.dequant_w_scale,
                        &dequant_x_scale,
                        &self.quant_scale,
                        self.lin.bias(),
                        None,
                        beta,
                        None,
                        None,
                    )?
                    .reshape(tgt_shape)
            }
            None => {
                // Dequantize matmul
                let dequant_x = x.clone();
                let lin = self.dequantize(x.dtype())?;
                lin.forward(&dequant_x)
            }
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize(delta.dtype())?;
        let new = Linear::new((dequant.weight() + delta)?, dequant.bias().cloned());
        Ok(Arc::new(Self::new(QuantMethodConfig::FP8 {
            lin: new,
            dtype: self.dtype,
        })?))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (DType::F8E4M3, self.lin.weight().device().clone())
    }

    fn plan_isq(&self, request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        Ok(crate::plan_weight_isq(
            self.dtype,
            self.lin.weight().device().clone(),
            self.lin.weight().dims().to_vec(),
            request,
            true,
        ))
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                let dequant = self.dequantize(DType::F32)?;
                let w = dequant.weight().to_device(&device)?;
                let b = dequant.bias().map(|b| b.to_device(&device)).transpose()?;
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            _ => todo!(),
        }
    }
}

impl QuantizedSerde for FP8Linear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "fp8-linear"
    }
    fn serialize_uqff(&self, prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        if ty != IsqType::F8E4M3 {
            candle_core::bail!("Cannot serialize FP8 layer as {ty}; actual type is F8E4M3.");
        }

        let mut data = vec![
            UqffTensor::from_u8_scalar(
                format!("{prefix}.weight.format"),
                QuantizedSerdeType::Fp8 as u8,
            ),
            UqffTensor::from_tensor(format!("{prefix}.weight"), self.lin.weight())?,
            UqffTensor::from_tensor(
                format!("{prefix}.weight.dequant_w_scale"),
                &self.dequant_w_scale,
            )?,
            UqffTensor::from_tensor(
                format!("{prefix}.weight.dequant_x_scale"),
                &self.dequant_x_scale,
            )?,
            UqffTensor::from_tensor(format!("{prefix}.weight.quant_scale"), &self.quant_scale)?,
            UqffTensor::from_u32_scalar(
                format!("{prefix}.weight.dtype"),
                dtype_to_uqff_code(self.dtype)?,
            ),
        ];
        if let Some(bias) = self.lin.bias() {
            data.push(UqffTensor::from_tensor(format!("{prefix}.bias"), bias)?);
        }
        Ok(data)
    }
    fn deserialize_uqff(
        reader: &UqffReader,
        prefix: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self::from_uqff(reader, prefix, device, shard)?))
    }
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType> {
        Ok(IsqType::F8E4M3)
    }
}
