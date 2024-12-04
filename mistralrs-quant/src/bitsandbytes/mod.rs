use std::{
    borrow::Cow,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{Context, DType, Device, Result, Shape, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::{IsqType, QuantMethod, QuantMethodConfig, QuantizedSerde};

#[cfg(feature = "cuda")]
mod ffi;

mod op;
mod quantize;

const SUPPORTED_BLOCKSIZE: [usize; 7] = [2048, 4096, 1024, 512, 256, 128, 64];
pub(crate) const ISQ_BNB_BLOCKSIZE: usize = 64;

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum BnbDType {
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "bfloat16")]
    BF16,
    #[serde(rename = "float16")]
    F16,
}

impl From<BnbDType> for DType {
    fn from(value: BnbDType) -> Self {
        match value {
            BnbDType::F32 => Self::F32,
            BnbDType::BF16 => Self::BF16,
            BnbDType::F16 => Self::F16,
        }
    }
}

impl From<DType> for BnbDType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => Self::F32,
            DType::BF16 => Self::BF16,
            DType::F16 => Self::F16,
            _ => panic!("impossible dtype!"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BnbQuantType {
    Int8,
    Fp4,
    Nf4,
}

#[derive(Debug, Deserialize)]
pub struct BnbQuantState {
    pub blocksize: usize,
    pub shape: Vec<usize>,
    pub dtype: BnbDType,
    pub nested_blocksize: Option<usize>,
    pub nested_offset: Option<f64>,
    pub nested_dtype: Option<BnbDType>,
}

#[derive(Debug, Clone)]
pub struct BnbQuantParmas {
    pub absmax: Tensor,
    pub code: Tensor,
    pub blocksize: usize,
    pub shape: Option<Shape>,
    pub nested: Option<Arc<BnbQuantParmas>>,
    pub offset: Option<f64>,
    pub dtype: BnbDType,
}

#[derive(Debug)]
pub struct BnbLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    params: BnbQuantParmas,
    quant_ty: BnbQuantType,
}

impl BnbLinear {
    pub fn linear_b(_in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_unchecked_dtype("weight", DType::U8)?;

        let vb_w = vb.pp("weight");

        if !vb_w.contains_tensor("quant_state.bitsandbytes__nf4")
            && !vb_w.contains_tensor("quant_state.bitsandbytes__fp4")
        {
            candle_core::bail!("`BnbLinear` expects either `...__nf4` or `...__fp4` tensors, this means the layer is not 4bit.");
        }

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let quant_ty = if vb_w.contains_tensor("quant_state.bitsandbytes__nf4") {
            BnbQuantType::Nf4
        } else if vb_w.contains_tensor("quant_state.bitsandbytes__fp4") {
            BnbQuantType::Fp4
        } else {
            BnbQuantType::Int8
        };

        let state = match quant_ty {
            BnbQuantType::Nf4 => {
                Some(vb_w.get_unchecked_dtype("quant_state.bitsandbytes__nf4", DType::U8)?)
            }
            BnbQuantType::Fp4 => {
                Some(vb_w.get_unchecked_dtype("quant_state.bitsandbytes__fp4", DType::U8)?)
            }
            BnbQuantType::Int8 => None,
        };
        let Some(state) = state else {
            candle_core::bail!("Only fp8/nf4 quantization is supported for now.")
        };

        let state_str = String::from_utf8(state.to_vec1::<u8>()?)?;
        let state: BnbQuantState =
            serde_json::from_str(&state_str).map_err(candle_core::Error::msg)?;

        let nested = if vb_w.contains_tensor("nested_absmax") {
            // TODO: can `nested_blocksize` be None, default to 64 like bnb?
            Some(Arc::new(BnbQuantParmas {
                absmax: vb_w.get_unchecked_dtype("nested_absmax", DType::F32)?,
                code: vb_w.get_unchecked_dtype("nested_quant_map", DType::F32)?,
                blocksize: state
                    .nested_blocksize
                    .context("`nested_blocksize` must be present.")?,
                shape: None,
                nested: None,
                offset: None, // Put it in the outer one!
                dtype: state
                    .nested_dtype
                    .context("`nested_dtype` must be present.")?,
            }))
        } else {
            None
        };

        let absmax = if nested.is_some() {
            vb_w.get_unchecked_dtype("absmax", DType::U8)?
        } else {
            vb_w.get_unchecked_dtype("absmax", DType::F32)?
        };

        let params = BnbQuantParmas {
            absmax,
            code: vb_w.get_unchecked_dtype("quant_map", DType::F32)?,
            blocksize: state.blocksize,
            shape: Some(Shape::from_dims(&state.shape)),
            nested,
            offset: state.nested_offset,
            dtype: state.dtype,
        };

        Ok(Self {
            weight,
            bias,
            params,
            quant_ty,
        })
    }

    /// Dequantize input (u8). Handles nested absmax dequantization.
    pub(crate) fn dequantize(
        input: &Tensor,
        params: &BnbQuantParmas,
        quant_ty: BnbQuantType,
    ) -> Result<Tensor> {
        let mut absmax = params.absmax.clone();
        if let Some(nested) = &params.nested {
            absmax = Self::dequantize(&params.absmax, nested, BnbQuantType::Int8)?;
            absmax = (absmax + params.offset.context("`offset` must be present.")?)?;
        }

        let out_shape = params.shape.clone().unwrap_or(input.shape().clone());
        let out_dtype: DType = params.dtype.into();

        if !SUPPORTED_BLOCKSIZE.contains(&params.blocksize) {
            candle_core::bail!(
                "Blocksize of {} is not supported, {SUPPORTED_BLOCKSIZE:?} are.",
                params.blocksize
            );
        }

        op::dequantize(
            input,
            &absmax,
            &params.code,
            out_shape,
            params.blocksize,
            quant_ty,
            params.dtype,
        )?
        .to_dtype(out_dtype)
    }

    pub fn with_bias(mut self, bias: Tensor) -> Self {
        self.bias = Some(bias);
        self
    }
}

impl QuantMethod for BnbLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::FP8 { .. } => unreachable!(),
            QuantMethodConfig::Bnb {
                weight,
                bias,
                params,
                quant_ty,
            } => Ok(Self {
                weight,
                bias,
                params,
                quant_ty,
            }),
        }
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = Self::dequantize(&self.weight, &self.params, self.quant_ty)?
            .t()?
            .to_dtype(xs.dtype())?;
        // dbg!(&w.mean_all());
        let res = xs.broadcast_matmul(&w)?;
        if let Some(bias) = &self.bias {
            res + bias
        } else {
            Ok(res)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        candle_core::bail!("HQQ quantization does not support adding weight delta.")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.params.dtype.into(), self.weight.device().clone())
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
    ) -> Result<Arc<dyn QuantMethod>> {
        if imatrix_weight.is_some() {
            // TODO just warn?
            candle_core::bail!("HQQ does not support imatrix.");
        }

        n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bits = match dtype {
            Some(IsqType::FP4) => BnbQuantType::Fp4,
            Some(IsqType::NF4) => BnbQuantType::Nf4,
            Some(IsqType::INT8) => BnbQuantType::Int8,
            _ => candle_core::bail!("Expected a BNB ISQ type."),
        };
        let dequant = Self::dequantize(&self.weight, &self.params, self.quant_ty)?;
        let res = Self::quantize_onto(
            &dequant,
            bits,
            self.params.dtype,
            self.params.blocksize,
            dequant.device(),
        )?;
        if let Some(ref bias) = self.bias {
            let bias = bias
                .to_device(&device)?
                .to_dtype(res.dtype_and_device().0)?;
            Ok(Arc::new(res.with_bias(bias)))
        } else {
            Ok(Arc::new(res))
        }
    }

    fn get_max_isq_cpu_threads(&self, _dtype: IsqType) -> Option<NonZeroUsize> {
        None
    }
}

impl QuantizedSerde for BnbLinear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "bnb-linear"
    }
    fn serialize(&self) -> Result<Cow<[u8]>> {
        todo!()
    }

    fn deserialize(_data: Cow<[u8]>, _device: &Device) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        todo!()
    }
}
