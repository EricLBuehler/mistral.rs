use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{Context, DType, Device, Result, Shape, Tensor};
use serde::Deserialize;

use crate::{
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, ShardedVarBuilder,
};

#[cfg(feature = "cuda")]
mod ffi;

mod op;

const SUPPORTED_BLOCKSIZE: [usize; 7] = [2048, 4096, 1024, 512, 256, 128, 64];

#[derive(Debug, Deserialize, Clone, Copy)]
pub enum BnbDType {
    #[serde(rename = "float32")]
    F32,
    #[serde(rename = "bfloat16")]
    BF16,
    #[serde(rename = "float16")]
    F16,
}

#[derive(Debug, Clone, Copy)]
pub enum BnbQuantType {
    Int8,
    Fp4,
    Nf4,
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
pub struct BnbQuantParams {
    pub absmax: Tensor,
    pub code: Tensor,
    pub blocksize: usize,
    pub shape: Option<Shape>,
    pub nested: Option<Arc<BnbQuantParams>>,
    pub offset: Option<f64>,
    pub dtype: BnbDType,
}

#[derive(Debug)]
pub struct BnbLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    params: BnbQuantParams,
    quant_ty: BnbQuantType,
}

impl BnbLinear {
    pub fn linear_b(
        _in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
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
            Some(Arc::new(BnbQuantParams {
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

        let params = BnbQuantParams {
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
    fn dequantize(
        input: &Tensor,
        params: &BnbQuantParams,
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
}

impl QuantMethod for BnbLinear {
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
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
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

    fn dequantize_w(&self) -> Result<Tensor> {
        Self::dequantize(&self.weight, &self.params, self.quant_ty)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w = Self::dequantize(&self.weight, &self.params, self.quant_ty)?
            .t()?
            .to_dtype(xs.dtype())?;
        let res = xs.broadcast_matmul(&w)?;
        if let Some(bias) = &self.bias {
            res.broadcast_add(bias)
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

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }
}

impl QuantizedSerde for BnbLinear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "bnb-linear"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        candle_core::bail!("BitsAndBytes quantization does not support UQFF serialization")
    }

    fn deserialize(
        _data: Cow<[u8]>,
        _device: &Device,
        _comm: &Arc<crate::Comm>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!("BitsAndBytes quantization does not support UQFF deserialization")
    }
}
