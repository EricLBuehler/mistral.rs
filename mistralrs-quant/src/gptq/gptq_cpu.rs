use crate::{
    DummyLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    QuantizedSerde, ShardedVarBuilder,
};
use candle_core::{DType, Device, Result, Tensor};
use std::sync::{atomic::AtomicUsize, Arc};

#[derive(Debug)]
pub struct GptqLayer;

impl QuantMethod for GptqLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::GptqAwq { .. } => {
                candle_core::bail!("GPTQ is only supported on CUDA.")
            }
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => {
                unreachable!()
            }
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        todo!()
    }

    fn forward(&self, _a: &Tensor) -> Result<Tensor> {
        todo!()
    }

    fn quantized_act_type(&self) -> Option<DType> {
        todo!()
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        todo!()
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

impl QuantizedSerde for GptqLayer {
    fn name(&self) -> &'static str {
        "gptq"
    }
}

macro_rules! pack_factor {
    ($bits:expr) => {
        32 / $bits
    };
}

pub fn gptq_linear(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::GptqAwq {
        bits,
        group_size,
        checkpoint_format: _,
        is_awq,
    } = config
    else {
        candle_core::bail!("Unexpected quantization config.")
    };

    let is_awq = *is_awq;
    // Handle the case where we actually have an unquantized
    if vb.contains_tensor("weight") {
        return crate::linear_b(in_dim, out_dim, false, &None, vb);
    }

    // Handle the case where the layer is dummy (no tensors)
    if !vb.contains_tensor("qweight")
        || !vb.contains_tensor("qzeros")
        || !vb.contains_tensor("scales")
        || !is_awq && !vb.contains_tensor("g_idx")
    {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    let qw_shape = if !is_awq {
        //quantized gptq (k/pack_factor, n) format
        (in_dim / pack_factor!(bits), out_dim)
    } else {
        //quantized awq (k, n/pack_factor) format
        (in_dim, out_dim / pack_factor!(bits))
    };

    let qweight = vb.get_with_hints_dtype(qw_shape, "qweight", Default::default(), DType::I32)?;
    let scale_and_zero_size = in_dim / group_size;
    let qzeros = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim / pack_factor!(bits)),
        "qzeros",
        Default::default(),
        DType::I32,
    )?;
    let g_idx = if is_awq {
        None
    } else {
        Some(vb.get_with_hints_dtype((in_dim,), "g_idx", Default::default(), DType::I32)?)
    };
    let scales = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim),
        "scales",
        Default::default(),
        DType::F16,
    )?;
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get_with_hints_dtype((out_dim,), "bias", Default::default(), DType::F16)?)
    } else {
        None
    };

    let config = QuantMethodConfig::GptqAwq {
        bits: *bits as i32,
        use_exllama: false,
        q_weight: qweight,
        qzeros: Some(qzeros),
        scales,
        g_idx,
        bias,
        workspace: None,
        is_marlin: false,
        is_awq,
    };
    Ok(Arc::new(GptqLayer::new(config)?))
}
