use crate::{DummyLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizedConfig, QuantizedSerde};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

#[derive(Debug)]
pub struct GptqLayer;

impl QuantMethod for GptqLayer {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gptq { .. } => candle_core::bail!("GPTQ is only supported on CUDA."),
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. } => {
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

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        todo!()
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
    ) -> Result<Arc<dyn QuantMethod>> {
        todo!()
    }

    fn get_max_isq_cpu_threads(&self, _dtype: IsqType) -> Option<NonZeroUsize> {
        todo!()
    }

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        Ok(self.clone())
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
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    // Handle the case where the layer is dummy (no tensors)
    if !(vb.contains_tensor("qweight")
        && vb.contains_tensor("qzeros")
        && vb.contains_tensor("g_idx")
        && vb.contains_tensor("scales"))
    {
        let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
        return Ok(Arc::new(layer) as Arc<dyn QuantMethod>);
    }

    let bits = config.bits.expect("GPTQ requires bits in config");
    let qweight = vb.get_with_hints_dtype(
        (in_dim / pack_factor!(bits), out_dim),
        "qweight",
        Default::default(),
        DType::I32,
    )?;
    let scale_and_zero_size = in_dim
        / config
            .group_size
            .expect("GPTQ requires group size in config");
    let qzeros = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim / pack_factor!(bits)),
        "qzeros",
        Default::default(),
        DType::I32,
    )?;
    let g_idx = vb.get_with_hints_dtype((in_dim,), "g_idx", Default::default(), DType::I32)?;
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

    let config = QuantMethodConfig::Gptq {
        bits: bits as i32,
        use_exllama: false,
        q_weight: qweight,
        gptq_qzeros: Some(qzeros),
        gptq_scales: scales,
        g_idx: Some(g_idx),
        bias,
        workspace: None,
        is_marlin: false,
    };
    Ok(Arc::new(GptqLayer::new(config)?))
}
