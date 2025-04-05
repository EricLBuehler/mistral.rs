use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Linear;

use crate::{
    Comm, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, UnquantLinear,
};

mod ops;

#[derive(Debug, Clone, Copy)]
pub enum AfqBits {
    Two = 2,
    Three = 3,
    Four = 4,
    Six = 6,
    Eight = 8,
}

#[derive(Debug, Clone, Copy, Default)]
pub enum AfqGroupSize {
    #[default]
    Low = 32,
    Med = 64,
    High = 128,
}

#[derive(Debug)]
pub struct AfqLayer {
    w_q: Tensor,
    scales: Tensor,
    biases: Tensor,
    bias: Option<Tensor>,
    bits: AfqBits,
    group_size: AfqGroupSize,
}

impl QuantMethod for AfqLayer {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::Unquantized(_) => unreachable!(),
            QuantMethodConfig::Afq {
                weight,
                bias,
                bits,
                group_size,
            } => {
                let (w_q, scales, biases) = ops::afq_quantize_op(&weight, group_size, bits)?;

                Ok(Self {
                    w_q,
                    scales,
                    biases,
                    bias,
                    bits,
                    group_size,
                })
            }
        }
    }

    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        ops::afq_dequantize_op(
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        return ops::afq_mm_op(
            x,
            &self.w_q,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
            true,
        );

        // Dequantize matmul always.
        // TODO: add a specific kernel?
        let weight = self.dequantize_w()?;
        // Dispatch to unquant. This uses some cublaslt for bias & on cuda always, so it is better
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight,
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        let dequant = self.dequantize_w()?;
        Ok(Arc::new(Self::new(QuantMethodConfig::Afq {
            weight: (dequant + delta)?,
            bias: self.bias.clone(),
            bits: self.bits,
            group_size: self.group_size,
        })?))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.scales.dtype(), self.scales.device().clone())
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

impl QuantizedSerde for AfqLayer {
    fn name(&self) -> &'static str {
        "afq-layer"
    }
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn serialize_with_bias(&self, _bias: Option<Tensor>) -> Result<Cow<[u8]>> {
        todo!()
    }
    fn deserialize(
        _data: Cow<[u8]>,
        _device: &Device,
        _comm: &Arc<Comm>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        todo!()
    }
    fn deserialize_ext_bias(
        _data: Cow<[u8]>,
        _device: &Device,
        _guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        todo!()
    }
}
