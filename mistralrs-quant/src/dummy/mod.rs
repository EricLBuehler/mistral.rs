use std::sync::Arc;

use candle_core::Result;

use crate::{QuantMethod, QuantizedSerde};

#[derive(Debug)]
pub struct DummyLayer;

impl QuantMethod for DummyLayer {
    fn new(_method: crate::QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self)
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        candle_core::bail!("DummyLayer cannot be dequantized!")
    }
    fn add_delta_w(
        &self,
        _delta: &candle_core::Tensor,
    ) -> candle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        candle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn apply_isq(
        self: std::sync::Arc<Self>,
        _dtype: Option<crate::IsqType>,
        _device: candle_core::Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
    ) -> candle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        candle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        (candle_core::DType::F64, candle_core::Device::Cpu)
    }
    fn forward(&self, _a: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn forward_via_half(
        &self,
        _a: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn get_bias_mut(&mut self) -> Option<&mut candle_core::Tensor> {
        None
    }
    fn get_max_isq_cpu_threads(&self, _dtype: crate::IsqType) -> Option<std::num::NonZeroUsize> {
        None
    }
    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        None
    }

    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>> {
        Ok(self.clone())
    }
}

impl QuantizedSerde for DummyLayer {
    fn name(&self) -> &'static str {
        "dummy"
    }
}
