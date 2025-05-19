use candle_core::Result;

use crate::{QuantMethod, QuantizeOntoGuard, QuantizedSerde};

#[derive(Debug, Copy, Clone)]
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
        _guard: QuantizeOntoGuard,
    ) -> candle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        // This is necessary for the immediate ISQ
        Ok(self)
    }
    fn dtype_and_device(&self) -> (candle_core::DType, candle_core::Device) {
        (candle_core::DType::F32, candle_core::Device::Cpu)
    }
    fn forward(&self, _a: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        None
    }
}

impl QuantizedSerde for DummyLayer {
    fn name(&self) -> &'static str {
        "dummy"
    }
}
