use crate::{QuantMethod, QuantizedSerde};

#[derive(Debug)]
pub struct DummyLayer;

impl QuantMethod for DummyLayer {
    fn new(_method: crate::QuantMethodConfig) -> mcandle_core::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self)
    }
    fn add_delta_w(
        &self,
        _delta: &mcandle_core::Tensor,
    ) -> mcandle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        mcandle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn apply_isq(
        self: std::sync::Arc<Self>,
        _dtype: Option<crate::IsqType>,
        _device: mcandle_core::Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
    ) -> mcandle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        mcandle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn dtype_and_device(&self) -> (mcandle_core::DType, mcandle_core::Device) {
        (mcandle_core::DType::F64, mcandle_core::Device::Cpu)
    }
    fn forward(&self, _a: &mcandle_core::Tensor) -> mcandle_core::Result<mcandle_core::Tensor> {
        mcandle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn forward_via_half(
        &self,
        _a: &mcandle_core::Tensor,
    ) -> mcandle_core::Result<mcandle_core::Tensor> {
        mcandle_core::bail!("DummyLayer should not ever be present in forward pass!")
    }
    fn get_bias_mut(&mut self) -> Option<&mut mcandle_core::Tensor> {
        None
    }
    fn get_max_isq_cpu_threads(&self, _dtype: crate::IsqType) -> Option<std::num::NonZeroUsize> {
        None
    }
    fn quantized_act_type(&self) -> Option<mcandle_core::DType> {
        None
    }
}

impl QuantizedSerde for DummyLayer {
    fn name(&self) -> &'static str {
        "dummy"
    }
}
