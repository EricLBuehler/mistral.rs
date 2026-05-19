use candle_core::Result;

use crate::{QuantMethod, QuantizeOntoGuard, QuantizedSerde};

#[derive(Debug, Clone)]
pub struct DummyLayerInfo {
    pub context: String,
    pub prefix: String,
    pub missing_tensors: Vec<String>,
}

impl DummyLayerInfo {
    pub fn unknown() -> Self {
        Self {
            context: "unknown".to_string(),
            prefix: "<unknown>".to_string(),
            missing_tensors: Vec::new(),
        }
    }

    pub fn message(&self, action: &str) -> String {
        let missing = if self.missing_tensors.is_empty() {
            "<unknown>".to_string()
        } else {
            self.missing_tensors.join(", ")
        };
        format!(
            "DummyLayer reached {action} for {} at prefix `{}`. Missing tensor path(s): {missing}. Dummy layers are only valid as temporary UQFF placeholders and must be replaced before inference.",
            self.context, self.prefix
        )
    }
}

#[derive(Debug, Clone)]
pub struct DummyLayer {
    info: DummyLayerInfo,
}

impl DummyLayer {
    pub fn placeholder(info: DummyLayerInfo) -> Self {
        Self { info }
    }

    pub fn info(&self) -> &DummyLayerInfo {
        &self.info
    }
}

impl QuantMethod for DummyLayer {
    fn new(_method: crate::QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        Ok(Self {
            info: DummyLayerInfo::unknown(),
        })
    }
    fn dequantize_w(&self) -> Result<candle_core::Tensor> {
        candle_core::bail!("{}", self.info.message("dequantization"))
    }
    fn add_delta_w(
        &self,
        _delta: &candle_core::Tensor,
    ) -> candle_core::Result<std::sync::Arc<dyn QuantMethod>> {
        candle_core::bail!("{}", self.info.message("LoRA delta application"))
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
    fn forward_raw(&self, _a: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        candle_core::bail!("{}", self.info.message("forward pass"))
    }
    fn quantized_act_type(&self) -> Option<candle_core::DType> {
        None
    }

    fn dummy_info(&self) -> Option<&crate::DummyLayerInfo> {
        Some(&self.info)
    }
}

impl QuantizedSerde for DummyLayer {
    fn name(&self) -> &'static str {
        "dummy"
    }
}
