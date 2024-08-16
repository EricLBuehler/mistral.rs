use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{
    quantized::{GgmlDType, QMatMul},
    DType, Device, Result, Tensor,
};
use candle_nn::Module;

use crate::{generate_isq, IsqType, QuantMethod, QuantMethodConfig};

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self {
                w: QMatMul::from_arc(q_weight)?,
                b,
            }),
            QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. } => unreachable!(),
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward(a)?;
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }

    fn forward_via_half(&self, a: &Tensor) -> Result<Tensor> {
        let x = self.w.forward_via_f16(a)?;
        if let Some(ref b) = self.b {
            x.broadcast_add(b)
        } else {
            Ok(x)
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        Some(DType::F32)
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self {
                w: QMatMul::Tensor(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::Tensor((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::TensorF16(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::TensorF16((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::QTensor(w),
                b,
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        self.b.as_mut()
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: IsqType,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        let t = match &self.w {
            QMatMul::QTensor(q) => q.dequantize(&q.device())?,
            QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
        };
        let dtype = dtype.try_into()?;
        let res = generate_isq!(t, device, dtype, n_quantized);
        Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: res,
            b: self.b.clone(),
        })?))
    }

    fn get_max_isq_cpu_threads(&self, _dtype: IsqType) -> Option<NonZeroUsize> {
        None
    }
}
