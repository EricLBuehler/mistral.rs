use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module};

use crate::{
    hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer, ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    generate_isq, GgufMatMul, IsqType, IsqType, QuantMethod, QuantMethodConfig,
};

#[derive(Debug)]
pub struct UnquantLinear(Linear);

impl QuantMethod for UnquantLinear {
    fn new(method: QuantMethodConfig) -> candle_core::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::Gptq { .. }
            | QuantMethodConfig::Hqq { .. } => unreachable!(),
            QuantMethodConfig::Unquantized(l) => Ok(Self(l)),
        }
    }

    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        self.0.forward(a)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self(Linear::new(
            (self.0.weight() + delta)?,
            self.0.bias().cloned(),
        ))))
    }

    fn dtype_and_device(&self) -> (DType, candle_core::Device) {
        (self.0.weight().dtype(), self.0.weight().device().clone())
    }

    fn get_bias_mut(&mut self) -> Option<&mut Tensor> {
        None
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: IsqType,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        if dtype.is_hqq() {
            n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let bits = match dtype {
                IsqType::HQQ8 => HqqBits::Eight,
                IsqType::HQQ4 => HqqBits::Four,
                IsqType::HQQ3 => HqqBits::Three,
                IsqType::HQQ2 => HqqBits::Two,
                IsqType::HQQ1 => HqqBits::One,
                _ => unreachable!(),
            };
            let cfg = HqqConfig {
                bits,
                group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                axis: HqqAxis::Zero,
                optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                round_zeros: false,
                channel_wise: true,
            };
            let res = HqqLayer::quantize(&self.0.weight().to_device(&device)?, &device, cfg)?;
            if let Some(bias) = self.0.bias() {
                let bias = bias
                    .to_device(&device)?
                    .to_dtype(res.dtype_and_device().0)?;
                Ok(Arc::new(res.with_bias(bias)))
            } else {
                Ok(Arc::new(res))
            }
        } else if dtype.is_gguf() {
            let dtype = dtype.try_into()?;
            let res = generate_isq!(self.0.weight(), device, dtype, n_quantized);
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: res,
                b: self
                    .0
                    .bias()
                    .cloned()
                    .map(|b| b.to_dtype(DType::F32).unwrap()),
            })?))
        } else {
            candle_core::bail!("Unsupported dtype for ISQ on UnquantLinear {dtype:?}")
        }
    }

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        if dtype.is_hqq() {
            // Use 1 because our HQQ quantizes on the GPU
            Some(1.try_into().unwrap())
        } else if dtype.is_gguf() {
            None
        } else {
            unreachable!()
        }
    }
}
