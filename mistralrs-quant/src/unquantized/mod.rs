use std::{
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{quantized::GgmlDType, DType, Device, Result, Shape, Tensor, D};
use candle_nn::{Linear, Module};

use crate::{
    generate_isq,
    hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer, ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    GgufMatMul, IsqType, QuantMethod, QuantMethodConfig, CUBLASLT_HANDLE,
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
        if let Some(bias) = self.0.bias() {
            // If we have a bias, use a fused GEMM if possible!
            let w = match *a.dims() {
                [b1, b2, _, _] => self.0.weight().broadcast_left((b1, b2))?,
                [bsize, _, _] => self.0.weight().broadcast_left(bsize)?,
                _ => self.0.weight().clone(),
            };
            let mut tgt_shape = a.dims().to_vec();
            tgt_shape[a.dims().len() - 1] = w.dim(D::Minus2)?;
            let b = bias.broadcast_as(Shape::from_dims(&tgt_shape))?;

            if let (Device::Cuda(_), Some(cublaslt)) =
                (a.device(), *CUBLASLT_HANDLE.lock().unwrap())
            {
                cublaslt
                    .batch_matmul(
                        a,
                        &w,
                        Some(&b.t()?.contiguous()?),
                        None,
                        Some(1.0),
                        None,
                        None,
                    )?
                    .t()
            } else {
                a.matmul(&w.t()?)? + b
            }
        } else {
            self.0.forward(a)
        }
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
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>> {
        match dtype {
            /*Some(IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    // IsqType::HQQ3 => HqqBits::Three,
                    // IsqType::HQQ2 => HqqBits::Two,
                    // IsqType::HQQ1 => HqqBits::One,
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
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
                let res = generate_isq!(self.0.weight(), device, dtype, n_quantized);
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .0
                        .bias()
                        .cloned()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            None => {
                let w = self.0.weight().to_device(&device)?;
                let b = if let Some(b) = self.0.bias() {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize> {
        match dtype {
            /*IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            IsqType::HQQ4 | IsqType::HQQ8 => {
                // Use 1 because our HQQ quantizes on the GPU
                Some(1.try_into().unwrap())
            }
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1 => None,
        }
    }

    fn cast_to_device(self: Arc<Self>, device: Device) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(Self(Linear::new(
            self.0.weight().to_device(&device)?,
            self.0.bias().map(|b| b.to_device(&device).unwrap()),
        ))))
    }
}
