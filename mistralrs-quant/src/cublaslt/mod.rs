// https://github.com/huggingface/text-embeddings-inference/blob/cc1c510e8d8af8447c01e6b14c417473cf2dfda9/backends/candle/src/layers/cublaslt.rs

#![allow(unused_variables, unused_imports, dead_code)]

use candle_core::{Device, Result, Tensor};
use candle_nn::Activation as CandleActivation;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, Once};

/// Controller for the CUBLASLT handle and inhibition flag.
pub struct CublasLtController {
    handle: Mutex<Option<&'static CublasLtWrapper>>,
    inhibit: AtomicBool,
}

impl CublasLtController {
    /// Set whether to inhibit CUBLASLT usage.
    pub fn set_inhibit(&self, value: bool) {
        self.inhibit.store(value, Ordering::SeqCst);
    }

    /// Get the handle if not inhibited.
    pub fn get(&self) -> Option<&'static CublasLtWrapper> {
        let handle_opt = self.handle.lock().unwrap();
        if self.inhibit.load(Ordering::SeqCst) {
            None
        } else {
            *handle_opt
        }
    }
}

pub static CUBLASLT_CONTROLLER: Lazy<CublasLtController> = Lazy::new(|| CublasLtController {
    handle: Mutex::new(None),
    inhibit: AtomicBool::new(false),
});

#[cfg(feature = "cuda")]
mod api;
#[cfg(feature = "cuda")]
mod matmul;
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;

#[cfg(feature = "cuda")]
pub use api::{fused_batch_matmul, fused_batch_matmul_f8, CublasLt};

pub enum F8MatmulOutType {
    F8,
    BF16,
}

pub fn maybe_init_cublas_lt_wrapper(device: Device) {
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        #[cfg(feature = "cuda")]
        {
            match device {
                Device::Cuda(_) => {
                    let wrapper = Box::new(CublasLtWrapper {
                        cublaslt: CublasLt::new(&device).unwrap(),
                    });
                    let wrapper_ptr = Box::leak(wrapper) as &'static CublasLtWrapper;

                    // Set the controller handle
                    let mut handle_lock = CUBLASLT_CONTROLLER.handle.lock().unwrap();
                    *handle_lock = Some(wrapper_ptr);
                }
                _ => {
                    let mut handle_lock = CUBLASLT_CONTROLLER.handle.lock().unwrap();
                    *handle_lock = None;
                }
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let mut handle_lock = CUBLASLT_CONTROLLER.handle.lock().unwrap();
            *handle_lock = None;
        }
    });
}

#[derive(Debug, Clone)]
pub struct CublasLtWrapper {
    #[cfg(feature = "cuda")]
    pub cublaslt: CublasLt,
}

impl CublasLtWrapper {
    /// Fused batch matmul + add + Relu/Gelu activation using CublasLt for F8 dtypes.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of size BxMxK
    /// * `b` - Input tensor of size BxNxK
    /// * `dequant_a_scale` - F32 scalar tensor, used to `a` the out tensor.
    /// * `dequant_b_scale` - F32 scalar tensor, used to `b` the out tensor.
    /// * `quantize_scale` - F32 scalar tensor, used to requantize.
    /// * `out` - Optional Output tensor of size BxNxK. If set and beta != 0, will be added to the end result of A*B before `act`
    /// * `alpha` - Optional scaling factor for A*B
    /// * `beta` - Optional scaling factor for C
    /// * `bias` - Optional bias tensor of size M
    /// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
    ///
    /// The resulting tensor is of shape NxM
    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul_f8(
        &self,
        a: &Tensor,
        b: &Tensor,
        dequant_a_scale: &Tensor,
        dequant_b_scale: &Tensor,
        quantize_scale: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<CandleActivation>,
        out_dtype: F8MatmulOutType,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = act.map(|a| match a {
                CandleActivation::Relu => matmul::Activation::Relu,
                CandleActivation::Gelu => matmul::Activation::Gelu,
                _ => unreachable!("Unsupported activation in cublaslt matmul"),
            });
            let mut result = fused_batch_matmul_f8(
                a,
                b,
                dequant_a_scale,
                dequant_b_scale,
                quantize_scale,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                out_dtype,
                self.cublaslt.clone(),
            )?;

            if Some(CandleActivation::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!("`cuda` feature is not enabled")
        }
    }

    /// Fused batch matmul + add + Relu/Gelu activation using CublasLt.
    ///
    /// # Arguments
    ///
    /// * `a` - Input tensor of size BxMxK
    /// * `b` - Input tensor of size BxNxK
    /// * `out` - Optional Output tensor of size BxNxK. If set and beta != 0, will be added to the end result of A*B before `act`
    /// * `alpha` - Optional scaling factor for A*B
    /// * `beta` - Optional scaling factor for C
    /// * `bias` - Optional bias tensor of size M
    /// * `act` - Optional Gelu or Relu activation. If set, will be added to the end result
    ///
    /// The resulting tensor is of shape NxM
    #[allow(clippy::too_many_arguments)]
    pub fn batch_matmul(
        &self,
        a: &Tensor,
        b: &Tensor,
        out: Option<&Tensor>,
        alpha: Option<f32>,
        beta: Option<f32>,
        bias: Option<&Tensor>,
        act: Option<CandleActivation>,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let inner_act = act.map(|a| match a {
                CandleActivation::Relu => matmul::Activation::Relu,
                CandleActivation::Gelu => matmul::Activation::Gelu,
                _ => unreachable!("Unsupported activation in cublaslt matmul"),
            });
            let mut result = fused_batch_matmul(
                a,
                b,
                out,
                alpha,
                beta,
                bias,
                inner_act,
                self.cublaslt.clone(),
            )?;

            if Some(CandleActivation::Swiglu) == act {
                result = candle_nn::ops::swiglu(&result)?;
            }
            Ok(result)
        }
        #[cfg(not(feature = "cuda"))]
        {
            candle_core::bail!("`cuda` feature is not enabled")
        }
    }
}
