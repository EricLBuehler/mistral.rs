use std::sync::Arc;

use candle_core::{
    quantized::{gguf_file, QMatMul, QTensor},
    Device, Result, Tensor,
};
use candle_nn::{Linear, Module};

#[derive(Debug, Clone)]
pub struct QLinear {
    inner: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    pub fn new<R: std::io::Read + std::io::Seek>(
        ct: &gguf_file::Content,
        r: &mut R,
        name: &str,
        device: &Device,
    ) -> Result<Self> {
        let w = ct.tensor(r, &format!("{name}.weight"), device)?;
        let b = ct.tensor(r, &format!("{name}.bias"), device)?;
        let inner = QMatMul::from_qtensor(w)?;
        let bias = b.dequantize(device)?;
        Ok(Self {
            inner,
            bias: Some(bias),
        })
    }

    pub fn from_linear(linear: Linear) -> Self {
        Self {
            inner: QMatMul::Tensor(linear.weight().clone()),
            bias: Some(linear.bias().unwrap().clone()),
        }
    }

    pub fn from_parts(w: Tensor, b: Option<Tensor>) -> Self {
        Self {
            inner: QMatMul::Tensor(w),
            bias: b,
        }
    }

    pub fn from_qparts(w: QTensor, b: Option<Tensor>) -> Self {
        Self {
            inner: QMatMul::QTensor(Arc::new(w)),
            bias: b,
        }
    }

    pub fn inner(&mut self) -> &mut QMatMul {
        &mut self.inner
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if let Some(bias) = &self.bias {
            self.inner.forward(xs)?.broadcast_add(bias)
        } else {
            self.inner.forward(xs)
        }
    }
}
