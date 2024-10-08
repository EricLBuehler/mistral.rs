use candle_core::{DType, Result, Tensor};
use float8::F8E4M3;

use super::FP8Linear;

impl FP8Linear {
    fn quantize(&self, w: &Tensor) -> Result<Tensor> {
        let mut w = w.to_dtype(DType::F32)?.mean_all()?;
        while !w.dims().is_empty() {
            w = w.min(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        todo!()
    }
}