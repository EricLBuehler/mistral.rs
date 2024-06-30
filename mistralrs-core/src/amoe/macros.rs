#[macro_export]
#[doc(hidden)]
macro_rules! get_delta_from_lora_ab {
    ($vb_mlp:expr, $rank:expr, $alpha:expr, ($in_d:expr, $out_d:expr), $name:expr) => {{
        let proj_a = $vb_mlp
            .pp($name)
            .pp("lora_A")
            .get(($rank, $in_d), "weight")?;
        let proj_b = $vb_mlp
            .pp($name)
            .pp("lora_B")
            .get(($out_d, $rank), "weight")?;
        (proj_b.matmul(&proj_a)? * $alpha)?
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! merge_delta {
    ($qmatmul:expr, $delta:expr) => {
        match &$qmatmul {
            QMatMul::Tensor(w) => QMatMul::Tensor((w + $delta)?),
            QMatMul::TensorF16(w) => QMatMul::TensorF16((w + $delta)?),
            QMatMul::QTensor(w) => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + $delta)?, dtype)?,
                ))
            }
        }
    };
}
