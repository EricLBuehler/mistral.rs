#[macro_export]
#[doc(hidden)]
macro_rules! get_delta_from_lora_ab {
    ($vb_mlp:expr, $rank:expr, $alpha:expr, $h_sz:expr, $i_sz:expr, $name:expr) => {{
        let proj_a = $vb_mlp
            .pp($name)
            .pp("lora_A")
            .get(($rank, $h_sz), "weight")?;
        let proj_b = $vb_mlp
            .pp($name)
            .pp("lora_B")
            .get(($i_sz, $rank), "weight")?;
        (proj_b.matmul(&proj_a)? * $alpha)?
    }};
}

#[macro_export]
#[doc(hidden)]
macro_rules! merge_delta {
    ($qmatmul:expr, $delta:expr) => {
        match &$qmatmul {
            QMatMul::Tensor(w) => QMatMul::Tensor((w + &$delta)?),
            QMatMul::TensorF16(w) => QMatMul::TensorF16((w + &$delta)?),
            QMatMul::QTensor(w) => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                QMatMul::QTensor(std::sync::Arc::new(
                    candle_core::quantized::QTensor::quantize(&(w + &$delta)?, dtype)?,
                ))
            }
        }
    };
}
