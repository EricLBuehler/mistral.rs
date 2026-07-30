use candle_core::{DType, Result, WithDType};

pub(crate) fn dtype_to_uqff_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::U8 => Ok(0),
        DType::U32 => Ok(1),
        DType::I32 => Ok(2),
        DType::I64 => Ok(3),
        DType::F16 => Ok(4),
        DType::BF16 => Ok(5),
        DType::F32 => Ok(6),
        DType::F64 => Ok(7),
        DType::I16 => Ok(8),
        DType::F8E4M3 => Ok(9),
        DType::F6E2M3 => Ok(10),
        DType::F6E3M2 => Ok(11),
        DType::F4 => Ok(12),
        DType::F8E8M0 => Ok(13),
        other => candle_core::bail!("Unsupported dtype for UQFF serialization: {other:?}"),
    }
}

pub(crate) fn uqff_code_to_dtype(dtype: u32) -> Result<DType> {
    match dtype {
        0 => Ok(DType::U8),
        1 => Ok(DType::U32),
        2 => Ok(DType::I32),
        3 => Ok(DType::I64),
        4 => Ok(DType::F16),
        5 => Ok(DType::BF16),
        6 => Ok(DType::F32),
        7 => Ok(DType::F64),
        8 => Ok(DType::I16),
        9 => Ok(DType::F8E4M3),
        10 => Ok(DType::F6E2M3),
        11 => Ok(DType::F6E3M2),
        12 => Ok(DType::F4),
        13 => Ok(DType::F8E8M0),
        _ => candle_core::bail!("unknown dtype for quantized tensor {dtype}"),
    }
}

pub(crate) fn data_to_bytes<T: WithDType>(mut vs: Vec<T>) -> Vec<u8> {
    let size_in_bytes = T::DTYPE.size_in_bytes();
    let length = vs.len() * size_in_bytes;
    let capacity = vs.capacity() * size_in_bytes;
    let ptr = vs.as_mut_ptr() as *mut u8;
    std::mem::forget(vs);
    unsafe { Vec::from_raw_parts(ptr, length, capacity) }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dtype_variant_count_unchanged() {
        assert_eq!(
            std::mem::size_of::<candle_core::DType>(),
            1,
            "DType repr size changed, check if the discriminant size is the same"
        );
        const EXPECTED_VARIANTS: usize = 14;
        let count = [
            candle_core::DType::U8,
            candle_core::DType::U32,
            candle_core::DType::I16,
            candle_core::DType::I32,
            candle_core::DType::I64,
            candle_core::DType::BF16,
            candle_core::DType::F16,
            candle_core::DType::F32,
            candle_core::DType::F64,
            candle_core::DType::F8E4M3,
            candle_core::DType::F6E2M3,
            candle_core::DType::F6E3M2,
            candle_core::DType::F4,
            candle_core::DType::F8E8M0,
        ]
        .len();
        assert_eq!(
            count, EXPECTED_VARIANTS,
            "Update this list and the UQFF match arms when DType variants change"
        );
    }
}
