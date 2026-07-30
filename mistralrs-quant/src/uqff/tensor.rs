use std::borrow::Cow;

use candle_core::{DType, Result, Tensor};
use float8::F8E4M3;
use safetensors::tensor::{Dtype, View};

use crate::utils::data_to_bytes;

#[derive(Debug, Clone)]
pub struct UqffTensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl UqffTensor {
    pub fn from_tensor(name: impl Into<String>, tensor: &Tensor) -> Result<Self> {
        let shape = tensor.dims().to_vec();
        let dtype = to_safetensors_dtype(tensor.dtype())?;
        let flat = tensor.flatten_all()?;
        let data = match flat.dtype() {
            DType::U8 => data_to_bytes::<u8>(flat.to_vec1()?),
            DType::U32 => data_to_bytes::<u32>(flat.to_vec1()?),
            DType::I16 => data_to_bytes::<i16>(flat.to_vec1()?),
            DType::I32 => data_to_bytes::<i32>(flat.to_vec1()?),
            DType::I64 => data_to_bytes::<i64>(flat.to_vec1()?),
            DType::F16 => data_to_bytes::<half::f16>(flat.to_vec1()?),
            DType::BF16 => data_to_bytes::<half::bf16>(flat.to_vec1()?),
            DType::F32 => data_to_bytes::<f32>(flat.to_vec1()?),
            DType::F64 => data_to_bytes::<f64>(flat.to_vec1()?),
            DType::F8E4M3 => data_to_bytes::<F8E4M3>(flat.to_vec1()?),
            other => candle_core::bail!("Unsupported UQFF tensor dtype: {other:?}"),
        };
        Ok(Self {
            name: name.into(),
            dtype,
            shape,
            data,
        })
    }

    pub fn from_raw_u8(name: impl Into<String>, data: Vec<u8>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            dtype: Dtype::U8,
            shape,
            data,
        }
    }

    pub fn from_u8_scalar(name: impl Into<String>, value: u8) -> Self {
        Self::from_raw_u8(name, vec![value], vec![])
    }

    pub fn from_u32_scalar(name: impl Into<String>, value: u32) -> Self {
        Self::from_u32_vec(name, vec![value], vec![])
    }

    pub fn from_u32_vec(name: impl Into<String>, values: Vec<u32>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            dtype: Dtype::U32,
            shape,
            data: data_to_bytes(values),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn scalar_u8(&self) -> Result<u8> {
        if self.dtype != Dtype::U8 || !self.shape.is_empty() || self.data.len() != 1 {
            candle_core::bail!("UQFF tensor `{}` is not a u8 scalar.", self.name);
        }
        Ok(self.data[0])
    }

    pub fn scalar_u32(&self) -> Result<u32> {
        if self.dtype != Dtype::U32 || !self.shape.is_empty() || self.data.len() != 4 {
            candle_core::bail!("UQFF tensor `{}` is not a u32 scalar.", self.name);
        }
        Ok(u32::from_le_bytes(
            self.data
                .as_slice()
                .try_into()
                .expect("u32 scalar is four bytes"),
        ))
    }

    pub fn u32_values(&self) -> Result<Vec<usize>> {
        if self.dtype != Dtype::U32 || !self.data.len().is_multiple_of(4) {
            candle_core::bail!("UQFF tensor `{}` is not a u32 vector.", self.name);
        }
        Ok(self
            .data
            .chunks_exact(4)
            .map(|chunk| {
                u32::from_le_bytes(chunk.try_into().expect("chunk is four bytes")) as usize
            })
            .collect())
    }

    pub fn nbytes(&self) -> usize {
        self.data.len()
    }
}

impl View for &UqffTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn to_safetensors_dtype(dtype: DType) -> Result<Dtype> {
    match dtype {
        DType::U8 => Ok(Dtype::U8),
        DType::U32 => Ok(Dtype::U32),
        DType::I16 => Ok(Dtype::I16),
        DType::I32 => Ok(Dtype::I32),
        DType::I64 => Ok(Dtype::I64),
        DType::F16 => Ok(Dtype::F16),
        DType::BF16 => Ok(Dtype::BF16),
        DType::F32 => Ok(Dtype::F32),
        DType::F64 => Ok(Dtype::F64),
        DType::F8E4M3 => Ok(Dtype::F8_E4M3),
        other => candle_core::bail!("Unsupported UQFF safetensors dtype: {other:?}"),
    }
}
