use candle_core::{DType, Device, IndexOp, Result, Tensor};

pub struct DiaKvCache {
    k: Tensor,
    v: Tensor,
    current_index: usize,
}

impl DiaKvCache {
    pub fn new(shape: (usize, usize, usize, usize), dtype: DType, device: &Device) -> Result<Self> {
        Ok(Self {
            k: Tensor::zeros(shape, dtype, device)?,
            v: Tensor::zeros(shape, dtype, device)?,
            current_index: 0,
        })
    }

    pub fn k_v(&self) -> (Tensor, Tensor) {
        (self.k.clone(), self.v.clone())
    }

    pub fn from_kv(k: Tensor, v: Tensor) -> Self {
        Self {
            k,
            v,
            current_index: 0,
        }
    }

    pub fn update(&self, k: &Tensor, v: &Tensor, current_index: usize) -> Result<(Tensor, Tensor)> {
        self.k.slice_set(k, 2, current_index)?;
        self.v.slice_set(v, 2, current_index)?;
        Ok((self.k.clone(), self.v.clone()))
    }

    pub fn prefill(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.k.slice_set(k, 2, 0)?;
        self.v.slice_set(v, 2, 0)?;
        let prefill_len = k.dim(2)?;
        self.current_index = prefill_len - 1;
        Ok((
            self.k.i((.., .., ..prefill_len, ..))?,
            self.v.i((.., .., ..prefill_len, ..))?,
        ))
    }
}
