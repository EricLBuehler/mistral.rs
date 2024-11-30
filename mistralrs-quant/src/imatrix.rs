use std::sync::{Arc, RwLock};

use candle_core::{Context, DType, Device, Result, Tensor, D};

#[derive(Debug)]
struct ImatrixLayerStats_ {
    row_counts: usize,
    ncalls: usize,
    row_accum: Tensor,
}

#[derive(Debug, Clone)]
pub struct ImatrixLayerStats(Arc<RwLock<Option<ImatrixLayerStats_>>>);

impl ImatrixLayerStats {
    pub fn new(w: &Tensor, device: &Device) -> Result<Self> {
        Ok(Self(Arc::new(RwLock::new(Some(ImatrixLayerStats_ {
            row_counts: 0,
            ncalls: 0,
            row_accum: Tensor::zeros((w.dim(1)?,), DType::F32, device)?,
        })))))
    }

    pub fn process(&self, inp: &Tensor) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        let this = handle.as_mut().context("Layer stats were dinitialized!")?;

        let inp = inp.reshape(((), inp.dim(D::Minus1)?))?;
        this.ncalls += 1;
        this.row_counts += inp.dim(D::Minus1)?;
        this.row_accum = (&this.row_accum + inp.to_dtype(DType::F32)?.sqr()?.sum(0)?)?;
        Ok(())
    }

    pub fn compute_imatrix(&self) -> Result<Tensor> {
        let handle = self.0.read().unwrap();
        let this = handle.as_ref().context("Layer stats were dinitialized!")?;
        (&this.row_accum / this.row_counts as f64)? * this.ncalls as f64
    }

    pub fn clear(&self) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        *handle = None;
        Ok(())
    }
}
