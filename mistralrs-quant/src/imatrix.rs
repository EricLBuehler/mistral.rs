use std::{
    collections::HashMap,
    fs,
    path::Path,
    sync::{Arc, RwLock},
};

use candle_core::{Context, DType, Device, Result, Tensor, D};
use serde::{Deserialize, Serialize};

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

#[derive(Serialize, Deserialize)]
pub struct CollectedImatrixData(pub HashMap<usize, Option<Vec<f32>>>);

impl CollectedImatrixData {
    pub fn save_imatrix<P: AsRef<Path>>(&self, fname: P) -> Result<()> {
        if let Some(ext) = fname.as_ref().extension() {
            if ext != "cimatrix" {
                candle_core::bail!(
                    "Expected a .cimatrix file to save collectd imatrix data to, got {:?}",
                    ext
                );
            }
        }
        let ser = serde_json::to_string(&self.0).map_err(candle_core::Error::msg)?;
        fs::write(fname, ser)?;
        Ok(())
    }

    pub fn load_imatrix<P: AsRef<Path>>(fname: P) -> Result<Self> {
        let ser = fs::read_to_string(fname)?;
        serde_json::from_str(&ser).map_err(candle_core::Error::msg)
    }
}
