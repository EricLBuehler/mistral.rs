use std::{
    collections::HashMap,
    fs,
    io::Cursor,
    path::Path,
    sync::{Arc, RwLock},
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
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
        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        // Number of entries
        cursor.write_u64::<LittleEndian>(self.0.len() as u64)?;

        for (i, data) in &self.0 {
            // i
            cursor.write_u64::<LittleEndian>(*i as u64)?;
            // has data
            cursor.write_u8(data.is_some() as u8)?;
            if let Some(data) = data {
                // data len
                cursor.write_u64::<LittleEndian>(data.len() as u64)?;
                // data
                for x in data {
                    cursor.write_f32::<LittleEndian>(*x)?;
                }
            }
        }

        fs::write(fname, buf)?;
        Ok(())
    }

    pub fn load_imatrix<P: AsRef<Path>>(fname: P) -> Result<Self> {
        let buf = fs::read(fname)?;
        let mut cursor = Cursor::new(buf);

        let mut entries = HashMap::new();
        let num_entries = cursor.read_u64::<LittleEndian>()?;

        for _ in 0..num_entries {
            let i = cursor.read_u64::<LittleEndian>()?;
            let has_data = cursor.read_u8()? != 0;
            if has_data {
                let len_data = cursor.read_u64::<LittleEndian>()?;
                let mut data = Vec::new();
                for _ in 0..len_data {
                    data.push(cursor.read_f32::<LittleEndian>()?);
                }
                entries.insert(i as usize, Some(data));
            } else {
                entries.insert(i as usize, None);
            }
        }

        Ok(Self(entries))
    }
}
