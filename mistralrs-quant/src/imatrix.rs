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
    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }

    /// Start collecting; safe on shared layers since the state is interior-mutable.
    pub fn enable(&self, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_ {
            row_counts: 0,
            ncalls: 0,
            row_accum: Tensor::zeros((in_dim,), DType::F32, device)?,
        });
        Ok(())
    }

    pub fn is_enabled(&self) -> bool {
        self.0.read().unwrap().is_some()
    }

    /// (forward calls, token rows) accumulated so far, when enabled.
    pub fn snapshot(&self) -> Option<(usize, usize)> {
        self.0
            .read()
            .unwrap()
            .as_ref()
            .map(|s| (s.ncalls, s.row_counts))
    }

    pub fn process(&self, inp: &Tensor) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        let this = handle.as_mut().context("Layer stats were dinitialized!")?;

        let inp = inp.reshape(((), inp.dim(D::Minus1)?))?;
        this.ncalls += 1;
        // Counts are token rows, yielding mean square per input column.
        this.row_counts += inp.dim(0)?;
        this.row_accum = (&this.row_accum + inp.to_dtype(DType::F32)?.sqr()?.sum(0)?)?;
        Ok(())
    }

    pub fn compute_imatrix(&self) -> Result<Tensor> {
        let handle = self.0.read().unwrap();
        let this = handle.as_ref().context("Layer stats were dinitialized!")?;
        if this.row_counts == 0 {
            candle_core::bail!("No activations were recorded for this layer.");
        }
        (&this.row_accum / this.row_counts as f64)? * this.ncalls as f64
    }

    pub fn clear(&self) -> Result<()> {
        let mut handle = self.0.write().unwrap();
        *handle = None;
        Ok(())
    }
}

/// Collected imatrix data keyed by layer tracking key (`.cimatrix` format).
#[derive(Serialize, Deserialize)]
pub struct CollectedImatrixData(pub HashMap<String, Vec<f32>>);

impl CollectedImatrixData {
    pub fn save_imatrix<P: AsRef<Path>>(&self, fname: P) -> Result<()> {
        if let Some(ext) = fname.as_ref().extension() {
            if ext != "cimatrix" {
                candle_core::bail!(
                    "Expected a .cimatrix file to save collected imatrix data to, got {:?}",
                    ext
                );
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut buf);

        cursor.write_u64::<LittleEndian>(self.0.len() as u64)?;
        for (key, data) in &self.0 {
            cursor.write_u64::<LittleEndian>(key.len() as u64)?;
            std::io::Write::write_all(&mut cursor, key.as_bytes())?;
            cursor.write_u64::<LittleEndian>(data.len() as u64)?;
            for x in data {
                cursor.write_f32::<LittleEndian>(*x)?;
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
            let key_len = cursor.read_u64::<LittleEndian>()? as usize;
            let mut key = vec![0u8; key_len];
            std::io::Read::read_exact(&mut cursor, &mut key)?;
            let key = String::from_utf8(key)
                .map_err(|_| candle_core::Error::Msg("Invalid cimatrix key".to_string()))?;
            let len_data = cursor.read_u64::<LittleEndian>()? as usize;
            let mut data = Vec::with_capacity(len_data);
            for _ in 0..len_data {
                data.push(cursor.read_f32::<LittleEndian>()?);
            }
            entries.insert(key, data);
        }

        Ok(Self(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QuantMethod, QuantMethodConfig};
    use candle_core::quantized::{GgmlDType, QTensor};

    fn manual_imatrix(rows: &[Vec<f32>], ncalls: usize) -> Vec<f32> {
        let in_dim = rows[0].len();
        let mut sums = vec![0f32; in_dim];
        for row in rows {
            for (s, x) in sums.iter_mut().zip(row) {
                *s += x * x;
            }
        }
        sums.iter()
            .map(|s| s / rows.len() as f32 * ncalls as f32)
            .collect()
    }

    #[test]
    fn gguf_layer_collects_imatrix() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;

        layer.begin_track_stats()?;
        let x1 = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        let x2 = Tensor::randn(0f32, 1f32, (3, 32), &device)?;
        layer.forward_raw(&x1)?;
        layer.forward_raw(&x2)?;
        let imatrix = layer.end_track_stats()?.to_vec1::<f32>()?;

        let mut rows: Vec<Vec<f32>> = x1.to_vec2()?;
        rows.extend(x2.to_vec2::<f32>()?);
        let expected = manual_imatrix(&rows, 2);
        for (a, b) in imatrix.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-4, "got {a}, expected {b}");
        }
        Ok(())
    }

    #[test]
    fn snapshot_visible_through_layer() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;
        assert!(layer.stats_snapshot().is_none());
        layer.begin_track_stats()?;
        assert_eq!(layer.stats_snapshot(), Some((0, 0)));
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        layer.forward_raw(&x)?;
        assert_eq!(layer.stats_snapshot(), Some((1, 4)));
        Ok(())
    }

    #[test]
    fn disabled_stats_are_free_of_state() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &device)?;
        let q = QTensor::quantize(&w, GgmlDType::Q8_0)?;
        let layer = crate::GgufMatMul::new(QuantMethodConfig::Gguf {
            q_weight: std::sync::Arc::new(q),
            b: None,
        })?;
        let x = Tensor::randn(0f32, 1f32, (4, 32), &device)?;
        layer.forward_raw(&x)?;
        assert!(layer.end_track_stats().is_err());
        Ok(())
    }
}
