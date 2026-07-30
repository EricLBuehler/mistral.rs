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
enum ImatrixLayerStats_ {
    Dense {
        row_counts: usize,
        ncalls: usize,
        row_accum: Tensor,
    },
    /// Routed (MoE): `accum [E, in]`, `counts [E]`; `rows` mirrors counts' total on the CPU so
    /// snapshots never sync the device.
    Routed {
        ncalls: usize,
        rows: usize,
        counts: Tensor,
        accum: Tensor,
    },
}

#[derive(Debug, Clone)]
pub struct ImatrixLayerStats(Arc<RwLock<Option<ImatrixLayerStats_>>>);

impl ImatrixLayerStats {
    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }

    /// Start collecting; safe on shared layers since the state is interior-mutable.
    pub fn enable(&self, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Dense {
            row_counts: 0,
            ncalls: 0,
            row_accum: Tensor::zeros((in_dim,), DType::F32, device)?,
        });
        Ok(())
    }

    /// Start collecting per-expert statistics for a routed (stacked `[E, out, in]`) layer.
    pub fn enable_routed(&self, num_experts: usize, in_dim: usize, device: &Device) -> Result<()> {
        *self.0.write().unwrap() = Some(ImatrixLayerStats_::Routed {
            ncalls: 0,
            rows: 0,
            counts: Tensor::zeros((num_experts,), DType::F32, device)?,
            accum: Tensor::zeros((num_experts, in_dim), DType::F32, device)?,
        });
        Ok(())
    }

    pub fn is_enabled(&self) -> bool {
        self.0.read().unwrap().is_some()
    }

    /// (forward calls, token rows) so far; routed rows are token x top-k slots.
    pub fn snapshot(&self) -> Option<(usize, usize)> {
        self.0.read().unwrap().as_ref().map(|s| match s {
            ImatrixLayerStats_::Dense {
                ncalls, row_counts, ..
            } => (*ncalls, *row_counts),
            ImatrixLayerStats_::Routed { ncalls, rows, .. } => (*ncalls, *rows),
        })
    }

    pub fn process(&self, inp: &Tensor) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        // routed layers are observed by their router; a stray plain forward contributes nothing
        let Some(ImatrixLayerStats_::Dense {
            row_counts,
            ncalls,
            row_accum,
        }) = handle.as_mut()
        else {
            return Ok(());
        };

        let inp = inp.reshape(((), inp.dim(D::Minus1)?))?;
        *ncalls += 1;
        // Counts are token rows, yielding mean square per input column.
        *row_counts += inp.dim(0)?;
        *row_accum = (&*row_accum + inp.to_dtype(DType::F32)?.sqr()?.sum(0)?)?;
        Ok(())
    }

    /// Accumulate routed activations: `ids` is `(n, k)`; `x` is `(n, in)` (row scatters into all
    /// its experts) or `(n, k, in)` (slot `(t, s)` scatters into `ids[t, s]` only).
    pub fn process_routed(&self, x: &Tensor, ids: &Tensor) -> Result<()> {
        if self.0.read().unwrap().is_none() {
            return Ok(());
        }
        let mut handle = self.0.write().unwrap();
        let Some(ImatrixLayerStats_::Routed {
            ncalls,
            rows,
            counts,
            accum,
        }) = handle.as_mut()
        else {
            return Ok(());
        };

        let (n, k) = ids.dims2()?;
        let in_dim = x.dim(D::Minus1)?;
        let x2 = match x.dims().len() {
            2 => x
                .to_dtype(DType::F32)?
                .sqr()?
                .unsqueeze(1)?
                .broadcast_as((n, k, in_dim))?
                .force_contiguous()?
                .reshape((n * k, in_dim))?,
            3 => x.to_dtype(DType::F32)?.sqr()?.reshape((n * k, in_dim))?,
            other => candle_core::bail!("process_routed expects rank 2 or 3 input, got {other}"),
        };
        let ids_flat = ids.flatten_all()?.to_dtype(DType::U32)?;
        *accum = accum.index_add(&ids_flat, &x2, 0)?;
        let ones = Tensor::ones((n * k,), DType::F32, counts.device())?;
        *counts = counts.index_add(&ids_flat, &ones, 0)?;
        *rows += n * k;
        *ncalls += 1;
        Ok(())
    }

    /// Dense: `[in]`. Routed: `[E, in]` with all-zero rows for zero-traffic experts.
    pub fn compute_imatrix(&self) -> Result<Tensor> {
        let handle = self.0.read().unwrap();
        match handle.as_ref().context("Layer stats were deinitialized!")? {
            ImatrixLayerStats_::Dense {
                row_counts,
                ncalls,
                row_accum,
            } => {
                if *row_counts == 0 {
                    candle_core::bail!("No activations were recorded for this layer.");
                }
                (row_accum / *row_counts as f64)? * *ncalls as f64
            }
            ImatrixLayerStats_::Routed {
                ncalls,
                counts,
                accum,
                ..
            } => {
                let total = counts.sum_all()?.to_scalar::<f32>()?;
                if total == 0.0 {
                    candle_core::bail!("No activations were recorded for this layer.");
                }
                // Per-expert mean square; zero-traffic experts divide to zero, not NaN.
                let safe_counts = counts.maximum(1.0)?.unsqueeze(1)?;
                accum.broadcast_div(&safe_counts)? * *ncalls as f64
            }
        }
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
    fn routed_stats_scatter_per_expert() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(3, 2, &device)?;

        // two tokens, k=2: token0 -> experts {0, 2}, token1 -> experts {2, 1}
        let x = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
        let ids = Tensor::new(&[[0u32, 2], [2, 1]], &device)?;
        stats.process_routed(&x, &ids)?;

        assert_eq!(stats.snapshot(), Some((1, 4)));
        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        // expert 0: token0 only; expert 1: token1 only; expert 2: mean of both
        assert_eq!(m[0], [1., 4.]);
        assert_eq!(m[1], [9., 16.]);
        assert_eq!(m[2], [(1. + 9.) / 2., (4. + 16.) / 2.]);
        Ok(())
    }

    #[test]
    fn routed_stats_rank3_per_slot() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(2, 2, &device)?;

        // per-slot inputs: slot (0,0)->e0 with [1,0]; slot (0,1)->e1 with [0,2]
        let x = Tensor::new(&[[[1f32, 0.], [0., 2.]]], &device)?;
        let ids = Tensor::new(&[[0u32, 1]], &device)?;
        stats.process_routed(&x, &ids)?;

        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        assert_eq!(m[0], [1., 0.]);
        assert_eq!(m[1], [0., 4.]);
        Ok(())
    }

    #[test]
    fn dead_expert_rows_are_zero_not_nan() -> Result<()> {
        let device = Device::Cpu;
        let stats = ImatrixLayerStats::empty();
        stats.enable_routed(3, 2, &device)?;
        let x = Tensor::new(&[[1f32, 1.]], &device)?;
        let ids = Tensor::new(&[[0u32, 0]], &device)?;
        stats.process_routed(&x, &ids)?;
        let m = stats.compute_imatrix()?.to_vec2::<f32>()?;
        assert!(m[1].iter().all(|v| *v == 0.0));
        assert!(m.iter().flatten().all(|v| v.is_finite()));
        Ok(())
    }

    #[test]
    fn expert_stack_per_slab_commutes() -> Result<()> {
        let device = Device::Cpu;
        // in-dim 256 so Q4K applies without fallback
        let stack = Tensor::randn(0f32, 1f32, (3, 8, 256), &device)?;
        let shared: Vec<f32> = (0..256).map(|i| 1.0 + (i % 7) as f32).collect();

        // shared vector through per-slab assembly is byte-identical to whole-stack quantize
        let per_slab = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            crate::IsqType::Q4K,
            Some(&shared),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;
        let whole = QTensor::quantize_imatrix(&stack, &shared, GgmlDType::Q4K)?;
        assert_eq!(
            per_slab.data()?.to_vec(),
            whole.data()?.to_vec(),
            "per-slab assembly diverged from whole-stack quantization"
        );

        // per-expert vectors actually differentiate slabs
        let mut per_expert = shared.clone();
        per_expert.extend((0..256).map(|i| 100.0 + i as f32));
        per_expert.extend((0..256).map(|_| 1.0));
        let routed = crate::GgufMatMul::quantize_expert_stack(
            &stack,
            crate::IsqType::Q4K,
            Some(&per_expert),
            &device,
            crate::QuantizeOntoGuard::new(),
        )?;
        assert_ne!(
            routed.data()?.to_vec(),
            whole.data()?.to_vec(),
            "per-expert vectors had no effect"
        );
        // expert 0 shares its vector with the shared case and must be bit-identical
        let block_bytes = whole.data()?.len() / 3;
        assert_eq!(
            &routed.data()?[..block_bytes],
            &whole.data()?[..block_bytes],
            "expert 0 should match the shared quantization"
        );
        Ok(())
    }

    #[test]
    fn apply_isq_routes_rank3_imatrix_per_slab() -> Result<()> {
        use std::sync::{atomic::AtomicUsize, Arc};
        let device = Device::Cpu;
        let stack = Tensor::randn(0f32, 1f32, (2, 8, 256), &device)?;
        // expert 1 gets a wildly skewed importance vector so its blocks must quantize differently
        let mut routed: Vec<f32> = vec![1.0; 256];
        routed.extend((0..256).map(|i| if i < 128 { 1000.0 } else { 0.001 }));

        // unquantized resident (capture mode / from-source)
        let make = |im: Option<Vec<f32>>| -> Result<Tensor> {
            let unquant = Arc::new(crate::UnquantLinear::new(QuantMethodConfig::Unquantized(
                candle_nn::Linear::new(stack.clone(), None),
            ))?) as Arc<dyn QuantMethod>;
            unquant
                .apply_isq(
                    Some(crate::IsqType::Q4K),
                    device.clone(),
                    &AtomicUsize::new(0),
                    im,
                    crate::QuantizeOntoGuard::new(),
                )?
                .dequantize_w()
        };
        let with_imatrix = make(Some(routed.clone()))?;
        assert_eq!(with_imatrix.dims(), [2, 8, 256]);
        let without = make(None)?;
        let diff = (&with_imatrix - &without)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff > 0.0, "routed imatrix had no effect on quantization");

        // quantized resident (apply without source weights)
        let q = Arc::new(crate::GgufMatMul::from_qtensor(
            QTensor::quantize(&stack, GgmlDType::Q8_0)?,
            None,
        )) as Arc<dyn QuantMethod>;
        let q2 = q.apply_isq(
            Some(crate::IsqType::Q4K),
            device,
            &AtomicUsize::new(0),
            Some(routed),
            crate::QuantizeOntoGuard::new(),
        )?;
        assert_eq!(q2.dequantize_w()?.dims(), [2, 8, 256]);
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
