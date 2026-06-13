//! The online calibration lifecycle: begin/status/apply on a live model, with from-source
//! requantization (dense via carried shards, expert stacks via the moe layout reader).

use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use mistralrs_quant::{IsqType, QuantMethod, TrackedModule};
use tracing::info;

use super::{harvest_imatrix, module_imatrix, requantize_and_swap};

#[derive(Clone, Debug, serde::Serialize)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct CalibrationStatus {
    pub collecting: bool,
    pub layers: usize,
    pub layers_tracking: usize,
    pub total_rows: usize,
    pub min_rows: usize,
    pub max_rows: usize,
}

/// Start collecting activation statistics on every tracked layer of a live model.
pub(crate) fn begin_calibration(modules: &[TrackedModule]) -> Result<usize> {
    if modules.is_empty() {
        anyhow::bail!("Online calibration requires the model to have been loaded with ISQ.");
    }
    for (i, module) in modules.iter().enumerate() {
        if let Err(e) = module.ct.begin_track_stats() {
            // half-enabled collection skews statistics and slows serving; unwind fully
            for enabled in &modules[..i] {
                let _ = enabled.ct.end_track_stats();
            }
            return Err(e.into());
        }
    }
    info!(
        "Collecting activation statistics on {} layers.",
        modules.len()
    );
    Ok(modules.len())
}

pub(crate) fn calibration_status(modules: &[TrackedModule]) -> CalibrationStatus {
    let mut tracking = 0usize;
    let mut total = 0usize;
    let mut min_rows = usize::MAX;
    let mut max_rows = 0usize;
    for module in modules {
        if let Some((_, rows)) = module.ct.stats_snapshot() {
            tracking += 1;
            total += rows;
            min_rows = min_rows.min(rows);
            max_rows = max_rows.max(rows);
        }
    }
    CalibrationStatus {
        collecting: tracking > 0,
        layers: modules.len(),
        layers_tracking: tracking,
        total_rows: total,
        min_rows: if tracking == 0 { 0 } else { min_rows },
        max_rows,
    }
}

/// Harvest collected statistics, requantize every tracked layer with the resulting imatrix,
/// and swap each into the live model. Layers without data quantize plainly with a warning.
pub(crate) fn apply_calibration(
    modules: &[TrackedModule],
    source_files: &[std::path::PathBuf],
    save_cimatrix: Option<&std::path::Path>,
) -> Result<CalibrationStatus> {
    if modules.is_empty() {
        anyhow::bail!("Online calibration requires the model to have been loaded with ISQ.");
    }
    let status = calibration_status(modules);
    if !status.collecting {
        anyhow::bail!("No calibration data collected; call start first.");
    }
    // harvest destroys the collected state; reject a bad save path before touching it
    if let Some(path) = save_cimatrix {
        if path.extension().is_none_or(|ext| ext != "cimatrix") {
            anyhow::bail!(
                "save_cimatrix path `{}` must end in .cimatrix",
                path.display()
            );
        }
    }

    let map = harvest_imatrix(modules)?;
    if let Some(path) = save_cimatrix {
        mistralrs_quant::CollectedImatrixData(map.clone()).save_imatrix(path)?;
        info!("Saved collected imatrix to `{}`.", path.display());
    }

    let pool_ty = modules
        .iter()
        .find_map(|m| m.ty)
        .context("No ISQ types recorded for tracked layers.")?;
    info!(
        "Requantizing {} layers with traffic-collected imatrix ({} layers have data).",
        modules.len(),
        map.len()
    );
    if source_files.is_empty() {
        tracing::warn!(
            "No source weights available; requantizing from resident quantized weights (reduced quality)."
        );
        requantize_and_swap(modules, pool_ty, |m| m.ty.unwrap_or(pool_ty), &|key| {
            map.get(key).cloned()
        })?;
    } else {
        requantize_from_source(modules, source_files, pool_ty, &map)?;
    }
    Ok(status)
}

/// Bare from-source replacement cannot reproduce a multi-rank RowParallel's all-reduce; those
/// modules go to dequant-requant instead, whose apply_isq rebuilds the wrapper.
fn needs_distributed_wrapper(module: &TrackedModule) -> bool {
    let multi_rank = matches!(
        module.shard,
        Some(mistralrs_quant::Shard::Simple { world_size, .. }) if world_size > 1
    );
    multi_rank
        && matches!(
            module.ct.is_distributed(),
            Some(mistralrs_quant::DistributedKind::RowParallel)
        )
}

/// Mmap-backed view of the original checkpoint for from-source requantization.
struct SourceWeights {
    mmap: std::sync::Arc<mistralrs_quant::safetensors::MmapedSafetensors>,
    shapes: HashMap<String, Vec<usize>>,
}

impl SourceWeights {
    fn open(files: &[std::path::PathBuf]) -> Result<Self> {
        // mmap safety: source checkpoint files are not mutated while serving
        let mmap = std::sync::Arc::new(unsafe {
            mistralrs_quant::safetensors::MmapedSafetensors::multi(files)?
        });
        // non-float tensors are pre-quantized (FP8, BnB) and unusable without their scales
        let shapes = mmap
            .tensors()
            .into_iter()
            .filter(|(_, view)| {
                matches!(
                    view.dtype(),
                    safetensors::Dtype::F64
                        | safetensors::Dtype::F32
                        | safetensors::Dtype::F16
                        | safetensors::Dtype::BF16
                )
            })
            .map(|(name, view)| (name, view.shape().to_vec()))
            .collect();
        Ok(Self { mmap, shapes })
    }

    fn has_dense(&self, key: &str) -> bool {
        self.shapes.contains_key(&format!("{key}.weight"))
    }

    fn has_expert_stack(&self, key: &str) -> bool {
        crate::moe::expert_stack_available(&self.shapes, key)
    }
}

fn quantize_source_tensor(
    w: Tensor,
    b: Option<Tensor>,
    ty: IsqType,
    device: Device,
    imatrix: Option<Vec<f32>>,
    guard: mistralrs_quant::QuantizeOntoGuard,
) -> Result<std::sync::Arc<dyn QuantMethod>> {
    let unquant = std::sync::Arc::new(mistralrs_quant::UnquantLinear::new(
        mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(w, b)),
    )?) as std::sync::Arc<dyn QuantMethod>;
    Ok(unquant.apply_isq(
        Some(ty),
        device,
        &std::sync::atomic::AtomicUsize::new(0),
        imatrix,
        guard,
    )?)
}

/// Requantize tracked modules from the original source weights on the ISQ pool. Expert stacks
/// rebuild serially on this thread (one [E, out, in] tensor in memory at a time); layers absent
/// from the source fall back to dequant-requant.
pub(crate) fn requantize_from_source(
    modules: &[TrackedModule],
    source_files: &[std::path::PathBuf],
    pool_ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let source = SourceWeights::open(source_files)?;

    let mut dense = Vec::new();
    let mut experts = Vec::new();
    let mut fallback = Vec::new();
    for module in modules {
        if source.has_dense(&module.key)
            && module.shard.is_some()
            && !needs_distributed_wrapper(module)
        {
            dense.push(module.clone());
        } else if source.has_expert_stack(&module.key) && module.shard.is_some() {
            experts.push(module.clone());
        } else {
            fallback.push(module.clone());
        }
    }
    info!(
        "Requantizing from source weights: {} dense, {} expert stacks ({} fall back to resident weights).",
        dense.len(),
        experts.len(),
        fallback.len()
    );

    let (pool, _) = mistralrs_quant::create_isq_thread_pool(Some(pool_ty));
    let guard = mistralrs_quant::QuantizeOntoGuard::new();
    let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
    let n_jobs = dense.len();
    for module in dense {
        let mmap = source.mmap.clone();
        let guard = guard.clone();
        let tx = tx.clone();
        let (ty, imatrix) = module_imatrix(&module, pool_ty, imatrix_map);
        let source_has_bias = source.shapes.contains_key(&format!("{}.bias", module.key));
        pool.spawn(move || {
            let job = || -> Result<()> {
                let resident = module.ct.resolve()?;
                let (_, device) = resident.dtype_and_device();
                let shard = module.shard.expect("partition requires a shard");
                let w = mmap.load(&format!("{}.weight", module.key), &Device::Cpu, None)?;
                // force_contiguous: offset views share storage, and QTensor::quantize reads raw storage
                let w = shard.apply_to(&w)?.force_contiguous()?;
                let b = if resident.has_bias() && source_has_bias {
                    let b = mmap.load(&format!("{}.bias", module.key), &Device::Cpu, None)?;
                    // Out-dim shards slice the bias the same way; in-dim shards leave it whole.
                    let sharded_dim0 = matches!(
                        shard,
                        mistralrs_quant::Shard::Simple { dim: 0, .. }
                            | mistralrs_quant::Shard::Offset { dim: 0, .. }
                    );
                    Some(if sharded_dim0 {
                        shard.apply_to(&b)?.force_contiguous()?
                    } else {
                        b
                    })
                } else {
                    None
                };
                module
                    .ct
                    .replace(quantize_source_tensor(w, b, ty, device, imatrix, guard)?);
                Ok(())
            };
            let _ = tx.send(job());
        });
    }
    drop(tx);

    let mut errors: Vec<String> = Vec::new();
    for module in &experts {
        let job = || -> Result<()> {
            let stack =
                crate::moe::rebuild_expert_stack(&source.mmap, &source.shapes, &module.key)?
                    .context("Expert stack probe succeeded but rebuild failed.")?;
            let resident = module.ct.resolve()?;
            let (_, device) = resident.dtype_and_device();
            let (ty, imatrix) = module_imatrix(module, pool_ty, imatrix_map);
            module.ct.replace(mistralrs_quant::quantize_expert_stack(
                stack,
                ty,
                imatrix,
                &device,
                guard.clone(),
            )?);
            Ok(())
        };
        if let Err(e) = job() {
            errors.push(format!("{}: {e:#}", module.key));
        }
    }

    // drain everything; failed layers keep their prior resident, so a partial apply stays consistent
    let mut received = 0usize;
    for result in rx {
        received += 1;
        if let Err(e) = result {
            errors.push(format!("{e:#}"));
        }
    }
    anyhow::ensure!(
        received == n_jobs,
        "From-source requantize jobs died early."
    );
    if !errors.is_empty() {
        anyhow::bail!(
            "{} of {} from-source requantize jobs failed; first: {}",
            errors.len(),
            n_jobs + experts.len(),
            errors[0]
        );
    }

    if !fallback.is_empty() {
        requantize_and_swap(&fallback, pool_ty, |m| m.ty.unwrap_or(pool_ty), &|key| {
            imatrix_map.get(key).cloned()
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    const E: usize = 4;
    const INTER: usize = 8;

    fn write_st(path: &std::path::Path, tensors: Vec<(String, Tensor)>) {
        candle_core::safetensors::save(&tensors.into_iter().collect(), path).unwrap();
    }

    #[test]
    fn from_source_respects_shard() -> Result<()> {
        use mistralrs_quant::{QuantMethod, Shard, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let truth = Tensor::randn(0f32, 1f32, (8, 32), &Device::Cpu)?;
        candle_core::safetensors::save(
            &[("m.lin.weight".to_string(), truth.clone())]
                .into_iter()
                .collect(),
            &file,
        )?;

        // resident is rank 1 of 2: rows 4..8, quantized zeros until the swap
        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((4, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = std::sync::Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: std::sync::Arc::new(zeros),
                b: None,
            },
        )?) as std::sync::Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(resident)).unwrap();
        let ct = std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.lin".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            shard: Some(Shard::Simple {
                dim: 0,
                rank: 1,
                world_size: 2,
            }),
        };

        requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())?;

        let swapped = ct.resolve()?.dequantize_w()?;
        assert_eq!(swapped.dims(), [4, 32]);
        let expected = truth.narrow(0, 4, 4)?;
        let diff = (&swapped - &expected)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(diff < 0.05, "max diff {diff}");
        Ok(())
    }

    #[test]
    fn from_source_replaces_expert_layer() -> Result<()> {
        use mistralrs_quant::{QuantMethod, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        // in-dim 32 so q8_0 blocks (32) divide evenly
        let truth = Tensor::randn(0f32, 1f32, (E, INTER, 32), &Device::Cpu)?;
        let slabs: Vec<Tensor> = (0..E).map(|i| truth.get(i).unwrap()).collect();
        let mut tensors = Vec::new();
        for (i, s) in slabs.iter().enumerate() {
            tensors.push((format!("m.experts.{i}.w1.weight"), s.clone()));
        }
        write_st(&file, tensors);

        // resident starts as quantized zeros; from-source must replace it with real weights
        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((E, INTER, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = std::sync::Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: std::sync::Arc::new(zeros),
                b: None,
            },
        )?) as std::sync::Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(resident)).unwrap();
        let ct = std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.experts.gate_proj".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            shard: Some(mistralrs_quant::Shard::default()),
        };

        requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())?;

        let swapped = ct.resolve()?.dequantize_w()?;
        let cos = {
            let a = swapped.flatten_all()?;
            let b = truth.flatten_all()?;
            let dot = (&a * &b)?.sum_all()?.to_scalar::<f32>()?;
            let na = (&a * &a)?.sum_all()?.to_scalar::<f32>()?.sqrt();
            let nb = (&b * &b)?.sum_all()?.to_scalar::<f32>()?.sqrt();
            dot / (na * nb)
        };
        assert!(cos > 0.99, "cos {cos}");
        Ok(())
    }
}
