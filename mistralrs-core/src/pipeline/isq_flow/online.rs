//! The online calibration lifecycle: begin/status/apply on a live model, with from-source
//! requantization (dense via carried shards, expert stacks via the moe layout reader).

use std::{collections::HashMap, sync::atomic::AtomicUsize};

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
    weight_source: Option<&dyn mistralrs_quant::QuantizedWeightSource>,
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
    if let Some(source) = weight_source {
        requantize_from_weight_source(modules, source, pool_ty, &map)?;
    } else if source_files.is_empty() {
        tracing::warn!(
            "No source weights available; requantizing from resident quantized weights (reduced quality)."
        );
        requantize_and_swap(modules, pool_ty, |m| m.resolve_type(pool_ty), &|key| {
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

fn requantize_from_weight_source(
    modules: &[TrackedModule],
    source: &dyn mistralrs_quant::QuantizedWeightSource,
    pool_ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let mut from_source = Vec::new();
    let mut fallback = Vec::new();
    for module in modules {
        if module.shard.is_some()
            && !needs_distributed_wrapper(module)
            && source.contains(&format!("{}.weight", module.key))
        {
            from_source.push(module);
        } else {
            fallback.push(module.clone());
        }
    }
    info!(
        "Requantizing from quantized source weights: {} layers ({} fall back to resident weights).",
        from_source.len(),
        fallback.len()
    );
    if !fallback.is_empty() {
        tracing::warn!(
            "{} layers cannot be reloaded from the source; requantizing from resident weights.",
            fallback.len()
        );
    }

    let guard = mistralrs_quant::QuantizeOntoGuard::new();
    for module in from_source {
        let resident = module.ct.resolve()?;
        let (_, device) = resident.dtype_and_device();
        let Some(source_layer) = source.load_linear(
            &module.key,
            &Device::Cpu,
            module.shard.expect("source reload requires a shard"),
        )?
        else {
            fallback.push(module.clone());
            continue;
        };
        let (ty, imatrix) = module_imatrix(module, pool_ty, imatrix_map);
        let replacement = source_layer.apply_isq(
            Some(ty),
            device,
            &AtomicUsize::new(0),
            imatrix,
            guard.clone().with_module_key(module.key.clone()),
        )?;
        module
            .ct
            .replace(resident.preserve_dynamic_lora(replacement));
    }

    if !fallback.is_empty() {
        requantize_and_swap(&fallback, pool_ty, |m| m.resolve_type(pool_ty), &|key| {
            imatrix_map.get(key).cloned()
        })?;
    }
    Ok(())
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
        } else if source.has_expert_stack(&module.key)
            && module.shard.is_some()
            && !needs_distributed_wrapper(module)
        {
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
    if !fallback.is_empty() {
        tracing::warn!(
            "{} layers cannot be reloaded from the source; requantizing from resident weights.",
            fallback.len()
        );
    }

    let (executor, _) = mistralrs_quant::create_isq_executor(
        mistralrs_quant::IsqExecutorConfig::new(Some(pool_ty)),
    );
    let guard = mistralrs_quant::QuantizeOntoGuard::new();
    let n_jobs = dense.len();
    let mut dense_receivers = Vec::with_capacity(n_jobs);
    for module in dense {
        let mmap = source.mmap.clone();
        let guard = guard.clone().with_module_key(module.key.clone());
        let (ty, imatrix) = module_imatrix(&module, pool_ty, imatrix_map);
        let source_shape = source
            .shapes
            .get(&format!("{}.weight", module.key))
            .expect("dense source probe requires a weight")
            .clone();
        let source_has_bias = source.shapes.contains_key(&format!("{}.bias", module.key));
        let resident = module.ct.resolve()?;
        let (_, device) = resident.dtype_and_device();
        let resident_has_bias = resident.has_bias();
        let request = mistralrs_quant::IsqRequest {
            ty: Some(ty),
            device: device.clone(),
            has_imatrix: imatrix.is_some(),
            capture: mistralrs_quant::IsqCaptureMode::Immediate,
            consumer: mistralrs_quant::IsqConsumer::RuntimeSwap,
            module_key: module.key.clone(),
        };
        let plan = resident.plan_isq(&request)?;
        let key = module.key.clone();
        let rx = executor.submit(plan, request.consumer, move || {
            let job = || -> Result<()> {
                let shard = module.shard.expect("partition requires a shard");
                let w = mmap.load(&format!("{}.weight", module.key), &Device::Cpu, None)?;
                // force_contiguous: offset views share storage, and QTensor::quantize reads raw storage
                let w = shard.apply_to(&w)?.force_contiguous()?;
                let range = mistralrs_quant::shard_range(shard, &source_shape)?;
                let b = if source_has_bias && (resident_has_bias || source_shape.len() == 3) {
                    let b = mmap.load(&format!("{}.bias", module.key), &Device::Cpu, None)?;
                    match mistralrs_quant::bias_shard(range, source_shape.len()) {
                        mistralrs_quant::BiasShard::Skip if source_shape.len() == 3 => {
                            anyhow::bail!(
                                "Biased stacked expert `{}` cannot be input-sharded; load replicated experts.",
                                module.key
                            );
                        }
                        mistralrs_quant::BiasShard::Skip => None,
                        mistralrs_quant::BiasShard::Full => Some(b),
                        mistralrs_quant::BiasShard::Narrow { dim, start, len } => {
                            Some(b.narrow(dim, start, len)?.force_contiguous()?)
                        }
                    }
                } else {
                    None
                };
                let replacement = if w.rank() == 3 {
                    mistralrs_quant::quantize_expert_stack_with_bias(
                        w,
                        b,
                        ty,
                        imatrix,
                        &device,
                        &AtomicUsize::new(0),
                        guard,
                    )?
                } else {
                    quantize_source_tensor(w, b, ty, device, imatrix, guard)?
                };
                module
                    .ct
                    .replace(resident.preserve_dynamic_lora(replacement));
                Ok(())
            };
            job().map_err(|e| candle_core::Error::msg(format!("{e:#}")))
        });
        dense_receivers.push((key, rx));
    }

    let mut errors: Vec<String> = Vec::new();
    for module in &experts {
        let job = || -> Result<()> {
            let projection =
                crate::moe::rebuild_expert_projection(&source.mmap, &source.shapes, &module.key)?
                    .context("Expert stack probe succeeded but rebuild failed.")?;
            let resident = module.ct.resolve()?;
            let (_, device) = resident.dtype_and_device();
            let (ty, imatrix) = module_imatrix(module, pool_ty, imatrix_map);
            let shard = module.shard.expect("expert source reload requires a shard");
            let range = mistralrs_quant::shard_range(shard, projection.weight.dims())?;
            let weight_rank = projection.weight.rank();
            let weight = shard.apply_to(&projection.weight)?.force_contiguous()?;
            let bias = match (
                projection.bias,
                mistralrs_quant::bias_shard(range, weight_rank),
            ) {
                (Some(_), mistralrs_quant::BiasShard::Skip) => {
                    anyhow::bail!(
                        "Biased stacked expert `{}` cannot be input-sharded; load replicated experts.",
                        module.key
                    );
                }
                (None, mistralrs_quant::BiasShard::Skip) => None,
                (bias, mistralrs_quant::BiasShard::Full) => bias,
                (Some(bias), mistralrs_quant::BiasShard::Narrow { dim, start, len }) => {
                    Some(bias.narrow(dim, start, len)?.contiguous()?)
                }
                (None, mistralrs_quant::BiasShard::Narrow { .. }) => None,
            };
            let replacement = mistralrs_quant::quantize_expert_stack_with_bias(
                weight,
                bias,
                ty,
                imatrix,
                &device,
                &AtomicUsize::new(0),
                guard.clone().with_module_key(module.key.clone()),
            )?;
            module
                .ct
                .replace(resident.preserve_dynamic_lora(replacement));
            Ok(())
        };
        if let Err(e) = job() {
            errors.push(format!("{}: {e:#}", module.key));
        }
    }

    // drain everything; failed layers keep their prior resident, so a partial apply stays consistent
    let mut received = 0usize;
    for (key, rx) in dense_receivers {
        received += 1;
        match rx.recv() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => errors.push(format!("{key}: {e:#}")),
            Err(e) => errors.push(format!("{key}: channel error: {e}")),
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
        requantize_and_swap(&fallback, pool_ty, |m| m.resolve_type(pool_ty), &|key| {
            imatrix_map.get(key).cloned()
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::sync::Arc;

    const E: usize = 4;
    const INTER: usize = 8;

    struct TestWeightSource {
        weight: Tensor,
    }

    impl mistralrs_quant::QuantizedWeightSource for TestWeightSource {
        fn contains(&self, name: &str) -> bool {
            name == "m.lin.weight"
        }

        fn load_linear(
            &self,
            key: &str,
            device: &Device,
            shard: mistralrs_quant::Shard,
        ) -> candle_core::Result<Option<Arc<dyn QuantMethod>>> {
            if key != "m.lin" {
                return Ok(None);
            }
            let weight = shard.apply_to(&self.weight)?.to_device(device)?;
            Ok(Some(Arc::new(mistralrs_quant::UnquantLinear::new(
                mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                    weight, None,
                )),
            )?)))
        }

        fn load_optional_tensor(
            &self,
            _name: &str,
            _device: &Device,
        ) -> candle_core::Result<Option<Tensor>> {
            Ok(None)
        }

        fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    fn write_st(path: &std::path::Path, tensors: Vec<(String, Tensor)>) {
        candle_core::safetensors::save(&tensors.into_iter().collect(), path).unwrap();
    }

    #[test]
    fn calibration_prefers_quantized_source_over_checkpoint_paths() -> Result<()> {
        use mistralrs_quant::{QuantMethod, Shard, TrackedModule};

        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: Arc::new(zeros),
                b: None,
            },
        )?) as Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))?;
        let ct = Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.lin".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
            shard: Some(Shard::default()),
        };
        let modules = [module];
        begin_calibration(&modules)?;
        ct.forward_raw(&Tensor::ones((1, 32), DType::F32, &Device::Cpu)?)?;

        let source: Arc<dyn mistralrs_quant::QuantizedWeightSource> = Arc::new(TestWeightSource {
            weight: Tensor::ones((2, 32), DType::F32, &Device::Cpu)?,
        });
        apply_calibration(
            &modules,
            &[std::path::PathBuf::from("not-a-safetensors-file.gguf")],
            Some(source.as_ref()),
            None,
        )?;

        let swapped = ct.resolve()?.dequantize_w()?;
        let diff = (swapped - 1f64)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 0.01, "max diff {diff}");
        Ok(())
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
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))
            .unwrap();
        let ct = std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.lin".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
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
    fn from_source_preserves_dynamic_lora() -> Result<()> {
        use mistralrs_quant::{
            maybe_wrap_dynamic_lora, with_lora_execution, LoraExecution, LoraLayerRegistry,
            LoraLinearSpec, LoraWeights, QuantMethod, Shard, ShardedSafeTensors, TrackedModule,
            UnquantLinear,
        };

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let source_weight = Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?;
        candle_core::safetensors::save(
            &[("m.lin.weight".to_string(), source_weight)]
                .into_iter()
                .collect(),
            &file,
        )?;

        let registry = std::sync::Arc::new(LoraLayerRegistry::new());
        let vb = ShardedSafeTensors::wrap_with_dummy_regexes(
            HashMap::<String, Tensor>::new(),
            DType::F32,
            Device::Cpu,
            None,
        )
        .with_lora_registry(registry.clone())
        .pp("m.lin");
        let base = std::sync::Arc::new(UnquantLinear::new(
            mistralrs_quant::QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                Tensor::zeros((2, 32), DType::F32, &Device::Cpu)?,
                None,
            )),
        )?) as std::sync::Arc<dyn QuantMethod>;
        let resident = maybe_wrap_dynamic_lora(&vb, base, LoraLinearSpec::replicated(32, 2))?;
        let site = registry.sites().pop().expect("registered LoRA site");
        registry.finalize()?;

        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))
            .unwrap();
        let ct = std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.lin".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
            shard: Some(Shard::default()),
        };

        requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())?;

        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::ones((1, 32), DType::F32, &Device::Cpu)?,
                Tensor::ones((2, 1), DType::F32, &Device::Cpu)?,
                1.0,
            )?,
        )?;
        let input = Tensor::ones((1, 32), DType::F32, &Device::Cpu)?;
        let output =
            with_lora_execution(Some(std::sync::Arc::new(execution)), || ct.forward(&input))?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![32.0, 32.0]]);
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
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))
            .unwrap();
        let ct = std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.experts.gate_proj".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
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

    #[test]
    fn from_source_preserves_expert_bias() -> Result<()> {
        use mistralrs_quant::{QuantMethod, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let mut tensors = Vec::new();
        let mut biases = Vec::new();
        for expert in 0..E {
            tensors.push((
                format!("m.experts.{expert}.w1.weight"),
                Tensor::zeros((INTER, 32), DType::F32, &Device::Cpu)?,
            ));
            let bias = Tensor::from_vec(
                (0..INTER)
                    .map(|output| {
                        f32::from(u16::try_from(expert * INTER + output).expect("test value"))
                    })
                    .collect::<Vec<_>>(),
                INTER,
                &Device::Cpu,
            )?;
            tensors.push((format!("m.experts.{expert}.w1.bias"), bias.clone()));
            biases.push(bias);
        }
        write_st(&file, tensors);

        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((E, INTER, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: Arc::new(zeros),
                b: Some(Tensor::zeros((E, INTER), DType::F32, &Device::Cpu)?),
            },
        )?) as Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))?;
        let ct = Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.experts.gate_proj".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
            shard: Some(mistralrs_quant::Shard::default()),
        };

        requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())?;

        let indices = Tensor::from_vec(vec![0u32, 2, 1, 3], (2, 2), &Device::Cpu)?;
        let input = Tensor::zeros((2, 1, 32), DType::F32, &Device::Cpu)?;
        let actual = ct.gather_forward(&input, &indices)?;
        let expected = Tensor::stack(&biases, 0)?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((2, 2, INTER))?;
        let diff = (actual - expected)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "max diff {diff}");
        Ok(())
    }

    #[test]
    fn from_source_respects_stacked_expert_output_shard_and_bias() -> Result<()> {
        use mistralrs_quant::{QuantMethod, Shard, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let truth = Tensor::randn(0f32, 1f32, (E, INTER, 32), &Device::Cpu)?;
        let bias = Tensor::from_vec(
            (0..E * INTER)
                .map(|value| f32::from(u16::try_from(value).expect("test value")))
                .collect::<Vec<_>>(),
            (E, INTER),
            &Device::Cpu,
        )?;
        write_st(
            &file,
            vec![
                ("m.experts.gate_proj.weight".to_string(), truth.clone()),
                ("m.experts.gate_proj.bias".to_string(), bias.clone()),
            ],
        );

        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((E, INTER / 2, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: Arc::new(zeros),
                b: Some(Tensor::zeros((E, INTER / 2), DType::F32, &Device::Cpu)?),
            },
        )?) as Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))?;
        let ct = Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.experts.gate_proj".to_string(),
            ct: ct.clone(),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
            shard: Some(Shard::Simple {
                dim: 1,
                rank: 1,
                world_size: 2,
            }),
        };

        requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())?;

        let swapped = ct.resolve()?;
        assert_eq!(swapped.dequantize_w()?.dims(), [E, INTER / 2, 32]);
        let indices = Tensor::from_vec(vec![0u32, 2, 1, 3], (2, 2), &Device::Cpu)?;
        let input = Tensor::zeros((2, 1, 32), DType::F32, &Device::Cpu)?;
        let actual = swapped.gather_forward(&input, &indices)?;
        let expected_bias = bias
            .narrow(1, INTER / 2, INTER / 2)?
            .contiguous()?
            .index_select(&indices.flatten_all()?, 0)?
            .reshape((2, 2, INTER / 2))?;
        let expected_weight = truth.narrow(1, INTER / 2, INTER / 2)?;
        let weight_diff = (swapped.dequantize_w()? - expected_weight)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let bias_diff = (actual - expected_bias)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert!(weight_diff < 0.05, "weight max diff {weight_diff}");
        assert!(bias_diff < 1e-6, "bias max diff {bias_diff}");
        Ok(())
    }

    #[test]
    fn from_source_rejects_biased_expert_input_shard() -> Result<()> {
        use mistralrs_quant::{QuantMethod, Shard, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        write_st(
            &file,
            vec![
                (
                    "m.experts.down_proj.weight".to_string(),
                    Tensor::zeros((E, INTER, 64), DType::F32, &Device::Cpu)?,
                ),
                (
                    "m.experts.down_proj.bias".to_string(),
                    Tensor::zeros((E, INTER), DType::F32, &Device::Cpu)?,
                ),
            ],
        );
        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((E, INTER, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: Arc::new(zeros),
                b: None,
            },
        )?) as Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))?;
        let module = TrackedModule {
            key: "m.experts.down_proj".to_string(),
            ct: Arc::new(mistralrs_quant::PendingIsqLayer::new(rx)),
            ty: Some(IsqType::Q8_0),
            promote_default: false,
            shard: Some(Shard::Simple {
                dim: 2,
                rank: 0,
                world_size: 2,
            }),
        };

        let error = requantize_from_source(&[module], &[file], IsqType::Q8_0, &HashMap::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot be input-sharded"), "{error}");
        Ok(())
    }

    #[test]
    fn from_source_rejects_stacked_expert_target_without_gather() -> Result<()> {
        use mistralrs_quant::{QuantMethod, TrackedModule};

        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        write_st(
            &file,
            vec![(
                "m.experts.gate_proj.weight".to_string(),
                Tensor::zeros((E, INTER, 32), DType::F32, &Device::Cpu)?,
            )],
        );
        let zeros = candle_core::quantized::QTensor::quantize(
            &Tensor::zeros((E, INTER, 32), DType::F32, &Device::Cpu)?,
            candle_core::quantized::GgmlDType::Q8_0,
        )?;
        let resident = Arc::new(mistralrs_quant::GgufMatMul::new(
            mistralrs_quant::QuantMethodConfig::Gguf {
                q_weight: Arc::new(zeros),
                b: None,
            },
        )?) as Arc<dyn QuantMethod>;
        let (tx, rx) = mistralrs_quant::pending_isq_channel();
        tx.send(Ok(mistralrs_quant::IsqJobOutput::ready(resident)))?;
        let ct = Arc::new(mistralrs_quant::PendingIsqLayer::new(rx));
        let module = TrackedModule {
            key: "m.experts.gate_proj".to_string(),
            ct,
            ty: Some(IsqType::HQQ4),
            promote_default: false,
            shard: Some(mistralrs_quant::Shard::default()),
        };

        let error = requantize_from_source(&[module], &[file], IsqType::HQQ4, &HashMap::new())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("does not support stacked expert gather"),
            "{error}"
        );
        Ok(())
    }
}
