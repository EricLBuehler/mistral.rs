//! Shared post-load ISQ orchestration: flag validation and capture-mode planning, the
//! calibration text drive, and requantize-and-swap used by both load completion and runtime
//! re-ISQ. Pipeline loaders (normal, multimodal, embedding) call into this instead of
//! duplicating the flow.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use mistralrs_quant::{IsqType, QuantMethod, TrackedModule};
use tokenizers::Tokenizer;
use tracing::info;

use super::isq::{format_isq_types, load_imatrix_map, IsqModelLoader, IsqOrganization};
use super::text_models_inputs_processor::{make_prompt_chunk, InputMetadata};
use super::{EitherCache, EmbeddingModel, ModelForwardContext, MultimodalModel, NormalModel};

pub(crate) struct IsqPlanInputs<'a> {
    pub in_situ_quant: Option<IsqType>,
    pub has_imatrix: bool,
    pub has_calibration: bool,
    pub write_uqff_types: Option<Vec<IsqType>>,
    pub has_write_uqff: bool,
    pub loading_from_uqff: bool,
    pub organization: IsqOrganization,
    pub topology_overrides: Vec<mistralrs_quant::ImmediateIsqOverride>,
    pub loader: &'a dyn IsqModelLoader,
    pub config: &'a str,
    pub device: &'a Device,
}

pub(crate) struct IsqLoadPlan {
    pub wants_imatrix: bool,
    pub immediate_isq_installed: bool,
    pub capture: mistralrs_quant::IsqCaptureMode,
    pub write_types: Option<Vec<IsqType>>,
    pub loading_isq: bool,
    pub load_device: Device,
}

/// Validate the ISQ/imatrix/UQFF flag combination, install the immediate-ISQ thread pool and
/// capture mode, and resolve the load device. Shared by all pipeline loaders.
pub(crate) fn resolve_and_install_isq_plan(i: IsqPlanInputs<'_>) -> Result<IsqLoadPlan> {
    let wants_imatrix = i.has_imatrix || i.has_calibration;
    if i.has_imatrix && i.has_calibration {
        anyhow::bail!("`imatrix` and `calibration_file` were both specified, this is not allowed.");
    }
    // UQFF writes carry their ISQ types in `write_uqff.types` rather than `in_situ_quant`.
    if wants_imatrix && i.in_situ_quant.is_none() && !i.has_write_uqff {
        anyhow::bail!("imatrix quantization requires an ISQ type (e.g. `--isq q4k`).");
    }
    if i.has_write_uqff
        && i.write_uqff_types.as_ref().is_some_and(|t| t.is_empty())
        && i.in_situ_quant.is_none()
    {
        anyhow::bail!("UQFF serialization requires at least one ISQ type.");
    }
    if i.has_write_uqff && i.loading_from_uqff {
        anyhow::bail!(
            "Writing UQFF (`write_uqff`) while loading from UQFF (`from_uqff`) is not supported."
        );
    }

    let allow_immediate_cli = i.in_situ_quant.is_some() || i.has_write_uqff;
    let write_types = if i.has_write_uqff {
        i.write_uqff_types.map(|types| {
            if types.is_empty() {
                i.in_situ_quant.into_iter().collect()
            } else {
                types
            }
        })
    } else {
        None
    };

    let mut immediate_ty = None;
    if allow_immediate_cli {
        immediate_ty = if i.has_write_uqff {
            None
        } else {
            i.in_situ_quant
        };
        let immediate_predicates = if matches!(i.organization, IsqOrganization::MoeExpertsOnly) {
            i.loader.immediate_isq_predicates_moqe(i.config)?
        } else {
            i.loader.immediate_isq_predicates(i.config)?
        };
        if let Some(types) = &write_types {
            info!("Preparing UQFF output for [{}].", format_isq_types(types));
        } else if let Some(ty) = i.in_situ_quant {
            info!("Quantizing model weights to {ty}.");
        }
        if immediate_predicates.is_empty() {
            tracing::warn!("No predicates for this model and ISQ setting detected. ISQ will not be applied to any weights!");
        }

        let capture = capture_mode(i.has_write_uqff, wants_imatrix);
        let (pool, num_threads) = mistralrs_quant::create_isq_thread_pool(immediate_ty);
        tracing::debug!("Using {num_threads} worker thread(s) for weight quantization.");
        mistralrs_quant::set_immediate_isq_with_pool(
            immediate_ty,
            immediate_predicates,
            i.topology_overrides.clone(),
            capture,
            pool,
        );
    } else if !i.topology_overrides.is_empty() {
        let (pool, num_threads) = mistralrs_quant::create_isq_thread_pool(immediate_ty);
        tracing::debug!("Using {num_threads} worker thread(s) for weight quantization.");
        mistralrs_quant::set_immediate_isq_with_pool(
            immediate_ty,
            Vec::new(),
            i.topology_overrides.clone(),
            capture_mode(i.has_write_uqff, wants_imatrix),
            pool,
        );
    }

    let use_immediate = allow_immediate_cli || !i.topology_overrides.is_empty();
    let mut loading_isq = if use_immediate {
        false
    } else {
        i.in_situ_quant.is_some()
    };

    // Load onto the regular device if not using isq.
    // For immediate ISQ on discrete GPUs, load to CPU: the mapper will set the correct target
    // device per-layer, and linear constructors will override to CPU for ISQ-targeted weights.
    // On integrated/unified memory systems (e.g. Grace Blackwell), CPU and GPU share memory,
    // so we load directly to the device.
    let load_device = if !loading_isq {
        loading_isq = false;
        if use_immediate && !crate::utils::normal::is_integrated_gpu(i.device) {
            Device::Cpu
        } else {
            i.device.clone()
        }
    } else {
        Device::Cpu
    };

    Ok(IsqLoadPlan {
        wants_imatrix,
        immediate_isq_installed: use_immediate,
        capture: capture_mode(i.has_write_uqff, wants_imatrix),
        write_types,
        loading_isq,
        load_device,
    })
}

fn capture_mode(has_write_uqff: bool, wants_imatrix: bool) -> mistralrs_quant::IsqCaptureMode {
    if has_write_uqff {
        mistralrs_quant::IsqCaptureMode::CaptureAll
    } else if wants_imatrix {
        mistralrs_quant::IsqCaptureMode::CaptureMatches
    } else {
        mistralrs_quant::IsqCaptureMode::Immediate
    }
}

/// Minimal model surface for driving calibration text through a pipeline's model.
pub(crate) trait CalibrationDrive {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()>;
    fn reset_cache(&self) {}
    fn sliding_window(&self) -> Option<usize> {
        None
    }
}

pub(crate) struct NormalCalibrationDrive<'a>(pub &'a dyn NormalModel);

impl CalibrationDrive for NormalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        self.0.forward(&input, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) {
        reset_either_cache(self.0.cache());
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct MultimodalCalibrationDrive<'a>(pub &'a dyn MultimodalModel);

impl CalibrationDrive for MultimodalCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        let mut ctx = ModelForwardContext::new(
            &inputs.positions,
            &inputs.context_lens,
            &inputs.position_ids,
            None,
            &inputs.flash_meta,
        );
        // Text-only drive: the vision tower sees no calibration data, so its
        // layers quantize without imatrix weights.
        let args = self.0.default_model_specific_args(&input);
        self.0.forward(&input, None, args, &mut ctx)?;
        Ok(())
    }

    fn reset_cache(&self) {
        reset_either_cache(self.0.cache());
    }

    fn sliding_window(&self) -> Option<usize> {
        self.0.config().sliding_window
    }
}

pub(crate) struct EmbeddingCalibrationDrive<'a>(pub &'a dyn EmbeddingModel);

impl CalibrationDrive for EmbeddingCalibrationDrive<'_> {
    fn calibration_forward(&self, inputs: &InputMetadata) -> candle_core::Result<()> {
        let input = inputs.input.to_device(self.0.device())?;
        self.0.forward(&input, &inputs.flash_meta)?;
        Ok(())
    }
}

fn reset_either_cache(cache: &EitherCache) {
    match cache {
        EitherCache::Full(full) => {
            for layer in &mut *full.lock() {
                *layer = None
            }
        }
        EitherCache::Normal(normal) => {
            for layer in &mut *normal.lock().unwrap().0 {
                layer.reset();
            }
        }
        EitherCache::Hybrid(hybrid) => {
            hybrid.lock().unwrap().reset();
        }
    }
}

pub(crate) struct CalibrationCtx<'a> {
    pub tokenizer: &'a Tokenizer,
    pub bos_tok_id: Option<u32>,
    pub load_device: &'a Device,
    pub mapper: Option<&'a dyn crate::device_map::DeviceMapper>,
}

const CALIBRATION_CHUNK_SIZE: usize = 1024;

/// Produce the per-layer imatrix map: collect via calibration forwards, or load from a file.
pub(crate) fn resolve_imatrix_map(
    drive: &dyn CalibrationDrive,
    modules: &[TrackedModule],
    imatrix_path: Option<&PathBuf>,
    calibration_file: Option<&PathBuf>,
    ctx: &CalibrationCtx<'_>,
) -> Result<HashMap<String, Vec<f32>>> {
    if let Some(calibration_file) = calibration_file {
        let calibration_data = std::fs::read_to_string(calibration_file)?;
        // Tokenize without bos; it is inserted per chunk below
        let tokens = ctx
            .tokenizer
            .encode_fast(calibration_data, false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        info!(
            "Collecting imatrix from calibration file `{}` of {} tokens.",
            calibration_file.display(),
            tokens.len()
        );

        for module in modules {
            module.ct.begin_track_stats()?;
        }

        let n_chunks = tokens.len().div_ceil(CALIBRATION_CHUNK_SIZE);
        let collect_start = std::time::Instant::now();
        for (i, chunk) in tokens.chunks(CALIBRATION_CHUNK_SIZE).enumerate() {
            let mut chunk = chunk.to_vec();
            if let Some(bos_tok_id) = ctx.bos_tok_id {
                chunk.insert(0, bos_tok_id);
            }
            let chunk_len = chunk.len();

            let chunk_start = std::time::Instant::now();
            let inputs = make_prompt_chunk(
                0,
                vec![&chunk],
                &[0],
                ctx.load_device,
                None,
                false,
                None,
                ctx.mapper,
                None,
                drive.sliding_window(),
            )?;
            drive.calibration_forward(&inputs)?;
            drive.reset_cache();

            info!(
                "Processed chunk {}/{n_chunks} ({chunk_len} tokens), {:.2}s",
                i + 1,
                chunk_start.elapsed().as_secs_f32()
            );
        }
        ctx.load_device.synchronize()?;
        info!(
            "Finished collecting imatrix in {:.2}s",
            collect_start.elapsed().as_secs_f32()
        );

        let mut map = HashMap::new();
        for module in modules {
            if let Ok(stats) = module.ct.end_track_stats() {
                map.insert(module.key.clone(), stats.to_vec1::<f32>()?);
            }
        }
        Ok(map)
    } else if let Some(imatrix_path) = imatrix_path {
        load_imatrix_map(imatrix_path, modules)
    } else {
        unreachable!("wants_imatrix requires imatrix or calibration_file")
    }
}

/// Requantize tracked modules on a fresh pool and swap each result into the live model.
pub(crate) fn requantize_and_swap(
    modules: &[TrackedModule],
    pool_ty: IsqType,
    ty_for: impl Fn(&TrackedModule) -> IsqType,
    imatrix_for: &dyn Fn(&str) -> Option<Vec<f32>>,
) -> Result<()> {
    let handles = mistralrs_quant::requantize_tracked(
        modules,
        pool_ty,
        mistralrs_quant::RequantizeResults::Resident,
        ty_for,
        imatrix_for,
    )?;
    for (module, rx) in modules.iter().zip(handles.receivers) {
        let layer = rx
            .recv()
            .map_err(|e| anyhow::anyhow!("Requantize channel error: {e}"))??;
        module.ct.replace(layer);
    }
    Ok(())
}

/// Finish a CaptureMatches load: quantize the deferred layers with imatrix data and swap them in.
pub(crate) fn complete_isq_capture(
    modules: &[TrackedModule],
    ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> Result<()> {
    let missing = modules
        .iter()
        .filter(|module| !imatrix_map.contains_key(&module.key))
        .count();
    if missing > 0 {
        tracing::warn!(
            "{missing} of {} layers have no imatrix data; quantizing those without weights.",
            modules.len()
        );
    }
    info!("Quantizing {} layers to {ty} with imatrix.", modules.len());
    requantize_and_swap(modules, ty, |m| m.ty.unwrap_or(ty), &|key| {
        imatrix_map.get(key).cloned()
    })
}

#[derive(Clone, Debug, serde::Serialize)]
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
    for module in modules {
        module.ct.begin_track_stats()?;
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

/// Harvest collected statistics and requantize every tracked layer with the resulting imatrix,
/// swapping each into the live model. Layers without data quantize plainly with a warning.
/// Weights currently come from dequantizing the resident layer; from-source requantization
/// replaces this for full quality.
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

    let mut map = HashMap::new();
    for module in modules {
        if let Ok(stats) = module.ct.end_track_stats() {
            map.insert(module.key.clone(), stats.to_vec1::<f32>()?);
        }
    }
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
        let shapes = mmap
            .tensors()
            .into_iter()
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

/// Quantize a freshly read source tensor to `ty` on `device` and return the layer to swap in.
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

fn module_imatrix(
    module: &TrackedModule,
    pool_ty: IsqType,
    imatrix_map: &HashMap<String, Vec<f32>>,
) -> (IsqType, Option<Vec<f32>>) {
    let ty = module.ty.unwrap_or(pool_ty);
    let imatrix = ty
        .supports_imatrix()
        .then(|| imatrix_map.get(&module.key).cloned())
        .flatten();
    (ty, imatrix)
}

/// Requantize tracked modules from the original source weights, overlapping mmap reads,
/// imatrix-weighted quantization, and per-layer hot-swaps on the ISQ pool. Expert stacks
/// rebuild serially on this thread (one canonical [E, out, in] tensor in memory at a time);
/// layers absent from the source fall back to dequant-requant.
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
        if source.has_dense(&module.key) && module.shard.is_some() {
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
        pool.spawn(move || {
            let job = || -> Result<()> {
                let resident = module.ct.resolve()?;
                let (_, device) = resident.dtype_and_device();
                let shard = module.shard.expect("partition requires a shard");
                let w = mmap.load(&format!("{}.weight", module.key), &Device::Cpu, None)?;
                // force_contiguous: offset views share storage, and QTensor::quantize reads raw storage
                let w = shard.apply_to(&w)?.force_contiguous()?;
                let b = if resident.has_bias() {
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

    for module in &experts {
        let stack = crate::moe::rebuild_expert_stack(&source.mmap, &source.shapes, &module.key)?
            .context("Expert stack probe succeeded but rebuild failed.")?;
        let resident = module.ct.resolve()?;
        let (_, device) = resident.dtype_and_device();
        let (ty, imatrix) = module_imatrix(module, pool_ty, imatrix_map);
        module.ct.replace(quantize_source_tensor(
            stack,
            None,
            ty,
            device,
            imatrix,
            guard.clone(),
        )?);
    }

    let mut received = 0usize;
    for result in rx {
        result?;
        received += 1;
    }
    anyhow::ensure!(
        received == n_jobs,
        "From-source requantize jobs died early."
    );

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
