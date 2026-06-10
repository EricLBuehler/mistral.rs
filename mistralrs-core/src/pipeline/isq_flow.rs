//! Shared post-load ISQ orchestration: flag validation and capture-mode planning, the
//! calibration text drive, and requantize-and-swap used by both load completion and runtime
//! re-ISQ. Pipeline loaders (normal, multimodal, embedding) call into this instead of
//! duplicating the flow.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;
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
