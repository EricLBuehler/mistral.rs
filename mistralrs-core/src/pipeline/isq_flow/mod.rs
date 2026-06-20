//! Shared post-load ISQ orchestration. [`plan`] resolves flags into a capture plan,
//! [`drive`] runs offline calibration text through a model, [`online`] is the live
//! calibration lifecycle; the requantize-and-swap primitives here serve all of them
//! plus runtime re-ISQ.

mod drive;
mod online;
mod plan;

pub(crate) use drive::{
    resolve_imatrix_map, CalibrationCtx, EmbeddingCalibrationDrive, MultimodalCalibrationDrive,
    NormalCalibrationDrive,
};
pub use online::CalibrationStatus;
pub(crate) use online::{apply_calibration, begin_calibration, calibration_status};
pub(crate) use plan::{resolve_and_install_isq_plan, IsqPlanInputs};

use std::collections::HashMap;

use anyhow::Result;
use mistralrs_quant::{IsqType, QuantMethod, TrackedModule};
use tracing::info;

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
        None,
    )?;
    // drain everything; failed layers keep their prior resident, so a partial swap stays consistent
    let mut errors: Vec<String> = Vec::new();
    for (module, rx) in modules.iter().zip(handles.receivers) {
        match rx.recv() {
            Ok(Ok(layer)) => module.ct.replace(layer),
            Ok(Err(e)) => errors.push(format!("{}: {e}", module.key)),
            Err(e) => errors.push(format!("{}: channel error: {e}", module.key)),
        }
    }
    if !errors.is_empty() {
        anyhow::bail!(
            "{} of {} layers failed to requantize; first: {}",
            errors.len(),
            modules.len(),
            errors[0]
        );
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

/// Drain collected statistics into a key -> imatrix map; layers without data are absent.
fn harvest_imatrix(modules: &[TrackedModule]) -> Result<HashMap<String, Vec<f32>>> {
    let mut map = HashMap::new();
    for module in modules {
        if let Ok(stats) = module.ct.end_track_stats() {
            map.insert(module.key.clone(), stats.flatten_all()?.to_vec1::<f32>()?);
        }
    }
    Ok(map)
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
