//! Load-time ISQ planning: flag validation, capture-mode selection, pool install.

use anyhow::Result;
use candle_core::{DType, Device};
use mistralrs_quant::IsqType;
use tracing::info;

use crate::{device_map::DeviceMapper, TryIntoDType};

use super::super::isq::{format_isq_types, IsqModelLoader, IsqOrganization};

fn resolve_isq_predicates(
    loader: &dyn IsqModelLoader,
    config: &str,
    organization: IsqOrganization,
) -> Result<(Vec<regex::Regex>, Vec<regex::Regex>)> {
    let promoted = loader.promoted_isq_predicates(config)?;
    let mut selected = if matches!(organization, IsqOrganization::MoeExpertsOnly) {
        loader.immediate_isq_predicates_moqe(config)?
    } else {
        loader.immediate_isq_predicates(config)?
    };
    if !matches!(organization, IsqOrganization::MoeExpertsOnly) {
        selected.extend(promoted.iter().cloned());
    }
    Ok((selected, promoted))
}

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

pub(crate) fn resolve_weight_load_dtype(
    dtype: &dyn TryIntoDType,
    mapper: &dyn DeviceMapper,
    available_devices: &[Device],
    write_uqff: bool,
) -> Result<DType> {
    if write_uqff {
        dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())
    } else {
        Ok(mapper.get_min_dtype(dtype)?)
    }
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
    let (immediate_predicates, promoted_predicates) =
        resolve_isq_predicates(i.loader, i.config, i.organization)?;

    let mut immediate_ty = None;
    if allow_immediate_cli {
        immediate_ty = if i.has_write_uqff {
            None
        } else {
            i.in_situ_quant
        };
        if let Some(types) = &write_types {
            info!("Preparing UQFF output for [{}].", format_isq_types(types));
        } else if let Some(ty) = i.in_situ_quant {
            let sensitive_ty = ty.promote_for_sensitive_tensor();
            if sensitive_ty == ty || promoted_predicates.is_empty() {
                info!("Quantizing model weights to {ty}.");
            } else {
                info!("Quantizing model weights to {ty}, with sensitive tensors using {sensitive_ty}.");
            }
        }
        if immediate_predicates.is_empty() {
            tracing::warn!("No predicates for this model and ISQ setting detected. ISQ will not be applied to any weights!");
        }

        let capture = capture_mode(i.has_write_uqff, wants_imatrix);
        let (executor, num_threads) = mistralrs_quant::create_isq_executor(
            mistralrs_quant::IsqExecutorConfig::new(immediate_ty),
        );
        tracing::debug!("Using {num_threads} worker thread(s) for weight quantization.");
        mistralrs_quant::set_immediate_isq_config(
            mistralrs_quant::ImmediateIsqConfig::new(immediate_ty, immediate_predicates, capture)
                .with_promoted_predicates(promoted_predicates.clone())
                .with_overrides(i.topology_overrides.clone()),
            executor,
        );
    } else if !i.topology_overrides.is_empty() {
        let (executor, num_threads) = mistralrs_quant::create_isq_executor(
            mistralrs_quant::IsqExecutorConfig::new(immediate_ty),
        );
        tracing::debug!("Using {num_threads} worker thread(s) for weight quantization.");
        mistralrs_quant::set_immediate_isq_config(
            mistralrs_quant::ImmediateIsqConfig::new(
                immediate_ty,
                Vec::new(),
                capture_mode(i.has_write_uqff, wants_imatrix),
            )
            .with_promoted_predicates(promoted_predicates)
            .with_overrides(i.topology_overrides.clone()),
            executor,
        );
    }

    let use_immediate = allow_immediate_cli || !i.topology_overrides.is_empty();
    let loading_isq = if use_immediate {
        false
    } else {
        i.in_situ_quant.is_some()
    };

    // Load onto the regular device if not using isq.
    // For immediate ISQ on discrete GPUs, load to CPU: the mapper will set the correct target
    // device per-layer, and linear constructors will override to CPU for ISQ-targeted weights.
    // On integrated/unified memory systems (e.g. Grace Blackwell), CPU and GPU share memory,
    // so we load directly to the device.
    let load_device = if i.has_write_uqff {
        Device::Cpu
    } else if !loading_isq {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Loader;

    impl IsqModelLoader for Loader {
        fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<regex::Regex>> {
            Ok(vec![
                regex::Regex::new(r"^model\.embed_tokens\.weight$")?,
                regex::Regex::new(r"^lm_head\.weight$")?,
            ])
        }

        fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<regex::Regex>> {
            Ok(vec![regex::Regex::new(r"^model\.layers\.")?])
        }

        fn immediate_isq_predicates_moqe(&self, _config: &str) -> Result<Vec<regex::Regex>> {
            Ok(vec![regex::Regex::new(r"^lm_head\.weight$")?])
        }
    }

    fn matches(predicates: &[regex::Regex], name: &str) -> bool {
        predicates.iter().any(|predicate| predicate.is_match(name))
    }

    #[test]
    fn default_selection_includes_model_declared_promoted_tensors() -> Result<()> {
        let (selected, promoted) = resolve_isq_predicates(&Loader, "", IsqOrganization::Default)?;
        assert!(matches(&selected, "model.layers.0.self_attn.q_proj.weight"));
        assert!(matches(&selected, "model.embed_tokens.weight"));
        assert!(matches(&promoted, "model.embed_tokens.weight"));
        Ok(())
    }

    #[test]
    fn moqe_selection_does_not_add_embeddings_but_keeps_classification() -> Result<()> {
        let (selected, promoted) =
            resolve_isq_predicates(&Loader, "", IsqOrganization::MoeExpertsOnly)?;
        assert!(matches(&selected, "lm_head.weight"));
        assert!(!matches(&selected, "model.embed_tokens.weight"));
        assert!(matches(&promoted, "model.embed_tokens.weight"));
        assert!(matches(&promoted, "lm_head.weight"));
        Ok(())
    }
}
