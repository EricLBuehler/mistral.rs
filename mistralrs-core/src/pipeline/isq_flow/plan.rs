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
    pub uqff_quantize_predicates: Option<Vec<regex::Regex>>,
    pub loading_isq: bool,
    pub load_device: Device,
    expects_quantized_selection: bool,
    _scope: ImmediateIsqScope,
}

impl IsqLoadPlan {
    pub fn validate_tracked_selection(
        &self,
        modules: &[mistralrs_quant::TrackedModule],
    ) -> Result<()> {
        if !self.expects_quantized_selection {
            return Ok(());
        }

        let selected = if let Some(predicates) = &self.uqff_quantize_predicates {
            modules.iter().any(|module| {
                let weight_key = format!("{}.weight", module.key);
                module.ty.is_some()
                    || predicates
                        .iter()
                        .any(|predicate| predicate.is_match(&weight_key))
            })
        } else {
            !modules.is_empty()
        };
        if !selected {
            anyhow::bail!(
                "ISQ was requested, but no model weights matched the selected quantization predicates or typed topology overrides. This model's ISQ bindings may be incomplete or out of date."
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoDeviceMapSizing {
    Uqff,
    Isq(IsqType),
    PreparedWeightSource,
    Checkpoint,
}

pub(crate) fn resolve_auto_device_map_sizing(
    loading_from_uqff: bool,
    has_prepared_weight_source: bool,
    in_situ_quant: Option<IsqType>,
) -> AutoDeviceMapSizing {
    if loading_from_uqff {
        AutoDeviceMapSizing::Uqff
    } else if let Some(isq) = in_situ_quant {
        AutoDeviceMapSizing::Isq(isq)
    } else if has_prepared_weight_source {
        AutoDeviceMapSizing::PreparedWeightSource
    } else {
        AutoDeviceMapSizing::Checkpoint
    }
}

struct ImmediateIsqScope;

impl Drop for ImmediateIsqScope {
    fn drop(&mut self) {
        mistralrs_quant::clear_immediate_isq();
    }
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

fn moqe_is_inert(
    organization: IsqOrganization,
    in_situ_quant: Option<IsqType>,
    has_write_uqff: bool,
    loading_from_uqff: bool,
) -> bool {
    matches!(organization, IsqOrganization::MoeExpertsOnly)
        && (loading_from_uqff || (in_situ_quant.is_none() && !has_write_uqff))
}

/// Validate the ISQ/imatrix/UQFF flag combination, install the immediate-ISQ thread pool and
/// capture mode, and resolve the load device. Shared by all pipeline loaders.
pub(crate) fn resolve_and_install_isq_plan(i: IsqPlanInputs<'_>) -> Result<IsqLoadPlan> {
    mistralrs_quant::clear_immediate_isq();

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
    if let Some(types) = &i.write_uqff_types {
        let mut seen = std::collections::HashSet::new();
        for ty in types {
            if !seen.insert(*ty) {
                anyhow::bail!("Duplicate UQFF output type `{ty}` was requested.");
            }
        }
    }
    if i.has_write_uqff && i.loading_from_uqff {
        anyhow::bail!(
            "Writing UQFF (`write_uqff`) while loading from UQFF (`from_uqff`) is not supported."
        );
    }
    if wants_imatrix && i.loading_from_uqff {
        anyhow::bail!(
            "Imatrix or calibration input cannot be combined with loading from UQFF; UQFF weights take precedence over applying ISQ."
        );
    }
    if moqe_is_inert(
        i.organization,
        i.in_situ_quant,
        i.has_write_uqff,
        i.loading_from_uqff,
    ) {
        tracing::warn!(
            "ISQ organization `moqe` has no effect without an active ISQ or UQFF-write target; pre-quantized GGUF/UQFF weights are loaded as-is. Use `--isq <type>` to quantize routed experts."
        );
    }
    let topology_overrides = if i.loading_from_uqff {
        i.topology_overrides
            .iter()
            .filter_map(|override_entry| {
                override_entry
                    .device
                    .as_ref()
                    .map(|device| mistralrs_quant::ImmediateIsqOverride {
                        predicate: override_entry.predicate.clone(),
                        layer_range: override_entry.layer_range.clone(),
                        ty: None,
                        device: Some(device.clone()),
                    })
            })
            .collect::<Vec<_>>()
    } else {
        i.topology_overrides.clone()
    };

    let allow_immediate_cli =
        !i.loading_from_uqff && (i.in_situ_quant.is_some() || i.has_write_uqff);
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
    let has_typed_topology_override = topology_overrides
        .iter()
        .any(|override_entry| override_entry.ty.is_some());
    if allow_immediate_cli
        && matches!(i.organization, IsqOrganization::MoeExpertsOnly)
        && immediate_predicates.is_empty()
        && !has_typed_topology_override
    {
        anyhow::bail!(
            "MoQE quantization is not supported for this model because no routed-expert weights were identified. Use `--isq-organization default`, add a typed topology override, or choose a model with MoQE support."
        );
    }
    let uqff_quantize_predicates = (i.has_write_uqff
        && matches!(i.organization, IsqOrganization::MoeExpertsOnly))
    .then(|| immediate_predicates.clone());

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
                .with_overrides(topology_overrides.clone()),
            executor,
        );
    } else if !topology_overrides.is_empty() {
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
            .with_overrides(topology_overrides.clone()),
            executor,
        );
    }

    let use_immediate = allow_immediate_cli || !topology_overrides.is_empty();
    let loading_isq = !use_immediate && !i.loading_from_uqff && i.in_situ_quant.is_some();

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
        uqff_quantize_predicates,
        loading_isq,
        load_device,
        expects_quantized_selection: allow_immediate_cli,
        _scope: ImmediateIsqScope,
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

    struct NoMoqeLoader;

    impl IsqModelLoader for NoMoqeLoader {
        fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<regex::Regex>> {
            Ok(Vec::new())
        }
    }

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
            Ok(vec![regex::Regex::new(
                r"^model\.layers\.\d+\.mlp\.experts\.",
            )?])
        }
    }

    fn matches(predicates: &[regex::Regex], name: &str) -> bool {
        predicates.iter().any(|predicate| predicate.is_match(name))
    }

    fn tracked_module(key: &str, ty: Option<IsqType>) -> mistralrs_quant::TrackedModule {
        let (_tx, rx) = mistralrs_quant::pending_isq_channel();
        mistralrs_quant::TrackedModule {
            key: key.to_string(),
            ct: std::sync::Arc::new(mistralrs_quant::PendingIsqLayer::new(rx)),
            ty,
            promote_default: false,
            shard: None,
        }
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
    fn moqe_selection_only_includes_experts() -> Result<()> {
        let (selected, promoted) =
            resolve_isq_predicates(&Loader, "", IsqOrganization::MoeExpertsOnly)?;
        assert!(matches(
            &selected,
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
        assert!(!matches(&selected, "lm_head.weight"));
        assert!(!matches(&selected, "model.embed_tokens.weight"));
        assert!(matches(&promoted, "model.embed_tokens.weight"));
        assert!(matches(&promoted, "lm_head.weight"));
        Ok(())
    }

    #[test]
    fn moqe_without_a_quantization_target_is_inert() {
        assert!(moqe_is_inert(
            IsqOrganization::MoeExpertsOnly,
            None,
            false,
            false
        ));
        assert!(moqe_is_inert(
            IsqOrganization::MoeExpertsOnly,
            Some(IsqType::Q4K),
            false,
            true
        ));
        assert!(!moqe_is_inert(
            IsqOrganization::MoeExpertsOnly,
            Some(IsqType::Q4K),
            false,
            false
        ));
        assert!(!moqe_is_inert(
            IsqOrganization::MoeExpertsOnly,
            None,
            true,
            false
        ));
        assert!(!moqe_is_inert(IsqOrganization::Default, None, false, false));
    }

    #[test]
    fn uqff_moqe_plan_preserves_expert_selection() -> Result<()> {
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: Some(vec![IsqType::Q4K, IsqType::Q8_0]),
            has_write_uqff: true,
            loading_from_uqff: false,
            organization: IsqOrganization::MoeExpertsOnly,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        let predicates = plan
            .uqff_quantize_predicates
            .as_deref()
            .expect("MoQE UQFF writes must retain their expert selection");
        assert!(matches(
            predicates,
            "model.layers.0.mlp.experts.0.down_proj.weight"
        ));
        for native in [
            "lm_head.weight",
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ] {
            assert!(!matches(predicates, native), "{native} must remain native");
        }
        Ok(())
    }

    #[test]
    fn uqff_rejects_duplicate_output_types() {
        let result = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: Some(vec![IsqType::Q4K, IsqType::Q4K]),
            has_write_uqff: true,
            loading_from_uqff: false,
            organization: IsqOrganization::Default,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        });

        let error = result.err().expect("duplicate UQFF types must fail");
        assert!(error
            .to_string()
            .contains("Duplicate UQFF output type `q4k`"));
    }

    #[test]
    fn uqff_moqe_rejects_models_without_expert_predicates() {
        let result = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: Some(vec![IsqType::Q4K]),
            has_write_uqff: true,
            loading_from_uqff: false,
            organization: IsqOrganization::MoeExpertsOnly,
            topology_overrides: Vec::new(),
            loader: &NoMoqeLoader,
            config: "",
            device: &Device::Cpu,
        });
        let err = result.err().expect("unsupported MoQE UQFF must fail");
        assert!(err.to_string().contains("no routed-expert weights"));
        assert!(err.to_string().contains("--isq-organization default"));
    }

    #[test]
    fn direct_moqe_rejects_models_without_expert_predicates() {
        let result = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: false,
            organization: IsqOrganization::MoeExpertsOnly,
            topology_overrides: Vec::new(),
            loader: &NoMoqeLoader,
            config: "",
            device: &Device::Cpu,
        });
        let err = result.err().expect("unsupported direct MoQE must fail");
        assert!(err.to_string().contains("no routed-expert weights"));
    }

    #[test]
    fn typed_topology_override_allows_moqe_without_model_predicates() -> Result<()> {
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: Some(vec![IsqType::Q4K]),
            has_write_uqff: true,
            loading_from_uqff: false,
            organization: IsqOrganization::MoeExpertsOnly,
            topology_overrides: vec![mistralrs_quant::ImmediateIsqOverride {
                predicate: Some(regex::Regex::new(r"^model\.layers\.0\.self_attn\.q_proj")?),
                layer_range: None,
                ty: Some(IsqType::Q8_0),
                device: None,
            }],
            loader: &NoMoqeLoader,
            config: "",
            device: &Device::Cpu,
        })?;

        plan.validate_tracked_selection(&[tracked_module(
            "model.layers.0.self_attn.q_proj",
            Some(IsqType::Q8_0),
        )])?;
        Ok(())
    }

    #[test]
    fn post_load_validation_rejects_predicates_that_match_no_modules() -> Result<()> {
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: Some(vec![IsqType::Q4K]),
            has_write_uqff: true,
            loading_from_uqff: false,
            organization: IsqOrganization::MoeExpertsOnly,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        let error = plan
            .validate_tracked_selection(&[tracked_module("model.layers.0.self_attn.q_proj", None)])
            .unwrap_err();
        assert!(error.to_string().contains("no model weights matched"));
        Ok(())
    }

    #[test]
    fn device_only_topology_does_not_require_quantized_modules() -> Result<()> {
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: false,
            organization: IsqOrganization::Default,
            topology_overrides: vec![mistralrs_quant::ImmediateIsqOverride {
                predicate: Some(regex::Regex::new(r"^model\.layers\.0\.")?),
                layer_range: None,
                ty: None,
                device: Some(Device::Cpu),
            }],
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        plan.validate_tracked_selection(&[])?;
        Ok(())
    }

    #[test]
    fn uqff_precedence_suppresses_isq_topology_overrides() -> Result<()> {
        mistralrs_quant::clear_immediate_isq();
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: true,
            organization: IsqOrganization::Default,
            topology_overrides: vec![mistralrs_quant::ImmediateIsqOverride {
                predicate: Some(regex::Regex::new(r"^model\.layers\.0\.")?),
                layer_range: None,
                ty: Some(IsqType::Q6K),
                device: None,
            }],
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        assert!(!plan.immediate_isq_installed);
        assert!(!plan.loading_isq);
        assert!(mistralrs_quant::get_immediate_isq().is_none());
        Ok(())
    }

    #[test]
    fn uqff_precedence_preserves_topology_device_overrides() -> Result<()> {
        mistralrs_quant::clear_immediate_isq();
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: true,
            organization: IsqOrganization::Default,
            topology_overrides: vec![mistralrs_quant::ImmediateIsqOverride {
                predicate: Some(regex::Regex::new(r"^model\.vision_proj\.weight$")?),
                layer_range: None,
                ty: Some(IsqType::Q6K),
                device: Some(Device::Cpu),
            }],
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        assert!(plan.immediate_isq_installed);
        assert!(!plan.loading_isq);
        let params = mistralrs_quant::get_immediate_isq().expect("device override is installed");
        assert_eq!(params.ty, None);
        assert!(params.predicates.is_empty());
        assert_eq!(params.overrides.len(), 1);
        assert_eq!(params.overrides[0].ty, None);
        assert!(params.overrides[0]
            .device
            .as_ref()
            .is_some_and(Device::is_cpu));
        assert!(params.overrides[0]
            .predicate
            .as_ref()
            .is_some_and(|predicate| predicate.is_match("model.vision_proj.weight")));
        mistralrs_quant::clear_immediate_isq();
        Ok(())
    }

    #[test]
    fn quantized_prepared_sources_keep_immediate_isq_enabled() -> Result<()> {
        mistralrs_quant::clear_immediate_isq();
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: false,
            organization: IsqOrganization::Default,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        assert!(plan.immediate_isq_installed);
        assert!(!plan.loading_isq);
        assert!(mistralrs_quant::get_immediate_isq().is_some());
        mistralrs_quant::clear_immediate_isq();
        Ok(())
    }

    #[test]
    fn inactive_plan_clears_stale_immediate_isq() -> Result<()> {
        mistralrs_quant::set_immediate_isq(
            Some(IsqType::Q4K),
            vec![regex::Regex::new(r"^model\.layers\.")?],
            mistralrs_quant::IsqCaptureMode::Immediate,
        );

        let _plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: None,
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: false,
            organization: IsqOrganization::Default,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        assert!(mistralrs_quant::get_immediate_isq().is_none());
        Ok(())
    }

    #[test]
    fn dropping_plan_clears_immediate_isq() -> Result<()> {
        let plan = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: false,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: false,
            organization: IsqOrganization::Default,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        })?;

        assert!(mistralrs_quant::get_immediate_isq().is_some());
        drop(plan);
        assert!(mistralrs_quant::get_immediate_isq().is_none());
        Ok(())
    }

    #[test]
    fn prepared_weight_source_requantization_sizes_for_target() {
        let dtype = DType::BF16;
        for target in [IsqType::Q8_0, IsqType::Q2K] {
            let sizing = resolve_auto_device_map_sizing(false, true, Some(target));
            let AutoDeviceMapSizing::Isq(selected) = sizing else {
                panic!("prepared source requantization should size for the target")
            };
            assert_eq!(selected.pack_factor(dtype), target.pack_factor(dtype));
        }
    }

    #[test]
    fn uqff_and_prepared_source_sizing_precedence_is_stable() {
        assert_eq!(
            resolve_auto_device_map_sizing(true, true, Some(IsqType::Q2K)),
            AutoDeviceMapSizing::Uqff
        );
        assert_eq!(
            resolve_auto_device_map_sizing(false, true, None),
            AutoDeviceMapSizing::PreparedWeightSource
        );
        assert_eq!(
            resolve_auto_device_map_sizing(false, false, None),
            AutoDeviceMapSizing::Checkpoint
        );
    }

    #[test]
    fn uqff_rejects_calibration_input() {
        let result = resolve_and_install_isq_plan(IsqPlanInputs {
            in_situ_quant: Some(IsqType::Q4K),
            has_imatrix: false,
            has_calibration: true,
            write_uqff_types: None,
            has_write_uqff: false,
            loading_from_uqff: true,
            organization: IsqOrganization::Default,
            topology_overrides: Vec::new(),
            loader: &Loader,
            config: "",
            device: &Device::Cpu,
        });
        let Err(err) = result else {
            panic!("UQFF calibration input should be rejected");
        };

        assert!(err.to_string().contains("cannot be combined"));
        assert!(err.to_string().contains("UQFF weights take precedence"));
    }
}
