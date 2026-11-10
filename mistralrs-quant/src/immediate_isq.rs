use super::*;

#[derive(Clone, Debug)]
pub struct ImmediateIsqParams {
    pub guard: QuantizeOntoGuard,
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub promoted_predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
    pub executor: IsqExecutor,
    pub capture: IsqCaptureMode,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqConfig {
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub promoted_predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
    pub capture: IsqCaptureMode,
}

impl ImmediateIsqConfig {
    pub fn new(ty: Option<IsqType>, predicates: Vec<Regex>, capture: IsqCaptureMode) -> Self {
        Self {
            ty,
            predicates,
            promoted_predicates: Vec::new(),
            overrides: Vec::new(),
            capture,
        }
    }

    pub fn with_promoted_predicates(mut self, promoted_predicates: Vec<Regex>) -> Self {
        self.promoted_predicates = promoted_predicates;
        self
    }

    pub fn with_overrides(mut self, overrides: Vec<ImmediateIsqOverride>) -> Self {
        self.overrides = overrides;
        self
    }
}

/// Whether load-time ISQ quantizes layers or captures them unquantized for later quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IsqCaptureMode {
    /// Quantize matching layers as they load.
    #[default]
    Immediate,
    /// Capture every layer unquantized (UQFF serialization needs all of them).
    CaptureAll,
    /// Capture matching layers unquantized for deferred quantization (e.g. calibration).
    CaptureMatches,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqOverride {
    pub predicate: Option<Regex>,
    /// Decoder layer index range, matched via the `layers.N` segment of the weight prefix.
    pub layer_range: Option<std::ops::Range<usize>>,
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
}

impl ImmediateIsqOverride {
    pub(crate) fn matches(&self, prefix: &str) -> bool {
        if let Some(predicate) = &self.predicate {
            if predicate.is_match(prefix) {
                return true;
            }
        }
        if let Some(range) = &self.layer_range {
            if let Some(index) = layer_index_from_prefix(prefix) {
                return range.contains(&index);
            }
        }
        false
    }
}

pub(crate) static LAYER_INDEX_REGEX: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

/// Extract the decoder layer index from a weight prefix like `model.layers.12.self_attn.q_proj`.
pub fn layer_index_from_prefix(prefix: &str) -> Option<usize> {
    let regex = LAYER_INDEX_REGEX
        .get_or_init(|| Regex::new(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)").expect("valid regex"));
    regex.captures(prefix)?.get(1)?.as_str().parse().ok()
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqMatch {
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
    pub promote_default: bool,
}

thread_local! {
    static ENGINE_IMMEDIATE_ISQ: std::cell::RefCell<Option<ImmediateIsqParams>> = const { std::cell::RefCell::new(None) } ;
}

pub fn set_immediate_isq(isq: Option<IsqType>, predicates: Vec<Regex>, capture: IsqCaptureMode) {
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(isq));
    set_immediate_isq_with_executor(isq, predicates, Vec::new(), capture, executor);
}

pub fn set_immediate_isq_with_executor(
    isq: Option<IsqType>,
    predicates: Vec<Regex>,
    overrides: Vec<ImmediateIsqOverride>,
    capture: IsqCaptureMode,
    executor: IsqExecutor,
) {
    set_immediate_isq_config(
        ImmediateIsqConfig::new(isq, predicates, capture).with_overrides(overrides),
        executor,
    );
}

pub fn set_immediate_isq_config(config: ImmediateIsqConfig, executor: IsqExecutor) {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = Some(ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty: config.ty,
            predicates: config.predicates,
            promoted_predicates: config.promoted_predicates,
            overrides: config.overrides,
            executor,
            capture: config.capture,
        });
    });
}

#[cfg(target_os = "macos")]
pub(crate) unsafe fn set_isq_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#[cfg(not(target_os = "macos"))]
pub(crate) unsafe fn set_isq_thread_affinity() {}

pub fn create_isq_executor(config: IsqExecutorConfig) -> (IsqExecutor, usize) {
    let executor = IsqExecutor::new(config);
    let num_threads = executor.worker_threads();
    (executor, num_threads)
}

pub fn get_immediate_isq() -> Option<ImmediateIsqParams> {
    ENGINE_IMMEDIATE_ISQ.with(|cell| cell.borrow().clone())
}

pub fn clear_immediate_isq() {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

pub fn should_apply_immediate_isq(vb: &ShardedVarBuilder) -> bool {
    immediate_isq_match(vb).is_some()
}

pub fn weight_source_load_device(vb: &ShardedVarBuilder) -> Device {
    if should_apply_immediate_isq(vb) {
        Device::Cpu
    } else {
        vb.device().clone()
    }
}

pub fn immediate_isq_match(vb: &ShardedVarBuilder) -> Option<ImmediateIsqMatch> {
    let immediate_isq = get_immediate_isq()?;
    // Add a .weight to match the ISQ regexes!
    let prefix = format!("{}.weight", vb.prefix());
    resolve_immediate_isq(&immediate_isq, &prefix)
}

pub(crate) fn resolve_immediate_isq(
    params: &ImmediateIsqParams,
    prefix: &str,
) -> Option<ImmediateIsqMatch> {
    let promote_default = params
        .promoted_predicates
        .iter()
        .any(|predicate| predicate.is_match(prefix));
    let default_ty = params.ty.map(|ty| {
        if promote_default {
            ty.promote_for_sensitive_tensor()
        } else {
            ty
        }
    });
    if params.capture == IsqCaptureMode::CaptureAll {
        // Capture everything; topology overrides still pin per-layer ty/device.
        if let Some(override_hit) = params
            .overrides
            .iter()
            .find(|override_entry| override_entry.matches(prefix))
        {
            return Some(ImmediateIsqMatch {
                ty: override_hit.ty.or(default_ty),
                device: override_hit.device.clone(),
                promote_default,
            });
        }
        return Some(ImmediateIsqMatch {
            ty: None,
            device: None,
            promote_default,
        });
    }

    if let Some(override_hit) = params
        .overrides
        .iter()
        .find(|override_entry| override_entry.matches(prefix))
    {
        let ty = override_hit.ty.or(default_ty);
        // Device-only overrides still need a match so the layer gets relocated
        if ty.is_some() || override_hit.device.is_some() {
            return Some(ImmediateIsqMatch {
                ty,
                device: override_hit.device.clone(),
                promote_default,
            });
        }
        return None;
    }

    if let Some(ty) = default_ty {
        if params
            .predicates
            .iter()
            .any(|predicate| predicate.is_match(prefix))
        {
            return Some(ImmediateIsqMatch {
                ty: Some(ty),
                device: None,
                promote_default,
            });
        }
    }

    None
}
