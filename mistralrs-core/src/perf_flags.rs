use std::sync::OnceLock;

const CUDA_GRAPHS_ENV: &str = "MISTRALRS_CUDA_GRAPHS";
const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";
const FLASHINFER_PREFILL_ENV: &str = "MISTRALRS_FLASHINFER_PREFILL";
const PAGED_PREFIX_VARLEN_SDPA_ENV: &str = "MISTRALRS_PAGED_PREFIX_VARLEN_SDPA";

static CUDA_GRAPHS_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_DECODE_ENABLED: OnceLock<bool> = OnceLock::new();
static FLASHINFER_PREFILL_ENABLED: OnceLock<bool> = OnceLock::new();
static PAGED_PREFIX_VARLEN_SDPA_ENABLED: OnceLock<bool> = OnceLock::new();

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .map(|value| {
            if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "on") {
                true
            } else if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                false
            } else {
                default
            }
        })
        .unwrap_or(default)
}

pub(crate) fn cuda_graphs_enabled() -> bool {
    *CUDA_GRAPHS_ENABLED.get_or_init(|| env_flag(CUDA_GRAPHS_ENV, true))
}

pub(crate) fn flashinfer_decode_enabled() -> bool {
    *FLASHINFER_DECODE_ENABLED.get_or_init(|| env_flag(FLASHINFER_DECODE_ENV, true))
}

pub(crate) fn flashinfer_prefill_enabled() -> bool {
    *FLASHINFER_PREFILL_ENABLED.get_or_init(|| env_flag(FLASHINFER_PREFILL_ENV, true))
}

pub(crate) fn paged_prefix_varlen_sdpa_enabled() -> bool {
    *PAGED_PREFIX_VARLEN_SDPA_ENABLED.get_or_init(|| env_flag(PAGED_PREFIX_VARLEN_SDPA_ENV, true))
}
