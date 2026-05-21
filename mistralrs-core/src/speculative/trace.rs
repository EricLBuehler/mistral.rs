use std::fmt;
use std::sync::OnceLock;

use candle_core::Tensor;

const ENV_FLAG: &str = "MISTRALRS_MTP_TRACE";
const MAX_TOKENS: usize = 24;

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(ENV_FLAG).is_ok_and(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

pub(crate) fn log(args: fmt::Arguments<'_>) {
    if enabled() {
        tracing::info!(target: "mistralrs_core::speculative::trace", "{args}");
    }
}

pub(crate) fn tensor(tensor: &Tensor) -> String {
    format!(
        "shape={:?}, dtype={:?}, device={:?}",
        tensor.dims(),
        tensor.dtype(),
        tensor.device()
    )
}

pub(crate) fn tokens(tokens: &[u32]) -> String {
    if tokens.len() <= MAX_TOKENS {
        return format!("{tokens:?}");
    }
    format!(
        "{:?} ... (+{} more)",
        &tokens[..MAX_TOKENS],
        tokens.len() - MAX_TOKENS
    )
}
