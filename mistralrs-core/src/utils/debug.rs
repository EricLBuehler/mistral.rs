use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

use crate::DEBUG;

// This should be called in each `Loader` when it is created.
pub(crate) fn setup_logger_and_debug() {
    let is_debug = std::env::var("MISTRALRS_DEBUG")
        .unwrap_or_default()
        .contains('1');
    DEBUG.store(is_debug, std::sync::atomic::Ordering::Relaxed);

    let filter = EnvFilter::builder()
        .with_default_directive(if is_debug {
            LevelFilter::DEBUG.into()
        } else {
            LevelFilter::INFO.into()
        })
        .from_env_lossy();
    tracing_subscriber::fmt().with_env_filter(filter).init();
}
