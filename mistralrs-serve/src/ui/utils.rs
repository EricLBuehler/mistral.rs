use std::path::PathBuf;

/// Determine the base cache directory for the UI.
/// Uses XDG_CACHE_HOME or falls back to ~/.cache/mistralrs.
pub fn get_cache_dir() -> PathBuf {
    let cache_home = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".cache"))
                .unwrap_or_else(|_| PathBuf::from(".cache"))
        });
    cache_home.join("mistralrs")
}
