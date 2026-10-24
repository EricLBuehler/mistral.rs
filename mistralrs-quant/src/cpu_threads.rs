//! CPU worker-thread sizing for decode-shaped work.
//!
//! mistral.rs's decode GEMV and its flash-attention pool read their size from here so the two
//! stay consistent, and so a caller can steer them with one variable. The default delegates
//! to Candle's core count; it deliberately does NOT impose a ratio.
//!
//! An earlier revision divided the core count by two. That was extrapolated from one 32-core
//! node and it is wrong in two separate ways:
//!
//! - It was measured *worse*. Sizing these two pools to 16 while Candle's global Rayon pool
//!   stayed at 32 is an inconsistent configuration, and the mismatch cost more than the ratio
//!   ever won (2.2 tok/s against 4.2 for a consistent 32).
//! - "Half the cores" is the wrong shape on a large host: it keeps shrinking as cores grow,
//!   long past the point where a batch-1, layer-serial decode can use more parallelism.
//!
//! See `examples/cpu_decode_gemv.rs` for the isolated measurements. There is a separate,
//! pre-existing effect where leaving `RAYON_NUM_THREADS` unset interacts with `numactl` CPU
//! pinning and under-uses the allowed cores; that lives in Candle's pool sizing and is out
//! of scope here, but `RAYON_NUM_THREADS` is the lever that resolves it.

/// Threads for mistral.rs's CPU decode pools, honouring `MISTRALRS_CPU_THREADS` then
/// `RAYON_NUM_THREADS`, else the core count Candle reports.
pub fn resolve() -> usize {
    read_env_usize("MISTRALRS_CPU_THREADS")
        .or_else(|| read_env_usize("RAYON_NUM_THREADS"))
        .unwrap_or_else(candle_core::utils::get_num_threads)
        .max(1)
}

fn read_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok().filter(|v| *v > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_returns_zero() {
        assert!(resolve() >= 1);
    }

    #[test]
    fn default_matches_candle_core_count() {
        // Neither override set: defer to Candle rather than apply a ratio, so these pools
        // agree with Candle's own.
        if std::env::var("MISTRALRS_CPU_THREADS").is_err()
            && std::env::var("RAYON_NUM_THREADS").is_err()
        {
            assert_eq!(resolve(), candle_core::utils::get_num_threads().max(1));
        }
    }
}
