//! CPU worker-thread sizing for decode-shaped work.
//!
//! Decode is batch-1 and layer-serial, so a thread per core spends most of its time at
//! barriers rather than streaming weights. Candle reports the core count via
//! `get_num_threads`, which for this class of workload measured flat at one core's bandwidth
//! from 1 thread to 32 (see `examples/cpu_decode_gemv.rs`). Sizing off an explicit override
//! keeps the knob discoverable without fighting `RAYON_NUM_THREADS`, which candle already
//! honours upstream of us.
//!
//! With the output-parallel GEMV the kernel became bandwidth-bound rather than
//! barrier-bound, and the measured optimum is half the available cores (16 of 32 on the test
//! node, 4.9 tok/s against 4.1 at 32), so the default is that ratio rather than the full count.

/// Fraction of available cores the decode GEMV uses when neither override is set.
const DEFAULT_CORE_FRACTION: usize = 2;

/// Threads for CPU decode-shaped work, honouring `MISTRALRS_CPU_THREADS` then
/// `RAYON_NUM_THREADS`, else half the cores candle reports.
pub fn resolve() -> usize {
    read_env_usize("MISTRALRS_CPU_THREADS")
        .or_else(|| read_env_usize("RAYON_NUM_THREADS"))
        .unwrap_or_else(|| (candle_core::utils::get_num_threads() / DEFAULT_CORE_FRACTION).max(1))
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
    fn default_is_half_the_cores() {
        let cores = candle_core::utils::get_num_threads();
        // Set neither override: the default must be the measured half-core ratio.
        assert_eq!(resolve(), (cores / DEFAULT_CORE_FRACTION).max(1));
    }
}
