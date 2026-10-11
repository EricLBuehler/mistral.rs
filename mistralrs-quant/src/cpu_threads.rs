//! CPU worker-thread sizing for decode-shaped work.
//!
//! Decode is batch-1 and layer-serial, so a thread per core spends most of its time at
//! barriers rather than streaming weights. Candle reports the core count via
//! `get_num_threads`, which for this class of workload measured flat at one core's bandwidth
//! from 1 thread to 32 (see `examples/cpu_decode_gemv.rs`). Sizing off an explicit override
//! keeps the knob discoverable without fighting `RAYON_NUM_THREADS`, which candle already
//! honours upstream of us.

/// Threads to use for CPU decode GEMV, honouring `MISTRALRS_CPU_THREADS` then
/// `RAYON_NUM_THREADS`, else candle's core count.
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
}
