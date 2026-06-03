//! Tests for the parking_lot module.
//!
//! Admission-gate behavior (permits, semaphore cap, queue rejection,
//! cancellation safety) is tested in `worker_pool::admission_tests`. YAML/CLI
//! config parsing and validation is tested in `config`. This file covers the
//! `InferenceWorkerPoolConfig` surface used by the engine.

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn test_worker_pool_config_explicit() {
        let config = InferenceWorkerPoolConfig::new(4, 100);
        assert_eq!(config.worker_count, 4);
        assert_eq!(config.max_queue_depth, 100);
    }

    #[test]
    fn test_worker_pool_config_default() {
        let config = InferenceWorkerPoolConfig::default();
        assert!(config.worker_count >= 1);
        assert_eq!(config.max_queue_depth, 1000);
    }

    #[test]
    fn test_worker_pool_config_from_scheduler_config() {
        // An empty scheduler config falls back to defaults.
        let sched = ParkingLotSchedulerConfig::default();
        let config = InferenceWorkerPoolConfig::from_scheduler_config(sched);
        assert!(config.worker_count >= 1);
        assert_eq!(config.max_queue_depth, 1000);
    }
}
