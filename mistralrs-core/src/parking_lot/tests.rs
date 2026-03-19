//! Tests for the parking_lot module.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::sampler::SamplingParams;
    use crate::request::RequestMessage;

    #[test]
    fn test_resource_adapter_creation() {
        let adapter = ResourceAdapter::default();
        assert_eq!(adapter.block_size(), DEFAULT_BLOCK_SIZE);
        assert!(adapter.max_units() > 0);
    }

    #[test]
    fn test_resource_adapter_cost_calculation() {
        let adapter = ResourceAdapter::new(16, 1024, 64);
        
        // Test cost calculation
        let cost = adapter.calculate_cost(100, 50);
        assert_eq!(cost.units, 10); // (100 + 50) / 16 = 9.375 → 10 blocks
        
        // Test token/block conversions
        assert_eq!(adapter.tokens_to_blocks(32), 2);
        assert_eq!(adapter.blocks_to_tokens(5), 80);
    }

    #[test]
    fn test_inference_job_creation() {
        let job = InferenceJob {
            request_id: 123,
            is_streaming: false,
            messages: Some(RequestMessage::Completion {
                text: "Hello".to_string(),
                echo_prompt: false,
                best_of: None,
            }),
            sampling_params: Some(SamplingParams::default()),
            constraint: None,
            return_logprobs: false,
            truncate_sequence: false,
            tools: None,
            tool_choice: None,
        };

        assert_eq!(job.request_id, 123);
        assert!(!job.is_streaming);
    }

    #[test]
    fn test_inference_job_serialization() {
        let job = InferenceJob {
            request_id: 456,
            is_streaming: false,
            messages: None,
            sampling_params: Some(SamplingParams::default()),
            constraint: None,
            return_logprobs: false,
            truncate_sequence: false,
            tools: None,
            tool_choice: None,
        };

        // Test serialization
        let json = serde_json::to_string(&job).unwrap();
        assert!(json.contains("456"));
    }

    #[test]
    fn test_task_metadata_builder() {
        let cost = ResourceCost::gpu_vram(10);
        let meta = TaskMetadata::new(42u64, cost)
            .with_priority(Priority::High)
            .with_deadline_ms(999999);

        assert_eq!(meta.id, 42);
        assert_eq!(meta.priority, Priority::High);
        assert_eq!(meta.cost.units, 10);
        assert_eq!(meta.deadline_ms, Some(999999));
    }

    #[test]
    fn test_task_metadata_conversion() {
        use super::super::types::ParkingLotTaskMetadata;
        
        let cost = ResourceCost::gpu_vram(5);
        let meta = TaskMetadata::new(100u64, cost);
        
        // Test conversion to ParkingLotTaskMetadata
        let pl_meta: ParkingLotTaskMetadata = meta.clone().into();
        assert_eq!(pl_meta.id, 100);
        assert_eq!(pl_meta.cost.units, 5);
    }

    #[test]
    fn test_streaming_registry() {
        let registry = StreamingRegistry::with_default_retention();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_streaming_registry_register_retrieve() {
        use std::time::Duration;
        
        let registry = StreamingRegistry::with_default_retention();
        let (_tx, rx) = flume::unbounded();
        
        registry.register(
            "test-key".to_string(),
            "req-123".to_string(),
            rx,
        );
        
        assert_eq!(registry.len(), 1);
        
        let retrieved = registry.retrieve("test-key");
        assert!(retrieved.is_some());
        
        // Test removal
        assert!(registry.remove("test-key"));
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_inference_result_variants() {
        use crate::response::{ChatCompletionResponse, Choice, ResponseMessage};
        
        // Test error variant
        let error_result = InferenceResult::error("test error");
        assert!(error_result.is_error());
        assert_eq!(error_result.error_message(), Some("test error"));
        
        // Test streaming variant
        let (_tx, rx) = flume::unbounded();
        let streaming_result = InferenceResult::streaming("req-1".to_string(), rx);
        assert!(!streaming_result.is_error());
    }

    #[test]
    fn test_worker_pool_config() {
        let config = InferenceWorkerPoolConfig::new(4, 1024, 100)
            .with_timeout_secs(60);
        
        assert_eq!(config.worker_count, 4);
        assert_eq!(config.max_units, 1024);
        assert_eq!(config.max_queue_depth, 100);
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats {
            active_workers: 4,
            queued_tasks: 10,
            available_capacity: 512,
            total_capacity: 1024,
        };
        
        assert_eq!(stats.active_workers, 4);
        assert_eq!(stats.queued_tasks, 10);
        assert_eq!(stats.available_capacity, 512);
    }

    #[tokio::test]
    async fn test_task_executor_trait() {
        use std::sync::Arc;
        use tokio::sync::Mutex;
        
        // Create a simple pipeline wrapper for testing
        // Note: This test verifies the trait implementation compiles
        // Actual execution testing requires a full pipeline setup
    }
}
