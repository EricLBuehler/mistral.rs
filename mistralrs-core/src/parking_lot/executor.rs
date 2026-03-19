//! LLM inference executor for the parking-lot scheduler.
//!
//! This module implements the `TaskExecutor` trait for processing
//! LLM inference jobs through mistral.rs pipelines.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex as TokioMutex;
use tracing::{debug, error, info};

use super::job::{InferenceJob, InferenceResult, StreamingTokenResult};
use super::types::{ParkingLotTaskMetadata, PrometheusWorkerExecutor, TaskExecutor, TaskMetadata};
use crate::pipeline::Pipeline;
use crate::response::Response;

/// LLM inference executor that processes inference jobs.
///
/// This executor wraps a mistral.rs Pipeline and processes inference
/// requests through the worker pool.
#[derive(Clone)]
pub struct LlmExecutor {
    /// Pipeline for inference (uses tokio::sync::Mutex for async)
    pipeline: Arc<TokioMutex<dyn Pipeline + Send + Sync>>,
}

impl LlmExecutor {
    /// Create a new LLM executor.
    #[must_use]
    pub fn new(pipeline: Arc<TokioMutex<dyn Pipeline + Send + Sync>>) -> Self {
        Self { pipeline }
    }

    /// Process a completion (non-streaming) job.
    ///
    /// This processes the inference request through the Pipeline and
    /// collects the complete response.
    async fn process_completion(&self, job: &InferenceJob, meta: &TaskMetadata) -> InferenceResult {
        debug!(
            task_id = %meta.id,
            request_id = %job.request_id,
            "Processing completion job"
        );

        // Create a channel to receive responses
        let (tx, mut rx) = tokio::sync::mpsc::channel(100);

        // Convert job to Request and send to the response channel directly
        // NOTE: In the actual integration, this would go through the Engine's
        // handle_request method, but for now we create a direct completion response
        
        // Wait for complete response (this will be sent by the actual pipeline integration)
        while let Some(response) = rx.recv().await {
            match response {
                Response::Done(completion) => {
                    return InferenceResult::chat_completion(completion);
                }
                Response::CompletionDone(completion) => {
                    return InferenceResult::completion(completion);
                }
                Response::ModelError(msg, _) => {
                    return InferenceResult::error(msg);
                }
                Response::ValidationError(err) => {
                    return InferenceResult::error(format!("{}", err));
                }
                Response::InternalError(err) => {
                    return InferenceResult::error(format!("{}", err));
                }
                _ => {
                    // Ignore chunks for non-streaming
                }
            }
        }

        InferenceResult::error("No response received from pipeline")
    }

    /// Process a streaming job.
    ///
    /// This sets up a streaming channel and returns a receiver for tokens.
    async fn process_streaming(&self, job: &InferenceJob, meta: &TaskMetadata) -> InferenceResult {
        debug!(
            task_id = %meta.id,
            request_id = %job.request_id,
            "Processing streaming job"
        );

        // Create channels
        let (response_tx, mut response_rx) = tokio::sync::mpsc::channel(100);
        let (token_tx, token_rx) = flume::unbounded();

        // Convert job to Request
        let request = job.to_request(response_tx);

        // TODO: Send request to pipeline
        // This is a stub - needs proper implementation

        // Spawn a task to forward chunks to the token channel
        let request_id_clone = job.request_id.to_string();
        tokio::spawn(async move {
            while let Some(response) = response_rx.recv().await {
                match response {
                    Response::Chunk(chunk_response) => {
                        for (idx, choice) in chunk_response.choices.iter().enumerate() {
                            let is_finished = choice.finish_reason.is_some();
                            let token_result = StreamingTokenResult {
                                text: choice.delta.content.clone().unwrap_or_default(),
                                token_id: None, // Not available in chunk response
                                is_finished,
                                finish_reason: choice.finish_reason.clone(),
                                model: chunk_response.model.clone(),
                                id: chunk_response.id.clone(),
                                created: chunk_response.created as u64,
                                index: idx,
                            };
                            if token_tx.send(Ok(token_result)).is_err() {
                                break;
                            }
                            if is_finished {
                                break;
                            }
                        }
                    }
                    Response::Done(_) | Response::CompletionDone(_) => {
                        // Final chunk already sent above
                        break;
                    }
                    Response::ModelError(msg, _) => {
                        let _ = token_tx.send(Err(msg));
                        break;
                    }
                    Response::ValidationError(err) => {
                        let _ = token_tx.send(Err(format!("{}", err)));
                        break;
                    }
                    Response::InternalError(err) => {
                        let _ = token_tx.send(Err(format!("{}", err)));
                        break;
                    }
                    _ => {}
                }
            }
        });

        InferenceResult::streaming(job.request_id.to_string(), token_rx)
    }
}

// Implement prometheus_parking_lot's WorkerExecutor trait
#[async_trait]
impl PrometheusWorkerExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(
        &self,
        payload: InferenceJob,
        meta: ParkingLotTaskMetadata,
    ) -> InferenceResult {
        // Convert ParkingLotTaskMetadata to our local TaskMetadata
        let local_meta = TaskMetadata {
            id: meta.id,
            priority: meta.priority,
            cost: meta.cost,
            created_at_ms: meta.created_at_ms,
            deadline_ms: meta.deadline_ms,
            mailbox: meta.mailbox,
        };

        info!(
            task_id = %local_meta.id,
            request_id = %payload.request_id,
            is_streaming = payload.is_streaming,
            "Executing inference job via WorkerPool"
        );

        if payload.is_streaming {
            self.process_streaming(&payload, &local_meta).await
        } else {
            self.process_completion(&payload, &local_meta).await
        }
    }
}

// Also implement our local TaskExecutor trait for backward compatibility
#[async_trait]
impl TaskExecutor<InferenceJob, InferenceResult> for LlmExecutor {
    async fn execute(&self, payload: InferenceJob, meta: TaskMetadata) -> InferenceResult {
        info!(
            task_id = %meta.id,
            request_id = %payload.request_id,
            is_streaming = payload.is_streaming,
            "Executing inference job"
        );

        if payload.is_streaming {
            self.process_streaming(&payload, &meta).await
        } else {
            self.process_completion(&payload, &meta).await
        }
    }
}
