//! ## Background task management for the Responses API.
//!
//! This module handles background processing of responses when `background: true` is set.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use uuid::Uuid;

use crate::responses_types::{ResponseError, ResponseResource, ResponseStatus};

/// State of a background task
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum BackgroundTaskState {
    /// Task is queued
    Queued,
    /// Task is in progress
    InProgress,
    /// Task completed successfully
    Completed(ResponseResource),
    /// Task failed
    Failed(ResponseError),
    /// Task was cancelled
    Cancelled,
}

/// A background task for processing responses
#[derive(Debug)]
pub struct BackgroundTask {
    /// Task ID (same as response ID)
    pub id: String,
    /// Current state
    pub state: BackgroundTaskState,
    /// Created timestamp
    pub created_at: u64,
    /// Model name
    pub model: String,
    /// Cancellation flag
    pub cancel_requested: bool,
}

impl BackgroundTask {
    /// Create a new background task
    pub fn new(id: String, model: String) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            id,
            state: BackgroundTaskState::Queued,
            created_at,
            model,
            cancel_requested: false,
        }
    }

    /// Convert the current task state to a ResponseResource
    pub fn to_response_resource(&self) -> ResponseResource {
        let mut resource =
            ResponseResource::new(self.id.clone(), self.model.clone(), self.created_at);

        match &self.state {
            BackgroundTaskState::Queued => {
                resource.status = ResponseStatus::Queued;
            }
            BackgroundTaskState::InProgress => {
                resource.status = ResponseStatus::InProgress;
            }
            BackgroundTaskState::Completed(resp) => {
                return resp.clone();
            }
            BackgroundTaskState::Failed(error) => {
                resource.status = ResponseStatus::Failed;
                resource.error = Some(error.clone());
            }
            BackgroundTaskState::Cancelled => {
                resource.status = ResponseStatus::Cancelled;
            }
        }

        resource
    }
}

/// Manager for background tasks
#[derive(Debug, Default)]
pub struct BackgroundTaskManager {
    /// Map of task ID to task
    tasks: Arc<RwLock<HashMap<String, BackgroundTask>>>,
}

impl BackgroundTaskManager {
    /// Create a new background task manager
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new background task and return its ID
    pub fn create_task(&self, model: String) -> String {
        let id = format!("resp_{}", Uuid::new_v4());
        let task = BackgroundTask::new(id.clone(), model);

        let mut tasks = self.tasks.write().unwrap();
        tasks.insert(id.clone(), task);

        id
    }

    /// Get the current state of a task
    pub fn get_task(&self, id: &str) -> Option<BackgroundTask> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(id).cloned()
    }

    /// Get the response resource for a task
    pub fn get_response(&self, id: &str) -> Option<ResponseResource> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(id).map(|t| t.to_response_resource())
    }

    /// Update task to in_progress state
    pub fn mark_in_progress(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(id) {
            task.state = BackgroundTaskState::InProgress;
            true
        } else {
            false
        }
    }

    /// Update task to completed state
    pub fn mark_completed(&self, id: &str, response: ResponseResource) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(id) {
            task.state = BackgroundTaskState::Completed(response);
            true
        } else {
            false
        }
    }

    /// Update task to failed state
    pub fn mark_failed(&self, id: &str, error: ResponseError) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(id) {
            task.state = BackgroundTaskState::Failed(error);
            true
        } else {
            false
        }
    }

    /// Request cancellation of a task
    pub fn request_cancel(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(id) {
            if matches!(
                task.state,
                BackgroundTaskState::Queued | BackgroundTaskState::InProgress
            ) {
                task.cancel_requested = true;
                return true;
            }
        }
        false
    }

    /// Check if cancellation was requested for a task
    pub fn is_cancel_requested(&self, id: &str) -> bool {
        let tasks = self.tasks.read().unwrap();
        tasks.get(id).map(|t| t.cancel_requested).unwrap_or(false)
    }

    /// Mark task as cancelled
    pub fn mark_cancelled(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(task) = tasks.get_mut(id) {
            task.state = BackgroundTaskState::Cancelled;
            true
        } else {
            false
        }
    }

    /// Delete a task
    pub fn delete_task(&self, id: &str) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        tasks.remove(id).is_some()
    }

    /// List all task IDs
    pub fn list_tasks(&self) -> Vec<String> {
        let tasks = self.tasks.read().unwrap();
        tasks.keys().cloned().collect()
    }

    /// Clean up old completed/failed tasks older than the given duration (in seconds)
    pub fn cleanup_old_tasks(&self, max_age_secs: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut tasks = self.tasks.write().unwrap();
        tasks.retain(|_, task| {
            // Keep queued and in-progress tasks
            if matches!(
                task.state,
                BackgroundTaskState::Queued | BackgroundTaskState::InProgress
            ) {
                return true;
            }

            // Remove old completed/failed/cancelled tasks
            now - task.created_at < max_age_secs
        });
    }
}

impl Clone for BackgroundTask {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            state: self.state.clone(),
            created_at: self.created_at,
            model: self.model.clone(),
            cancel_requested: self.cancel_requested,
        }
    }
}

/// Global background task manager
static BACKGROUND_TASK_MANAGER: std::sync::LazyLock<BackgroundTaskManager> =
    std::sync::LazyLock::new(BackgroundTaskManager::new);

/// Get the global background task manager
pub fn get_background_task_manager() -> &'static BackgroundTaskManager {
    &BACKGROUND_TASK_MANAGER
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_task() {
        let manager = BackgroundTaskManager::new();
        let id = manager.create_task("test-model".to_string());

        let task = manager.get_task(&id).unwrap();
        assert_eq!(task.id, id);
        assert!(matches!(task.state, BackgroundTaskState::Queued));
    }

    #[test]
    fn test_task_state_transitions() {
        let manager = BackgroundTaskManager::new();
        let id = manager.create_task("test-model".to_string());

        // Move to in_progress
        assert!(manager.mark_in_progress(&id));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::InProgress));

        // Mark completed
        let response = ResponseResource::new(id.clone(), "test-model".to_string(), 0);
        assert!(manager.mark_completed(&id, response));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::Completed(_)));
    }

    #[test]
    fn test_cancel_task() {
        let manager = BackgroundTaskManager::new();
        let id = manager.create_task("test-model".to_string());

        // Request cancellation
        assert!(manager.request_cancel(&id));
        assert!(manager.is_cancel_requested(&id));

        // Mark as cancelled
        assert!(manager.mark_cancelled(&id));
        let task = manager.get_task(&id).unwrap();
        assert!(matches!(task.state, BackgroundTaskState::Cancelled));
    }
}
