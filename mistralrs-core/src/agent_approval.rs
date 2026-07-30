use std::{future::Future, pin::Pin, sync::Arc};

use mistralrs_mcp::AgentToolMetadata;

#[derive(Clone, Debug)]
pub struct AgentToolApproval {
    pub approval_id: String,
    pub session_id: String,
    pub round: usize,
    pub tool: AgentToolMetadata,
    pub arguments: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct AgentToolApprovalDecision {
    pub approve: bool,
    pub remember_for_session: bool,
    pub message: Option<String>,
}

impl AgentToolApprovalDecision {
    pub fn approve() -> Self {
        Self {
            approve: true,
            remember_for_session: false,
            message: None,
        }
    }

    pub fn approve_for_session() -> Self {
        Self {
            approve: true,
            remember_for_session: true,
            message: None,
        }
    }

    pub fn deny(message: Option<String>) -> Self {
        Self {
            approve: false,
            remember_for_session: false,
            message,
        }
    }

    pub fn deny_with_message(message: impl Into<String>) -> Self {
        Self {
            approve: false,
            remember_for_session: false,
            message: Some(message.into()),
        }
    }

    pub fn with_remember_for_session(mut self, remember_for_session: bool) -> Self {
        self.remember_for_session = remember_for_session;
        self
    }
}

pub type AgentToolApprovalCallback =
    Arc<dyn Fn(&AgentToolApproval) -> AgentToolApprovalDecision + Send + Sync + 'static>;

pub type AgentToolApprovalFuture =
    Pin<Box<dyn Future<Output = AgentToolApprovalDecision> + Send + 'static>>;

pub type AgentToolApprovalAsyncCallback =
    Arc<dyn Fn(AgentToolApproval) -> AgentToolApprovalFuture + Send + Sync + 'static>;

#[derive(Clone)]
pub enum AgentToolApprovalHandler {
    Sync(AgentToolApprovalCallback),
    Async(AgentToolApprovalAsyncCallback),
}

impl AgentToolApprovalHandler {
    pub fn from_sync(callback: AgentToolApprovalCallback) -> Self {
        Self::Sync(callback)
    }

    pub fn from_async(callback: AgentToolApprovalAsyncCallback) -> Self {
        Self::Async(callback)
    }
}
