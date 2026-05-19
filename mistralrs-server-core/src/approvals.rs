use std::{
    collections::{HashMap, HashSet},
    sync::{mpsc, Arc, Mutex},
    time::Duration,
};

use axum::{
    extract::{Json, Path},
    http::StatusCode,
    response::IntoResponse,
    Extension,
};
use mistralrs_core::{
    AgentToolApproval, AgentToolApprovalCallback, AgentToolApprovalDecision,
    AgentToolApprovalNotifier, AgentToolApprovalRequest, Response,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Default)]
pub struct ApprovalBroker {
    inner: Arc<Mutex<ApprovalState>>,
}

#[derive(Default)]
struct ApprovalState {
    pending: HashMap<String, PendingApproval>,
    early_decisions: HashMap<String, ApprovalDecisionState>,
    approved_sessions: HashSet<String>,
    notified: HashSet<String>,
}

struct PendingApproval {
    session_id: String,
    tx: mpsc::Sender<AgentToolApprovalDecision>,
}

#[derive(Clone)]
struct ApprovalDecisionState {
    approve: bool,
    remember_for_session: bool,
    message: Option<String>,
}

impl ApprovalBroker {
    pub fn callback(&self) -> AgentToolApprovalCallback {
        let broker = self.clone();
        Arc::new(move |approval| broker.wait_for_decision(approval))
    }

    pub fn notifier(&self, response: Sender<Response>) -> Arc<AgentToolApprovalNotifier> {
        let broker = self.clone();
        Arc::new(move |approval| broker.notify_approval_required(approval, response.clone()))
    }

    fn notify_approval_required(
        &self,
        approval: AgentToolApprovalRequest,
        response: Sender<Response>,
    ) {
        if self.is_session_approved(&approval.session_id) {
            return;
        }

        let approval_id = approval.approval_id;
        self.inner
            .lock()
            .unwrap()
            .notified
            .insert(approval_id.clone());
        let send_result = response.try_send(Response::AgenticToolApprovalRequired {
            approval_id: approval_id.clone(),
            session_id: approval.session_id,
            round: approval.round,
            tool: approval.tool,
            arguments: approval.arguments,
        });
        if send_result.is_err() {
            let _ = self.resolve(&approval_id, false, false, None);
        }
    }

    fn wait_for_decision(&self, approval: &AgentToolApproval) -> AgentToolApprovalDecision {
        if self.is_session_approved(&approval.session_id) {
            return AgentToolApprovalDecision::approve();
        }

        let (tx, rx) = mpsc::channel();
        {
            let mut state = self.inner.lock().unwrap();
            if let Some(decision) = state.early_decisions.remove(&approval.approval_id) {
                if decision.approve && decision.remember_for_session {
                    state.approved_sessions.insert(approval.session_id.clone());
                }
                return AgentToolApprovalDecision {
                    approve: decision.approve,
                    remember_for_session: decision.remember_for_session,
                    message: decision.message,
                };
            }
            state.pending.insert(
                approval.approval_id.clone(),
                PendingApproval {
                    session_id: approval.session_id.clone(),
                    tx,
                },
            );
        }

        let decision =
            rx.recv_timeout(APPROVAL_TIMEOUT)
                .unwrap_or_else(|_| AgentToolApprovalDecision {
                    approve: false,
                    remember_for_session: false,
                    message: Some("Approval timed out.".to_string()),
                });
        let mut state = self.inner.lock().unwrap();
        state.pending.remove(&approval.approval_id);
        state.notified.remove(&approval.approval_id);
        decision
    }

    fn resolve(
        &self,
        approval_id: &str,
        approve: bool,
        remember_for_session: bool,
        message: Option<String>,
    ) -> ApprovalResolveStatus {
        let mut state = self.inner.lock().unwrap();
        let Some(pending) = state.pending.remove(approval_id) else {
            if !state.notified.remove(approval_id) {
                return ApprovalResolveStatus::NotFound;
            }
            state.early_decisions.insert(
                approval_id.to_string(),
                ApprovalDecisionState {
                    approve,
                    remember_for_session,
                    message,
                },
            );
            return ApprovalResolveStatus::Queued;
        };

        state.notified.remove(approval_id);
        if approve && remember_for_session {
            state.approved_sessions.insert(pending.session_id);
        }
        let _ = pending.tx.send(AgentToolApprovalDecision {
            approve,
            remember_for_session,
            message,
        });
        ApprovalResolveStatus::Resolved
    }

    fn is_session_approved(&self, session_id: &str) -> bool {
        self.inner
            .lock()
            .unwrap()
            .approved_sessions
            .contains(session_id)
    }
}

enum ApprovalResolveStatus {
    Resolved,
    Queued,
    NotFound,
}

#[derive(Deserialize)]
pub struct ApprovalDecisionRequest {
    pub decision: ApprovalDecision,
    #[serde(default)]
    pub remember_for_session: bool,
    pub message: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approve,
    Deny,
}

#[derive(Serialize)]
pub struct ApprovalDecisionResponse {
    pub status: &'static str,
}

pub async fn resolve_agent_approval(
    Extension(broker): Extension<ApprovalBroker>,
    Path(approval_id): Path<String>,
    Json(request): Json<ApprovalDecisionRequest>,
) -> impl IntoResponse {
    let approve = matches!(request.decision, ApprovalDecision::Approve);
    let status = broker.resolve(
        &approval_id,
        approve,
        request.remember_for_session,
        request.message,
    );
    let (status_code, status) = match status {
        ApprovalResolveStatus::Resolved => (StatusCode::OK, "resolved"),
        ApprovalResolveStatus::Queued => (StatusCode::OK, "queued"),
        ApprovalResolveStatus::NotFound => (StatusCode::NOT_FOUND, "not_found"),
    };
    (status_code, Json(ApprovalDecisionResponse { status }))
}

#[cfg(test)]
mod tests {
    use mistralrs_core::{AgentToolKind, AgentToolMetadata, AgentToolSource};

    use super::*;

    #[test]
    fn unknown_approval_id_is_not_found() {
        let broker = ApprovalBroker::default();

        assert!(matches!(
            broker.resolve("missing", true, false, None),
            ApprovalResolveStatus::NotFound
        ));
    }

    #[test]
    fn early_http_decision_unblocks_callback() {
        let broker = ApprovalBroker::default();
        let approval_id = "appr_test".to_string();
        let session_id = "session".to_string();
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let notifier = broker.notifier(tx);

        notifier(AgentToolApprovalRequest {
            approval_id: approval_id.clone(),
            session_id: session_id.clone(),
            round: 0,
            tool: AgentToolMetadata {
                source: AgentToolSource::Mistralrs,
                kind: AgentToolKind::CodeExecution,
                label: "Python code".to_string(),
            },
            arguments: serde_json::json!({"code": "print('hello')"}),
        });

        assert!(matches!(
            rx.try_recv().unwrap(),
            Response::AgenticToolApprovalRequired { .. }
        ));
        assert!(matches!(
            broker.resolve(&approval_id, true, false, None),
            ApprovalResolveStatus::Queued
        ));

        let callback = broker.callback();
        assert!(
            callback(&AgentToolApproval {
                approval_id,
                session_id,
                round: 0,
                tool: AgentToolMetadata {
                    source: AgentToolSource::Mistralrs,
                    kind: AgentToolKind::CodeExecution,
                    label: "Python code".to_string(),
                },
                arguments: serde_json::json!({"code": "print('hello')"}),
            })
            .approve
        );
    }
}
