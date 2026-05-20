use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    extract::{Json, Path},
    http::StatusCode,
    response::IntoResponse,
    Extension,
};
use mistralrs_core::{
    AgentToolApproval, AgentToolApprovalAsyncCallback, AgentToolApprovalDecision,
    AgentToolApprovalNotifier, AgentToolApprovalRequest, Response,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc::Sender, oneshot};

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
    tx: oneshot::Sender<AgentToolApprovalDecision>,
}

#[derive(Clone)]
struct ApprovalDecisionState {
    approve: bool,
    remember_for_session: bool,
    message: Option<String>,
}

impl ApprovalBroker {
    pub fn callback(&self) -> AgentToolApprovalAsyncCallback {
        let broker = self.clone();
        Arc::new(move |approval| {
            let broker = broker.clone();
            Box::pin(async move { broker.wait_for_decision(approval).await })
        })
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

    async fn wait_for_decision(&self, approval: AgentToolApproval) -> AgentToolApprovalDecision {
        if self.is_session_approved(&approval.session_id) {
            return AgentToolApprovalDecision::approve();
        }

        let (tx, rx) = oneshot::channel();
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

        let decision = tokio::time::timeout(APPROVAL_TIMEOUT, rx)
            .await
            .ok()
            .and_then(Result::ok)
            .unwrap_or_else(|| AgentToolApprovalDecision {
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

    const TEST_PENDING_WAIT_TIMEOUT: Duration = Duration::from_secs(1);
    const TEST_PENDING_WAIT_RETRY: Duration = Duration::from_millis(1);

    async fn wait_for_pending(broker: &ApprovalBroker, approval_id: &str) {
        tokio::time::timeout(TEST_PENDING_WAIT_TIMEOUT, async {
            loop {
                if broker
                    .inner
                    .lock()
                    .unwrap()
                    .pending
                    .contains_key(approval_id)
                {
                    return;
                }
                tokio::time::sleep(TEST_PENDING_WAIT_RETRY).await;
            }
        })
        .await
        .unwrap();
    }

    #[test]
    fn unknown_approval_id_is_not_found() {
        let broker = ApprovalBroker::default();

        assert!(matches!(
            broker.resolve("missing", true, false, None),
            ApprovalResolveStatus::NotFound
        ));
    }

    #[tokio::test]
    async fn early_http_decision_unblocks_callback() {
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
                source: AgentToolSource::BuiltIn,
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
            callback(AgentToolApproval {
                approval_id,
                session_id,
                round: 0,
                tool: AgentToolMetadata {
                    source: AgentToolSource::BuiltIn,
                    kind: AgentToolKind::CodeExecution,
                    label: "Python code".to_string(),
                },
                arguments: serde_json::json!({"code": "print('hello')"}),
            })
            .await
            .approve
        );
    }

    #[tokio::test]
    async fn http_decision_resolves_waiting_callback() {
        let broker = ApprovalBroker::default();
        let approval_id = "appr_waiting".to_string();
        let session_id = "session".to_string();
        let callback = broker.callback();

        let decision_task = tokio::spawn({
            let approval_id = approval_id.clone();
            let session_id = session_id.clone();
            async move {
                callback(AgentToolApproval {
                    approval_id,
                    session_id,
                    round: 0,
                    tool: AgentToolMetadata {
                        source: AgentToolSource::BuiltIn,
                        kind: AgentToolKind::CodeExecution,
                        label: "Python code".to_string(),
                    },
                    arguments: serde_json::json!({"code": "print('hello')"}),
                })
                .await
            }
        });

        wait_for_pending(&broker, &approval_id).await;

        assert!(matches!(
            broker.resolve(&approval_id, true, true, None),
            ApprovalResolveStatus::Resolved
        ));
        let decision = decision_task.await.unwrap();
        assert!(decision.approve);
        assert!(decision.remember_for_session);
    }
}
