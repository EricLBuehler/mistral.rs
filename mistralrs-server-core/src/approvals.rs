use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    extract::{rejection::JsonRejection, Json, Path},
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
use utoipa::ToSchema;

use crate::handler_core::{openai_error_response, ApiError, ApiErrorKind};

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

/// Decision payload for a pending agentic tool approval.
#[derive(Deserialize, ToSchema)]
pub struct ApprovalDecisionRequest {
    pub decision: ApprovalDecision,
    /// Auto-approve all later tool calls in the same session.
    #[serde(default)]
    pub remember_for_session: bool,
    /// Optional note passed back to the model on denial.
    pub message: Option<String>,
}

#[derive(Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approve,
    Deny,
}

#[derive(Serialize, ToSchema)]
pub struct ApprovalDecisionResponse {
    /// "resolved" or "queued".
    pub status: &'static str,
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/agent/approvals/{approval_id}",
    params(("approval_id" = String, Path, description = "Approval ID from the approval-required SSE event")),
    request_body = ApprovalDecisionRequest,
    responses(
        (status = 200, description = "Decision applied or queued", body = ApprovalDecisionResponse),
        (status = 400, description = "Invalid decision payload"),
        (status = 404, description = "Unknown approval ID"),
        (status = 413, description = "Decision payload is too large"),
        (status = 415, description = "Unsupported content type"),
    )
)]
pub async fn resolve_agent_approval(
    Extension(broker): Extension<ApprovalBroker>,
    Path(approval_id): Path<String>,
    payload: Result<Json<ApprovalDecisionRequest>, JsonRejection>,
) -> axum::response::Response {
    let Json(request) = match payload {
        Ok(request) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    let approve = matches!(request.decision, ApprovalDecision::Approve);
    let status = broker.resolve(
        &approval_id,
        approve,
        request.remember_for_session,
        request.message,
    );
    let status = match status {
        ApprovalResolveStatus::Resolved => "resolved",
        ApprovalResolveStatus::Queued => "queued",
        ApprovalResolveStatus::NotFound => {
            return openai_error_response(ApiError::new(
                ApiErrorKind::NotFound,
                format!("Approval `{approval_id}` was not found."),
                Some("approval_not_found"),
                Some("approval_id"),
            ));
        }
    };
    (StatusCode::OK, Json(ApprovalDecisionResponse { status })).into_response()
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{to_bytes, Body},
        extract::FromRequest,
        http::{header::CONTENT_TYPE, Request as HttpRequest},
    };
    use mistralrs_core::{AgentToolKind, AgentToolMetadata, AgentToolSource};

    use super::*;

    const TEST_PENDING_WAIT_TIMEOUT: Duration = Duration::from_secs(1);
    const TEST_PENDING_WAIT_RETRY: Duration = Duration::from_millis(1);

    async fn error_body(response: axum::response::Response) -> serde_json::Value {
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    async fn approval_json_rejection(
        body: &'static str,
        content_type: Option<&'static str>,
    ) -> JsonRejection {
        let mut builder = HttpRequest::builder();
        if let Some(content_type) = content_type {
            builder = builder.header(CONTENT_TYPE, content_type);
        }
        let request = builder.body(Body::from(body)).unwrap();
        match Json::<ApprovalDecisionRequest>::from_request(request, &()).await {
            Ok(_) => panic!("expected JSON rejection"),
            Err(error) => error,
        }
    }

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
    async fn approval_json_rejections_use_openai_errors() {
        let cases = [
            (
                approval_json_rejection("{", Some("application/json")).await,
                StatusCode::BAD_REQUEST,
                "malformed_json",
            ),
            (
                approval_json_rejection("{}", Some("application/json")).await,
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
            ),
            (
                approval_json_rejection(r#"{"decision":"approve"}"#, None).await,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
            ),
        ];

        for (rejection, status, code) in cases {
            let response = resolve_agent_approval(
                Extension(ApprovalBroker::default()),
                Path("approval".to_string()),
                Err(rejection),
            )
            .await;
            assert_eq!(response.status(), status);
            let body = error_body(response).await;
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["code"], code);
        }
    }

    #[tokio::test]
    async fn unknown_approval_response_uses_openai_error() {
        let response = resolve_agent_approval(
            Extension(ApprovalBroker::default()),
            Path("missing".to_string()),
            Ok(Json(ApprovalDecisionRequest {
                decision: ApprovalDecision::Approve,
                remember_for_session: false,
                message: None,
            })),
        )
        .await;

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = error_body(response).await;
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "approval_not_found");
        assert_eq!(body["error"]["param"], "approval_id");
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
