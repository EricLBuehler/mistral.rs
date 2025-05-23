use anyhow::Result;
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, get_service, post},
    Json, Router,
};
use futures_util::stream::StreamExt;
use mistralrs::{
    IsqType, Model, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
    VisionModelBuilder,
};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::{net::TcpListener, sync::RwLock};
use tower_http::services::ServeDir;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    let app_state = AppState {
        model: Arc::new(RwLock::new(None)),
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/load_model", post(load_model))
        // serve static files from ./static at the root
        .nest_service(
            "/",
            get_service(ServeDir::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/static"
            ))),
        )
        .with_state(app_state);

    let addr: SocketAddr = ([127, 0, 0, 1], 3000).into();
    let listener = TcpListener::bind(addr).await?;
    info!("🔌 listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Application-wide shared state.
#[derive(Clone)]
struct AppState {
    model: Arc<RwLock<Option<Arc<Model>>>>,
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
    kind: String,
}

async fn load_model(
    State(app): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let result = (|| async {
        match req.kind.as_str() {
            "text" => TextModelBuilder::new(req.path)
                .with_isq(IsqType::Q8_0)
                .with_logging()
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                .build()
                .await
                .map(|m| Arc::new(m) as Arc<Model>),
            "vision" => VisionModelBuilder::new(req.path)
                .with_isq(IsqType::Q8_0)
                .with_logging()
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                .build()
                .await
                .map(|m| Arc::new(m) as Arc<Model>),
            _ => Err(anyhow::anyhow!("unknown model kind")),
        }
    })()
    .await;

    match result {
        Ok(model) => {
            let mut guard = app.model.write().await;
            *guard = Some(model);
            (StatusCode::OK, "Model loaded successfully").into_response()
        }
        Err(e) => {
            error!("load error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response()
        }
    }
}

/// Upgrades an HTTP request to a WebSocket connection.
async fn ws_handler(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

/// Per-connection task.
async fn handle_socket(mut socket: WebSocket, app: AppState) {
    // Store the running chat history for this connection
    let mut history = Vec::new();

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        // --- Add user message to history
        history.push((TextMessageRole::User, user_msg.clone()));

        // --- Build TextMessages from history
        let mut msgs = TextMessages::new();
        for (role, text) in &history {
            msgs = msgs.add_message(role.clone(), text);
        }

        let model_opt = { app.model.read().await.clone() };
        let Some(model) = model_opt else {
            let _ = socket
                .send(Message::Text(
                    "No model loaded. Use the UI to load one.".into(),
                ))
                .await;
            continue;
        };
        // --- Call mistralrs streaming API
        match model.stream_chat_request(msgs).await {
            Ok(mut stream) => {
                // Add the assistant role’s running content to `history`
                let mut assistant_reply = String::new();

                while let Some(chunk) = stream.next().await {
                    if let mistralrs::Response::Chunk(resp) = chunk {
                        if let Some(choice) = resp.choices.first() {
                            if let Some(token) = &choice.delta.content {
                                // Send token down the wire
                                if socket.send(Message::Text(token.clone())).await.is_err() {
                                    return;
                                }
                                assistant_reply.push_str(token);
                            }
                        }
                    }
                }
                history.push((TextMessageRole::Assistant, assistant_reply));
            }
            Err(e) => {
                error!("stream error: {}", e);
                let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
            }
        }
    }
}
