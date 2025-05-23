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
    VisionMessages, VisionModelBuilder,
};
use serde::Deserialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::{net::TcpListener, sync::RwLock};
use tower_http::services::ServeDir;
use tracing::{error, info};

/// Distinguish at runtime which kind of model we have loaded.
#[derive(Clone)]
enum LoadedModel {
    Text(Arc<Model>),
    Vision(Arc<Model>),
}

#[derive(Clone)]
struct AppState {
    model: Arc<RwLock<Option<LoadedModel>>>,
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
    kind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initially no model is loaded
    let app_state = AppState {
        model: Arc::new(RwLock::new(None)),
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/load_model", post(load_model))
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

/// Loads a model from a path and kind.
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
                .map(|m| LoadedModel::Text(Arc::new(m))),
            "vision" => VisionModelBuilder::new(req.path)
                .with_isq(IsqType::Q8_0)
                .with_logging()
                .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                .build()
                .await
                .map(|m| LoadedModel::Vision(Arc::new(m))),
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

/// Generic helper to stream tokens and forward them to the websocket.
async fn stream_and_forward<Msgs, F>(
    model: &Arc<Model>,
    msgs: Msgs,
    socket: &mut WebSocket,
    mut on_assistant_complete: F,
) -> Result<(), anyhow::Error>
where
    Msgs: mistralrs::RequestLike + Send + 'static,
    F: FnMut(&str),
{
    match model.stream_chat_request(msgs).await {
        Ok(mut stream) => {
            let mut assistant_reply = String::new();
            while let Some(chunk) = stream.next().await {
                if let mistralrs::Response::Chunk(resp) = chunk {
                    if let Some(choice) = resp.choices.first() {
                        if let Some(token) = &choice.delta.content {
                            if socket.send(Message::Text(token.clone())).await.is_err() {
                                break;
                            }
                            assistant_reply.push_str(token);
                        }
                    }
                }
            }
            on_assistant_complete(&assistant_reply);
            Ok(())
        }
        Err(e) => {
            let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
            Err(e)
        }
    }
}

/// Per-connection task.
async fn handle_socket(mut socket: WebSocket, app: AppState) {
    let mut text_history: Vec<(TextMessageRole, String)> = Vec::new();
    let mut vision_history: Vec<(TextMessageRole, String)> = Vec::new();

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        let model_opt = { app.model.read().await.clone() };
        let Some(model_loaded) = model_opt else {
            let _ = socket
                .send(Message::Text(
                    "No model loaded. Use the UI to load one.".into(),
                ))
                .await;
            continue;
        };

        match model_loaded {
            LoadedModel::Text(model) => {
                // --- Update history and build TextMessages
                text_history.push((TextMessageRole::User, user_msg.clone()));
                let mut msgs = TextMessages::new();
                for (role, text) in &text_history {
                    msgs = msgs.add_message(role.clone(), text);
                }

                if let Err(e) = stream_and_forward(&model, msgs, &mut socket, |tok| {
                    text_history.push((TextMessageRole::Assistant, tok.to_string()));
                })
                .await
                {
                    error!("stream error: {}", e);
                }
            }
            LoadedModel::Vision(model) => {
                // --- Vision message flow
                vision_history.push((TextMessageRole::User, user_msg.clone()));
                let mut msgs = VisionMessages::new();
                for (role, text) in &vision_history {
                    msgs = msgs.add_message(role.clone(), text);
                }

                if let Err(e) = stream_and_forward(&model, msgs, &mut socket, |tok| {
                    vision_history.push((TextMessageRole::Assistant, tok.to_string()));
                })
                .await
                {
                    error!("stream error: {}", e);
                }
            }
        }
    }
}
