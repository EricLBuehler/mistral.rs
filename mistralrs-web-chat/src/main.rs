use anyhow::Result;
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::{get, get_service},
    Router,
};
use futures_util::stream::StreamExt;
use mistralrs::{
    IsqType, Model, PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tower_http::services::ServeDir;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<()> {
    // ----  load a model once and share it behind Arc<RwLock<..>> ----
    let model = TextModelBuilder::new("../hf_models/llama3.2_3b".to_string())
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?; //  [oai_citation:0‡ericlbuehler.github.io](https://ericlbuehler.github.io/mistral.rs/mistralrs/)

    let app_state = AppState {
        model: Arc::new(model),
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        // serve static files from ./static at the root
        .nest_service(
            "/",
            get_service(
                ServeDir::new(concat!(env!("CARGO_MANIFEST_DIR"), "/static"))
            )
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
    model: Arc<Model>,
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

        // --- Call mistralrs streaming API
        match app.model.stream_chat_request(msgs).await {
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
