use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{sink::SinkExt, stream::StreamExt};
use mistralrs_core::{
    AudioInput, Constraint, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::warn;
use uuid::Uuid;

use crate::handler_core::create_response_channel;
use crate::types::SharedMistralRsState;

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate { session: Value },
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend { audio: String },
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit {},
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear {},
    #[serde(rename = "response.create")]
    ResponseCreate { response: Option<Value> },
    #[serde(rename = "response.cancel")]
    ResponseCancel {},
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "error")]
    Error { error: Value },
    #[serde(rename = "session.created")]
    SessionCreated { session: Value },
    #[serde(rename = "session.updated")]
    SessionUpdated { session: Value },
    #[serde(rename = "input_audio_buffer.committed")]
    InputAudioBufferCommitted { item_id: String },
    #[serde(rename = "input_audio_buffer.cleared")]
    InputAudioBufferCleared {},
    #[serde(rename = "response.created")]
    ResponseCreated { response: Value },
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta {
        delta: String,
        response_id: String,
        item_id: String,
        output_index: usize,
        content_index: usize,
    },
    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta {
        delta: String,
        response_id: String,
        item_id: String,
        output_index: usize,
        content_index: usize,
    },
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta {
        delta: String,
        response_id: String,
        item_id: String,
        output_index: usize,
        content_index: usize,
    },
    #[serde(rename = "response.done")]
    ResponseDone { response: Value },
}

pub async fn realtime_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedMistralRsState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: SharedMistralRsState) {
    let (mut sender, mut receiver) = socket.split();
    let mut audio_buffer = Vec::new();
    let mut session_config = json!({
        "modalities": ["text", "audio"],
        "instructions": "You are a helpful assistant.",
    });

    // Send session.created
    let session_id = format!("sess_{}", Uuid::new_v4());
    let _ = sender
        .send(Message::Text(
            serde_json::to_string(&ServerEvent::SessionCreated {
                session: json!({
                    "id": session_id,
                    "object": "realtime.session",
                    "model": "default",
                    "modalities": ["text", "audio"],
                    "instructions": "You are a helpful assistant.",
                }),
            })
            .unwrap()
            .into(),
        ))
        .await;

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                let event: Result<ClientEvent, _> = serde_json::from_str(&text);
                match event {
                    Ok(ClientEvent::SessionUpdate { session }) => {
                        session_config = session.clone();
                        let _ = sender
                            .send(Message::Text(
                                serde_json::to_string(&ServerEvent::SessionUpdated {
                                    session: session_config.clone(),
                                })
                                .unwrap()
                                .into(),
                            ))
                            .await;
                    }
                    Ok(ClientEvent::InputAudioBufferAppend { audio }) => {
                        use base64::Engine;
                        if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(audio)
                        {
                            audio_buffer.extend(decoded);
                        }
                    }
                    Ok(ClientEvent::InputAudioBufferCommit {}) => {
                        let item_id = format!("item_{}", Uuid::new_v4());
                        let _ = sender
                            .send(Message::Text(
                                serde_json::to_string(&ServerEvent::InputAudioBufferCommitted {
                                    item_id,
                                })
                                .unwrap()
                                .into(),
                            ))
                            .await;
                    }
                    Ok(ClientEvent::InputAudioBufferClear {}) => {
                        audio_buffer.clear();
                        let _ = sender
                            .send(Message::Text(
                                serde_json::to_string(&ServerEvent::InputAudioBufferCleared {})
                                    .unwrap()
                                    .into(),
                            ))
                            .await;
                    }
                    Ok(ClientEvent::ResponseCreate { .. }) => {
                        let response_id = format!("resp_{}", Uuid::new_v4());
                        let _ = sender.send(Message::Text(serde_json::to_string(&ServerEvent::ResponseCreated {
                            response: json!({ "id": response_id, "object": "realtime.response", "status": "in_progress" }),
                        }).unwrap().into())).await;

                        // Process audio and send to engine
                        let audio_input = if !audio_buffer.is_empty() {
                            match AudioInput::from_bytes(&audio_buffer) {
                                Ok(input) => {
                                    audio_buffer.clear();
                                    Some(input)
                                }
                                Err(e) => {
                                    let _ = sender.send(Message::Text(serde_json::to_string(&ServerEvent::Error {
                                        error: json!({ "message": format!("Failed to decode audio: {}", e) }),
                                    }).unwrap().into())).await;
                                    None
                                }
                            }
                        } else {
                            None
                        };

                        let (tx, mut rx) = create_response_channel(None);

                        let instructions = session_config["instructions"]
                            .as_str()
                            .unwrap_or("You are a helpful assistant.")
                            .to_string();

                        let mut messages = Vec::new();

                        // System message
                        let mut system_msg = indexmap::IndexMap::new();
                        system_msg.insert(
                            "role".to_string(),
                            either::Either::Left("system".to_string()),
                        );
                        system_msg
                            .insert("content".to_string(), either::Either::Left(instructions));
                        messages.push(system_msg);

                        // User message
                        let mut user_msg = indexmap::IndexMap::new();
                        user_msg
                            .insert("role".to_string(), either::Either::Left("user".to_string()));
                        user_msg.insert(
                            "content".to_string(),
                            either::Either::Left(
                                "Please respond to the audio if provided.".to_string(),
                            ),
                        );
                        messages.push(user_msg);

                        let request = Request::Normal(Box::new(NormalRequest {
                            id: state.next_request_id(),
                            messages: RequestMessage::VisionChat {
                                images: Vec::new(),
                                audios: audio_input.map(|a| vec![a]).unwrap_or_default(),
                                messages,
                                enable_thinking: None,
                                reasoning_effort: None,
                            },
                            sampling_params: SamplingParams::deterministic(),
                            response: tx,
                            return_logprobs: false,
                            is_streaming: true,
                            suffix: None,
                            constraint: Constraint::None,
                            tool_choice: None,
                            tools: None,
                            logits_processors: None,
                            return_raw_logits: false,
                            web_search_options: None,
                            model_id: None,
                            truncate_sequence: false,
                        }));

                        if let Err(e) = state.send_request(request) {
                            let _ = sender.send(Message::Text(serde_json::to_string(&ServerEvent::Error {
                                error: json!({ "message": format!("Failed to send request: {}", e) }),
                            }).unwrap().into())).await;
                            continue;
                        }

                        // Stream responses back
                        while let Some(resp) = rx.recv().await {
                            match resp {
                                Response::Chunk(chunk) => {
                                    for choice in chunk.choices {
                                        if let Some(content) = choice.delta.content {
                                            let _ = sender
                                                .send(Message::Text(
                                                    serde_json::to_string(
                                                        &ServerEvent::ResponseTextDelta {
                                                            delta: content,
                                                            response_id: response_id.clone(),
                                                            item_id: "item_output".to_string(),
                                                            output_index: 0,
                                                            content_index: 0,
                                                        },
                                                    )
                                                    .unwrap()
                                                    .into(),
                                                ))
                                                .await;
                                        }
                                    }
                                }
                                Response::Done(done) => {
                                    let _ = sender.send(Message::Text(serde_json::to_string(&ServerEvent::ResponseDone {
                                        response: json!({ "id": response_id.clone(), "object": "realtime.response", "status": "completed", "usage": done.usage }),
                                    }).unwrap().into())).await;
                                    break;
                                }
                                Response::ModelError(e, _) => {
                                    let _ = sender.send(Message::Text(serde_json::to_string(&ServerEvent::Error {
                                        error: json!({ "message": format!("Model error: {}", e) }),
                                    }).unwrap().into())).await;
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    Ok(ClientEvent::ResponseCancel {}) => {
                        // TODO: Implement cancel logic if needed
                    }
                    Err(e) => {
                        let _ = sender
                            .send(Message::Text(
                                serde_json::to_string(&ServerEvent::Error {
                                    error: json!({ "message": format!("Invalid event: {}", e) }),
                                })
                                .unwrap()
                                .into(),
                            ))
                            .await;
                    }
                }
            }
            Message::Binary(_) => {
                warn!("Received unexpected binary message");
            }
            _ => {}
        }
    }
}
