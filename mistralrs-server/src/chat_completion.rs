use serde_json::Value;
use std::{
    collections::HashMap,
    env,
    error::Error,
    ops::Deref,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::{
    openai::{ChatCompletionRequest, Grammar, MessageInnerContent, StopTokens},
    util,
};
use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use either::Either;
use indexmap::IndexMap;
use mistralrs_core::{
    ChatCompletionResponse, Constraint, DrySamplingParams, MistralRs, NormalRequest, Request,
    RequestMessage, Response, SamplingParams, StopTokens as InternalStopTokens,
};
use serde::Serialize;

#[derive(Debug)]
struct ModelErrorMessage(String);
impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ModelErrorMessage {}

enum DoneState {
    Running,
    SendingDone,
    Done,
}

pub struct Streamer {
    rx: Receiver<Response>,
    done_state: DoneState,
    state: Arc<MistralRs>,
}

impl futures::Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
                // https://platform.openai.com/docs/api-reference/completions/create
                // If true, returns a stream of events that happen during the Run as server-sent events, terminating when the Run enters a terminal state with a data: [DONE] message.
                self.done_state = DoneState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            DoneState::Done => {
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::ModelError(msg, _) => {
                    MistralRs::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.to_string()),
                    );
                    // Done now, just need to send the [DONE]
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(Event::default().data(msg))))
                }
                Response::ValidationError(e) => {
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::Chunk(response) => {
                    if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                        self.done_state = DoneState::SendingDone;
                    }
                    // Done now, just need to send the [DONE]
                    MistralRs::maybe_log_response(self.state.clone(), &response);
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::Raw { .. } => unreachable!(),
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

pub enum ChatCompletionResponder {
    Sse(Sse<Streamer>),
    Json(ChatCompletionResponse),
    ModelError(String, ChatCompletionResponse),
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
}

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

#[derive(Serialize)]
struct JsonModelError {
    message: String,
    partial_response: ChatCompletionResponse,
}

impl JsonModelError {
    fn new(message: String, partial_response: ChatCompletionResponse) -> Self {
        Self {
            message,
            partial_response,
        }
    }
}

impl ErrorToResponse for JsonModelError {}

impl IntoResponse for ChatCompletionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::Json(s) => Json(s).into_response(),
            ChatCompletionResponder::InternalError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatCompletionResponder::ValidationError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatCompletionResponder::ModelError(msg, response) => {
                JsonModelError::new(msg, response)
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

async fn parse_request(
    oairequest: ChatCompletionRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        None => None,
    };
    let messages = match oairequest.messages {
        Either::Left(req_messages) => {
            let mut messages = Vec::new();
            let mut image_urls = Vec::new();
            for message in req_messages {
                match message.content.deref() {
                    Either::Left(content) => {
                        // Handle tool call
                        let content = match content {
                            Some(content) => content.to_string(),
                            None => {
                                use anyhow::Context;
                                let calls = message.tool_calls.as_ref()
                                    .context("No content was provided, expected tool calls to be provided.")?
                                    .iter().map(|call| &call.function).collect::<Vec<_>>();

                                serde_json::to_string(&calls)?
                            }
                        };

                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role));
                        message_map.insert("content".to_string(), Either::Left(content));
                        messages.push(message_map);
                    }
                    Either::Right(image_messages) => {
                        // If there is only one message, it is possible a text message
                        // found when rig is used as client. In this case, we need to check if
                        // the message is a text message or an image message.
                        if image_messages.len() == 1 {
                            if !image_messages[0].contains_key("text") {
                                anyhow::bail!("Expected `text` key in input message.");
                            }
                            let content = match image_messages[0]["text"].deref() {
                                Either::Left(left) => left.to_string(),
                                Either::Right(right) => format!("{:?}", right),
                            };
                            let mut message_map: IndexMap<
                                String,
                                Either<String, Vec<IndexMap<String, Value>>>,
                            > = IndexMap::new();
                            message_map.insert("role".to_string(), Either::Left(message.role));
                            message_map.insert("content".to_string(), Either::Left(content));
                            messages.push(message_map);
                            continue;
                        }
                        if image_messages.len() != 2 {
                            anyhow::bail!(
                                "Expected 2 items for the content of a message with an image."
                            );
                        }
                        if message.role != "user" {
                            anyhow::bail!(
                                "Role for an image message must be `user`, but it is {}",
                                message.role
                            );
                        }

                        let mut items = Vec::new();
                        for image_message in image_messages {
                            if image_message.len() != 2 {
                                anyhow::bail!("Expected 2 items for the sub-content of a message with an image.");
                            }
                            if !image_message.contains_key("type") {
                                anyhow::bail!("Expected `type` key in input message.");
                            }
                            if image_message["type"].is_right() {
                                anyhow::bail!("Expected string value in `type`.");
                            }
                            items.push(image_message["type"].as_ref().unwrap_left().clone())
                        }

                        fn get_content_and_url(
                            text_idx: usize,
                            url_idx: usize,
                            image_messages: &[HashMap<String, MessageInnerContent>],
                        ) -> Result<(String, String)> {
                            if image_messages[text_idx]["text"].is_right() {
                                anyhow::bail!("Expected string value in `text`.");
                            }
                            let content = image_messages[text_idx]["text"]
                                .as_ref()
                                .unwrap_left()
                                .clone();
                            if image_messages[url_idx]["image_url"].is_left()
                                || !image_messages[url_idx]["image_url"]
                                    .as_ref()
                                    .unwrap_right()
                                    .contains_key("url")
                            {
                                anyhow::bail!("Expected content of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}")
                            }
                            let url = image_messages[url_idx]["image_url"].as_ref().unwrap_right()
                                ["url"]
                                .clone();
                            Ok((content, url))
                        }
                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role));
                        let (content, url) = if items[0] == "text" {
                            get_content_and_url(0, 1, image_messages)?
                        } else {
                            get_content_and_url(1, 0, image_messages)?
                        };

                        let mut content_map: Vec<IndexMap<String, Value>> = Vec::new();
                        let mut content_image_map = IndexMap::new();
                        content_image_map
                            .insert("type".to_string(), Value::String("image".to_string()));
                        content_map.push(content_image_map);
                        let mut content_text_map = IndexMap::new();
                        content_text_map
                            .insert("type".to_string(), Value::String("text".to_string()));
                        content_text_map.insert("text".to_string(), Value::String(content));
                        content_map.push(content_text_map);

                        message_map.insert("content".to_string(), Either::Right(content_map));
                        messages.push(message_map);
                        image_urls.push(url);
                    }
                }
            }
            if !image_urls.is_empty() {
                let mut images = Vec::new();
                for url_unparsed in image_urls {
                    use anyhow::Context;
                    let image = util::parse_image_url(&url_unparsed)
                        .await
                        .context(format!("Failed to parse image resource: {}", url_unparsed))?;

                    images.push(image);
                }
                RequestMessage::VisionChat { messages, images }
            } else {
                RequestMessage::Chat(messages)
            }
        }
        Either::Right(prompt) => {
            let mut messages = Vec::new();
            let mut message_map: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message_map.insert("role".to_string(), Either::Left("user".to_string()));
            message_map.insert("content".to_string(), Either::Left(prompt));
            messages.push(message_map);
            RequestMessage::Chat(messages)
        }
    };

    let dry_params = if let Some(dry_multiplier) = oairequest.dry_multiplier {
        Some(DrySamplingParams::new_with_defaults(
            dry_multiplier,
            oairequest.dry_sequence_breakers,
            oairequest.dry_base,
            oairequest.dry_allowed_length,
        )?)
    } else {
        None
    };

    let is_streaming = oairequest.stream.unwrap_or(false);
    Ok((
        Request::Normal(NormalRequest {
            id: state.next_request_id(),
            messages,
            sampling_params: SamplingParams {
                temperature: oairequest.temperature,
                top_k: oairequest.top_k,
                top_p: oairequest.top_p,
                min_p: oairequest.min_p,
                top_n_logprobs: oairequest.top_logprobs.unwrap_or(1),
                frequency_penalty: oairequest.frequency_penalty,
                presence_penalty: oairequest.presence_penalty,
                max_len: oairequest.max_tokens,
                stop_toks,
                logits_bias: oairequest.logit_bias,
                n_choices: oairequest.n_choices,
                dry_params,
            },
            response: tx,
            return_logprobs: oairequest.logprobs,
            is_streaming,
            suffix: None,
            constraint: match oairequest.grammar {
                Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
                Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
                Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
                Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
                None => Constraint::None,
            },
            adapters: oairequest.adapters,
            tool_choice: oairequest.tool_choice,
            tools: oairequest.tools,
            logits_processors: None,
            return_raw_logits: false,
        }),
        is_streaming,
    ))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chatcompletions(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<ChatCompletionRequest>,
) -> ChatCompletionResponder {
    let (tx, mut rx) = channel(10_000);
    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx).await {
        Ok(x) => x,
        Err(e) => {
            let e = anyhow::Error::msg(e.to_string());
            MistralRs::maybe_log_error(state, &*e);
            return ChatCompletionResponder::InternalError(e.into());
        }
    };
    let sender = state.get_sender().unwrap();

    if let Err(e) = sender.send(request).await {
        let e = anyhow::Error::msg(e.to_string());
        MistralRs::maybe_log_error(state, &*e);
        return ChatCompletionResponder::InternalError(e.into());
    }

    if is_streaming {
        let streamer = Streamer {
            rx,
            done_state: DoneState::Running,
            state,
        };

        let keep_alive_interval = env::var("KEEP_ALIVE_INTERVAL")
            .map(|val| val.parse::<u64>().unwrap_or(10000))
            .unwrap_or(10000);
        ChatCompletionResponder::Sse(
            Sse::new(streamer)
                .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval))),
        )
    } else {
        let response = match rx.recv().await {
            Some(response) => response,
            None => {
                let e = anyhow::Error::msg("No response received from the model.");
                MistralRs::maybe_log_error(state, &*e);
                return ChatCompletionResponder::InternalError(e.into());
            }
        };

        match response {
            Response::InternalError(e) => {
                MistralRs::maybe_log_error(state, &*e);
                ChatCompletionResponder::InternalError(e)
            }
            Response::ModelError(msg, response) => {
                MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
                MistralRs::maybe_log_response(state, &response);
                ChatCompletionResponder::ModelError(msg, response)
            }
            Response::ValidationError(e) => ChatCompletionResponder::ValidationError(e),
            Response::Done(response) => {
                MistralRs::maybe_log_response(state, &response);
                ChatCompletionResponder::Json(response)
            }
            Response::Chunk(_) => unreachable!(),
            Response::CompletionDone(_) => unreachable!(),
            Response::CompletionModelError(_, _) => unreachable!(),
            Response::CompletionChunk(_) => unreachable!(),
            Response::ImageGeneration(_) => unreachable!(),
            Response::Raw { .. } => unreachable!(),
        }
    }
}
