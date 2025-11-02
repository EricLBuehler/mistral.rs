//! OpenAI-compatible embeddings endpoint.

use anyhow::{anyhow, Context, Error as AnyhowError, Result};
use axum::{
    extract::{Json, State},
    http,
    response::IntoResponse,
};
use base64::{prelude::BASE64_STANDARD, Engine};
use futures::future::join_all;
use mistralrs_core::{
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::Receiver;

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request_with_model,
        ErrorToResponse, JsonError,
    },
    openai::{
        EmbeddingData, EmbeddingEncodingFormat, EmbeddingInput, EmbeddingRequest,
        EmbeddingResponse, EmbeddingUsage, EmbeddingVector,
    },
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::{sanitize_error_message, validate_model_name},
};

/// Represents different types of embeddings responses.
pub enum EmbeddingResponder {
    Json(EmbeddingResponse),
    InternalError(AnyhowError),
    ValidationError(AnyhowError),
}

struct EmbeddingWithUsage {
    embedding: Vec<f32>,
    prompt_tokens: usize,
    total_tokens: usize,
}

impl IntoResponse for EmbeddingResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            EmbeddingResponder::Json(s) => Json(s).into_response(),
            EmbeddingResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.root_cause()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            EmbeddingResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.root_cause()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/embeddings",
    request_body = EmbeddingRequest,
    responses((status = 200, description = "Embeddings", body = EmbeddingResponse))
)]
pub async fn embeddings(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<EmbeddingRequest>,
) -> EmbeddingResponder {
    let repr =
        serde_json::to_string(&oairequest).expect("Serialization of embedding request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    if let Err(e) = validate_model_name(&oairequest.model, state.clone()) {
        return validation_error(e);
    }

    if let Some(dimensions) = oairequest.dimensions {
        return validation_error(anyhow!(
            "Custom embedding dimensions ({dimensions}) are not supported."
        ));
    }

    let inputs = match normalize_inputs(oairequest.input) {
        Ok(inputs) => inputs,
        Err(e) => return validation_error(e),
    };

    if inputs.is_empty() {
        return validation_error(anyhow!("input must contain at least one entry."));
    }

    let model_override = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let encoding = oairequest.encoding_format.unwrap_or_default();
    let return_base64 = matches!(encoding, EmbeddingEncodingFormat::Base64);

    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens: usize = 0;
    let mut total_tokens: usize = 0;

    match inputs {
        Inputs::Prompt(prompts) => {
            let futures = prompts.into_iter().map(|prompt| {
                let state = state.clone();
                let model_override = model_override.clone();
                async move {
                    fetch_embedding(
                        state,
                        prompt,
                        model_override.as_deref(),
                        oairequest.truncate_sequence.unwrap_or(false),
                    )
                    .await
                }
            });

            let results = join_all(futures).await;
            for (index, result) in results.into_iter().enumerate() {
                match result {
                    Ok(EmbeddingWithUsage {
                        embedding,
                        prompt_tokens,
                        total_tokens: item_total_tokens,
                    }) => {
                        let embedding = if return_base64 {
                            EmbeddingVector::Base64(encode_embedding_base64(&embedding))
                        } else {
                            EmbeddingVector::Float(embedding)
                        };
                        data.push(EmbeddingData {
                            object: "embedding",
                            embedding,
                            index,
                        });
                        total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens);
                        total_tokens = total_tokens.saturating_add(item_total_tokens);
                    }
                    Err(e) => {
                        MistralRs::maybe_log_error(state.clone(), e.as_ref());
                        return internal_error(e);
                    }
                }
            }
        }
        Inputs::Tokens(batches) => {
            let futures = batches.into_iter().map(|tokens| {
                let state = state.clone();
                let model_override = model_override.clone();
                async move {
                    fetch_embedding_tokens(
                        state,
                        tokens,
                        model_override.as_deref(),
                        oairequest.truncate_sequence.unwrap_or(false),
                    )
                    .await
                }
            });

            let results = join_all(futures).await;
            for (index, result) in results.into_iter().enumerate() {
                match result {
                    Ok(EmbeddingWithUsage {
                        embedding,
                        prompt_tokens,
                        total_tokens: item_total_tokens,
                    }) => {
                        let embedding = if return_base64 {
                            EmbeddingVector::Base64(encode_embedding_base64(&embedding))
                        } else {
                            EmbeddingVector::Float(embedding)
                        };
                        data.push(EmbeddingData {
                            object: "embedding",
                            embedding,
                            index,
                        });
                        total_prompt_tokens = total_prompt_tokens.saturating_add(prompt_tokens);
                        total_tokens = total_tokens.saturating_add(item_total_tokens);
                    }
                    Err(e) => {
                        MistralRs::maybe_log_error(state.clone(), e.as_ref());
                        return internal_error(e);
                    }
                }
            }
        }
    }

    let usage = EmbeddingUsage {
        prompt_tokens: saturating_to_u32(total_prompt_tokens),
        total_tokens: saturating_to_u32(total_tokens),
    };

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: oairequest.model,
        usage,
    };

    MistralRs::maybe_log_response(state.clone(), &response);

    EmbeddingResponder::Json(response)
}

enum Inputs {
    Prompt(Vec<String>),
    Tokens(Vec<Vec<u32>>),
}

impl Inputs {
    fn is_empty(&self) -> bool {
        match self {
            Self::Prompt(x) => x.is_empty(),
            Self::Tokens(x) => x.is_empty(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Prompt(x) => x.len(),
            Self::Tokens(x) => x.len(),
        }
    }
}

fn normalize_inputs(input: EmbeddingInput) -> Result<Inputs> {
    match input {
        EmbeddingInput::Single(s) => Ok(Inputs::Prompt(vec![s])),
        EmbeddingInput::Multiple(items) => Ok(Inputs::Prompt(items)),
        EmbeddingInput::Tokens(t) => Ok(Inputs::Tokens(vec![t])),
        EmbeddingInput::TokensBatch(batch) => Ok(Inputs::Tokens(batch)),
    }
}

async fn fetch_embedding(
    state: SharedMistralRsState,
    prompt: String,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<EmbeddingWithUsage> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::Embedding { prompt },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: model_id.map(|m| m.to_string()),
        truncate_sequence,
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch embedding request")?;

    process_embedding_response(&mut rx, state.clone()).await
}

async fn fetch_embedding_tokens(
    state: SharedMistralRsState,
    tokens: Vec<u32>,
    model_id: Option<&str>,
    truncate_sequence: bool,
) -> Result<EmbeddingWithUsage> {
    let (tx, mut rx) = create_response_channel(Some(1));

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::EmbeddingTokens { prompt: tokens },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: model_id.map(|m| m.to_string()),
        truncate_sequence,
    }));

    send_request_with_model(&state, request, model_id)
        .await
        .context("Failed to dispatch embedding request")?;

    process_embedding_response(&mut rx, state.clone()).await
}

async fn process_embedding_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> Result<EmbeddingWithUsage> {
    base_process_non_streaming_response(
        rx,
        state.clone(),
        |_, response| match response {
            Response::Embeddings {
                embeddings,
                prompt_tokens,
                total_tokens,
            } => Ok(EmbeddingWithUsage {
                embedding: embeddings,
                prompt_tokens,
                total_tokens,
            }),
            Response::ValidationError(e) | Response::InternalError(e) => Err(anyhow!(e)),
            Response::ModelError(msg, _) => Err(anyhow!(msg)),
            Response::Done(_)
            | Response::Chunk(_)
            | Response::CompletionDone(_)
            | Response::CompletionChunk(_)
            | Response::CompletionModelError(_, _)
            | Response::ImageGeneration(_)
            | Response::Speech { .. }
            | Response::Raw { .. } => Err(anyhow!(
                "Received unexpected response type from embedding request."
            )),
        },
        |_, err| Err(anyhow!(err)),
    )
    .await
}

fn validation_error<E>(err: E) -> EmbeddingResponder
where
    E: Into<AnyhowError>,
{
    let err = err.into();
    EmbeddingResponder::ValidationError(err)
}

fn internal_error<E>(err: E) -> EmbeddingResponder
where
    E: Into<AnyhowError>,
{
    let err = err.into();
    EmbeddingResponder::InternalError(err)
}

fn encode_embedding_base64(embedding: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(embedding));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    BASE64_STANDARD.encode(bytes)
}

fn saturating_to_u32(value: usize) -> u32 {
    if value > u32::MAX as usize {
        u32::MAX
    } else {
        value as u32
    }
}
