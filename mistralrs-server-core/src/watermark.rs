use std::future::Future;

use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::{IntoResponse, Response},
};
use either::Either;
use mistralrs_core::{Request, TokenizationRequest, Watermark, WatermarkConfig, WatermarkEvidence};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::{
    handler_core::{openai_error_response, send_request_with_model, ApiError, ApiErrorKind},
    types::ExtractedMistralRsState,
};

const TOKENIZATION_CHANNEL_CAPACITY: usize = 1;
const MAX_DETECTION_VOCAB_SIZE: usize = 1_048_576;

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct WatermarkDetectionRequest {
    pub watermark: WatermarkConfig,
    pub input: WatermarkDetectionInput,
    /// Prefix length in tokens, or sentences for embedding input; excluded from evidence.
    #[serde(default)]
    pub prompt_len: usize,
    /// Stop token scoring at the first generated EOS; only valid for token or text input.
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum WatermarkDetectionInput {
    Tokens {
        /// Original prompt-plus-generation vocabulary IDs.
        tokens: Vec<u32>,
    },
    Text {
        /// Complete text to tokenize; no chat template is applied.
        text: String,
        /// Serving model or alias used for tokenization.
        model: String,
        #[serde(default)]
        add_special_tokens: bool,
    },
    Embeddings {
        /// Sentence embeddings from the same encoder used during SemStamp generation.
        embeddings: Vec<Vec<f32>>,
    },
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/watermark/detect",
    request_body = WatermarkDetectionRequest,
    responses(
        (status = 200, description = "Uncalibrated, scheme-specific watermark evidence", body = WatermarkEvidence),
        (status = 400, description = "Invalid watermark configuration or detection input"),
        (status = 404, description = "Text tokenization model was not found"),
        (status = 413, description = "Request body is too large"),
        (status = 415, description = "Request content type is not JSON"),
        (status = 500, description = "Detection or tokenization failed"),
        (status = 503, description = "Tokenization model is unavailable")
    )
)]
pub async fn detect_watermark(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<WatermarkDetectionRequest>, JsonRejection>,
) -> Response {
    detection_response(payload, |request, model| async move {
        send_request_with_model(&state, Request::Tokenize(request), model.as_deref())
            .await
            .map_err(|error| ApiError::from_error(&error, ApiErrorKind::Internal))
    })
    .await
}

async fn detection_response<F, Fut>(
    payload: Result<Json<WatermarkDetectionRequest>, JsonRejection>,
    send_tokenization: F,
) -> Response
where
    F: FnOnce(TokenizationRequest, Option<String>) -> Fut,
    Fut: Future<Output = Result<(), ApiError>>,
{
    let request = match payload {
        Ok(Json(request)) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    match run_detection(request, send_tokenization).await {
        Ok(evidence) => Json(evidence).into_response(),
        Err(error) => openai_error_response(error),
    }
}

async fn run_detection<F, Fut>(
    mut request: WatermarkDetectionRequest,
    send_tokenization: F,
) -> Result<WatermarkEvidence, ApiError>
where
    F: FnOnce(TokenizationRequest, Option<String>) -> Fut,
    Fut: Future<Output = Result<(), ApiError>>,
{
    request
        .watermark
        .validate()
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
    // A small HTTP request must not allocate a U32-sized vocabulary table.
    if request
        .watermark
        .vocab_size()
        .is_some_and(|size| size > MAX_DETECTION_VOCAB_SIZE)
    {
        return Err(ApiError::invalid_request(format!(
            "Detection vocab_size must not exceed {MAX_DETECTION_VOCAB_SIZE}."
        )));
    }
    let semantic = matches!(request.watermark, WatermarkConfig::Semstamp { .. });
    if semantic != matches!(request.input, WatermarkDetectionInput::Embeddings { .. }) {
        return Err(ApiError::invalid_request(
            "SemStamp requires embeddings input; token schemes require tokens or text input.",
        ));
    }
    if semantic && !request.eos_token_ids.is_empty() {
        return Err(ApiError::invalid_request(
            "eos_token_ids cannot be used with embeddings input.",
        ));
    }
    if let WatermarkDetectionInput::Text {
        text,
        model,
        add_special_tokens,
    } = request.input
    {
        let (response, mut receiver) = tokio::sync::mpsc::channel(TOKENIZATION_CHANNEL_CAPACITY);
        let tokenize = TokenizationRequest {
            text: Either::Right(text),
            tools: None,
            add_generation_prompt: false,
            add_special_tokens,
            enable_thinking: None,
            reasoning_effort: None,
            response,
        };
        send_tokenization(tokenize, (model != "default").then_some(model)).await?;
        let tokens = receiver
            .recv()
            .await
            .ok_or_else(ApiError::internal)?
            .map_err(|_| ApiError::internal())?;
        request.input = WatermarkDetectionInput::Tokens { tokens };
    }
    tokio::task::spawn_blocking(move || {
        let watermark = Watermark::new(&request.watermark)?;
        match request.input {
            WatermarkDetectionInput::Tokens { tokens } => {
                watermark.detect(&tokens, request.prompt_len, &request.eos_token_ids)
            }
            WatermarkDetectionInput::Embeddings { embeddings } => {
                watermark.detect_embeddings(&embeddings, request.prompt_len)
            }
            WatermarkDetectionInput::Text { .. } => unreachable!("text was tokenized"),
        }
    })
    .await
    .map_err(|_| ApiError::internal())?
    .map_err(|error| ApiError::invalid_request(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        extract::FromRequest,
        http::{Request as HttpRequest, StatusCode},
    };
    use candle_core::{Device, Tensor};
    use mistralrs_core::WatermarkTensor;
    use serde_json::{json, Value};

    const TEST_VOCAB_SIZE: usize = 8;
    const TEST_GENERATED_TOKENS: usize = 16;

    fn config(scheme: &str) -> Value {
        let mut config = json!({"scheme": scheme, "key": "42".repeat(32)});
        match scheme {
            "synthid" => {}
            "semstamp" => {
                config["embedding_dim"] = json!(3);
                config["num_hyperplanes"] = json!(2);
                config["margin"] = json!(0.0);
            }
            _ => config["vocab_size"] = json!(TEST_VOCAB_SIZE),
        }
        if scheme == "mpac" {
            config["payload"] = json!([1, 0, 1]);
        }
        if matches!(scheme, "exponential" | "inverse_transform") {
            config["sequence_len"] = json!(7);
            config["start_position"] = json!(5);
        }
        config
    }

    async fn payload(
        body: String,
        content_type: Option<&str>,
    ) -> Result<Json<WatermarkDetectionRequest>, JsonRejection> {
        let mut request = HttpRequest::builder()
            .method("POST")
            .uri(crate::route_registry::WATERMARK_DETECT_ROUTE.path);
        if let Some(content_type) = content_type {
            request = request.header("content-type", content_type);
        }
        Json::from_request(request.body(Body::from(body)).unwrap(), &()).await
    }

    async fn body(response: Response) -> (StatusCode, Value) {
        let status = response.status();
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    async fn post(request: Value) -> (StatusCode, Value) {
        body(
            detection_response(
                payload(request.to_string(), Some("application/json")).await,
                |_, _| async {
                    panic!("token and embedding inputs must not call the inference engine")
                },
            )
            .await,
        )
        .await
    }

    #[tokio::test]
    async fn watermark_detector_closes_generation_loop_for_all_token_schemes() -> anyhow::Result<()>
    {
        for scheme in [
            "synthid",
            "kgw",
            "unigram",
            "exponential",
            "inverse_transform",
            "mpac",
        ] {
            let config = config(scheme);
            let watermark = Watermark::new(&serde_json::from_value(config.clone())?)?;
            let mut tokens = vec![1, 2, 3, 4];
            let prompt_len = tokens.len();
            let weights =
                Tensor::new(&[0.1f32, 0.2, 0.1, 0.05, 0.15, 0.1, 0.1, 0.2], &Device::Cpu)?;
            for _ in 0..TEST_GENERATED_TOKENS {
                let values = match watermark.apply_tensor(&weights, &tokens, prompt_len)? {
                    WatermarkTensor::Probabilities(values)
                    | WatermarkTensor::SelectionScores(values) => values,
                };
                tokens.push(values.argmax(0)?.to_scalar::<u32>()?);
            }
            let expected = serde_json::to_value(watermark.detect(&tokens, prompt_len, &[])?)?;
            tokens.extend([99, 100]);
            let request = json!({"watermark": config, "input": {"type": "tokens", "tokens": tokens}, "prompt_len": prompt_len, "eos_token_ids": [99]});
            let (status, actual) = post(request.clone()).await;
            assert_eq!(status, StatusCode::OK, "{scheme}: {actual}");
            assert_eq!(actual, expected, "{scheme}");
            assert!(
                actual
                    .get("tokens_scored")
                    .or_else(|| actual.get("trials"))
                    .unwrap()
                    .as_u64()
                    .unwrap()
                    > 0
            );
            assert_eq!(post(request).await.1, actual);
        }
        Ok(())
    }

    #[tokio::test]
    async fn watermark_detector_semstamp_matches_embedding_evidence() -> anyhow::Result<()> {
        let config = config("semstamp");
        let embeddings = vec![
            vec![1.0f32, 0.3, 0.5],
            vec![-0.2, 0.9, 0.1],
            vec![0.4, -0.1, 1.0],
        ];
        let expected = Watermark::new(&serde_json::from_value(config.clone())?)?
            .detect_embeddings(&embeddings, 1)?;
        let (status, actual) = post(json!({"watermark": config, "input": {"type": "embeddings", "embeddings": embeddings}, "prompt_len": 1})).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(actual, serde_json::to_value(expected)?);
        Ok(())
    }

    #[tokio::test]
    async fn watermark_detector_text_uses_selected_model_and_token_boundaries() {
        let config = config("kgw");
        let tokens = vec![1, 2, 3, 4, 5, 6];
        let expected = post(json!({"watermark": config, "input": {"type": "tokens", "tokens": tokens}, "prompt_len": 2, "eos_token_ids": [6]})).await;
        for (model, add_special_tokens) in [("default", false), ("named-model", true)] {
            let request = json!({"watermark": config, "input": {"type": "text", "model": model, "text": "prompt and completion", "add_special_tokens": add_special_tokens}, "prompt_len": 2, "eos_token_ids": [6]});
            let actual = body(
                detection_response(
                    payload(request.to_string(), Some("application/json")).await,
                    |request, model_override| async move {
                        assert_eq!(
                            model_override.as_deref(),
                            (model != "default").then_some(model)
                        );
                    assert!(matches!(request.text, Either::Right(text) if text == "prompt and completion"));
                        assert_eq!(request.add_special_tokens, add_special_tokens);
                        assert!(!request.add_generation_prompt);
                        assert!(request.tools.is_none());
                        request
                            .response
                            .send(Ok(vec![1, 2, 3, 4, 5, 6]))
                            .await
                            .unwrap();
                        Ok(())
                    },
                )
                .await,
            )
            .await;
            assert_eq!(actual, expected);
        }
    }

    #[tokio::test]
    async fn watermark_detector_rejects_invalid_inputs_without_exposing_keys() {
        let valid = json!({"watermark": config("kgw"), "input": {"type": "tokens", "tokens": [1, 2, 3]}, "prompt_len": 1});
        let mut invalid = Vec::new();
        let mut request = valid.clone();
        request["watermark"]["key"] = json!("invalid-secret-key");
        invalid.push(request);
        let mut request = valid.clone();
        request["watermark"]["vocab_size"] = json!(MAX_DETECTION_VOCAB_SIZE + 1);
        invalid.push(request);
        let mut request = valid.clone();
        request["prompt_len"] = json!(4);
        invalid.push(request);
        let mut request = valid.clone();
        request["input"]["tokens"] = json!([1, 999]);
        invalid.push(request);
        let mut request = valid.clone();
        request["watermark"] = config("semstamp");
        invalid.push(request);
        let mut request = valid.clone();
        request["input"] = json!({"type": "embeddings", "embeddings": [[1.0, 0.0, 0.0]]});
        invalid.push(request.clone());
        request["watermark"] = config("semstamp");
        request["eos_token_ids"] = json!([1]);
        invalid.push(request.clone());
        request["eos_token_ids"] = json!([]);
        request["input"]["embeddings"] = json!([[1.0, 0.0]]);
        invalid.push(request);
        let mut request = valid;
        request["input"]["text"] = json!("ambiguous input");
        invalid.push(request);
        for request in invalid {
            let (status, body) = post(request).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert!(!body.to_string().contains(&"42".repeat(32)));
            assert!(!body.to_string().contains("invalid-secret-key"));
        }
    }

    #[tokio::test]
    async fn watermark_detector_json_errors_and_empty_evidence() {
        for (raw, content_type, status, code) in [
            (
                "{",
                Some("application/json"),
                StatusCode::BAD_REQUEST,
                "malformed_json",
            ),
            (
                "{}",
                Some("application/json"),
                StatusCode::BAD_REQUEST,
                "invalid_request_body",
            ),
            (
                "{}",
                None,
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "invalid_content_type",
            ),
        ] {
            let (actual_status, actual) = body(
                detection_response(payload(raw.into(), content_type).await, |_, _| async {
                    panic!("invalid JSON must not tokenize")
                })
                .await,
            )
            .await;
            assert_eq!(actual_status, status);
            assert_eq!(actual["error"]["code"], code);
        }
        let (status, actual) =
            post(json!({"watermark": config("kgw"), "input": {"type": "tokens", "tokens": []}}))
                .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(actual["trials"], 0);
        assert!(actual["z_score"].is_null());
    }

    #[tokio::test]
    async fn watermark_detector_tokenizer_errors_use_standard_http_errors() {
        for kind in [
            ApiErrorKind::NotFound,
            ApiErrorKind::Unavailable,
            ApiErrorKind::Internal,
        ] {
            let request = json!({"watermark": config("kgw"), "input": {"type": "text", "model": "missing", "text": "example"}});
            let (status, _) = body(
                detection_response(
                    payload(request.to_string(), Some("application/json")).await,
                    |_, _| async move { Err(ApiError::new(kind, "Model error.", None, None)) },
                )
                .await,
            )
            .await;
            assert_eq!(
                status,
                match kind {
                    ApiErrorKind::NotFound => StatusCode::NOT_FOUND,
                    ApiErrorKind::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
                    _ => StatusCode::INTERNAL_SERVER_ERROR,
                }
            );
        }
    }
}
