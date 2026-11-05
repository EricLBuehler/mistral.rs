use anyhow::{Context, Result};
use axum::{
    extract::{rejection::JsonRejection, Json, State},
    response::{IntoResponse, Response},
};
use indexmap::IndexMap;
use mistralrs_core::{
    AudioInput, DecisionInstructions, DecisionMedia, DecisionQuestion, DecisionRequest,
    DecisionResponse, MistralRs, ModelCategory, NormalRequest, Request, ResponseOk,
    VideoFrameSampling,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::{
    openapi::{RefOr, Schema},
    PartialSchema, ToSchema,
};

use crate::{
    handler_core::{
        create_response_channel, openai_error_from_error, openai_error_response,
        send_request_with_model, ApiError, ApiErrorKind,
    },
    types::ExtractedMistralRsState,
    util::{parse_audio_url_for_server, parse_image_url_for_server, validate_model_name},
    video::parse_video_url_for_server,
};

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionUrl {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct DecisionInputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionAudioSource {
    AudioUrl { audio_url: DecisionUrl },
    InputAudio { input_audio: DecisionInputAudio },
}

impl DecisionAudioSource {
    async fn load(self) -> Result<AudioInput> {
        let url = match self {
            Self::AudioUrl { audio_url } => audio_url.url,
            Self::InputAudio { input_audio } => {
                mistralrs_core::input_audio_data_url(&input_audio.data, &input_audio.format)?
            }
        };
        parse_audio_url_for_server(&url).await
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionMediaSource {
    ImageUrl { image_url: DecisionUrl },
    AudioUrl { audio_url: DecisionUrl },
    InputAudio { input_audio: DecisionInputAudio },
    VideoUrl { video_url: DecisionUrl },
}

impl DecisionMediaSource {
    async fn load(self, video_sampling: Option<VideoFrameSampling>) -> Result<DecisionMedia> {
        Ok(match self {
            Self::ImageUrl { image_url } => {
                DecisionMedia::Image(parse_image_url_for_server(&image_url.url).await?)
            }
            Self::AudioUrl { audio_url } => {
                DecisionMedia::Audio(DecisionAudioSource::AudioUrl { audio_url }.load().await?)
            }
            Self::InputAudio { input_audio } => DecisionMedia::Audio(
                DecisionAudioSource::InputAudio { input_audio }
                    .load()
                    .await?,
            ),
            Self::VideoUrl { video_url } => DecisionMedia::Video(
                parse_video_url_for_server(&video_url.url, video_sampling).await?,
            ),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum DecisionInstructionSource {
    Text(String),
    Audio(DecisionAudioSource),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(transparent)]
pub struct SystemOneQuestion(Value);

impl PartialSchema for SystemOneQuestion {
    fn schema() -> RefOr<Schema> {
        let mut schema = DecisionQuestion::schema();
        if let RefOr::T(Schema::OneOf(variants)) = &mut schema {
            for variant in &mut variants.items {
                if let RefOr::T(Schema::Object(object)) = variant {
                    object.properties.insert(
                        "instructions".into(),
                        utoipa::openapi::Ref::from_schema_name(DecisionInstructionSource::name())
                            .into(),
                    );
                }
            }
        }
        schema
    }
}

impl ToSchema for SystemOneQuestion {
    fn schemas(schemas: &mut Vec<(String, RefOr<Schema>)>) {
        schemas.push((
            DecisionInstructionSource::name().into(),
            DecisionInstructionSource::schema(),
        ));
        DecisionInstructionSource::schemas(schemas);
    }
}

impl SystemOneQuestion {
    async fn resolve(mut self) -> Result<DecisionQuestion> {
        let instructions: DecisionInstructionSource = serde_json::from_value(
            self.0
                .get("instructions")
                .context("Decision question requires instructions")?
                .clone(),
        )?;
        let instructions = match instructions {
            DecisionInstructionSource::Text(text) => DecisionInstructions::Text(text),
            DecisionInstructionSource::Audio(source) => {
                DecisionInstructions::Audio(source.load().await?)
            }
        };
        self.0["instructions"] = Value::String(String::new());
        let mut question: DecisionQuestion = serde_json::from_value(self.0)?;
        question.set_instructions(instructions);
        Ok(question)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct SystemOneRequest {
    pub model: Option<String>,
    #[serde(default)]
    pub state: Value,
    pub questions: IndexMap<String, SystemOneQuestion>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub media: Vec<DecisionMediaSource>,
    pub set_size: Option<usize>,
}

#[derive(Serialize, ToSchema)]
pub struct SystemOneResponse {
    pub model: String,
    #[serde(flatten)]
    pub result: DecisionResponse,
}

impl SystemOneRequest {
    async fn resolve(self, video_sampling: VideoFrameSampling) -> Result<DecisionRequest> {
        let mut media = Vec::with_capacity(self.images.len() + self.media.len());
        for image in self.images {
            media.push(DecisionMedia::Image(
                parse_image_url_for_server(&image).await?,
            ));
        }
        for source in self.media {
            media.push(source.load(Some(video_sampling)).await?);
        }
        let mut questions = IndexMap::with_capacity(self.questions.len());
        for (name, question) in self.questions {
            questions.insert(name, question.resolve().await?);
        }
        let request = DecisionRequest {
            state: self.state,
            questions,
            media,
            set_size: self.set_size,
        };
        request.validate()?;
        Ok(request)
    }
}

#[utoipa::path(
    post, tag = "Mistral.rs", path = "/v1/systemone", request_body = SystemOneRequest,
    responses((status = 200, description = "Named decision answers", body = SystemOneResponse),
              (status = 400, description = "Invalid question or unsupported model"))
)]
pub async fn systemone(
    State(state): ExtractedMistralRsState,
    payload: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Response {
    let Json(request) = match payload {
        Ok(request) => request,
        Err(error) => return openai_error_response(ApiError::from_json_rejection(error)),
    };
    MistralRs::maybe_log_request(
        state.clone(),
        serde_json::to_string(&request).expect("Decision request serializes"),
    );
    let parsed = async {
        let model = match request.model.as_deref().filter(|model| *model != "default") {
            Some(model) => model.to_owned(),
            None => state
                .get_default_model_id()
                .map_err(anyhow::Error::msg)?
                .context("No default model is loaded")?,
        };
        validate_model_name(&model, state.clone())?;
        let ModelCategory::Decision { video_sampling } = state.get_model_category(Some(&model))?
        else {
            anyhow::bail!("The selected model does not support decision requests");
        };
        Ok::<_, anyhow::Error>((model, request.resolve(video_sampling).await?))
    }
    .await;
    let (model, request) = match parsed {
        Ok(request) => request,
        Err(error) => return openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest),
    };
    let (tx, mut rx) = create_response_channel(Some(1));
    let request = Request::Normal(Box::new(NormalRequest::decision(
        request,
        tx,
        state.next_request_id(),
        Some(model.clone()),
    )));
    if let Err(error) = send_request_with_model(&state, request, Some(&model)).await {
        return openai_error_from_error(&error, ApiErrorKind::Internal);
    }
    match rx.recv().await {
        Some(mistralrs_core::Response::ValidationError(error)) => {
            openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest)
        }
        Some(response) => match response.as_result() {
            Ok(ResponseOk::Decision { result }) => {
                let response = SystemOneResponse { model, result };
                MistralRs::maybe_log_response(state, &response);
                Json(response).into_response()
            }
            Err(error) => openai_error_from_error(error.as_ref(), ApiErrorKind::Internal),
            _ => openai_error_response(ApiError::internal()),
        },
        None => openai_error_response(ApiError::internal()),
    }
}
