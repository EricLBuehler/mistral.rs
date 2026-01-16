//! Content types for the OpenResponses API.

use serde::{Deserialize, Serialize};
use utoipa::{
    openapi::{schema::SchemaType, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};

use super::enums::ImageDetail;

/// Image URL structure for image inputs
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ImageUrl {
    /// The URL of the image
    pub url: String,
    /// Optional detail level for processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

/// OpenResponses format input content types (with input_text, input_image, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenResponsesInputContent {
    /// Text input content
    #[serde(rename = "input_text")]
    InputText {
        /// The text content
        text: String,
    },
    /// Image input content
    #[serde(rename = "input_image")]
    InputImage {
        /// The image URL
        image_url: Option<String>,
        /// Base64 encoded image data
        #[serde(skip_serializing_if = "Option::is_none")]
        image_data: Option<String>,
        /// Optional detail level
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    /// Audio input content
    #[serde(rename = "input_audio")]
    InputAudio {
        /// Audio data (base64 encoded)
        data: String,
        /// Audio format (e.g., "wav", "mp3")
        format: String,
    },
    /// File input content
    #[serde(rename = "input_file")]
    InputFile {
        /// File ID (for previously uploaded files)
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// Base64 encoded file data
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        /// Filename
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
}

/// Audio input structure for OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct AudioInput {
    /// Base64 encoded audio data
    pub data: String,
    /// Audio format (e.g., "wav", "mp3")
    pub format: String,
}

/// OpenAI Chat format input content types (with text, image_url, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OpenAIInputContent {
    /// Text content (OpenAI format)
    #[serde(rename = "text")]
    Text {
        /// The text content
        text: String,
    },
    /// Image URL content (OpenAI format)
    #[serde(rename = "image_url")]
    ImageUrl {
        /// The image URL object
        image_url: ImageUrl,
    },
    /// Audio content (OpenAI format)
    #[serde(rename = "input_audio")]
    InputAudio {
        /// The audio input object
        input_audio: AudioInput,
    },
}

/// Input content types for the OpenResponses API.
///
/// Supports both OpenResponses format (input_text, input_image) and
/// OpenAI Chat format (text, image_url) for compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputContent {
    /// OpenResponses format content
    OpenResponses(OpenResponsesInputContent),
    /// OpenAI Chat format content
    OpenAI(OpenAIInputContent),
}

impl InputContent {
    /// Normalize to the internal format for processing
    pub fn into_normalized(self) -> NormalizedInputContent {
        match self {
            InputContent::OpenResponses(c) => match c {
                OpenResponsesInputContent::InputText { text } => {
                    NormalizedInputContent::Text { text }
                }
                OpenResponsesInputContent::InputImage {
                    image_url,
                    image_data,
                    detail,
                } => NormalizedInputContent::Image {
                    image_url,
                    image_data,
                    detail,
                },
                OpenResponsesInputContent::InputAudio { data, format } => {
                    NormalizedInputContent::Audio { data, format }
                }
                OpenResponsesInputContent::InputFile {
                    file_id,
                    file_data,
                    filename,
                } => NormalizedInputContent::File {
                    file_id,
                    file_data,
                    filename,
                },
            },
            InputContent::OpenAI(c) => match c {
                OpenAIInputContent::Text { text } => NormalizedInputContent::Text { text },
                OpenAIInputContent::ImageUrl { image_url } => NormalizedInputContent::Image {
                    image_url: Some(image_url.url),
                    image_data: None,
                    detail: image_url.detail,
                },
                OpenAIInputContent::InputAudio { input_audio } => NormalizedInputContent::Audio {
                    data: input_audio.data,
                    format: input_audio.format,
                },
            },
        }
    }
}

/// Normalized input content for internal processing
#[derive(Debug, Clone)]
pub enum NormalizedInputContent {
    /// Text content
    Text {
        /// The text content
        text: String,
    },
    /// Image content
    Image {
        /// The image URL
        image_url: Option<String>,
        /// Base64 encoded image data
        image_data: Option<String>,
        /// Optional detail level
        detail: Option<ImageDetail>,
    },
    /// Audio content
    Audio {
        /// Audio data (base64 encoded)
        data: String,
        /// Audio format (e.g., "wav", "mp3")
        format: String,
    },
    /// File content
    File {
        /// File ID (for previously uploaded files)
        file_id: Option<String>,
        /// Base64 encoded file data
        file_data: Option<String>,
        /// Filename
        filename: Option<String>,
    },
}

impl PartialSchema for InputContent {
    fn schema() -> RefOr<Schema> {
        RefOr::T(input_content_schema())
    }
}

impl ToSchema for InputContent {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((InputContent::name().into(), InputContent::schema()));
    }
}

fn input_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "input_text".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "text",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("text")
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "input_image".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "image_url",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "input_audio".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "data",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "format",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("data")
                    .required("format")
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "input_file".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "file_id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .build(),
            ))
            .build(),
    )
}

/// Annotation for output text content
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(tag = "type")]
pub enum Annotation {
    /// File citation annotation
    #[serde(rename = "file_citation")]
    FileCitation {
        /// The text that is annotated
        text: String,
        /// Start index in the text
        start_index: usize,
        /// End index in the text
        end_index: usize,
        /// File citation details
        file_citation: FileCitation,
    },
    /// URL citation annotation
    #[serde(rename = "url_citation")]
    UrlCitation {
        /// The text that is annotated
        text: String,
        /// Start index in the text
        start_index: usize,
        /// End index in the text
        end_index: usize,
        /// URL citation details
        url_citation: UrlCitation,
    },
    /// File path annotation
    #[serde(rename = "file_path")]
    FilePath {
        /// The text that is annotated
        text: String,
        /// Start index in the text
        start_index: usize,
        /// End index in the text
        end_index: usize,
        /// File path details
        file_path: FilePathInfo,
    },
}

/// File citation details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FileCitation {
    /// File ID
    pub file_id: String,
    /// Quote from the file
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quote: Option<String>,
}

/// URL citation details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UrlCitation {
    /// The URL
    pub url: String,
    /// Title of the page
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// File path information
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FilePathInfo {
    /// File ID
    pub file_id: String,
}

/// Output content types for the OpenResponses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContent {
    /// Text output content
    #[serde(rename = "output_text")]
    OutputText {
        /// The text content
        text: String,
        /// Optional annotations
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Annotation>>,
    },
    /// Refusal output content
    #[serde(rename = "refusal")]
    Refusal {
        /// The refusal message
        refusal: String,
    },
}

impl PartialSchema for OutputContent {
    fn schema() -> RefOr<Schema> {
        RefOr::T(output_content_schema())
    }
}

impl ToSchema for OutputContent {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((OutputContent::name().into(), OutputContent::schema()));
    }
}

fn output_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "output_text".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "text",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("text")
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .property(
                        "type",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .enum_values(Some(vec![serde_json::Value::String(
                                    "refusal".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "refusal",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("refusal")
                    .build(),
            ))
            .build(),
    )
}

impl OutputContent {
    /// Create a new text output content
    pub fn text(text: String) -> Self {
        OutputContent::OutputText {
            text,
            annotations: None,
        }
    }

    /// Create a new text output content with annotations
    pub fn text_with_annotations(text: String, annotations: Vec<Annotation>) -> Self {
        OutputContent::OutputText {
            text,
            annotations: Some(annotations),
        }
    }

    /// Create a new refusal output content
    pub fn refusal(refusal: String) -> Self {
        OutputContent::Refusal { refusal }
    }

    /// Get the text content if this is a text output
    pub fn get_text(&self) -> Option<&str> {
        match self {
            OutputContent::OutputText { text, .. } => Some(text),
            OutputContent::Refusal { refusal } => Some(refusal),
        }
    }
}
