//! ## OpenAI compatible functionality.

use std::{collections::HashMap, ops::Deref};

use either::Either;
use mistralrs_core::{
    ImageGenerationResponseFormat, LlguidanceGrammar, Tool, ToolChoice, ToolType, WebSearchOptions,
};
use serde::{Deserialize, Serialize};
use utoipa::{
    openapi::{schema::SchemaType, ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};

/// Inner content structure for messages that can be either a string or key-value pairs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] pub Either<String, HashMap<String, String>>,
);

// The impl Deref was preventing the Derive ToSchema and #[schema] macros from
// properly working, so manually impl ToSchema
impl PartialSchema for MessageInnerContent {
    fn schema() -> RefOr<Schema> {
        RefOr::T(message_inner_content_schema())
    }
}

impl ToSchema for MessageInnerContent {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            MessageInnerContent::name().into(),
            MessageInnerContent::schema(),
        ));
    }
}

impl Deref for MessageInnerContent {
    type Target = Either<String, HashMap<String, String>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Function for MessageInnerContent Schema generation to handle `Either`
fn message_inner_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            // Either::Left - simple string
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .build(),
            ))
            // Either::Right - object with string values
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .additional_properties(Some(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::String))
                            .build(),
                    ))))
                    .build(),
            ))
            .build(),
    )
}

/// Message content that can be either simple text or complex structured content
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageContent(
    #[serde(with = "either::serde_untagged")]
    Either<String, Vec<HashMap<String, MessageInnerContent>>>,
);

// The impl Deref was preventing the Derive ToSchema and #[schema] macros from
// properly working, so manually impl ToSchema
impl PartialSchema for MessageContent {
    fn schema() -> RefOr<Schema> {
        RefOr::T(message_content_schema())
    }
}

impl ToSchema for MessageContent {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((MessageContent::name().into(), MessageContent::schema()));
    }
}

impl Deref for MessageContent {
    type Target = Either<String, Vec<HashMap<String, MessageInnerContent>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Function for MessageContent Schema generation to handle `Either`
fn message_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::Object))
                            .additional_properties(Some(RefOr::Ref(
                                utoipa::openapi::Ref::from_schema_name("MessageInnerContent"),
                            )))
                            .build(),
                    )))
                    .build(),
            ))
            .build(),
    )
}

/// Represents a function call made by the assistant
///
/// When using tool calling, this structure contains the details of a function
/// that the model has decided to call, including the function name and its parameters.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
pub struct FunctionCalled {
    /// The name of the function to call
    pub name: String,
    /// The function arguments
    #[serde(alias = "arguments")]
    pub parameters: String,
}

/// Represents a tool call made by the assistant
///
/// This structure wraps a function call with its type information.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
pub struct ToolCall {
    /// The type of tool being called
    #[serde(rename = "type")]
    pub tp: ToolType,
    ///  The function call details
    pub function: FunctionCalled,
}

/// Represents a single message in a conversation
///
/// ### Examples
///
/// ```ignore
/// use either::Either;
/// use mistralrs_server_core::openai::{Message, MessageContent};
///
/// // User message
/// let user_msg = Message {
///     content: Some(MessageContent(Either::Left("What's 2+2?".to_string()))),
///     role: "user".to_string(),
///     name: None,
///     tool_calls: None,
/// };
///
/// // System message
/// let system_msg = Message {
///     content: Some(MessageContent(Either::Left("You are a helpful assistant.".to_string()))),
///     role: "system".to_string(),
///     name: None,
///     tool_calls: None,
/// };
/// ```
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct Message {
    /// The message content
    pub content: Option<MessageContent>,
    /// The role of the message sender ("user", "assistant", "system", "tool", etc.)
    pub role: String,
    pub name: Option<String>,
    /// Optional list of tool calls
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Stop token configuration for generation
///
/// Defines when the model should stop generating text, either with a single
/// stop token or multiple possible stop sequences.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum StopTokens {
    ///  Multiple possible stop sequences
    Multi(Vec<String>),
    /// Single stop sequence
    Single(String),
}

/// Default value helper
fn default_false() -> bool {
    false
}

/// Default value helper
fn default_1usize() -> usize {
    1
}

/// Default value helper
fn default_720usize() -> usize {
    720
}

/// Default value helper
fn default_1280usize() -> usize {
    1280
}

/// Default value helper
fn default_model() -> String {
    "default".to_string()
}

/// Default value helper
fn default_response_format() -> ImageGenerationResponseFormat {
    ImageGenerationResponseFormat::Url
}

/// Grammar specification for structured generation
///
/// Defines different types of grammars that can be used to constrain model output,
/// ensuring it follows specific formats or structures.
///
/// ### Examples
///
/// ```ignore
/// use mistralrs_server_core::openai::Grammar;
///
/// // Regex grammar for phone numbers
/// let phone_regex = Grammar::Regex(r"\d{3}-\d{3}-\d{4}".to_string());
///
/// // JSON schema for structured data
/// let json_schema = Grammar::JsonSchema(serde_json::json!({
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "integer"}
///     },
///     "required": ["name", "age"]
/// }));
///
/// // Lark grammar for arithmetic expressions
/// let lark_grammar = Grammar::Lark(r#"
///     ?start: expr
///     expr: term ("+" term | "-" term)*
///     term: factor ("*" factor | "/" factor)*
///     factor: NUMBER | "(" expr ")"
///     %import common.NUMBER
/// "#.to_string());
/// ```
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", content = "value")]
pub enum Grammar {
    /// Regular expression grammar
    #[serde(rename = "regex")]
    Regex(String),
    /// JSON schema grammar
    #[serde(rename = "json_schema")]
    JsonSchema(serde_json::Value),
    /// LLGuidance grammar
    #[serde(rename = "llguidance")]
    Llguidance(LlguidanceGrammar),
    /// Lark parser grammar
    #[serde(rename = "lark")]
    Lark(String),
}

// Implement ToSchema manually to handle `LlguidanceGrammar`
impl PartialSchema for Grammar {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(create_grammar_variant_schema(
                    "regex",
                    Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::String))
                            .build(),
                    ),
                ))
                .item(create_grammar_variant_schema(
                    "json_schema",
                    Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::Object))
                            .build(),
                    ),
                ))
                .item(create_grammar_variant_schema(
                    "llguidance",
                    llguidance_schema(),
                ))
                .item(create_grammar_variant_schema(
                    "lark",
                    Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::String))
                            .build(),
                    ),
                ))
                .build(),
        ))
    }
}

impl ToSchema for Grammar {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((Grammar::name().into(), Grammar::schema()));
    }
}

/// Helper function to create a grammar variant schema
fn create_grammar_variant_schema(type_value: &str, value_schema: Schema) -> Schema {
    Schema::Object(
        ObjectBuilder::new()
            .schema_type(SchemaType::Type(Type::Object))
            .property(
                "type",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::Type(Type::String))
                        .enum_values(Some(vec![serde_json::Value::String(
                            type_value.to_string(),
                        )]))
                        .build(),
                )),
            )
            .property("value", RefOr::T(value_schema))
            .required("type")
            .required("value")
            .build(),
    )
}

/// Helper function to generate LLGuidance schema
fn llguidance_schema() -> Schema {
    let grammar_with_lexer_schema = Schema::Object(
        ObjectBuilder::new()
            .schema_type(SchemaType::Type(Type::Object))
            .property(
                "name",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::from_iter([Type::String, Type::Null]))
                        .description(Some(
                            "The name of this grammar, can be used in GenGrammar nodes",
                        ))
                        .build(),
                )),
            )
            .property(
                "json_schema",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::from_iter([Type::Object, Type::Null]))
                        .description(Some("The JSON schema that the grammar should generate"))
                        .build(),
                )),
            )
            .property(
                "lark_grammar",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::from_iter([Type::String, Type::Null]))
                        .description(Some("The Lark grammar that the grammar should generate"))
                        .build(),
                )),
            )
            .description(Some("Grammar configuration with lexer settings"))
            .build(),
    );

    Schema::Object(
        ObjectBuilder::new()
            .schema_type(SchemaType::Type(Type::Object))
            .property(
                "grammars",
                RefOr::T(Schema::Array(
                    ArrayBuilder::new()
                        .items(RefOr::T(grammar_with_lexer_schema))
                        .description(Some("List of grammar configurations"))
                        .build(),
                )),
            )
            .property(
                "max_tokens",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::from_iter([Type::Integer, Type::Null]))
                        .description(Some("Maximum number of tokens to generate"))
                        .build(),
                )),
            )
            .required("grammars")
            .description(Some("Top-level grammar configuration for LLGuidance"))
            .build(),
    )
}

/// JSON Schema for structured responses
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct JsonSchemaResponseFormat {
    pub name: String,
    pub schema: serde_json::Value,
}

/// Response format for model output
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    /// Free-form text response
    #[serde(rename = "text")]
    Text,
    /// Structured response following a JSON schema
    #[serde(rename = "json_schema")]
    JsonSchema {
        json_schema: JsonSchemaResponseFormat,
    },
}

/// Chat completion request following OpenAI's specification
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    #[schema(
        schema_with = messages_schema,
        example = json!(vec![Message{content:Some(MessageContent{0: either::Left(("Why did the crab cross the road?".to_string()))}), role:"user".to_string(), name: None, tool_calls: None}])
    )]
    #[serde(with = "either::serde_untagged")]
    pub messages: Either<Vec<Message>, String>,
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    #[schema(example = json!(Option::None::<usize>))]
    pub top_logprobs: Option<usize>,
    #[schema(example = 256)]
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    #[schema(example = true)]
    pub stream: Option<bool>,
    #[schema(example = json!(Option::None::<Vec<Tool>>))]
    pub tools: Option<Vec<Tool>>,
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,
    #[schema(example = json!(Option::None::<ResponseFormat>))]
    pub response_format: Option<ResponseFormat>,
    #[schema(example = json!(Option::None::<WebSearchOptions>))]
    pub web_search_options: Option<WebSearchOptions>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[schema(example = json!(Option::None::<bool>))]
    pub enable_thinking: Option<bool>,
}

/// Function for ChatCompletionRequest.messages Schema generation to handle `Either`
fn messages_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::Ref(utoipa::openapi::Ref::from_schema_name(
                        "Message",
                    )))
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .build(),
            ))
            .build(),
    )
}

/// Model information metadata about an available mode
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// Whether tools are available through MCP or tool callbacks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools_available: Option<bool>,
    /// Number of tools available from MCP servers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_tools_count: Option<usize>,
    /// Number of connected MCP servers
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers_connected: Option<usize>,
}

/// Collection of available models
#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObjects {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// Legacy OpenAI compatible text completion request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct CompletionRequest {
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(example = "Say this is a test.")]
    pub prompt: String,
    #[schema(example = 1)]
    pub best_of: Option<usize>,
    #[serde(rename = "echo")]
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub echo_prompt: bool,
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[schema(example = json!(Option::None::<usize>))]
    pub logprobs: Option<usize>,
    #[schema(example = 16)]
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    pub stream: Option<bool>,
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    #[schema(example = json!(Option::None::<String>))]
    pub suffix: Option<String>,
    #[serde(rename = "user")]
    pub _user: Option<String>,
    #[schema(example = json!(Option::None::<Vec<Tool>>))]
    pub tools: Option<Vec<Tool>>,
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
}

/// Image generation request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ImageGenerationRequest {
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(example = "Draw a picture of a majestic, snow-covered mountain.")]
    pub prompt: String,
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    #[serde(default = "default_response_format")]
    pub response_format: ImageGenerationResponseFormat,
    #[serde(default = "default_720usize")]
    #[schema(example = 720)]
    pub height: usize,
    #[serde(default = "default_1280usize")]
    #[schema(example = 1280)]
    pub width: usize,
}

/// Audio format options for speech generation responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum AudioResponseFormat {
    /// Widely compatible, lossy compression, good for web streaming
    #[default]
    Mp3,
    /// Good compression efficiency, ideal for real-time communication
    Opus,
    /// High-quality lossy compression, commonly used in mobile applications
    Aac,
    /// Lossless compression, larger file sizes but good audio quality
    Flac,
    /// Uncompressed, largest file sizes but maximum compatibility
    Wav,
    ///  Raw audio data, requires additional format specification
    Pcm,
}

impl AudioResponseFormat {
    /// Generate the appropriate MIME content type string for this audio format.
    pub fn audio_content_type(
        &self,
        pcm_rate: usize,
        pcm_channels: usize,
        pcm_format: &'static str,
    ) -> String {
        let content_type = match &self {
            AudioResponseFormat::Mp3 => "audio/mpeg".to_string(),
            AudioResponseFormat::Opus => "audio/ogg; codecs=opus".to_string(),
            AudioResponseFormat::Aac => "audio/aac".to_string(),
            AudioResponseFormat::Flac => "audio/flac".to_string(),
            AudioResponseFormat::Wav => "audio/wav".to_string(),
            AudioResponseFormat::Pcm => format!("audio/pcm; codecs=1; format={pcm_format}"),
        };

        format!("{content_type}; rate={pcm_rate}; channels={pcm_channels}")
    }
}

/// Speech generation request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SpeechGenerationRequest {
    /// The TTS model to use for audio generation.
    #[schema(example = "nari-labs/Dia-1.6B")]
    #[serde(default = "default_model")]
    pub model: String,
    /// The text content to convert to speech.
    #[schema(
        example = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
    )]
    pub input: String,
    // `voice` and `instructions` are ignored.
    /// The desired audio format for the generated speech.
    #[schema(example = "mp3")]
    pub response_format: AudioResponseFormat,
}
