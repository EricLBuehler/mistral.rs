//! ## OpenAI compatible functionality.

use std::{collections::HashMap, ops::Deref};

use either::Either;
use mistralrs_core::{
    ImageGenerationResponseFormat, LlguidanceGrammar, Tool, ToolChoice, ToolType, WebSearchOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use utoipa::{
    openapi::{schema::SchemaType, ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};
use uuid::Uuid;

/// Inner content structure for messages that can be either a string or key-value pairs
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] pub Either<String, HashMap<String, Value>>,
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
    type Target = Either<String, HashMap<String, Value>>;
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
            // Either::Right - object with arbitrary values
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Object))
                    .additional_properties(Some(RefOr::T(Schema::Object(
                        ObjectBuilder::new().build(),
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

impl MessageContent {
    /// Create a new MessageContent from a string
    pub fn from_text(text: String) -> Self {
        MessageContent(Either::Left(text))
    }

    pub fn from_parts(parts: Vec<HashMap<String, MessageInnerContent>>) -> Self {
        MessageContent(Either::Right(parts))
    }

    /// Extract text from MessageContent
    pub fn to_text(&self) -> Option<String> {
        match &self.0 {
            Either::Left(text) => Some(text.clone()),
            Either::Right(parts) => {
                // For complex content, try to extract text from parts
                let mut text_parts = Vec::new();
                for part in parts {
                    for (key, value) in part {
                        if key == "text" {
                            if let Either::Left(text) = &**value {
                                text_parts.push(text.clone());
                            }
                        }
                    }
                }
                if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join(" "))
                }
            }
        }
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
    #[serde(rename = "arguments", alias = "parameters")]
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
    /// When role is `tool`, this associates the message with a tool call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
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

fn deserialize_truncation<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum TruncationRepr {
        String(String),
        Obj {
            #[serde(rename = "type")]
            tp: Option<String>,
        },
    }

    let repr = Option::<TruncationRepr>::deserialize(d)?;
    Ok(match repr {
        None => None,
        Some(TruncationRepr::String(s)) => Some(s),
        Some(TruncationRepr::Obj { tp }) => tp,
    })
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
        example = json!(vec![Message{content:Some(MessageContent{0: either::Left(("Why did the crab cross the road?".to_string()))}), role:"user".to_string(), name: None, tool_call_id: None, tool_calls: None}])
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
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
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
    #[schema(example = json!(Option::None::<bool>))]
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
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
    pub repetition_penalty: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[schema(example = json!(Option::None::<bool>))]
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
    Tokens(Vec<u32>),
    TokensBatch(Vec<Vec<u32>>),
}

impl PartialSchema for EmbeddingInput {
    fn schema() -> RefOr<Schema> {
        RefOr::T(embedding_input_schema())
    }
}

impl ToSchema for EmbeddingInput {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((EmbeddingInput::name().into(), EmbeddingInput::schema()));
    }
}

fn embedding_input_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .description(Some("Single input string"))
                    .build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::String))
                            .build(),
                    )))
                    .description(Some("Multiple input strings"))
                    .build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::Integer))
                            .build(),
                    )))
                    .description(Some("Single token array"))
                    .build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Array(
                        ArrayBuilder::new()
                            .items(RefOr::T(Schema::Object(
                                ObjectBuilder::new()
                                    .schema_type(SchemaType::Type(Type::Integer))
                                    .build(),
                            )))
                            .build(),
                    )))
                    .description(Some("Multiple token arrays"))
                    .build(),
            ))
            .build(),
    )
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingEncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct EmbeddingRequest {
    #[schema(example = "default")]
    #[serde(default = "default_model")]
    pub model: String,
    pub input: EmbeddingInput,
    #[schema(example = "float")]
    #[serde(default)]
    pub encoding_format: Option<EmbeddingEncodingFormat>,
    #[schema(example = json!(Option::None::<usize>))]
    pub dimensions: Option<usize>,
    #[schema(example = json!(Option::None::<String>))]
    #[serde(rename = "user")]
    pub _user: Option<String>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<bool>))]
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl PartialSchema for EmbeddingVector {
    fn schema() -> RefOr<Schema> {
        RefOr::T(embedding_vector_schema())
    }
}

impl ToSchema for EmbeddingVector {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((EmbeddingVector::name().into(), EmbeddingVector::schema()));
    }
}

fn embedding_vector_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Type(Type::Number))
                            .build(),
                    )))
                    .description(Some("Embedding returned as an array of floats"))
                    .build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .description(Some("Embedding returned as a base64-encoded string"))
                    .build(),
            ))
            .build(),
    )
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub embedding: EmbeddingVector,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
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

/// Helper type for messages field in ResponsesCreateRequest
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum ResponsesInputContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    // Codex (and some OpenAI tooling) may include prior assistant messages as `output_text` parts
    // inside `message.content`. For prompt-building, treat this equivalently to `input_text`.
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(default)]
        image_url: Option<ResponsesImageUrl>,
        #[serde(default)]
        file_id: Option<String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ResponsesImageUrl {
    Url(String),
    Object {
        url: String,
        #[serde(flatten)]
        extra: HashMap<String, Value>,
    },
}

impl ResponsesImageUrl {
    pub fn url(&self) -> &str {
        match self {
            ResponsesImageUrl::Url(u) => u,
            ResponsesImageUrl::Object { url, .. } => url,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ResponsesInputMessageContent {
    Text(String),
    Parts(Vec<ResponsesInputContentPart>),
}

impl ResponsesInputMessageContent {
    pub fn into_parts(self) -> Vec<ResponsesInputContentPart> {
        match self {
            ResponsesInputMessageContent::Text(t) => {
                vec![ResponsesInputContentPart::InputText { text: t }]
            }
            ResponsesInputMessageContent::Parts(p) => p,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummaryItem {
    SummaryText {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningContentItem {
    ReasoningText {
        text: String,
    },
    Text {
        text: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesWebSearchAction {
    Search {
        #[serde(default)]
        query: Option<String>,
    },
    OpenPage {
        #[serde(default)]
        url: Option<String>,
    },
    FindInPage {
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        pattern: Option<String>,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum ResponsesInputItem {
    /// A message-style input item, e.g. a user/system message with typed content parts.
    #[serde(rename = "message")]
    Message {
        /// Unique identifier for this input item.
        #[serde(default = "default_msg_id")]
        id: String,
        #[schema(example = "user")]
        role: String,
        content: ResponsesInputMessageContent,
    },
    /// Tool output provided by the client for a prior function call.
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        /// Unique identifier for this input item.
        #[serde(default = "default_fco_id")]
        id: String,
        #[schema(example = "call_abc123")]
        call_id: String,
        /// Tool output. Some clients send a plain string; others send an array of content items
        /// (e.g. `input_text` + `input_image`).
        output: ResponsesInputMessageContent,
    },
    /// A function/tool call emitted by the assistant (may be echoed back by some clients in `input` history).
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default = "default_msg_id")]
        id: String,
        #[schema(example = "call_abc123")]
        call_id: String,
        name: String,
        /// JSON string arguments (OpenAI-style).
        arguments: String,
    },
    /// Codex-specific tool call item (treated like a function call for prompt-building).
    #[serde(rename = "custom_tool_call")]
    CustomToolCall {
        #[serde(default = "default_msg_id")]
        id: String,
        #[serde(default)]
        status: Option<String>,
        #[schema(example = "call_abc123")]
        call_id: String,
        name: String,
        input: String,
    },
    /// Codex-specific tool call output item (treated like a function_call_output for prompt-building).
    #[serde(rename = "custom_tool_call_output")]
    CustomToolCallOutput {
        #[serde(default = "default_fco_id")]
        id: String,
        #[schema(example = "call_abc123")]
        call_id: String,
        output: ResponsesInputMessageContent,
    },
    /// Codex local shell call item. We preserve it for history but treat it as a tool call for prompt-building.
    #[serde(rename = "local_shell_call")]
    LocalShellCall {
        #[serde(default = "default_msg_id")]
        id: String,
        #[serde(default)]
        call_id: Option<String>,
        #[serde(default)]
        status: Option<String>,
        #[serde(default)]
        action: Option<Value>,
    },
    /// Reasoning item emitted by some Responses-compatible clients.
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default = "default_msg_id")]
        id: String,
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummaryItem>,
        #[serde(default)]
        content: Option<Vec<ResponsesReasoningContentItem>>,
        #[serde(default)]
        encrypted_content: Option<String>,
    },
    /// Web search call item emitted by the Responses API.
    #[serde(rename = "web_search_call")]
    WebSearchCall {
        #[serde(default = "default_msg_id")]
        id: String,
        #[serde(default)]
        status: Option<String>,
        action: ResponsesWebSearchAction,
    },
    /// Generated by the Codex harness but considered as model response.
    #[serde(rename = "ghost_snapshot")]
    GhostSnapshot {
        #[serde(default = "default_msg_id")]
        id: String,
        ghost_commit: Value,
    },
    /// Conversation compaction item (alias: `compaction_summary`).
    #[serde(rename = "compaction")]
    Compaction {
        #[serde(default = "default_msg_id")]
        id: String,
        encrypted_content: String,
    },
    /// Unknown / ignored Responses input item.
    #[serde(rename = "other")]
    Other {
        #[serde(default = "default_msg_id")]
        id: String,
        #[serde(default)]
        kind: String,
        #[serde(skip_serializing)]
        raw: Value,
    },
}

impl<'de> Deserialize<'de> for ResponsesInputItem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TypedMessage {
            #[serde(default = "default_msg_id")]
            id: String,
            role: String,
            content: ResponsesInputMessageContent,
        }

        #[derive(Deserialize)]
        struct TypedFunctionCallOutput {
            #[serde(default = "default_fco_id")]
            id: String,
            call_id: String,
            output: ResponsesInputMessageContent,
        }

        #[derive(Deserialize)]
        struct TypedFunctionCall {
            #[serde(default = "default_msg_id")]
            id: String,
            call_id: String,
            name: String,
            arguments: String,
        }

        #[derive(Deserialize)]
        struct TypedCustomToolCall {
            #[serde(default = "default_msg_id")]
            id: String,
            #[serde(default)]
            status: Option<String>,
            call_id: String,
            name: String,
            input: String,
        }

        #[derive(Deserialize)]
        struct TypedCustomToolCallOutput {
            #[serde(default = "default_fco_id")]
            id: String,
            call_id: String,
            output: ResponsesInputMessageContent,
        }

        #[derive(Deserialize)]
        struct TypedLocalShellCall {
            #[serde(default = "default_msg_id")]
            id: String,
            #[serde(default)]
            call_id: Option<String>,
            #[serde(default)]
            status: Option<String>,
            #[serde(default)]
            action: Option<Value>,
        }

        #[derive(Deserialize)]
        struct TypedReasoning {
            #[serde(default = "default_msg_id")]
            id: String,
            #[serde(default)]
            summary: Vec<ResponsesReasoningSummaryItem>,
            #[serde(default)]
            content: Option<Vec<ResponsesReasoningContentItem>>,
            #[serde(default)]
            encrypted_content: Option<String>,
        }

        #[derive(Deserialize)]
        struct TypedWebSearchCall {
            #[serde(default = "default_msg_id")]
            id: String,
            #[serde(default)]
            status: Option<String>,
            action: ResponsesWebSearchAction,
        }

        #[derive(Deserialize)]
        struct TypedGhostSnapshot {
            #[serde(default = "default_msg_id")]
            id: String,
            ghost_commit: Value,
        }

        #[derive(Deserialize)]
        struct TypedCompaction {
            #[serde(default = "default_msg_id")]
            id: String,
            encrypted_content: String,
        }

        let mut v = serde_json::Value::deserialize(deserializer)?;
        let kind = v
            .get("type")
            .and_then(|t| t.as_str())
            .map(|s| s.to_string());

        let kind = match kind {
            Some(k) => k,
            None => {
                // OpenAI's Responses API accepts shorthand message items without the `type` discriminator,
                // e.g. `{ "role": "user", "content": "hello" }`.
                if let Some(obj) = v.as_object_mut() {
                    obj.insert(
                        "type".to_string(),
                        serde_json::Value::String("message".to_string()),
                    );
                }
                "message".to_string()
            }
        };

        match kind.as_str() {
            "message" => {
                let msg: TypedMessage =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::Message {
                    id: msg.id,
                    role: msg.role,
                    content: msg.content,
                })
            }
            "function_call_output" => {
                let fco: TypedFunctionCallOutput =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::FunctionCallOutput {
                    id: fco.id,
                    call_id: fco.call_id,
                    output: fco.output,
                })
            }
            "function_call" => {
                let fc: TypedFunctionCall =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::FunctionCall {
                    id: fc.id,
                    call_id: fc.call_id,
                    name: fc.name,
                    arguments: fc.arguments,
                })
            }
            "custom_tool_call" => {
                let tc: TypedCustomToolCall =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::CustomToolCall {
                    id: tc.id,
                    status: tc.status,
                    call_id: tc.call_id,
                    name: tc.name,
                    input: tc.input,
                })
            }
            "custom_tool_call_output" => {
                let tco: TypedCustomToolCallOutput =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::CustomToolCallOutput {
                    id: tco.id,
                    call_id: tco.call_id,
                    output: tco.output,
                })
            }
            "local_shell_call" => {
                let sc: TypedLocalShellCall =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::LocalShellCall {
                    id: sc.id,
                    call_id: sc.call_id,
                    status: sc.status,
                    action: sc.action,
                })
            }
            "reasoning" => {
                let r: TypedReasoning =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::Reasoning {
                    id: r.id,
                    summary: r.summary,
                    content: r.content,
                    encrypted_content: r.encrypted_content,
                })
            }
            "web_search_call" => {
                let ws: TypedWebSearchCall =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::WebSearchCall {
                    id: ws.id,
                    status: ws.status,
                    action: ws.action,
                })
            }
            "ghost_snapshot" => {
                let gs: TypedGhostSnapshot =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::GhostSnapshot {
                    id: gs.id,
                    ghost_commit: gs.ghost_commit,
                })
            }
            "compaction" | "compaction_summary" => {
                let c: TypedCompaction =
                    serde_json::from_value(v).map_err(serde::de::Error::custom)?;
                Ok(ResponsesInputItem::Compaction {
                    id: c.id,
                    encrypted_content: c.encrypted_content,
                })
            }
            other => {
                let id = v
                    .get("id")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(default_msg_id);
                Ok(ResponsesInputItem::Other {
                    id,
                    kind: other.to_string(),
                    raw: v,
                })
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

impl ResponsesInput {
    pub fn into_items(self) -> Vec<ResponsesInputItem> {
        match self {
            ResponsesInput::Text(s) => vec![ResponsesInputItem::Message {
                id: default_msg_id(),
                role: "user".to_string(),
                content: ResponsesInputMessageContent::Text(s),
            }],
            ResponsesInput::Items(items) => items,
        }
    }
}

fn default_msg_id() -> String {
    format!("msg_{}", Uuid::new_v4())
}

fn default_fco_id() -> String {
    format!("fco_{}", Uuid::new_v4())
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTextFormatParam {
    Text,
    JsonSchema {
        /// Friendly name for the schema, used for telemetry/debugging.
        name: String,
        /// JSON schema for the desired output.
        schema: Value,
        /// When true, the server is expected to strictly validate responses.
        #[serde(default)]
        strict: bool,
    },
}

impl<'de> Deserialize<'de> for ResponsesTextFormatParam {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct NestedJsonSchema {
            name: String,
            schema: Value,
            #[serde(default)]
            strict: bool,
        }

        let v = serde_json::Value::deserialize(deserializer)?;
        let Some(obj) = v.as_object() else {
            return Err(serde::de::Error::custom("text.format must be an object"));
        };

        let format_type = obj.get("type").and_then(|t| t.as_str()).unwrap_or("text");

        match format_type {
            "text" => Ok(ResponsesTextFormatParam::Text),
            "json_schema" => {
                // OpenAI-style nested shape:
                // { "type": "json_schema", "json_schema": { "name": ..., "schema": ..., "strict": ... } }
                if let Some(nested) = obj.get("json_schema") {
                    let ns: NestedJsonSchema =
                        serde_json::from_value(nested.clone()).map_err(serde::de::Error::custom)?;
                    return Ok(ResponsesTextFormatParam::JsonSchema {
                        name: ns.name,
                        schema: ns.schema,
                        strict: ns.strict,
                    });
                }

                // Codex-style flat shape:
                // { "type": "json_schema", "name": ..., "schema": ..., "strict": ... }
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::custom("text.format.name is required"))?
                    .to_string();
                let schema = obj
                    .get("schema")
                    .cloned()
                    .ok_or_else(|| serde::de::Error::custom("text.format.schema is required"))?;
                let strict = obj.get("strict").and_then(|v| v.as_bool()).unwrap_or(false);

                Ok(ResponsesTextFormatParam::JsonSchema {
                    name,
                    schema,
                    strict,
                })
            }
            other => Err(serde::de::Error::custom(format!(
                "Unsupported text.format type: {other}"
            ))),
        }
    }
}

/// Controls the `text` field for the Responses API, combining verbosity and optional JSON schema formatting.
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ResponsesTextControls {
    #[serde(default)]
    pub verbosity: Option<String>,
    #[serde(default)]
    pub format: Option<ResponsesTextFormatParam>,
}

/// Response creation request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ResponsesCreateRequest {
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    pub input: ResponsesInput,
    #[schema(example = json!(Option::None::<String>))]
    pub instructions: Option<String>,
    #[schema(example = json!(Option::None::<Vec<String>>))]
    pub modalities: Option<Vec<String>>,
    #[schema(example = json!(Option::None::<String>))]
    pub previous_response_id: Option<String>,
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    #[schema(example = json!(Option::None::<usize>))]
    pub top_logprobs: Option<usize>,
    #[schema(example = 256)]
    #[serde(alias = "max_completion_tokens", alias = "max_output_tokens")]
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
    #[schema(example = false)]
    pub stream: Option<bool>,
    #[schema(example = json!(Option::None::<Vec<Tool>>))]
    pub tools: Option<Vec<Tool>>,
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,
    #[schema(example = json!(Option::None::<ResponsesTextControls>))]
    pub text: Option<ResponsesTextControls>,
    #[schema(example = json!(Option::None::<ResponseFormat>))]
    pub response_format: Option<ResponseFormat>,
    #[schema(example = json!(Option::None::<WebSearchOptions>))]
    pub web_search_options: Option<WebSearchOptions>,
    #[schema(example = json!(Option::None::<Value>))]
    pub metadata: Option<Value>,
    #[schema(example = json!(Option::None::<bool>))]
    pub output_token_details: Option<bool>,
    #[schema(example = json!(Option::None::<bool>))]
    pub parallel_tool_calls: Option<bool>,
    #[schema(example = json!(Option::None::<bool>))]
    pub store: Option<bool>,
    #[schema(example = json!(Option::None::<usize>))]
    pub max_tool_calls: Option<usize>,
    #[schema(example = json!(Option::None::<bool>))]
    pub reasoning_enabled: Option<bool>,
    #[schema(example = json!(Option::None::<usize>))]
    pub reasoning_max_tokens: Option<usize>,
    #[schema(example = json!(Option::None::<usize>))]
    pub reasoning_top_logprobs: Option<usize>,
    #[schema(example = json!(Option::None::<Vec<String>>))]
    #[serde(default, deserialize_with = "deserialize_truncation")]
    pub truncation: Option<String>,

    // mistral.rs additional
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
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
    #[schema(example = json!(Option::None::<bool>))]
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
    #[schema(example = json!(Option::None::<bool>))]
    pub compact_history_with_summary: Option<bool>,
}

#[cfg(test)]
mod responses_input_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn responses_message_shorthand_with_image_url_string_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is in this image?"},
                        {"type": "input_image", "image_url": "https://example.com/image.jpg"}
                    ]
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        assert_eq!(items.len(), 1);

        match &items[0] {
            ResponsesInputItem::Message { role, content, .. } => {
                assert_eq!(role, "user");
                let parts = content.clone().into_parts();
                assert_eq!(parts.len(), 2);
                assert!(matches!(
                    parts[0],
                    ResponsesInputContentPart::InputText { .. }
                ));
                match &parts[1] {
                    ResponsesInputContentPart::InputImage { image_url, .. } => {
                        let url = image_url.as_ref().expect("image_url");
                        assert_eq!(url.url(), "https://example.com/image.jpg");
                    }
                    other => panic!("expected input_image, got {other:?}"),
                }
            }
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn responses_message_accepts_output_text_parts() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "hello"}
                    ]
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::Message { role, content, .. } => {
                assert_eq!(role, "assistant");
                let parts = content.clone().into_parts();
                assert!(matches!(
                    parts[0],
                    ResponsesInputContentPart::OutputText { ref text } if text == "hello"
                ));
            }
            other => panic!("expected message, got {other:?}"),
        }
    }

    #[test]
    fn responses_function_call_output_accepts_array_output() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call1",
                    "output": [
                        {"type": "input_text", "text": "note"},
                        {"type": "input_image", "image_url": "data:image/png;base64,XYZ"}
                    ]
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::FunctionCallOutput {
                call_id, output, ..
            } => {
                assert_eq!(call_id, "call1");
                let parts = output.clone().into_parts();
                assert_eq!(parts.len(), 2);
                assert!(matches!(
                    parts[0],
                    ResponsesInputContentPart::InputText { .. }
                ));
                assert!(matches!(
                    parts[1],
                    ResponsesInputContentPart::InputImage { .. }
                ));
            }
            other => panic!("expected function_call_output, got {other:?}"),
        }
    }

    #[test]
    fn responses_function_call_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call1",
                    "name": "shell",
                    "arguments": "{\"cmd\":\"ls\"}"
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => {
                assert_eq!(call_id, "call1");
                assert_eq!(name, "shell");
                assert_eq!(arguments, "{\"cmd\":\"ls\"}");
            }
            other => panic!("expected function_call, got {other:?}"),
        }
    }

    #[test]
    fn responses_local_shell_call_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "local_shell_call",
                    "call_id": "call1",
                    "status": "completed",
                    "action": {
                        "type": "exec",
                        "command": ["ls"],
                        "timeout_ms": 1000
                    }
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::LocalShellCall {
                call_id,
                status,
                action,
                ..
            } => {
                assert_eq!(call_id.as_deref(), Some("call1"));
                assert_eq!(status.as_deref(), Some("completed"));
                assert!(action.is_some());
            }
            other => panic!("expected local_shell_call, got {other:?}"),
        }
    }

    #[test]
    fn responses_reasoning_item_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "reasoning",
                    "id": "rsn_1",
                    "summary": [{"type": "summary_text", "text": "short summary"}],
                    "content": [
                        {"type": "reasoning_text", "text": "hidden"},
                        {"type": "text", "text": "visible"}
                    ],
                    "encrypted_content": null
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::Reasoning {
                id,
                summary,
                content,
                encrypted_content,
                ..
            } => {
                assert_eq!(id, "rsn_1");
                assert_eq!(summary.len(), 1);
                assert!(content.as_ref().is_some_and(|c| c.len() == 2));
                assert!(encrypted_content.is_none());
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
    }

    #[test]
    fn responses_web_search_call_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "weather: SF"}
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::WebSearchCall { id, status, action } => {
                assert_eq!(id, "ws_1");
                assert_eq!(status.as_deref(), Some("completed"));
                assert!(matches!(action, ResponsesWebSearchAction::Search { .. }));
            }
            other => panic!("expected web_search_call, got {other:?}"),
        }
    }

    #[test]
    fn responses_compaction_summary_alias_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "compaction_summary",
                    "encrypted_content": "abc"
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::Compaction {
                encrypted_content, ..
            } => {
                assert_eq!(encrypted_content, "abc");
            }
            other => panic!("expected compaction, got {other:?}"),
        }
    }

    #[test]
    fn responses_ghost_snapshot_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "ghost_snapshot",
                    "ghost_commit": {"sha": "deadbeef"}
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::GhostSnapshot { ghost_commit, .. } => {
                assert_eq!(
                    ghost_commit.get("sha").and_then(|v| v.as_str()),
                    Some("deadbeef")
                );
            }
            other => panic!("expected ghost_snapshot, got {other:?}"),
        }
    }

    #[test]
    fn responses_unknown_item_is_accepted_as_other() {
        let v = json!({
            "model": "gpt-4.1",
            "input": [
                {
                    "type": "mystery_item",
                    "foo": "bar"
                }
            ]
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let items = req.input.into_items();
        match &items[0] {
            ResponsesInputItem::Other { kind, raw, .. } => {
                assert_eq!(kind, "mystery_item");
                assert_eq!(raw.get("foo").and_then(|v| v.as_str()), Some("bar"));
            }
            other => panic!("expected other, got {other:?}"),
        }
    }

    #[test]
    fn responses_text_format_json_schema_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": "hi",
            "text": {
                "verbosity": "medium",
                "format": {
                    "type": "json_schema",
                    "name": "codex_output_schema",
                    "strict": true,
                    "schema": {"type": "object", "properties": {"x": {"type": "string"}}}
                }
            }
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let text = req.text.expect("text");
        let fmt = text.format.expect("format");
        match fmt {
            ResponsesTextFormatParam::JsonSchema {
                name,
                schema,
                strict,
            } => {
                assert_eq!(name, "codex_output_schema");
                assert!(strict);
                assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
            }
            other => panic!("expected json_schema, got {other:?}"),
        }
    }

    #[test]
    fn responses_text_format_json_schema_nested_deserializes() {
        let v = json!({
            "model": "gpt-4.1",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "nested",
                        "strict": true,
                        "schema": {"type": "object"}
                    }
                }
            }
        });

        let req: ResponsesCreateRequest = serde_json::from_value(v).expect("deserialize");
        let fmt = req.text.unwrap().format.unwrap();
        match fmt {
            ResponsesTextFormatParam::JsonSchema { name, strict, .. } => {
                assert_eq!(name, "nested");
                assert!(strict);
            }
            other => panic!("expected json_schema, got {other:?}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesTextFormat {
    #[serde(rename = "type")]
    pub format_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesTextConfig {
    pub format: ResponsesTextFormat,
}

/// Response object
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponsesOutput>,
    pub output_text: Option<String>,
    pub usage: Option<ResponsesUsage>,
    pub error: Option<ResponsesError>,
    pub metadata: Option<Value>,
    pub instructions: Option<String>,
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    pub previous_response_id: Option<String>,
    pub store: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub truncation: Option<String>,
    pub tool_choice: Option<Value>,
    pub tools: Option<Vec<Tool>>,
    pub parallel_tool_calls: Option<bool>,
    pub text: Option<ResponsesTextConfig>,
    pub max_output_tokens: Option<usize>,
    pub max_tool_calls: Option<usize>,
}

/// Response usage information
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
    pub input_tokens_details: Option<ResponsesInputTokensDetails>,
    pub output_tokens_details: Option<ResponsesOutputTokensDetails>,
}

/// Input tokens details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesInputTokensDetails {
    pub audio_tokens: Option<usize>,
    pub cached_tokens: Option<usize>,
    pub image_tokens: Option<usize>,
    pub text_tokens: Option<usize>,
}

/// Output tokens details
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesOutputTokensDetails {
    pub audio_tokens: Option<usize>,
    pub text_tokens: Option<usize>,
    pub reasoning_tokens: Option<usize>,
}

/// Response error
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesError {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// Incomplete details for incomplete responses
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesIncompleteDetails {
    pub reason: String,
}

/// Response output item
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesOutput {
    pub id: String,
    #[serde(rename = "type")]
    pub output_type: String,
    #[serde(default)]
    pub role: String,
    pub status: Option<String>,
    pub content: Vec<ResponsesContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// Response content item
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
    pub annotations: Option<Vec<ResponsesAnnotation>>,
}

/// Response annotation
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesAnnotation {
    #[serde(rename = "type")]
    pub annotation_type: String,
    pub text: String,
    pub start_index: usize,
    pub end_index: usize,
}

/// Response streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesChunk {
    pub id: String,
    pub object: &'static str,
    pub created_at: f64,
    pub model: String,
    pub chunk_type: String,
    pub delta: Option<ResponsesDelta>,
    pub usage: Option<ResponsesUsage>,
    pub metadata: Option<Value>,
}

/// Response delta for streaming
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesDelta {
    pub output: Option<Vec<ResponsesDeltaOutput>>,
    pub status: Option<String>,
}

/// Response delta output item
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesDeltaOutput {
    pub id: String,
    #[serde(rename = "type")]
    pub output_type: String,
    pub content: Option<Vec<ResponsesDeltaContent>>,
}

/// Response delta content item
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesDeltaContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
}
