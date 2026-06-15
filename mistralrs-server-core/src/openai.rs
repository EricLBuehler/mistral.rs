//! ## OpenAI compatible functionality.

use std::{collections::HashMap, ops::Deref};

use anyhow::{bail, Result};
use either::Either;
use mistralrs_core::{
    AgentPermission, ApproximateUserLocation, CodeExecutionPermission,
    ImageGenerationResponseFormat, LlguidanceGrammar, SearchContextSize, Tool, ToolChoice,
    ToolType, WebSearchContentType, WebSearchFilters, WebSearchImageSettings, WebSearchOptions,
    WebSearchReturnTokenBudget, WebSearchUserLocation,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
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

impl MessageContent {
    /// Create a new MessageContent from a string
    pub fn from_text(text: String) -> Self {
        MessageContent(Either::Left(text))
    }

    /// Create a new MessageContent from multimodal parts
    pub fn from_parts(parts: Vec<HashMap<String, MessageInnerContent>>) -> Self {
        MessageContent(Either::Right(parts))
    }

    /// Create a text content part for multimodal messages
    pub fn text_part(text: String) -> HashMap<String, MessageInnerContent> {
        let mut part = HashMap::new();
        part.insert(
            "type".to_string(),
            MessageInnerContent(Either::Left("text".to_string())),
        );
        part.insert("text".to_string(), MessageInnerContent(Either::Left(text)));
        part
    }

    /// Create an image URL content part for multimodal messages
    pub fn image_url_part(url: String) -> HashMap<String, MessageInnerContent> {
        let mut part = HashMap::new();
        part.insert(
            "type".to_string(),
            MessageInnerContent(Either::Left("image_url".to_string())),
        );
        let mut image_url_obj = HashMap::new();
        image_url_obj.insert("url".to_string(), url);
        part.insert(
            "image_url".to_string(),
            MessageInnerContent(Either::Right(image_url_obj)),
        );
        part
    }

    /// Create an image URL content part with detail level
    pub fn image_url_part_with_detail(
        url: String,
        detail: String,
    ) -> HashMap<String, MessageInnerContent> {
        let mut part = HashMap::new();
        part.insert(
            "type".to_string(),
            MessageInnerContent(Either::Left("image_url".to_string())),
        );
        let mut image_url_obj = HashMap::new();
        image_url_obj.insert("url".to_string(), url);
        image_url_obj.insert("detail".to_string(), detail);
        part.insert(
            "image_url".to_string(),
            MessageInnerContent(Either::Right(image_url_obj)),
        );
        part
    }

    pub fn file_part(
        file_id: Option<String>,
        file_data: Option<String>,
        file_url: Option<String>,
        filename: Option<String>,
    ) -> HashMap<String, MessageInnerContent> {
        let mut part = HashMap::new();
        part.insert(
            "type".to_string(),
            MessageInnerContent(Either::Left("file".to_string())),
        );
        let mut file_obj = HashMap::new();
        if let Some(file_id) = file_id {
            file_obj.insert("file_id".to_string(), file_id);
        }
        if let Some(file_data) = file_data {
            file_obj.insert("file_data".to_string(), file_data);
        }
        if let Some(file_url) = file_url {
            file_obj.insert("file_url".to_string(), file_url);
        }
        if let Some(filename) = filename {
            file_obj.insert("filename".to_string(), filename);
        }
        part.insert(
            "file".to_string(),
            MessageInnerContent(Either::Right(file_obj)),
        );
        part
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
    /// The function arguments (JSON string)
    #[serde(alias = "parameters")]
    pub arguments: String,
}

/// Represents a tool call made by the assistant
///
/// This structure wraps a function call with its type information.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    #[serde(default)]
    pub id: Option<String>,
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
    /// Optional participant name for this message
    pub name: Option<String>,
    /// Optional list of tool calls (for assistant messages)
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call ID this message is responding to (for tool messages)
    pub tool_call_id: Option<String>,
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

const MAX_WEB_SEARCH_FILTER_DOMAINS: usize = 100;

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

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum OpenAiFunctionToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum OpenAiWebSearchToolType {
    #[serde(rename = "web_search")]
    WebSearch,
    #[serde(rename = "web_search_preview")]
    WebSearchPreview,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum OpenAiCodeInterpreterToolType {
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum OpenAiShellToolType {
    #[serde(rename = "shell")]
    Shell,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAiTool {
    Function(Tool),
    ResponsesFunction(OpenAiResponsesFunctionTool),
    WebSearch(OpenAiWebSearchTool),
    CodeInterpreter(OpenAiCodeInterpreterTool),
    Shell(OpenAiShellTool),
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OpenAiResponsesFunctionTool {
    #[serde(rename = "type")]
    pub tp: OpenAiFunctionToolType,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<HashMap<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl OpenAiResponsesFunctionTool {
    fn into_core_tool(self) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: mistralrs_core::Function {
                description: self.description,
                name: self.name,
                parameters: self.parameters,
                strict: self.strict,
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OpenAiWebSearchTool {
    #[serde(rename = "type")]
    pub tp: OpenAiWebSearchToolType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<SearchContextSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<OpenAiWebSearchUserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<WebSearchFilters>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_web_access: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_token_budget: Option<WebSearchReturnTokenBudget>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_content_types: Option<Vec<WebSearchContentType>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_settings: Option<WebSearchImageSettings>,
}

impl OpenAiWebSearchTool {
    fn into_web_search_options(self) -> Result<WebSearchOptions> {
        let is_preview = matches!(&self.tp, OpenAiWebSearchToolType::WebSearchPreview);
        if is_preview && self.filters.is_some() {
            bail!("tools[].type=\"web_search_preview\" filters are not supported.");
        }
        if is_preview && self.return_token_budget.is_some() {
            bail!("tools[].type=\"web_search_preview\" return_token_budget is not supported.");
        }
        if !is_preview && self.external_web_access == Some(false) {
            bail!("tools[].type=\"web_search\" external_web_access=false is not supported.");
        }
        if self.image_settings.is_some()
            || self.search_content_types.as_ref().is_some_and(|types| {
                types
                    .iter()
                    .any(|tp| matches!(tp, WebSearchContentType::Image))
            })
        {
            bail!("tools[].type=\"web_search\" image search is not supported.");
        }
        if let Some(filters) = &self.filters {
            validate_web_search_filter_domains("allowed_domains", &filters.allowed_domains)?;
            validate_web_search_filter_domains("blocked_domains", &filters.blocked_domains)?;
        }

        Ok(WebSearchOptions {
            search_context_size: self.search_context_size,
            user_location: self.user_location.map(Into::into),
            filters: self.filters,
            external_web_access: (!is_preview).then_some(self.external_web_access).flatten(),
            return_token_budget: self.return_token_budget,
            search_content_types: self.search_content_types,
            image_settings: self.image_settings,
            search_description: None,
            extract_description: None,
        })
    }
}

fn validate_web_search_filter_domains(label: &str, domains: &Option<Vec<String>>) -> Result<()> {
    if domains
        .as_ref()
        .is_some_and(|domains| domains.len() > MAX_WEB_SEARCH_FILTER_DOMAINS)
    {
        bail!("tools[].type=\"web_search\" filters.{label} may contain at most {MAX_WEB_SEARCH_FILTER_DOMAINS} domains.");
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum OpenAiWebSearchUserLocation {
    #[serde(rename = "approximate")]
    Approximate {
        #[serde(skip_serializing_if = "Option::is_none")]
        city: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        country: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        region: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timezone: Option<String>,
    },
}

impl From<OpenAiWebSearchUserLocation> for WebSearchUserLocation {
    fn from(location: OpenAiWebSearchUserLocation) -> Self {
        match location {
            OpenAiWebSearchUserLocation::Approximate {
                city,
                country,
                region,
                timezone,
            } => Self::Approximate {
                approximate: ApproximateUserLocation {
                    city,
                    country,
                    region,
                    timezone,
                },
            },
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OpenAiCodeInterpreterTool {
    #[serde(rename = "type")]
    pub tp: OpenAiCodeInterpreterToolType,
    pub container: OpenAiCodeInterpreterContainer,
}

impl OpenAiCodeInterpreterTool {
    fn validate_supported(&self) -> Result<()> {
        match &self.container {
            OpenAiCodeInterpreterContainer::Id(_) => {
                bail!("tools[].type=\"code_interpreter\" container IDs are not supported.")
            }
            OpenAiCodeInterpreterContainer::Auto(container) => {
                if container.memory_limit.is_some() {
                    bail!(
                        "tools[].type=\"code_interpreter\" container.memory_limit is not supported."
                    );
                }
                if container
                    .file_ids
                    .as_ref()
                    .is_some_and(|file_ids| !file_ids.is_empty())
                {
                    bail!("tools[].type=\"code_interpreter\" container.file_ids is not supported.");
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum OpenAiCodeInterpreterContainer {
    Id(String),
    Auto(OpenAiCodeInterpreterAutoContainer),
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OpenAiCodeInterpreterAutoContainer {
    #[serde(rename = "type")]
    pub tp: OpenAiCodeInterpreterContainerType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_limit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_ids: Option<Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
pub enum OpenAiCodeInterpreterContainerType {
    #[serde(rename = "auto")]
    Auto,
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct OpenAiShellTool {
    #[serde(rename = "type")]
    pub tp: OpenAiShellToolType,
    pub environment: OpenAiShellEnvironment,
}

impl OpenAiShellTool {
    fn into_skill_references(self) -> Result<Vec<OpenAiShellSkillReference>> {
        match self.environment {
            OpenAiShellEnvironment::ContainerAuto { skills } => {
                let mut refs = Vec::new();
                for skill in skills {
                    match skill {
                        OpenAiShellSkill::SkillReference { skill_id, version } => {
                            refs.push(OpenAiShellSkillReference { skill_id, version });
                        }
                        OpenAiShellSkill::Local { .. } => {
                            bail!("tools[].type=\"shell\" local skills are not supported.");
                        }
                    }
                }
                Ok(refs)
            }
            OpenAiShellEnvironment::Local { .. } => {
                bail!("tools[].type=\"shell\" local environments are not supported.")
            }
            OpenAiShellEnvironment::ContainerReference { .. } => {
                bail!("tools[].type=\"shell\" container references are not supported.")
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum OpenAiShellEnvironment {
    #[serde(rename = "container_auto")]
    ContainerAuto {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        skills: Vec<OpenAiShellSkill>,
    },
    #[serde(rename = "local")]
    Local { path: String },
    #[serde(rename = "container_reference")]
    ContainerReference { container_id: String },
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum OpenAiShellSkill {
    #[serde(rename = "skill_reference")]
    SkillReference {
        skill_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        version: Option<Value>,
    },
    #[serde(rename = "local")]
    Local { path: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiShellSkillReference {
    pub skill_id: String,
    pub version: Option<Value>,
}

#[derive(Clone, Debug, Default)]
pub struct OpenAiToolNormalization {
    pub tools: Option<Vec<Tool>>,
    pub web_search_options: Option<WebSearchOptions>,
    pub enable_code_execution: bool,
    pub enable_shell: bool,
    pub shell_skill_references: Vec<OpenAiShellSkillReference>,
}

impl OpenAiToolNormalization {
    fn has_available_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
            || self.web_search_options.is_some()
            || self.enable_code_execution
            || self.enable_shell
    }

    fn has_function_tool(&self, name: &str) -> bool {
        self.tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| tool.function.name == name))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum OpenAiToolSurface {
    ChatCompletions,
    Responses,
}

pub fn normalize_chat_completion_tools(
    tools: Option<Vec<OpenAiTool>>,
    web_search_options: Option<WebSearchOptions>,
) -> Result<OpenAiToolNormalization> {
    normalize_openai_tools(
        tools,
        web_search_options,
        OpenAiToolSurface::ChatCompletions,
    )
}

pub fn normalize_responses_tools(
    tools: Option<Vec<OpenAiTool>>,
) -> Result<OpenAiToolNormalization> {
    normalize_openai_tools(tools, None, OpenAiToolSurface::Responses)
}

pub fn validate_openai_tool_choice(
    tool_choice: Option<&ToolChoice>,
    normalized_tools: &OpenAiToolNormalization,
) -> Result<()> {
    match tool_choice {
        Some(ToolChoice::Required) if !normalized_tools.has_available_tools() => {
            bail!("tool_choice=\"required\" requires at least one tool.")
        }
        Some(ToolChoice::Tool(tool))
            if !normalized_tools.has_function_tool(&tool.function.name) =>
        {
            bail!(
                "tool_choice references unknown function tool `{}`.",
                tool.function.name
            )
        }
        Some(ToolChoice::NamedFunction(choice))
            if !normalized_tools.has_function_tool(&choice.name) =>
        {
            bail!(
                "tool_choice references unknown function tool `{}`.",
                choice.name
            )
        }
        Some(ToolChoice::None | ToolChoice::Auto | ToolChoice::Required)
        | Some(ToolChoice::Tool(_))
        | Some(ToolChoice::NamedFunction(_))
        | None => Ok(()),
    }
}

fn normalize_openai_tools(
    tools: Option<Vec<OpenAiTool>>,
    web_search_options: Option<WebSearchOptions>,
    surface: OpenAiToolSurface,
) -> Result<OpenAiToolNormalization> {
    let mut function_tools = Vec::new();
    let mut normalized_web_search_options = web_search_options;
    let mut enable_code_execution = false;
    let mut enable_shell = false;
    let mut shell_skill_references = Vec::new();

    for tool in tools.unwrap_or_default() {
        match tool {
            OpenAiTool::Function(tool) => function_tools.push(tool),
            OpenAiTool::ResponsesFunction(tool) => function_tools.push(tool.into_core_tool()),
            OpenAiTool::WebSearch(tool) => {
                if matches!(surface, OpenAiToolSurface::ChatCompletions) {
                    bail!(
                        "tools[].type=\"web_search\" is only supported by the Responses API; \
                         use web_search_options with Chat Completions"
                    );
                }
                if normalized_web_search_options.is_some() {
                    bail!("Only one web search configuration may be provided.");
                }
                normalized_web_search_options = Some(tool.into_web_search_options()?);
            }
            OpenAiTool::CodeInterpreter(tool) => {
                tool.validate_supported()?;
                enable_code_execution = true;
            }
            OpenAiTool::Shell(tool) => {
                if matches!(surface, OpenAiToolSurface::ChatCompletions) {
                    bail!("tools[].type=\"shell\" is only supported by the Responses API.");
                }
                enable_shell = true;
                shell_skill_references.extend(tool.into_skill_references()?);
            }
        }
    }

    Ok(OpenAiToolNormalization {
        tools: if function_tools.is_empty() {
            None
        } else {
            Some(function_tools)
        },
        web_search_options: normalized_web_search_options,
        enable_code_execution,
        enable_shell,
        shell_skill_references,
    })
}

/// Chat completion request following OpenAI's specification
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ChatCompletionRequest {
    /// The conversation so far, or a single raw prompt string.
    #[schema(
        schema_with = messages_schema,
        example = json!(vec![Message{content:Some(MessageContent{0: either::Left(("Why did the crab cross the road?".to_string()))}), role:"user".to_string(), name: None, tool_calls: None}])
    )]
    #[serde(with = "either::serde_untagged")]
    pub messages: Either<Vec<Message>, String>,
    /// Model ID; "default" targets the only loaded model.
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    /// Bias added to the logits of these token IDs before sampling.
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Return log probabilities of the output tokens.
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    /// Number of most likely tokens to return per position; requires `logprobs`.
    #[schema(example = json!(Option::None::<usize>))]
    pub top_logprobs: Option<usize>,
    /// Maximum number of tokens to generate.
    #[schema(example = 256)]
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    /// How many choices to generate.
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    /// Penalize tokens that have already appeared; positive values push toward new topics.
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    /// Penalize tokens by how often they have appeared so far; positive values reduce repetition.
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    /// Multiplicative repetition penalty; 1.0 disables it.
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
    /// Sequences where generation stops.
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    /// Sampling temperature; higher values increase randomness.
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    /// Nucleus sampling: only tokens within the top cumulative probability mass are considered.
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    /// Stream the response as server-sent events.
    #[schema(example = true)]
    pub stream: Option<bool>,
    /// Tools the model may call.
    #[schema(example = json!(Option::None::<Vec<OpenAiTool>>))]
    pub tools: Option<Vec<OpenAiTool>>,
    /// Controls which (if any) tool the model must call.
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,
    /// Force plain text or JSON-schema constrained output.
    #[schema(example = json!(Option::None::<ResponseFormat>))]
    pub response_format: Option<ResponseFormat>,
    /// Enable the built-in web search tool.
    #[schema(example = json!(Option::None::<WebSearchOptions>))]
    pub web_search_options: Option<WebSearchOptions>,
    /// Permission policy for agentic tools.
    #[schema(value_type = Option<String>, example = json!(Option::None::<String>))]
    pub agent_permission: Option<AgentPermission>,
    /// Permission policy for code execution.
    #[schema(value_type = Option<String>, example = json!(Option::None::<String>))]
    pub code_execution_permission: Option<CodeExecutionPermission>,
    /// Enable the built-in shell execution tool.
    #[serde(default)]
    #[schema(example = false)]
    pub enable_shell: bool,
    /// Persistent agentic state. If `None`, a new session is created and the ID is returned in the response.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Required output files. The runtime asks the model to produce them and surfaces a `File` (or error placeholder) for each.
    #[serde(default)]
    #[schema(value_type = Option<Vec<serde_json::Value>>)]
    pub files: Option<Vec<mistralrs_core::RequestedFile>>,

    // mistral.rs additional
    /// Sample only from the k most likely tokens.
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    /// Constrain output with a regex, JSON schema, Lark, or LLGuidance grammar.
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    /// Drop tokens below this fraction of the top token's probability.
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    /// DRY repetition penalty multiplier; 0 disables DRY.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    /// Base for DRY's exponential penalty growth.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    /// Longest repeated sequence DRY leaves unpenalized.
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    /// Sequences that reset DRY repetition matching.
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Toggle thinking output for models that support it.
    #[schema(example = json!(Option::None::<bool>))]
    pub enable_thinking: Option<bool>,
    /// Reasoning effort level for Harmony-format models (GPT-OSS).
    /// Controls the depth of reasoning/analysis: "low", "medium", or "high".
    #[schema(example = json!(Option::None::<String>))]
    pub reasoning_effort: Option<String>,
    /// Maximum number of tool-call rounds the server will auto-execute.
    #[schema(example = json!(Option::None::<usize>))]
    pub max_tool_rounds: Option<usize>,
    /// Truncate inputs that exceed the model's context length instead of erroring.
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
    /// Model status: "loaded", "unloaded", or "reloading"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
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
    /// Model ID; "default" targets the only loaded model.
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(example = "Say this is a test.")]
    pub prompt: String,
    #[schema(example = 1)]
    pub best_of: Option<usize>,
    /// Echo the prompt back alongside the completion.
    #[serde(rename = "echo")]
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub echo_prompt: bool,
    /// Penalize tokens that have already appeared; positive values push toward new topics.
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    /// Penalize tokens by how often they have appeared so far; positive values reduce repetition.
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    /// Bias added to the logits of these token IDs before sampling.
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Include log probabilities of this many most likely tokens.
    #[schema(example = json!(Option::None::<usize>))]
    pub logprobs: Option<usize>,
    /// Maximum number of tokens to generate.
    #[schema(example = 16)]
    #[serde(alias = "max_completion_tokens")]
    pub max_tokens: Option<usize>,
    /// How many choices to generate.
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    /// Sequences where generation stops.
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    /// Stream the response as server-sent events.
    pub stream: Option<bool>,
    /// Sampling temperature; higher values increase randomness.
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    /// Nucleus sampling: only tokens within the top cumulative probability mass are considered.
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    /// Text appended after the completion.
    #[schema(example = json!(Option::None::<String>))]
    pub suffix: Option<String>,
    #[serde(rename = "user")]
    pub _user: Option<String>,
    /// Tools the model may call.
    #[schema(example = json!(Option::None::<Vec<Tool>>))]
    pub tools: Option<Vec<Tool>>,
    /// Controls which (if any) tool the model must call.
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,

    // mistral.rs additional
    /// Sample only from the k most likely tokens.
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    /// Constrain output with a regex, JSON schema, Lark, or LLGuidance grammar.
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    /// Drop tokens below this fraction of the top token's probability.
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    /// Multiplicative repetition penalty; 1.0 disables it.
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
    /// DRY repetition penalty multiplier; 0 disables DRY.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    /// Base for DRY's exponential penalty growth.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    /// Longest repeated sequence DRY leaves unpenalized.
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    /// Sequences that reset DRY repetition matching.
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Truncate inputs that exceed the model's context length instead of erroring.
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
    /// Model ID; "default" targets the only loaded model.
    #[schema(example = "default")]
    #[serde(default = "default_model")]
    pub model: String,
    /// Text or token inputs to embed.
    pub input: EmbeddingInput,
    /// Return embeddings as float arrays or base64-encoded strings.
    #[schema(example = "float")]
    #[serde(default)]
    pub encoding_format: Option<EmbeddingEncodingFormat>,
    /// Truncate embeddings to this dimensionality, if the model supports it.
    #[schema(example = json!(Option::None::<usize>))]
    pub dimensions: Option<usize>,
    #[schema(example = json!(Option::None::<String>))]
    #[serde(rename = "user")]
    pub _user: Option<String>,

    // mistral.rs additional
    /// Truncate inputs that exceed the model's context length instead of erroring.
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
    /// Model ID; "default" targets the only loaded model.
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(example = "Draw a picture of a majestic, snow-covered mountain.")]
    pub prompt: String,
    /// How many choices to generate.
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    /// Return generated images as URLs or base64 data.
    #[serde(default = "default_response_format")]
    pub response_format: ImageGenerationResponseFormat,
    /// Image height in pixels
    #[serde(default = "default_720usize")]
    #[schema(example = 720)]
    pub height: usize,
    /// Image width in pixels
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
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ResponsesMessages {
    Messages(Vec<Message>),
    String(String),
}

impl ResponsesMessages {
    pub fn into_either(self) -> Either<Vec<Message>, String> {
        match self {
            ResponsesMessages::Messages(msgs) => Either::Left(msgs),
            ResponsesMessages::String(s) => Either::Right(s),
        }
    }
}

impl PartialSchema for ResponsesMessages {
    fn schema() -> RefOr<Schema> {
        RefOr::T(messages_schema())
    }
}

impl ToSchema for ResponsesMessages {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            ResponsesMessages::name().into(),
            ResponsesMessages::schema(),
        ));
    }
}

/// Response creation request
#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct ResponsesCreateRequest {
    /// Model ID; "default" targets the only loaded model.
    #[schema(example = "mistral")]
    #[serde(default = "default_model")]
    pub model: String,
    /// Input messages or a single raw prompt string.
    pub input: ResponsesMessages,
    /// System instructions prepended to the conversation.
    #[schema(example = json!(Option::None::<String>))]
    pub instructions: Option<String>,
    /// Requested output modalities.
    #[schema(example = json!(Option::None::<Vec<String>>))]
    pub modalities: Option<Vec<String>>,
    /// Continue the conversation from a stored previous response.
    #[schema(example = json!(Option::None::<String>))]
    pub previous_response_id: Option<String>,
    /// Bias added to the logits of these token IDs before sampling.
    #[schema(example = json!(Option::None::<HashMap<u32, f32>>))]
    pub logit_bias: Option<HashMap<u32, f32>>,
    /// Return log probabilities of the output tokens.
    #[serde(default = "default_false")]
    #[schema(example = false)]
    pub logprobs: bool,
    /// Number of most likely tokens to return per position; requires `logprobs`.
    #[schema(example = json!(Option::None::<usize>))]
    pub top_logprobs: Option<usize>,
    /// Maximum number of tokens to generate.
    #[schema(example = 256)]
    #[serde(alias = "max_completion_tokens", alias = "max_output_tokens")]
    pub max_tokens: Option<usize>,
    /// How many choices to generate.
    #[serde(rename = "n")]
    #[serde(default = "default_1usize")]
    #[schema(example = 1)]
    pub n_choices: usize,
    /// Penalize tokens that have already appeared; positive values push toward new topics.
    #[schema(example = json!(Option::None::<f32>))]
    pub presence_penalty: Option<f32>,
    /// Penalize tokens by how often they have appeared so far; positive values reduce repetition.
    #[schema(example = json!(Option::None::<f32>))]
    pub frequency_penalty: Option<f32>,
    /// Sequences where generation stops.
    #[serde(rename = "stop")]
    #[schema(example = json!(Option::None::<StopTokens>))]
    pub stop_seqs: Option<StopTokens>,
    /// Sampling temperature; higher values increase randomness.
    #[schema(example = 0.7)]
    pub temperature: Option<f64>,
    /// Nucleus sampling: only tokens within the top cumulative probability mass are considered.
    #[schema(example = json!(Option::None::<f64>))]
    pub top_p: Option<f64>,
    /// Stream the response as server-sent events.
    #[schema(example = false)]
    pub stream: Option<bool>,
    /// Tools the model may call.
    #[schema(example = json!(Option::None::<Vec<OpenAiTool>>))]
    pub tools: Option<Vec<OpenAiTool>>,
    /// Controls which (if any) tool the model must call.
    #[schema(example = json!(Option::None::<ToolChoice>))]
    pub tool_choice: Option<ToolChoice>,
    /// Force plain text or JSON-schema constrained output.
    #[schema(example = json!(Option::None::<ResponseFormat>))]
    pub response_format: Option<ResponseFormat>,
    /// Arbitrary metadata stored with the response.
    #[schema(example = json!(Option::None::<Value>))]
    pub metadata: Option<Value>,
    /// Include a detailed token breakdown in usage.
    #[schema(example = json!(Option::None::<bool>))]
    pub output_token_details: Option<bool>,
    /// Whether tool calls may run in parallel.
    #[schema(example = json!(Option::None::<bool>))]
    pub parallel_tool_calls: Option<bool>,
    /// Persist the response so it can be fetched later by ID.
    #[schema(example = json!(Option::None::<bool>))]
    pub store: Option<bool>,
    /// Maximum number of tool calls the model may make.
    #[schema(example = json!(Option::None::<usize>))]
    pub max_tool_calls: Option<usize>,
    /// Toggle reasoning output for models that support it.
    #[schema(example = json!(Option::None::<bool>))]
    pub reasoning_enabled: Option<bool>,
    /// Token budget for reasoning.
    #[schema(example = json!(Option::None::<usize>))]
    pub reasoning_max_tokens: Option<usize>,
    /// Number of top logprobs to return for reasoning tokens.
    #[schema(example = json!(Option::None::<usize>))]
    pub reasoning_top_logprobs: Option<usize>,
    /// OpenAI-style truncation strategy object.
    #[schema(example = json!(Option::None::<Vec<String>>))]
    pub truncation: Option<HashMap<String, Value>>,

    // mistral.rs additional
    /// Sample only from the k most likely tokens.
    #[schema(example = json!(Option::None::<usize>))]
    pub top_k: Option<usize>,
    /// Constrain output with a regex, JSON schema, Lark, or LLGuidance grammar.
    #[schema(example = json!(Option::None::<Grammar>))]
    pub grammar: Option<Grammar>,
    /// Drop tokens below this fraction of the top token's probability.
    #[schema(example = json!(Option::None::<f64>))]
    pub min_p: Option<f64>,
    /// Multiplicative repetition penalty; 1.0 disables it.
    #[schema(example = json!(Option::None::<f32>))]
    pub repetition_penalty: Option<f32>,
    /// DRY repetition penalty multiplier; 0 disables DRY.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_multiplier: Option<f32>,
    /// Base for DRY's exponential penalty growth.
    #[schema(example = json!(Option::None::<f32>))]
    pub dry_base: Option<f32>,
    /// Longest repeated sequence DRY leaves unpenalized.
    #[schema(example = json!(Option::None::<usize>))]
    pub dry_allowed_length: Option<usize>,
    /// Sequences that reset DRY repetition matching.
    #[schema(example = json!(Option::None::<String>))]
    pub dry_sequence_breakers: Option<Vec<String>>,
    /// Toggle thinking output for models that support it.
    #[schema(example = json!(Option::None::<bool>))]
    pub enable_thinking: Option<bool>,
    /// Truncate inputs that exceed the model's context length instead of erroring.
    #[schema(example = json!(Option::None::<bool>))]
    #[serde(default)]
    pub truncate_sequence: Option<bool>,
    /// Reasoning effort level for models that support extended thinking.
    /// Valid values: "low", "medium", "high"
    #[schema(example = json!(Option::None::<String>))]
    pub reasoning_effort: Option<String>,
}

/// Response object
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ResponsesObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: f64,
    pub model: String,
    pub status: String,
    pub output: Vec<ResponsesOutput>,
    pub output_text: Option<String>,
    pub usage: Option<ResponsesUsage>,
    pub error: Option<ResponsesError>,
    pub metadata: Option<Value>,
    pub instructions: Option<String>,
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
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
    pub role: String,
    pub status: Option<String>,
    pub content: Vec<ResponsesContent>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn assert_tool_roundtrips(value: serde_json::Value) -> OpenAiTool {
        let tool: OpenAiTool = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(&tool).unwrap(), value);
        tool
    }

    #[test]
    fn openai_tool_deserializes_chat_function_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string" }
                    }
                },
                "strict": true
            }
        }));

        let OpenAiTool::Function(tool) = tool else {
            panic!("expected nested chat function tool");
        };
        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(tool.function.strict, Some(true));
    }

    #[test]
    fn openai_tool_deserializes_minimal_chat_function_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "function",
            "function": {
                "name": "get_weather"
            }
        }));

        let OpenAiTool::Function(tool) = tool else {
            panic!("expected nested chat function tool");
        };
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn openai_tool_deserializes_responses_function_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "function",
            "name": "get_customer",
            "description": "Look up a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": { "type": "string" }
                },
                "required": ["customer_id"]
            },
            "strict": true
        }));

        let OpenAiTool::ResponsesFunction(tool) = tool else {
            panic!("expected flat responses function tool");
        };
        assert_eq!(tool.name, "get_customer");
        assert_eq!(tool.strict, Some(true));
    }

    #[test]
    fn openai_tool_deserializes_web_search_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "web_search",
            "search_context_size": "low",
            "filters": {
                "allowed_domains": ["openai.com"]
            },
            "return_token_budget": "default",
            "search_content_types": ["text"]
        }));

        let OpenAiTool::WebSearch(tool) = tool else {
            panic!("expected web search tool");
        };
        assert_eq!(tool.tp, OpenAiWebSearchToolType::WebSearch);
        assert_eq!(
            tool.return_token_budget,
            Some(WebSearchReturnTokenBudget::Default)
        );
    }

    #[test]
    fn openai_tool_deserializes_code_interpreter_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "code_interpreter",
            "container": { "type": "auto" }
        }));

        let OpenAiTool::CodeInterpreter(tool) = tool else {
            panic!("expected code interpreter tool");
        };
        assert_eq!(tool.tp, OpenAiCodeInterpreterToolType::CodeInterpreter);
    }

    #[test]
    fn openai_tool_deserializes_shell_tool() {
        let tool = assert_tool_roundtrips(json!({
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [{
                    "type": "skill_reference",
                    "skill_id": "skill_abc",
                    "version": "latest"
                }]
            }
        }));

        let OpenAiTool::Shell(tool) = tool else {
            panic!("expected shell tool");
        };
        assert_eq!(tool.tp, OpenAiShellToolType::Shell);
    }

    #[test]
    fn normalizes_responses_function_tool() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "function",
            "name": "get_customer",
            "description": "Look up a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": { "type": "string" }
                },
                "required": ["customer_id"]
            },
            "strict": true
        }]))
        .unwrap();

        let normalized = normalize_responses_tools(Some(tools)).unwrap();
        let tools = normalized.tools.unwrap();
        assert_eq!(tools[0].function.name, "get_customer");
        assert_eq!(tools[0].function.strict, Some(true));
        assert!(!normalized.enable_code_execution);
    }

    #[test]
    fn normalizes_code_interpreter_tool() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "code_interpreter",
            "container": { "type": "auto" }
        }]))
        .unwrap();

        let normalized = normalize_chat_completion_tools(Some(tools), None).unwrap();
        assert!(normalized.enable_code_execution);
        assert!(normalized.tools.is_none());
    }

    #[test]
    fn validates_required_tool_choice_has_tools() {
        let normalized = normalize_responses_tools(None).unwrap();
        assert!(validate_openai_tool_choice(Some(&ToolChoice::Required), &normalized).is_err());

        let function_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "function",
            "name": "get_customer"
        }]))
        .unwrap();
        let normalized = normalize_responses_tools(Some(function_tools)).unwrap();
        assert!(validate_openai_tool_choice(Some(&ToolChoice::Required), &normalized).is_ok());

        let normalized =
            normalize_chat_completion_tools(None, Some(WebSearchOptions::default())).unwrap();
        assert!(validate_openai_tool_choice(Some(&ToolChoice::Required), &normalized).is_ok());

        let code_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "code_interpreter",
            "container": { "type": "auto" }
        }]))
        .unwrap();
        let normalized = normalize_responses_tools(Some(code_tools)).unwrap();
        assert!(validate_openai_tool_choice(Some(&ToolChoice::Required), &normalized).is_ok());

        let shell_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": { "type": "container_auto" }
        }]))
        .unwrap();
        let normalized = normalize_responses_tools(Some(shell_tools)).unwrap();
        assert!(validate_openai_tool_choice(Some(&ToolChoice::Required), &normalized).is_ok());
    }

    #[test]
    fn validates_specific_function_tool_choice_references_declared_tool() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "function",
            "name": "get_customer"
        }]))
        .unwrap();
        let normalized = normalize_responses_tools(Some(tools)).unwrap();

        let valid_choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_customer" })).unwrap();
        assert!(validate_openai_tool_choice(Some(&valid_choice), &normalized).is_ok());

        let invalid_choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "missing" })).unwrap();
        assert!(validate_openai_tool_choice(Some(&invalid_choice), &normalized).is_err());

        let chat_choice: ToolChoice = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "get_customer" }
        }))
        .unwrap();
        assert!(validate_openai_tool_choice(Some(&chat_choice), &normalized).is_ok());
    }

    #[test]
    fn rejects_unsupported_code_interpreter_options() {
        let file_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "file_ids": ["file-abc"]
            }
        }]))
        .unwrap();
        assert!(normalize_chat_completion_tools(Some(file_tools), None).is_err());

        let memory_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "code_interpreter",
            "container": {
                "type": "auto",
                "memory_limit": "4g"
            }
        }]))
        .unwrap();
        assert!(normalize_chat_completion_tools(Some(memory_tools), None).is_err());

        let id_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "code_interpreter",
            "container": "cntr_123"
        }]))
        .unwrap();
        assert!(normalize_chat_completion_tools(Some(id_tools), None).is_err());
    }

    #[test]
    fn normalizes_responses_shell_tool() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [{
                    "type": "skill_reference",
                    "skill_id": "skill_abc",
                    "version": 2
                }]
            }
        }]))
        .unwrap();

        let normalized = normalize_responses_tools(Some(tools)).unwrap();
        assert!(normalized.enable_shell);
        assert_eq!(normalized.shell_skill_references.len(), 1);
        assert_eq!(normalized.shell_skill_references[0].skill_id, "skill_abc");
        assert_eq!(normalized.shell_skill_references[0].version, Some(json!(2)));
    }

    #[test]
    fn rejects_unsupported_shell_tool_forms() {
        let local_env_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": { "type": "local", "path": "./skill" }
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(local_env_tools)).is_err());

        let container_ref_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": { "type": "container_reference", "container_id": "cntr_123" }
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(container_ref_tools)).is_err());

        let local_skill_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": {
                "type": "container_auto",
                "skills": [{ "type": "local", "path": "./skill" }]
            }
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(local_skill_tools)).is_err());

        let chat_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "shell",
            "environment": { "type": "container_auto" }
        }]))
        .unwrap();
        assert!(normalize_chat_completion_tools(Some(chat_tools), None).is_err());
    }

    #[test]
    fn normalizes_responses_web_search_tool() {
        let tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "web_search",
            "search_context_size": "low",
            "user_location": {
                "type": "approximate",
                "country": "GB",
                "city": "London",
                "region": "London"
            },
            "filters": {
                "allowed_domains": ["openai.com"],
                "blocked_domains": ["example.com"]
            },
            "return_token_budget": "unlimited",
            "search_content_types": ["text"]
        }]))
        .unwrap();

        let normalized = normalize_responses_tools(Some(tools)).unwrap();
        assert_eq!(
            normalized
                .web_search_options
                .as_ref()
                .and_then(|opts| opts.search_context_size),
            Some(SearchContextSize::Low)
        );

        let opts = normalized.web_search_options.unwrap();
        assert_eq!(
            opts.return_token_budget,
            Some(WebSearchReturnTokenBudget::Unlimited)
        );
        assert_eq!(opts.external_web_access, None);
        assert_eq!(
            opts.filters
                .as_ref()
                .and_then(|filters| filters.allowed_domains.as_deref()),
            Some(&["openai.com".to_string()][..])
        );
        assert_eq!(
            opts.search_content_types.as_deref(),
            Some(&[WebSearchContentType::Text][..])
        );

        let approximate = match opts.user_location.unwrap() {
            WebSearchUserLocation::Approximate { approximate } => approximate,
        };
        assert_eq!(approximate.country.as_deref(), Some("GB"));
        assert_eq!(approximate.timezone, None);

        let preview_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "web_search_preview",
            "external_web_access": false
        }]))
        .unwrap();
        let preview_opts = normalize_responses_tools(Some(preview_tools))
            .unwrap()
            .web_search_options
            .unwrap();
        assert_eq!(preview_opts.external_web_access, None);
    }

    #[test]
    fn rejects_unsupported_responses_web_search_options() {
        let image_tools: Vec<OpenAiTool> = serde_json::from_value(
            json!([{ "type": "web_search", "search_content_types": ["image"] }]),
        )
        .unwrap();
        assert!(normalize_responses_tools(Some(image_tools)).is_err());

        let external_access_tools: Vec<OpenAiTool> =
            serde_json::from_value(json!([{ "type": "web_search", "external_web_access": false }]))
                .unwrap();
        assert!(normalize_responses_tools(Some(external_access_tools)).is_err());

        let preview_filters_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "web_search_preview",
            "filters": { "allowed_domains": ["openai.com"] }
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(preview_filters_tools)).is_err());

        let preview_budget_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "web_search_preview",
            "return_token_budget": "default"
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(preview_budget_tools)).is_err());

        let too_many_domains = (0..=MAX_WEB_SEARCH_FILTER_DOMAINS)
            .map(|idx| format!("example{idx}.com"))
            .collect::<Vec<_>>();
        let domain_filter_tools: Vec<OpenAiTool> = serde_json::from_value(json!([{
            "type": "web_search",
            "filters": { "allowed_domains": too_many_domains }
        }]))
        .unwrap();
        assert!(normalize_responses_tools(Some(domain_filter_tools)).is_err());
    }

    #[test]
    fn rejects_web_search_tool_for_chat_completions() {
        let tools: Vec<OpenAiTool> =
            serde_json::from_value(json!([{ "type": "web_search" }])).unwrap();

        assert!(normalize_chat_completion_tools(Some(tools), None).is_err());
    }
}
