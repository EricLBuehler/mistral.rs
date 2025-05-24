use either::Either;
use mistralrs_core::{
    ImageGenerationResponseFormat, LlguidanceGrammar, Tool, ToolChoice, ToolType, WebSearchOptions,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, ops::Deref};
use utoipa::{
    openapi::{ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, SchemaType},
    ToSchema,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageInnerContent(
    #[serde(with = "either::serde_untagged")] pub Either<String, HashMap<String, String>>,
);

// The impl Deref was preventing the Derive ToSchema and schema macros from
// properly working, so manually impl ToSchema
impl ToSchema<'_> for MessageInnerContent {
    fn schema() -> (&'static str, RefOr<Schema>) {
        (
            "MessageInnerContent",
            RefOr::T(message_inner_content_schema()),
        )
    }
}

impl Deref for MessageInnerContent {
    type Target = Either<String, HashMap<String, String>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Since `Either` doesn't implement ToSchema, manually implement one
fn message_inner_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new().schema_type(SchemaType::String).build(),
            ))
            .item(Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Object)
                    .additional_properties(Some(RefOr::T(Schema::Object(
                        ObjectBuilder::new().schema_type(SchemaType::String).build(),
                    ))))
                    .build(),
            ))
            .build(),
    )
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageContent(
    #[serde(with = "either::serde_untagged")]
    Either<String, Vec<HashMap<String, MessageInnerContent>>>,
);

// The impl Deref was preventing the Derive ToSchema and schema macros from
// properly working, so manually impl ToSchema
impl ToSchema<'_> for MessageContent {
    fn schema() -> (&'static str, RefOr<Schema>) {
        ("MessageContent", RefOr::T(message_content_schema()))
    }
}

impl Deref for MessageContent {
    type Target = Either<String, Vec<HashMap<String, MessageInnerContent>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Since `Either` doesn't implement ToSchema, manually implement one
fn message_content_schema() -> Schema {
    Schema::OneOf(
        OneOfBuilder::new()
            .item(Schema::Object(
                ObjectBuilder::new().schema_type(SchemaType::String).build(),
            ))
            .item(Schema::Array(
                ArrayBuilder::new()
                    .items(RefOr::T(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
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

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
pub struct FunctionCalled {
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, ToSchema)]
pub struct ToolCall {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: FunctionCalled,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct Message {
    pub content: Option<MessageContent>,
    pub role: String,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum StopTokens {
    Multi(Vec<String>),
    Single(String),
}

fn default_false() -> bool {
    false
}

fn default_1usize() -> usize {
    1
}

fn default_720usize() -> usize {
    720
}

fn default_1280usize() -> usize {
    1280
}

fn default_model() -> String {
    "default".to_string()
}

fn default_response_format() -> ImageGenerationResponseFormat {
    ImageGenerationResponseFormat::Url
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", content = "value")]
pub enum Grammar {
    #[serde(rename = "regex")]
    Regex(String),
    #[serde(rename = "json_schema")]
    JsonSchema(serde_json::Value),
    #[serde(rename = "llguidance")]
    Llguidance(LlguidanceGrammar),
    #[serde(rename = "lark")]
    Lark(String),
}

// Implement ToSchema manually to handle LlguidanceGrammar
impl utoipa::ToSchema<'_> for Grammar {
    fn schema() -> (&'static str, RefOr<Schema>) {
        (
            "Grammar",
            RefOr::T(Schema::OneOf(
                OneOfBuilder::new()
                    .item(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
                            .property(
                                "type",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new()
                                        .schema_type(SchemaType::String)
                                        .enum_values(Some(vec![serde_json::Value::String(
                                            "regex".to_string(),
                                        )]))
                                        .build(),
                                )),
                            )
                            .property(
                                "value",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new().schema_type(SchemaType::String).build(),
                                )),
                            )
                            .required("type")
                            .required("value")
                            .build(),
                    ))
                    .item(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
                            .property(
                                "type",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new()
                                        .schema_type(SchemaType::String)
                                        .enum_values(Some(vec![serde_json::Value::String(
                                            "json_schema".to_string(),
                                        )]))
                                        .build(),
                                )),
                            )
                            .property(
                                "value",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new().schema_type(SchemaType::Object).build(),
                                )),
                            )
                            .required("type")
                            .required("value")
                            .build(),
                    ))
                    .item(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
                            .property(
                                "type",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new()
                                        .schema_type(SchemaType::String)
                                        .enum_values(Some(vec![serde_json::Value::String(
                                            "llguidance".to_string(),
                                        )]))
                                        .build(),
                                )),
                            )
                            .property("value", RefOr::T(llguidance_schema()))
                            .required("type")
                            .required("value")
                            .build(),
                    ))
                    .item(Schema::Object(
                        ObjectBuilder::new()
                            .schema_type(SchemaType::Object)
                            .property(
                                "type",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new()
                                        .schema_type(SchemaType::String)
                                        .enum_values(Some(vec![serde_json::Value::String(
                                            "lark".to_string(),
                                        )]))
                                        .build(),
                                )),
                            )
                            .property(
                                "value",
                                RefOr::T(Schema::Object(
                                    ObjectBuilder::new().schema_type(SchemaType::String).build(),
                                )),
                            )
                            .required("type")
                            .required("value")
                            .build(),
                    ))
                    .build(),
            )),
        )
    }
}

fn llguidance_schema() -> Schema {
    let grammar_with_lexer_schema = Schema::Object(
        ObjectBuilder::new()
            .schema_type(SchemaType::Object)
            .property(
                "name",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::String)
                        .nullable(true)
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
                        .schema_type(SchemaType::Object)
                        .nullable(true)
                        .description(Some("The JSON schema that the grammar should generate"))
                        .build(),
                )),
            )
            .property(
                "lark_grammar",
                RefOr::T(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::String)
                        .nullable(true)
                        .description(Some("The Lark grammar that the grammar should generate"))
                        .build(),
                )),
            )
            .description(Some("Grammar configuration with lexer settings"))
            .build(),
    );

    Schema::Object(
        ObjectBuilder::new()
            .schema_type(SchemaType::Object)
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
                        .schema_type(SchemaType::Integer)
                        .nullable(true)
                        .description(Some("Maximum number of tokens to generate"))
                        .build(),
                )),
            )
            .required("grammars")
            .description(Some("Top-level grammar configuration for LLGuidance"))
            .build(),
    )
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct JsonSchemaResponseFormat {
    pub name: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema {
        json_schema: JsonSchemaResponseFormat,
    },
}

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

// Since `Either` doesn't implement ToSchema, manually implement one
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
                ObjectBuilder::new().schema_type(SchemaType::String).build(),
            ))
            .build(),
    )
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelObjects {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum AudioResponseFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl AudioResponseFormat {
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

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SpeechGenerationRequest {
    #[schema(example = "nari-labs/Dia-1.6B")]
    #[serde(default = "default_model")]
    pub model: String,
    #[schema(
        example = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
    )]
    pub input: String,
    // `voice` and `instructions` are ignored.
    #[schema(example = "mp3")]
    pub response_format: AudioResponseFormat,
}
