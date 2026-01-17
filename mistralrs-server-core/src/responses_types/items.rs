//! Input and output item types for the OpenResponses API.

use serde::{Deserialize, Serialize};
use utoipa::{
    openapi::{schema::SchemaType, ArrayBuilder, ObjectBuilder, OneOfBuilder, RefOr, Schema, Type},
    PartialSchema, ToSchema,
};

use super::{
    content::{InputContent, OutputContent},
    enums::ItemStatus,
};

/// Message content that can be a string or array of content parts
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContentParam {
    /// Simple string content
    Text(String),
    /// Array of content parts
    Parts(Vec<InputContent>),
}

impl PartialSchema for MessageContentParam {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::Type(Type::String))
                        .build(),
                ))
                .item(Schema::Array(
                    ArrayBuilder::new().items(InputContent::schema()).build(),
                ))
                .build(),
        ))
    }
}

impl ToSchema for MessageContentParam {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((
            MessageContentParam::name().into(),
            MessageContentParam::schema(),
        ));
    }
}

impl MessageContentParam {
    /// Convert to a text string, extracting text from parts if needed
    pub fn to_text(&self) -> String {
        use super::content::NormalizedInputContent;

        match self {
            MessageContentParam::Text(s) => s.clone(),
            MessageContentParam::Parts(parts) => {
                let mut texts = Vec::new();
                for part in parts {
                    if let NormalizedInputContent::Text { text } = part.clone().into_normalized() {
                        texts.push(text);
                    }
                }
                texts.join(" ")
            }
        }
    }

    /// Check if this is simple text content
    pub fn is_text(&self) -> bool {
        matches!(self, MessageContentParam::Text(_))
    }
}

/// Message item parameter for input
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct MessageItemParam {
    /// The role of the message sender
    pub role: String,
    /// The content of the message
    pub content: MessageContentParam,
    /// Optional name for the message sender
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Tagged input item types (with explicit "type" field)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TaggedInputItem {
    /// A message item with explicit type
    #[serde(rename = "message")]
    Message(MessageItemParam),
    /// Reference to a previous item
    #[serde(rename = "item_reference")]
    ItemReference {
        /// ID of the referenced item
        id: String,
    },
    /// A function call made by the assistant
    #[serde(rename = "function_call")]
    FunctionCall {
        /// Unique identifier for this function call
        call_id: String,
        /// Name of the function to call
        name: String,
        /// JSON-encoded arguments for the function
        arguments: String,
    },
    /// Output from a function call
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        /// ID of the function call this is a response to
        call_id: String,
        /// Output from the function
        output: String,
    },
}

/// Input item types for the OpenResponses API.
///
/// This enum supports both:
/// - Explicitly typed items with `"type": "message"`, `"type": "item_reference"`, etc.
/// - Bare messages with just `role` and `content` (no type field) for compatibility with OpenAI's format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InputItem {
    /// Explicitly typed input item (has "type" field)
    Tagged(TaggedInputItem),
    /// A bare message (just role + content, no type field) - OpenAI compatibility
    Message(MessageItemParam),
}

impl InputItem {
    /// Convert to a normalized form for internal processing
    pub fn into_tagged(self) -> TaggedInputItem {
        match self {
            InputItem::Tagged(t) => t,
            InputItem::Message(m) => TaggedInputItem::Message(m),
        }
    }
}

impl PartialSchema for InputItem {
    fn schema() -> RefOr<Schema> {
        RefOr::T(input_item_schema())
    }
}

impl ToSchema for InputItem {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((InputItem::name().into(), InputItem::schema()));
    }
}

fn input_item_schema() -> Schema {
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
                                    "message".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "role",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property("content", MessageContentParam::schema())
                    .required("type")
                    .required("role")
                    .required("content")
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
                                    "item_reference".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("id")
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
                                    "function_call".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "call_id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "name",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "arguments",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("call_id")
                    .required("name")
                    .required("arguments")
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
                                    "function_call_output".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "call_id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "output",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("call_id")
                    .required("output")
                    .build(),
            ))
            .build(),
    )
}

/// Output item types for the OpenResponses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    /// A message item
    #[serde(rename = "message")]
    Message {
        /// Unique identifier for this item
        id: String,
        /// The role (always "assistant" for output)
        role: String,
        /// The content of the message
        content: Vec<OutputContent>,
        /// Status of the item
        status: ItemStatus,
    },
    /// A function call item
    #[serde(rename = "function_call")]
    FunctionCall {
        /// Unique identifier for this item
        id: String,
        /// Unique identifier for the call (used to match with output)
        call_id: String,
        /// Name of the function to call
        name: String,
        /// JSON-encoded arguments for the function
        arguments: String,
        /// Status of the item
        status: ItemStatus,
    },
}

impl PartialSchema for OutputItem {
    fn schema() -> RefOr<Schema> {
        RefOr::T(output_item_schema())
    }
}

impl ToSchema for OutputItem {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((OutputItem::name().into(), OutputItem::schema()));
    }
}

fn output_item_schema() -> Schema {
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
                                    "message".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "role",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "content",
                        RefOr::T(Schema::Array(
                            ArrayBuilder::new().items(OutputContent::schema()).build(),
                        )),
                    )
                    .property(
                        "status",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("id")
                    .required("role")
                    .required("content")
                    .required("status")
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
                                    "function_call".to_string(),
                                )]))
                                .build(),
                        )),
                    )
                    .property(
                        "id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "call_id",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "name",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "arguments",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .property(
                        "status",
                        RefOr::T(Schema::Object(
                            ObjectBuilder::new()
                                .schema_type(SchemaType::Type(Type::String))
                                .build(),
                        )),
                    )
                    .required("type")
                    .required("id")
                    .required("call_id")
                    .required("name")
                    .required("arguments")
                    .required("status")
                    .build(),
            ))
            .build(),
    )
}

impl OutputItem {
    /// Get the ID of the output item
    pub fn id(&self) -> &str {
        match self {
            OutputItem::Message { id, .. } => id,
            OutputItem::FunctionCall { id, .. } => id,
        }
    }

    /// Get the status of the output item
    pub fn status(&self) -> ItemStatus {
        match self {
            OutputItem::Message { status, .. } => *status,
            OutputItem::FunctionCall { status, .. } => *status,
        }
    }

    /// Create a new message output item
    pub fn message(id: String, content: Vec<OutputContent>, status: ItemStatus) -> Self {
        OutputItem::Message {
            id,
            role: "assistant".to_string(),
            content,
            status,
        }
    }

    /// Create a new function call output item
    pub fn function_call(
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: ItemStatus,
    ) -> Self {
        OutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            status,
        }
    }
}

/// Input that can be a string, array of input items, or array of messages
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    /// Simple string input
    Text(String),
    /// Array of input items (OpenResponses format)
    Items(Vec<InputItem>),
}

impl PartialSchema for ResponsesInput {
    fn schema() -> RefOr<Schema> {
        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(Schema::Object(
                    ObjectBuilder::new()
                        .schema_type(SchemaType::Type(Type::String))
                        .build(),
                ))
                .item(Schema::Array(
                    ArrayBuilder::new().items(InputItem::schema()).build(),
                ))
                .build(),
        ))
    }
}

impl ToSchema for ResponsesInput {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        schemas.push((ResponsesInput::name().into(), ResponsesInput::schema()));
    }
}
