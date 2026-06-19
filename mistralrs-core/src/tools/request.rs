use mistralrs_mcp::Tool;
use mistralrs_mcp::ToolType;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    /// Disallow selection of tools.
    None,
    #[serde(rename = "auto")]
    /// Allow automatic selection of any given tool, or none.
    Auto,
    #[serde(rename = "required")]
    /// Require at least one tool call.
    Required,
    #[serde(untagged)]
    /// Restrict the tools the model may use.
    AllowedTools(AllowedToolsToolChoice),
    #[serde(untagged)]
    /// Force selection of a hosted tool.
    Builtin(BuiltinToolChoice),
    #[serde(untagged)]
    /// Force selection of a given tool.
    Tool(Tool),
    #[serde(untagged)]
    /// Force selection of a named function tool.
    NamedFunction(NamedFunctionToolChoice),
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct NamedFunctionToolChoice {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub name: String,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct AllowedToolsToolChoice {
    #[serde(rename = "type")]
    pub tp: AllowedToolsToolChoiceType,
    pub mode: AllowedToolsMode,
    pub tools: Vec<AllowedToolChoice>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub enum AllowedToolsToolChoiceType {
    #[serde(rename = "allowed_tools")]
    AllowedTools,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AllowedToolsMode {
    Auto,
    Required,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(tag = "type")]
pub enum AllowedToolChoice {
    #[serde(rename = "function")]
    Function { name: String },
    #[serde(rename = "web_search_preview", alias = "web_search")]
    WebSearch,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "shell")]
    Shell,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BuiltinToolChoice {
    #[serde(rename = "type")]
    pub tp: BuiltinToolChoiceType,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
pub enum BuiltinToolChoiceType {
    #[serde(rename = "web_search_preview", alias = "web_search")]
    WebSearch,
    #[serde(rename = "code_interpreter")]
    CodeInterpreter,
    #[serde(rename = "shell")]
    Shell,
}

#[cfg(feature = "utoipa")]
impl utoipa::PartialSchema for ToolChoice {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::{
            schema::{SchemaType, Type},
            ObjectBuilder, OneOfBuilder, Ref, RefOr, Schema,
        };

        fn string_choice(value: &'static str, description: &'static str) -> Schema {
            Schema::Object(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .description(Some(description))
                    .enum_values(Some(vec![serde_json::Value::String(value.to_string())]))
                    .build(),
            )
        }

        RefOr::T(Schema::OneOf(
            OneOfBuilder::new()
                .item(string_choice("none", "Disallow selection of tools."))
                .item(string_choice(
                    "auto",
                    "Allow automatic selection of any given tool, or none.",
                ))
                .item(string_choice("required", "Require at least one tool call."))
                .item(RefOr::Ref(Ref::from_schema_name("AllowedToolsToolChoice")))
                .item(RefOr::Ref(Ref::from_schema_name("BuiltinToolChoice")))
                .item(RefOr::Ref(Ref::from_schema_name("Tool")))
                .item(RefOr::Ref(Ref::from_schema_name("NamedFunctionToolChoice")))
                .build(),
        ))
    }
}

#[cfg(feature = "utoipa")]
impl utoipa::ToSchema for ToolChoice {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        fn push_schema<T>(
            schemas: &mut Vec<(
                String,
                utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
            )>,
            name: &str,
        ) where
            T: utoipa::PartialSchema,
        {
            schemas.push((name.to_string(), <T as utoipa::PartialSchema>::schema()));
        }

        schemas.push((
            "ToolChoice".to_string(),
            <ToolChoice as utoipa::PartialSchema>::schema(),
        ));
        push_schema::<AllowedToolsToolChoice>(schemas, "AllowedToolsToolChoice");
        push_schema::<AllowedToolsToolChoiceType>(schemas, "AllowedToolsToolChoiceType");
        push_schema::<AllowedToolsMode>(schemas, "AllowedToolsMode");
        push_schema::<AllowedToolChoice>(schemas, "AllowedToolChoice");
        push_schema::<BuiltinToolChoice>(schemas, "BuiltinToolChoice");
        push_schema::<BuiltinToolChoiceType>(schemas, "BuiltinToolChoiceType");
    }
}

impl ToolChoice {
    pub fn requires_tool_call(&self) -> bool {
        matches!(
            self,
            Self::Required | Self::Tool(_) | Self::NamedFunction(_) | Self::Builtin(_)
        ) || matches!(
            self,
            Self::AllowedTools(choice) if choice.mode == AllowedToolsMode::Required
        )
    }

    pub fn forced_function_name(&self) -> Option<&str> {
        match self {
            Self::Tool(tool) => Some(&tool.function.name),
            Self::NamedFunction(choice) => Some(&choice.name),
            Self::None | Self::Auto | Self::Required | Self::AllowedTools(_) | Self::Builtin(_) => {
                None
            }
        }
    }
}

impl AllowedToolChoice {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Function { .. } => "function",
            Self::WebSearch => "web_search_preview",
            Self::CodeInterpreter => "code_interpreter",
            Self::Shell => "shell",
        }
    }
}

impl BuiltinToolChoiceType {
    pub fn kind(self) -> &'static str {
        match self {
            Self::WebSearch => "web_search_preview",
            Self::CodeInterpreter => "code_interpreter",
            Self::Shell => "shell",
        }
    }
}
