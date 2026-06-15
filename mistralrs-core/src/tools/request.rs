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
        schemas.push((
            "ToolChoice".to_string(),
            <ToolChoice as utoipa::PartialSchema>::schema(),
        ));
    }
}

impl ToolChoice {
    pub fn requires_tool_call(&self) -> bool {
        matches!(
            self,
            Self::Required | Self::Tool(_) | Self::NamedFunction(_)
        )
    }

    pub fn forced_function_name(&self) -> Option<&str> {
        match self {
            Self::Tool(tool) => Some(&tool.function.name),
            Self::NamedFunction(choice) => Some(&choice.name),
            Self::None | Self::Auto | Self::Required => None,
        }
    }
}
