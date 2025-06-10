use mistralrs_mcp::Tool;

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    /// Disallow selection of tools.
    None,
    #[serde(rename = "auto")]
    /// Allow automatic selection of any given tool, or none.
    Auto,
    #[serde(untagged)]
    /// Force selection of a given tool.
    Tool(Tool),
}
