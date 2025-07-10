use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Display};

/// OpenAI-compatible tool schema for MCP tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolSchema {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: McpFunctionSchema,
}

/// Function schema for MCP tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpFunctionSchema {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// MCP tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    pub is_error: Option<bool>,
}

/// MCP content types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { data: String, mime_type: String },
    #[serde(rename = "resource")]
    Resource { resource: McpResourceReference },
}

/// Reference to an MCP resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceReference {
    pub uri: String,
    pub text: Option<String>,
}

/// MCP server capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerCapabilities {
    pub tools: Option<HashMap<String, serde_json::Value>>,
    pub resources: Option<HashMap<String, serde_json::Value>>,
    pub prompts: Option<HashMap<String, serde_json::Value>>,
    pub logging: Option<HashMap<String, serde_json::Value>>,
}

/// MCP initialization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpInitResult {
    pub protocol_version: String,
    pub capabilities: McpServerCapabilities,
    pub server_info: McpServerInfo,
}

/// MCP server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    pub version: String,
}

impl From<crate::McpToolInfo> for McpToolSchema {
    fn from(tool_info: crate::McpToolInfo) -> Self {
        Self {
            tool_type: "function".to_string(),
            function: McpFunctionSchema {
                name: tool_info.name,
                description: tool_info.description,
                parameters: tool_info.input_schema,
            },
        }
    }
}

impl Display for McpToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = self
            .content
            .iter()
            .map(|content| match content {
                McpContent::Text { text } => text.clone(),
                McpContent::Image { mime_type, .. } => {
                    format!("[Image: {mime_type}]")
                }
                McpContent::Resource { resource } => resource
                    .text
                    .clone()
                    .unwrap_or_else(|| format!("[Resource: {}]", resource.uri)),
            })
            .collect::<Vec<_>>()
            .join("\n");

        write!(f, "{res}")
    }
}
