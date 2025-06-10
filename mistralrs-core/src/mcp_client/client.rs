use crate::mcp_client::transport::{
    HttpTransport, McpTransport, ProcessTransport, WebSocketTransport,
};
use crate::mcp_client::types::McpToolResult;
use crate::mcp_client::{McpServerConnection, McpToolInfo};
use anyhow::Result;
use rust_mcp_schema::Resource;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// HTTP-based MCP server connection
pub struct HttpMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl HttpMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = HttpTransport::new(url, timeout_secs, headers)?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": "0.6.0"
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for HttpMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Tool execution failed: {}",
                tool_result.to_string()
            ));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({
            "uri": uri
        });

        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        let contents = result
            .get("contents")
            .and_then(|c| c.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resource read response format"))?;

        let mut text_parts = Vec::new();
        for content in contents {
            if let Some(content_type) = content.get("type").and_then(|t| t.as_str()) {
                match content_type {
                    "text" => {
                        if let Some(text) = content.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {
                        // Handle other content types as needed
                        text_parts.push(format!("[{}]", content_type));
                    }
                }
            }
        }

        Ok(text_parts.join("\n"))
    }

    async fn ping(&self) -> Result<()> {
        self.transport.ping().await
    }
}

/// Process-based MCP server connection
pub struct ProcessMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl ProcessMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        command: String,
        args: Vec<String>,
        work_dir: Option<String>,
        env: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = ProcessTransport::new(command, args, work_dir, env).await?;

        let connection = Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        };

        // Initialize the connection
        connection.initialize().await?;

        Ok(connection)
    }

    async fn initialize(&self) -> Result<()> {
        let init_params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "mistral.rs",
                "version": "0.6.0"
            }
        });

        self.transport
            .send_request("initialize", init_params)
            .await?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl McpServerConnection for ProcessMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        let result = self
            .transport
            .send_request("tools/list", Value::Null)
            .await?;

        let tools = result
            .get("tools")
            .and_then(|t| t.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid tools response format"))?;

        let mut tool_infos = Vec::new();
        for tool in tools {
            let name = tool
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or_else(|| anyhow::anyhow!("Tool missing name"))?
                .to_string();

            let description = tool
                .get("description")
                .and_then(|d| d.as_str())
                .map(|s| s.to_string());

            let input_schema = tool
                .get("inputSchema")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new()));

            tool_infos.push(McpToolInfo {
                name,
                description,
                input_schema,
                server_id: self.server_id.clone(),
                server_name: self.server_name.clone(),
            });
        }

        Ok(tool_infos)
    }

    async fn call_tool(&self, name: &str, arguments: Value) -> Result<String> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.transport.send_request("tools/call", params).await?;

        // Parse the MCP tool result
        let tool_result: McpToolResult = serde_json::from_value(result)?;

        // Check if the result indicates an error
        if tool_result.is_error.unwrap_or(false) {
            return Err(anyhow::anyhow!(
                "Tool execution failed: {}",
                tool_result.to_string()
            ));
        }

        Ok(tool_result.to_string())
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let result = self
            .transport
            .send_request("resources/list", Value::Null)
            .await?;

        let resources = result
            .get("resources")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resources response format"))?;

        let mut resource_list = Vec::new();
        for resource in resources {
            let mcp_resource: Resource = serde_json::from_value(resource.clone())?;
            resource_list.push(mcp_resource);
        }

        Ok(resource_list)
    }

    async fn read_resource(&self, uri: &str) -> Result<String> {
        let params = serde_json::json!({
            "uri": uri
        });

        let result = self
            .transport
            .send_request("resources/read", params)
            .await?;

        let contents = result
            .get("contents")
            .and_then(|c| c.as_array())
            .ok_or_else(|| anyhow::anyhow!("Invalid resource read response format"))?;

        let mut text_parts = Vec::new();
        for content in contents {
            if let Some(content_type) = content.get("type").and_then(|t| t.as_str()) {
                match content_type {
                    "text" => {
                        if let Some(text) = content.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    _ => {
                        // Handle other content types as needed
                        text_parts.push(format!("[{}]", content_type));
                    }
                }
            }
        }

        Ok(text_parts.join("\n"))
    }

    async fn ping(&self) -> Result<()> {
        self.transport.ping().await
    }
}

/// WebSocket-based MCP server connection
pub struct WebSocketMcpConnection {
    server_id: String,
    server_name: String,
    transport: Arc<dyn McpTransport>,
}

impl WebSocketMcpConnection {
    pub async fn new(
        server_id: String,
        server_name: String,
        url: String,
        timeout_secs: Option<u64>,
        headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let transport = WebSocketTransport::new(url, timeout_secs, headers).await?;

        Ok(Self {
            server_id,
            server_name,
            transport: Arc::new(transport),
        })
    }
}

#[async_trait::async_trait]
impl McpServerConnection for WebSocketMcpConnection {
    fn server_id(&self) -> &str {
        &self.server_id
    }

    fn server_name(&self) -> &str {
        &self.server_name
    }

    async fn list_tools(&self) -> Result<Vec<McpToolInfo>> {
        // WebSocket implementation would mirror HTTP implementation
        Err(anyhow::anyhow!("WebSocket transport not yet implemented"))
    }

    async fn call_tool(&self, _name: &str, _arguments: Value) -> Result<String> {
        // WebSocket implementation would mirror HTTP implementation
        Err(anyhow::anyhow!("WebSocket transport not yet implemented"))
    }

    async fn list_resources(&self) -> Result<Vec<Resource>> {
        // WebSocket implementation would mirror HTTP implementation
        Err(anyhow::anyhow!("WebSocket transport not yet implemented"))
    }

    async fn read_resource(&self, _uri: &str) -> Result<String> {
        // WebSocket implementation would mirror HTTP implementation
        Err(anyhow::anyhow!("WebSocket transport not yet implemented"))
    }

    async fn ping(&self) -> Result<()> {
        self.transport.ping().await
    }
}
