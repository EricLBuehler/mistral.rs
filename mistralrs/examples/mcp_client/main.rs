use anyhow::Result;
use mistralrs::{
    IsqType, McpClientConfig, McpServerConfig, McpServerSource, PagedAttentionMetaBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};
use std::collections::HashMap;

/// This example demonstrates how to use mistral.rs as an MCP client to connect
/// to external MCP servers and automatically register their tools for use in
/// automatic tool calling.
#[tokio::main]
async fn main() -> Result<()> {
    // Create MCP client configuration with example servers
    let mcp_config = McpClientConfig {
        servers: vec![
            // Example HTTP-based MCP server
            McpServerConfig {
                id: "example_server".to_string(),
                name: "Example MCP Server".to_string(),
                source: McpServerSource::Http {
                    url: "http://localhost:8080/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: Some({
                        let mut headers = HashMap::new();
                        headers.insert("Authorization".to_string(), "Bearer your-token".to_string());
                        headers
                    }),
                },
                enabled: true,
                tool_prefix: Some("example".to_string()),
                resources: None,
            },
            // Example process-based MCP server
            McpServerConfig {
                id: "filesystem_server".to_string(),
                name: "Filesystem MCP Server".to_string(),
                source: McpServerSource::Process {
                    command: "mcp-server-filesystem".to_string(),
                    args: vec!["--root".to_string(), "/tmp".to_string()],
                    work_dir: None,
                    env: None,
                },
                enabled: true,
                tool_prefix: Some("fs".to_string()),
                resources: Some(vec!["file://**".to_string()]),
            },
            // Example WebSocket-based MCP server (placeholder)
            McpServerConfig {
                id: "websocket_server".to_string(),
                name: "WebSocket MCP Server".to_string(),
                source: McpServerSource::WebSocket {
                    url: "ws://localhost:9090/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: None,
                },
                enabled: false, // Disabled since WebSocket transport is not yet implemented
                tool_prefix: Some("ws".to_string()),
                resources: None,
            },
        ],
        auto_register_tools: true,
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(5),
    };

    println!("Building model with MCP client support...");

    // Build the model with MCP client configuration
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct".to_string())
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .with_mcp_client(mcp_config) // Add MCP client configuration
        .build()
        .await?;

    println!("Model built successfully! MCP servers will be connected automatically.");
    println!("Any tools from the MCP servers will be available for automatic tool calling.");

    // Create a conversation that might trigger tool usage
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI assistant with access to external tools via MCP servers. \
             You can search the web, access filesystem operations, and use other tools \
             provided by connected MCP servers. Use these tools when appropriate to \
             help answer user questions.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! Can you help me list the files in the /tmp directory and then \
             search for information about Rust programming language?",
        );

    println!("\nSending chat request...");
    let response = model.send_chat_request(messages).await?;

    println!("\nResponse:");
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    // Display performance metrics
    println!("\nPerformance metrics:");
    println!(
        "Prompt tokens/sec: {:.2}",
        response.usage.avg_prompt_tok_per_sec
    );
    println!(
        "Completion tokens/sec: {:.2}",
        response.usage.avg_compl_tok_per_sec
    );

    // Display any tool calls that were made
    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        println!("\nTool calls made:");
        for tool_call in tool_calls {
            println!("- {}: {}", tool_call.function.name, tool_call.function.arguments);
        }
    }

    Ok(())
}