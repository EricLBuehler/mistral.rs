use anyhow::Result;
use mistralrs::{
    IsqType, McpClientConfig, McpServerConfig, McpServerSource, MemoryGpuConfig,
    PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
// use std::collections::HashMap; // Uncomment if using manual headers in examples below

/// This example demonstrates how to use mistral.rs as an MCP client to connect
/// to external MCP servers and automatically register their tools for use in
/// automatic tool calling.
#[tokio::main]
async fn main() -> Result<()> {
    // Create MCP client configuration with example servers
    let mcp_config = McpClientConfig {
        servers: vec![
            // Example HTTP-based MCP server with Bearer token authentication
            McpServerConfig {
                id: "example_server".to_string(),
                name: "Example MCP Server".to_string(),
                source: McpServerSource::Http {
                    url: "https://hf.co/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: None, // Additional headers can be specified here if needed
                },
                enabled: true,
                tool_prefix: Some("example".to_string()),
                resources: None,
                bearer_token: Some("hf_xxx".to_string()), // Bearer token for authentication
            },
            // // Example process-based MCP server (no authentication needed)
            // McpServerConfig {
            //     id: "filesystem_server".to_string(),
            //     name: "Filesystem MCP Server".to_string(),
            //     source: McpServerSource::Process {
            //         command: "mcp-server-filesystem".to_string(),
            //         args: vec!["--root".to_string(), "/tmp".to_string()],
            //         work_dir: None,
            //         env: None,
            //     },
            //     enabled: true,
            //     tool_prefix: Some("fs".to_string()),
            //     resources: Some(vec!["file://**".to_string()]),
            //     bearer_token: None, // Process servers typically don't need Bearer tokens
            // },
            //
            // // Example with both Bearer token and additional headers (uncomment HashMap import above)
            // McpServerConfig {
            //     id: "authenticated_server".to_string(),
            //     name: "Authenticated MCP Server".to_string(),
            //     source: McpServerSource::Http {
            //         url: "https://api.example.com/mcp".to_string(),
            //         timeout_secs: Some(60),
            //         headers: Some({
            //             let mut headers = HashMap::new();
            //             headers.insert("X-API-Version".to_string(), "v1".to_string());
            //             headers.insert("X-Client-ID".to_string(), "mistral-rs".to_string());
            //             headers
            //         }),
            //     },
            //     enabled: false,
            //     tool_prefix: Some("auth".to_string()),
            //     resources: None,
            //     bearer_token: Some("your-bearer-token".to_string()), // Will be added as Authorization: Bearer <token>
            // },
            // // Example WebSocket-based MCP server (placeholder)
            // McpServerConfig {
            //     id: "websocket_server".to_string(),
            //     name: "WebSocket MCP Server".to_string(),
            //     source: McpServerSource::WebSocket {
            //         url: "ws://localhost:9090/mcp".to_string(),
            //         timeout_secs: Some(30),
            //         headers: None,
            //     },
            //     enabled: false, // Disabled since WebSocket transport is not yet implemented
            //     tool_prefix: Some("ws".to_string()),
            //     resources: None,
            // },
        ],
        auto_register_tools: true,
        tool_timeout_secs: Some(30),
        max_concurrent_calls: Some(5),
    };

    println!("Building model with MCP client support...");

    // Build the model with MCP client configuration
    let model = TextModelBuilder::new("../hf_models/qwen3_4b".to_string())
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::ContextSize(8192))
                .build()
        })?
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
            "Hello! Can you help me get the top 10 HF models right now?",
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
            println!(
                "- {}: {}",
                tool_call.function.name, tool_call.function.arguments
            );
        }
    }

    Ok(())
}
