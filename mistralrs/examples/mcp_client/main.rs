// Example demonstrating MCP (Model Context Protocol) client usage with mistral.rs
//
// This example shows how to:
// - Configure MCP servers with different transport protocols (HTTP, WebSocket, Process)
// - Set up Bearer token authentication for secure connections
// - Use automatic tool discovery and registration
// - Integrate MCP tools with model tool calling
//
// The MCP client automatically discovers tools from connected servers and makes them
// available for the model to use during conversations.

use anyhow::Result;
use mistralrs::{
    IsqType, McpClientConfig, McpServerConfig, McpServerSource, MemoryGpuConfig,
    PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder,
};
// use std::collections::HashMap; // Uncomment if using manual headers in examples below

#[tokio::main]
async fn main() -> Result<()> {
    // Create MCP client configuration with example servers
    // This configuration demonstrates different transport types and authentication methods
    let mcp_config = McpClientConfig {
        servers: vec![
            // Example: HTTP-based MCP server with Bearer token authentication
            McpServerConfig {
                id: "example_server".to_string(),
                name: "Hugging Face MCP".to_string(),
                source: McpServerSource::Http {
                    url: "https://hf.co/mcp".to_string(),
                    timeout_secs: Some(30),
                    headers: None, // Additional headers can be specified here if needed
                },
                enabled: true,
                tool_prefix: Some("example".to_string()), // Prefixes tool names to avoid conflicts
                resources: None,
                bearer_token: Some("hf_xxx".to_string()), // Replace with your actual Hugging Face token
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
            // // Example WebSocket-based MCP server
            // McpServerConfig {
            //     id: "websocket_server".to_string(),
            //     name: "WebSocket MCP Server".to_string(),
            //     source: McpServerSource::WebSocket {
            //         url: "wss://api.example.com/mcp".to_string(),
            //         timeout_secs: Some(30),
            //         headers: None,
            //     },
            //     enabled: false, // Disabled for example - change to true to use
            //     tool_prefix: Some("ws".to_string()),
            //     resources: None,
            //     bearer_token: Some("your-websocket-token".to_string()), // WebSocket Bearer token support
            // },
        ],
        // Automatically discover and register tools from connected MCP servers
        auto_register_tools: true,
        // Timeout for individual tool calls (30 seconds)
        tool_timeout_secs: Some(30),
        // Maximum concurrent tool calls across all servers
        max_concurrent_calls: Some(5),
    };

    println!("Building model with MCP client support...");

    // Build the model with MCP client configuration
    // The MCP client will automatically connect to configured servers and discover available tools
    let model = TextModelBuilder::new("Qwen/Qwen3-4B".to_string())
        .with_isq(IsqType::Q8_0) // Use 8-bit quantization for efficiency
        .with_logging()
        .with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::ContextSize(8192))
                .build()
        })?
        .with_mcp_client(mcp_config) // This automatically connects to MCP servers and registers tools
        .build()
        .await?;

    println!("Model built successfully! MCP servers connected and tools registered.");
    println!("MCP tools are now available for automatic tool calling during conversations.");

    // Create a conversation that demonstrates MCP tool usage
    // The system message informs the model about available external tools
    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI assistant with access to external tools via MCP servers. \
             You can search the web, access filesystem operations, and use other tools \
             provided by connected MCP servers. Use these tools when appropriate to \
             help answer user questions. Tools are automatically available and you \
             can call them as needed.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! Can you help me get the top 10 HF models right now?",
        );

    println!("\nSending chat request...");
    println!("The model will automatically use MCP tools if needed to answer the question.");
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

    // Display any MCP tool calls that were made during the conversation
    if let Some(tool_calls) = &response.choices[0].message.tool_calls {
        println!("\nMCP tool calls made:");
        for tool_call in tool_calls {
            println!(
                "- Tool: {} | Arguments: {}",
                tool_call.function.name, tool_call.function.arguments
            );
        }
    } else {
        println!("\nNo tool calls were made for this request.");
    }

    Ok(())
}
