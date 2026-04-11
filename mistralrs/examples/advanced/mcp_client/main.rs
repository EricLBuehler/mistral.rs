//! MCP (Model Context Protocol) client usage with mistral.rs.
//!
//! Connects to an MCP server, auto-discovers tools, and makes them available
//! to the model during conversations.
//!
//! Run with: `cargo run --release --example mcp_client -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, McpClientConfig, McpServerConfig, McpServerSource, ModelBuilder, TextMessageRole,
    TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to a local filesystem MCP server.
    // Install it with: npx @modelcontextprotocol/server-filesystem . -y
    let mcp_config = McpClientConfig {
        servers: vec![McpServerConfig {
            name: "Filesystem Tools".to_string(),
            source: McpServerSource::Process {
                command: "npx".to_string(),
                args: vec![
                    "@modelcontextprotocol/server-filesystem".to_string(),
                    ".".to_string(),
                ],
                work_dir: None,
                env: None,
            },
            ..Default::default()
        }],
        ..Default::default()
    };

    // Other transport types are also supported:
    //
    // McpServerSource::Http {
    //     url: "https://hf.co/mcp".to_string(),
    //     timeout_secs: Some(30),
    //     headers: None,
    // }
    //
    // McpServerSource::WebSocket {
    //     url: "wss://api.example.com/mcp".to_string(),
    //     timeout_secs: Some(30),
    //     headers: None,
    // }
    //
    // For authentication, set `bearer_token: Some("token".to_string())`.
    // To avoid tool name conflicts, set `tool_prefix: Some("prefix".to_string())`.

    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_mcp_client(mcp_config)
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "List the files in the current directory.",
    );

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
