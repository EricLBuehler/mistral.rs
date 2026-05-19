//! Python code execution with an approval callback.
//!
//! Run with:
//! `cargo run --release --features code-execution --example code_execution_approval -p mistralrs`

use std::{
    io::{self, Write},
    sync::Arc,
};

use anyhow::Result;
use mistralrs::{
    AgentPermission, AgentToolApprovalCallback, AgentToolApprovalDecision, CodeExecutionConfig,
    IsqBits, ModelBuilder, RequestBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    let approval_callback: AgentToolApprovalCallback = Arc::new(|approval| {
        println!("\nAgent action approval required");
        println!("approval_id: {}", approval.approval_id);
        println!("session_id: {}", approval.session_id);
        println!("tool: {}", approval.tool.label);
        println!("\nCode:");
        println!(
            "{}",
            approval
                .arguments
                .get("code")
                .and_then(|value| value.as_str())
                .unwrap_or("<no code>")
        );

        loop {
            print!("\nRun this Python code? [y]es / [n]o / [a]lways: ");
            let _ = io::stdout().flush();

            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                return AgentToolApprovalDecision::deny(None);
            }
            match input.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => return AgentToolApprovalDecision::approve(),
                "a" | "always" => return AgentToolApprovalDecision::approve_for_session(),
                "" | "n" | "no" => {
                    return AgentToolApprovalDecision::deny_with_message(
                        "The user denied this action.",
                    );
                }
                _ => println!("Please enter y, n, or a."),
            }
        }
    });

    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_code_execution(CodeExecutionConfig::default())
        .build()
        .await?;

    let messages = TextMessages::new().add_message(
        TextMessageRole::User,
        "Use Python to calculate the first 20 Fibonacci numbers.",
    );
    let request = RequestBuilder::from(messages)
        .with_code_execution()
        .with_agent_permission(AgentPermission::Ask)
        .with_agent_approval_callback(approval_callback)
        .with_max_tool_rounds(4);

    let response = model.send_chat_request(request).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
