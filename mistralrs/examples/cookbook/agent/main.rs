/// Code review agent using the `#[tool]` macro and `AgentBuilder`.
///
/// Demonstrates:
/// - Defining a tool with the `#[tool]` proc macro
/// - Building an agent that can call the tool
/// - Running the agent loop for a code review task
///
/// Run with: `cargo run --release --example cookbook_agent -p mistralrs`
use anyhow::Result;
use mistralrs::{tool, AgentBuilder, IsqBits, ModelBuilder, PagedAttentionMetaBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Result of reading a file.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct FileContents {
    path: String,
    content: String,
}

/// Read a file from the local filesystem (mock implementation for safety).
#[tool(description = "Read the contents of a file at the given path")]
fn read_file(#[description = "Path to the file to read"] path: String) -> Result<FileContents> {
    // In a real tool you would read from disk; here we return a mock file.
    let content = match path.as_str() {
        "src/main.rs" => r#"
fn main() {
    let nums = vec![1, 2, 3, 4, 5];
    let mut total = 0;
    for i in 0..nums.len() {
        total += nums[i];
    }
    println!("Sum: {}", total);
    let x = 42;
}
"#
        .to_string(),
        _ => format!("// File not found: {path}"),
    };

    Ok(FileContents { path, content })
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let agent = AgentBuilder::new(model)
        .with_system_prompt(
            "You are an expert Rust code reviewer. When asked to review code, \
             use the read_file tool to read the file, then provide specific, \
             actionable feedback on code quality, idiomatic Rust usage, and \
             potential bugs.",
        )
        .with_max_iterations(3)
        .register_tool(read_file_tool_with_callback())
        .build();

    println!("=== Code Review Agent ===\n");

    let response = agent
        .run("Please review the file src/main.rs and suggest improvements.")
        .await?;

    if let Some(text) = &response.final_response {
        println!("Review:\n{text}");
    }

    println!("\nCompleted in {} iteration(s)", response.iterations);

    Ok(())
}
