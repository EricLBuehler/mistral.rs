//! Example demonstrating the agentic loop with streaming output
//!
//! This example shows how to:
//! 1. Define sync and async tools using the `#[tool]` macro
//! 2. Create an agent with registered tools
//! 3. Run the agentic loop with streaming output
//! 4. Process streaming events in real-time
//! 5. Execute tools in parallel
//!
//! For non-streaming output, see the `agent` example.
//!
//! Run with: `cargo run --release --example agent_streaming -p mistralrs`

use anyhow::Result;
use mistralrs::{
    tool, AgentBuilder, AgentEvent, AgentStopReason, IsqBits, ModelBuilder,
    PagedAttentionMetaBuilder,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::io::Write;

/// Weather information returned by the get_weather tool
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct WeatherInfo {
    /// Temperature in the requested unit
    temperature: f32,
    /// Weather conditions description
    conditions: String,
    /// Humidity percentage
    humidity: u8,
}

/// Search result from web search
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct SearchResult {
    /// Title of the search result
    title: String,
    /// Snippet from the result
    snippet: String,
}

/// Get the current weather for a location (sync tool)
#[tool(description = "Get the current weather for a location")]
fn get_weather(
    #[description = "The city name to get weather for"] city: String,
    #[description = "Temperature unit: 'celsius' or 'fahrenheit'"]
    #[default = "celsius"]
    unit: Option<String>,
) -> Result<WeatherInfo> {
    // Mock implementation - in a real application, this would call a weather API
    // Simulate some work
    std::thread::sleep(std::time::Duration::from_millis(100));

    let temp = match unit.as_deref() {
        Some("fahrenheit") => 72.5,
        _ => 22.5,
    };

    Ok(WeatherInfo {
        temperature: temp,
        conditions: format!("Sunny with clear skies in {}", city),
        humidity: 45,
    })
}

/// Search the web for information (async tool - demonstrates native async support)
#[tool(description = "Search the web for information on a topic")]
async fn web_search(
    #[description = "The search query"] query: String,
    #[description = "Maximum number of results to return"]
    #[default = 3u32]
    max_results: Option<u32>,
) -> Result<Vec<SearchResult>> {
    // Mock implementation - in a real application, this would call a search API
    // Simulate async I/O with tokio::time::sleep
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    let num_results = max_results.unwrap_or(3) as usize;

    let results: Vec<SearchResult> = (0..num_results)
        .map(|i| SearchResult {
            title: format!("Result {} for: {}", i + 1, query),
            snippet: format!(
                "This is a snippet of information about '{}' from result {}.",
                query,
                i + 1
            ),
        })
        .collect();

    Ok(results)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build the model
    // Using a model that supports tool calling (e.g., Llama 3.1, Qwen, Mistral)
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    // Create the agent with registered tools
    // - get_weather is a sync tool (runs in spawn_blocking)
    // - web_search is an async tool (runs natively async)
    // Both can execute in parallel when the model calls multiple tools
    let agent = AgentBuilder::new(model)
        .with_system_prompt(
            "You are a helpful assistant with access to weather and web search tools. \
             Use them when needed to answer user questions accurately.",
        )
        .with_max_iterations(5)
        .with_parallel_tool_execution(true) // Enable parallel tool execution (default)
        .register_tool(get_weather_tool_with_callback())
        .register_tool(web_search_tool_with_callback())
        .build();

    println!("=== Agent with Streaming Output ===\n");
    println!(
        "User: What's the weather like in Boston, and can you find me some good restaurants there?\n"
    );
    print!("Assistant: ");

    // Run the agent with streaming output
    let mut stream = agent
        .run_stream(
            "What's the weather like in Boston, and can you find me some good restaurants there?",
        )
        .await?;

    let stdout = std::io::stdout();
    let mut handle = stdout.lock();

    // Process streaming events
    while let Some(event) = stream.next().await {
        match event {
            AgentEvent::TextDelta(text) => {
                // Print text as it streams - this gives real-time output
                write!(handle, "{}", text)?;
                handle.flush()?;
            }
            AgentEvent::ToolCallsStart(calls) => {
                // Model is about to call tools
                writeln!(handle, "\n\n[Calling {} tool(s)...]", calls.len())?;
                for call in &calls {
                    writeln!(
                        handle,
                        "  - {}: {}",
                        call.function.name, call.function.arguments
                    )?;
                }
            }
            AgentEvent::ToolResult(result) => {
                // A single tool finished execution
                let status = if result.result.is_ok() { "OK" } else { "ERROR" };
                writeln!(
                    handle,
                    "  [Tool {} completed: {}]",
                    result.tool_name, status
                )?;
            }
            AgentEvent::ToolCallsComplete => {
                // All tools finished, model will continue generating
                writeln!(handle, "[All tools completed, continuing...]\n")?;
                write!(handle, "Assistant: ")?;
                handle.flush()?;
            }
            AgentEvent::Complete(response) => {
                // Agent finished executing
                writeln!(handle, "\n\n=== Agent Execution Summary ===")?;
                writeln!(handle, "Completed in {} iteration(s)", response.iterations)?;
                writeln!(handle, "Stop reason: {:?}", response.stop_reason)?;
                writeln!(handle, "Steps taken: {}", response.steps.len())?;

                match response.stop_reason {
                    AgentStopReason::TextResponse => {
                        writeln!(handle, "Final response delivered successfully.")?;
                    }
                    AgentStopReason::MaxIterations => {
                        writeln!(
                            handle,
                            "Agent reached maximum iterations without producing a final response."
                        )?;
                    }
                    AgentStopReason::NoAction => {
                        writeln!(handle, "Agent produced no response.")?;
                    }
                    AgentStopReason::Error(e) => {
                        writeln!(handle, "Agent encountered an error: {}", e)?;
                    }
                }
            }
        }
    }

    Ok(())
}
