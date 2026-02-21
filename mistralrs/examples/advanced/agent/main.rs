//! Example demonstrating the agentic loop with tool calling
//!
//! This example shows how to:
//! 1. Define sync and async tools using the `#[tool]` macro
//! 2. Create an agent with registered tools
//! 3. Run the agentic loop (non-streaming)
//! 4. Execute tools in parallel
//!
//! For streaming output, see the `agent_streaming` example.
//!
//! Run with: `cargo run --release --example agent -p mistralrs`

use anyhow::Result;
use mistralrs::{
    tool, AgentBuilder, AgentStopReason, IsqBits, ModelBuilder, PagedAttentionMetaBuilder,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
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

    println!("=== Agent Example (Non-Streaming) ===\n");

    let user_message =
        "What's the weather like in Boston, and can you find me some good restaurants there?";
    println!("User: {}\n", user_message);

    // Run the agent (waits for complete response)
    let response = agent.run(user_message).await?;

    // Print the final response
    if let Some(text) = &response.final_response {
        println!("Assistant: {}\n", text);
    }

    // Print execution summary
    println!("=== Execution Summary ===");
    println!("Completed in {} iteration(s)", response.iterations);
    println!("Stop reason: {:?}", response.stop_reason);
    println!("Steps taken: {}", response.steps.len());

    // Print details of each step
    for (i, step) in response.steps.iter().enumerate() {
        println!("\n--- Step {} ---", i + 1);
        if !step.tool_calls.is_empty() {
            println!("Tool calls:");
            for call in &step.tool_calls {
                println!("  - {}: {}", call.function.name, call.function.arguments);
            }
            println!("Tool results:");
            for result in &step.tool_results {
                let status = if result.result.is_ok() { "OK" } else { "ERROR" };
                println!("  - {}: {}", result.tool_name, status);
            }
        }
    }

    match response.stop_reason {
        AgentStopReason::TextResponse => {
            println!("\nFinal response delivered successfully.");
        }
        AgentStopReason::MaxIterations => {
            println!("\nAgent reached maximum iterations without producing a final response.");
        }
        AgentStopReason::NoAction => {
            println!("\nAgent produced no response.");
        }
        AgentStopReason::Error(e) => {
            println!("\nAgent encountered an error: {}", e);
        }
    }

    Ok(())
}
