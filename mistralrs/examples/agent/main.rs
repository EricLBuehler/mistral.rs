//! Example demonstrating the agentic loop with tool calling
//!
//! This example shows how to:
//! 1. Define tools using the `#[tool]` macro
//! 2. Create an agent with registered tools
//! 3. Run the agentic loop
//!
//! Run with: `cargo run --release --example agent -p mistralrs`

use anyhow::Result;
use mistralrs::{
    tool, AgentBuilder, AgentStopReason, IsqType, PagedAttentionMetaBuilder, TextModelBuilder,
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
    /// URL of the result
    url: String,
    /// Snippet from the result
    snippet: String,
}

/// Get the current weather for a location
#[tool(description = "Get the current weather for a location")]
fn get_weather(
    #[description = "The city name to get weather for"] city: String,
    #[description = "Temperature unit: 'celsius' or 'fahrenheit'"]
    #[default = "celsius"]
    unit: Option<String>,
) -> Result<WeatherInfo> {
    // Mock implementation - in a real application, this would call a weather API
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

/// Search the web for information
#[tool(description = "Search the web for information on a topic")]
fn web_search(
    #[description = "The search query"] query: String,
    #[description = "Maximum number of results to return"]
    #[default = 3u32]
    max_results: Option<u32>,
) -> Result<Vec<SearchResult>> {
    // Mock implementation - in a real application, this would call a search API
    let num_results = max_results.unwrap_or(3) as usize;

    let results: Vec<SearchResult> = (0..num_results)
        .map(|i| SearchResult {
            title: format!("Result {} for: {}", i + 1, query),
            url: format!("https://example.com/result/{}", i + 1),
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
    let model = TextModelBuilder::new("../hf_models/qwen3_4b")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    // Create the agent with registered tools
    let agent = AgentBuilder::new(model)
        .with_system_prompt(
            "You are a helpful assistant with access to weather and web search tools. \
             Use them when needed to answer user questions accurately.",
        )
        .with_max_iterations(5)
        .register_tool(get_weather_tool_with_callback())
        .register_tool(web_search_tool_with_callback())
        .build();

    // Run the agent with a user query
    let response = agent
        .run("What's the weather like in Boston, and can you find me some good restaurants there?")
        .await?;

    // Print the results
    println!("\n=== Agent Execution Summary ===");
    println!("Completed in {} iteration(s)", response.iterations);
    println!("Stop reason: {:?}", response.stop_reason);

    // Print each step
    for (i, step) in response.steps.iter().enumerate() {
        println!("\n--- Step {} ---", i + 1);

        if !step.tool_calls.is_empty() {
            println!("Tool calls:");
            for tc in &step.tool_calls {
                println!("  - {}: {}", tc.function.name, tc.function.arguments);
            }
        }

        if !step.tool_results.is_empty() {
            println!("Tool results:");
            for tr in &step.tool_results {
                match &tr.result {
                    Ok(result) => println!("  - {}: {}", tr.tool_name, result),
                    Err(e) => println!("  - {} (error): {}", tr.tool_name, e),
                }
            }
        }
    }

    // Print the final response
    println!("\n=== Final Response ===");
    match response.stop_reason {
        AgentStopReason::TextResponse => {
            if let Some(text) = &response.final_response {
                println!("{}", text);
            }
        }
        AgentStopReason::MaxIterations => {
            println!("Agent reached maximum iterations without producing a final response.");
        }
        AgentStopReason::NoAction => {
            println!("Agent produced no response.");
        }
        AgentStopReason::Error(e) => {
            println!("Agent encountered an error: {}", e);
        }
    }

    Ok(())
}
