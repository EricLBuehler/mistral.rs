//! Agentic loop implementation for mistral.rs
//!
//! This module provides an `Agent` that runs an agentic loop with tool calling.
//! The agent takes a model, registers tools, and automatically handles the
//! tool calling loop until the model produces a final response.
//!
//! # Example
//!
//! ```ignore
//! use mistralrs::{Agent, AgentBuilder, TextModelBuilder, IsqType, tool};
//! use schemars::JsonSchema;
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, JsonSchema)]
//! struct WeatherInfo { temperature: f32 }
//!
//! #[tool(description = "Get weather for a city")]
//! fn get_weather(city: String) -> anyhow::Result<WeatherInfo> {
//!     Ok(WeatherInfo { temperature: 22.5 })
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = TextModelBuilder::new("model-id")
//!         .build()
//!         .await?;
//!
//!     let agent = AgentBuilder::new(model)
//!         .with_system_prompt("You are a helpful assistant.")
//!         .register_tool(get_weather_tool_with_callback())
//!         .build();
//!
//!     let response = agent.run("What's the weather in Boston?").await?;
//!     println!("{:?}", response.final_response);
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    ChatCompletionResponse, Model, RequestBuilder, TextMessageRole, Tool, ToolCallResponse,
    ToolCallback, ToolChoice,
};

/// Configuration for the agentic loop
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Maximum number of iterations before stopping (default: 10)
    pub max_iterations: usize,
    /// Tool choice strategy (default: Auto)
    pub tool_choice: ToolChoice,
    /// Optional system prompt for the agent
    pub system_prompt: Option<String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tool_choice: ToolChoice::Auto,
            system_prompt: None,
        }
    }
}

/// Represents a single step in the agent execution
#[derive(Debug, Clone)]
pub struct AgentStep {
    /// The model's response for this step
    pub response: ChatCompletionResponse,
    /// Tool calls made in this step (if any)
    pub tool_calls: Vec<ToolCallResponse>,
    /// Results from tool executions
    pub tool_results: Vec<ToolResult>,
}

/// Result of a tool execution
#[derive(Debug, Clone)]
pub struct ToolResult {
    /// The tool call ID this result corresponds to
    pub tool_call_id: String,
    /// Name of the tool that was called
    pub tool_name: String,
    /// The result: Ok(output) or Err(error_message)
    pub result: Result<String, String>,
}

/// Final response from the agent
#[derive(Debug)]
pub struct AgentResponse {
    /// All steps taken during execution
    pub steps: Vec<AgentStep>,
    /// Final text response (if any)
    pub final_response: Option<String>,
    /// Total number of iterations performed
    pub iterations: usize,
    /// Why the agent stopped
    pub stop_reason: AgentStopReason,
}

/// Reason why the agent stopped executing
#[derive(Debug, Clone, PartialEq)]
pub enum AgentStopReason {
    /// Model produced a text response with no tool calls
    TextResponse,
    /// Maximum iterations reached
    MaxIterations,
    /// No tool calls and no text response
    NoAction,
    /// Error during execution
    Error(String),
}

/// An agent that runs an agentic loop with tool calling
pub struct Agent {
    model: Model,
    tools: Vec<Tool>,
    callbacks: HashMap<String, Arc<ToolCallback>>,
    config: AgentConfig,
}

impl Agent {
    /// Create a new agent with the given model and configuration
    pub fn new(model: Model, config: AgentConfig) -> Self {
        Self {
            model,
            tools: Vec::new(),
            callbacks: HashMap::new(),
            config,
        }
    }

    /// Add a tool with its callback
    pub fn with_tool(mut self, tool: Tool, callback: Arc<ToolCallback>) -> Self {
        let name = tool.function.name.clone();
        self.tools.push(tool);
        self.callbacks.insert(name, callback);
        self
    }

    /// Run the agentic loop with the given user message
    ///
    /// This method will:
    /// 1. Send the user message to the model
    /// 2. If the model returns tool calls, execute them and send results back
    /// 3. Repeat until the model returns a text response or max iterations is reached
    pub async fn run(&self, user_message: impl ToString) -> anyhow::Result<AgentResponse> {
        let mut steps = Vec::new();
        let mut messages = RequestBuilder::new();

        // Add system prompt if configured
        if let Some(ref system) = self.config.system_prompt {
            messages = messages.add_message(TextMessageRole::System, system);
        }

        // Add initial user message
        messages = messages.add_message(TextMessageRole::User, user_message.to_string());

        for iteration in 0..self.config.max_iterations {
            // Configure tools for this request
            let request = messages
                .clone()
                .set_tools(self.tools.clone())
                .set_tool_choice(self.config.tool_choice.clone());

            // Send request to model
            let response = self.model.send_chat_request(request).await?;

            let choice = response
                .choices
                .first()
                .ok_or_else(|| anyhow::anyhow!("No choices in response"))?;

            // Check for tool calls
            let tool_calls = choice.message.tool_calls.clone().unwrap_or_default();

            if tool_calls.is_empty() {
                // No tool calls - we're done
                let final_text = choice.message.content.clone();
                steps.push(AgentStep {
                    response: response.clone(),
                    tool_calls: vec![],
                    tool_results: vec![],
                });

                let stop_reason = if final_text.is_some() {
                    AgentStopReason::TextResponse
                } else {
                    AgentStopReason::NoAction
                };

                return Ok(AgentResponse {
                    steps,
                    final_response: final_text,
                    iterations: iteration + 1,
                    stop_reason,
                });
            }

            // Execute tool calls
            let mut tool_results = Vec::new();

            // Add assistant message with tool calls
            messages = messages.add_message_with_tool_call(
                TextMessageRole::Assistant,
                choice.message.content.clone().unwrap_or_default(),
                tool_calls.clone(),
            );

            for tool_call in &tool_calls {
                let result = self.execute_tool(tool_call);

                // Add tool result to messages
                let result_str = match &result.result {
                    Ok(s) => s.clone(),
                    Err(e) => format!("Error: {}", e),
                };
                messages = messages.add_tool_message(&result_str, &tool_call.id);

                tool_results.push(result);
            }

            steps.push(AgentStep {
                response: response.clone(),
                tool_calls: tool_calls.clone(),
                tool_results,
            });
        }

        // Max iterations reached
        Ok(AgentResponse {
            steps,
            final_response: None,
            iterations: self.config.max_iterations,
            stop_reason: AgentStopReason::MaxIterations,
        })
    }

    /// Execute a single tool call
    fn execute_tool(&self, tool_call: &ToolCallResponse) -> ToolResult {
        let tool_name = &tool_call.function.name;

        let result = match self.callbacks.get(tool_name) {
            Some(callback) => callback(&tool_call.function).map_err(|e| e.to_string()),
            None => Err(format!("Unknown tool: {}", tool_name)),
        };

        ToolResult {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_name.clone(),
            result,
        }
    }

    /// Get a reference to the underlying model
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Get a reference to the registered tools
    pub fn tools(&self) -> &[Tool] {
        &self.tools
    }

    /// Get a reference to the agent configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }
}

/// Builder for creating agents with a fluent API
pub struct AgentBuilder {
    model: Model,
    tools: Vec<Tool>,
    callbacks: HashMap<String, Arc<ToolCallback>>,
    config: AgentConfig,
}

impl AgentBuilder {
    /// Create a new agent builder with the given model
    pub fn new(model: Model) -> Self {
        Self {
            model,
            tools: Vec::new(),
            callbacks: HashMap::new(),
            config: AgentConfig::default(),
        }
    }

    /// Set the maximum number of iterations
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    /// Set the system prompt for the agent
    pub fn with_system_prompt(mut self, prompt: impl ToString) -> Self {
        self.config.system_prompt = Some(prompt.to_string());
        self
    }

    /// Set the tool choice strategy
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.config.tool_choice = choice;
        self
    }

    /// Add a tool with its callback
    pub fn with_tool(mut self, tool: Tool, callback: Arc<ToolCallback>) -> Self {
        let name = tool.function.name.clone();
        self.tools.push(tool);
        self.callbacks.insert(name, callback);
        self
    }

    /// Register a tool using a tuple of (Tool, Arc<ToolCallback>)
    ///
    /// This is designed to work with the `_tool_with_callback()` functions
    /// generated by the `#[tool]` macro.
    pub fn register_tool(self, (tool, callback): (Tool, Arc<ToolCallback>)) -> Self {
        self.with_tool(tool, callback)
    }

    /// Build the agent
    pub fn build(self) -> Agent {
        Agent {
            model: self.model,
            tools: self.tools,
            callbacks: self.callbacks,
            config: self.config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.max_iterations, 10);
        assert!(config.system_prompt.is_none());
    }

    #[test]
    fn test_agent_stop_reason_equality() {
        assert_eq!(AgentStopReason::TextResponse, AgentStopReason::TextResponse);
        assert_eq!(
            AgentStopReason::MaxIterations,
            AgentStopReason::MaxIterations
        );
        assert_ne!(
            AgentStopReason::TextResponse,
            AgentStopReason::MaxIterations
        );
    }
}
