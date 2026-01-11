//! Agentic loop implementation for mistral.rs
//!
//! This module provides an `Agent` that runs an agentic loop with tool calling.
//! The agent takes a model, registers tools, and automatically handles the
//! tool calling loop until the model produces a final response.
//!
//! # Features
//!
//! - **Async tools**: Native support for async tool functions
//! - **Parallel execution**: Execute multiple tool calls concurrently
//! - **Streaming**: Stream assistant responses and tool execution events
//!
//! # Example
//!
//! ```ignore
//! use mistralrs::{tool, AgentBuilder, AgentEvent};
//!
//! // Async tool - runs natively async
//! #[tool(description = "Fetch a URL")]
//! async fn fetch_url(url: String) -> Result<String> {
//!     reqwest::get(&url).await?.text().await.map_err(Into::into)
//! }
//!
//! // Sync tool
//! #[tool(description = "Get weather")]
//! fn get_weather(city: String) -> Result<WeatherInfo> {
//!     Ok(WeatherInfo { temperature: 22.5 })
//! }
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let model = TextModelBuilder::new("model-id").build().await?;
//!
//!     let agent = AgentBuilder::new(model)
//!         .with_system_prompt("You are helpful.")
//!         .register_tool(fetch_url_tool_with_callback())
//!         .register_tool(get_weather_tool_with_callback())
//!         .build();
//!
//!     // Streaming execution
//!     let mut stream = agent.run_stream("What's the weather?").await?;
//!     while let Some(event) = stream.next().await {
//!         match event {
//!             AgentEvent::TextDelta(text) => print!("{}", text),
//!             AgentEvent::Complete(response) => println!("\nDone!"),
//!             _ => {}
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::{
    CalledFunction, ChatCompletionChunkResponse, ChatCompletionResponse, ChunkChoice, Delta, Model,
    RequestBuilder, Response, TextMessageRole, Tool, ToolCallResponse, ToolCallback, ToolChoice,
};

/// Async tool callback type for native async tool support
pub type AsyncToolCallback = dyn Fn(CalledFunction) -> Pin<Box<dyn Future<Output = anyhow::Result<String>> + Send>>
    + Send
    + Sync;

/// Unified tool callback that can be sync or async
#[derive(Clone)]
pub enum ToolCallbackType {
    /// Synchronous callback (runs in spawn_blocking for parallel execution)
    Sync(Arc<ToolCallback>),
    /// Asynchronous callback (runs natively async)
    Async(Arc<AsyncToolCallback>),
}

/// Configuration for the agentic loop
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// Maximum number of iterations before stopping (default: 10)
    pub max_iterations: usize,
    /// Tool choice strategy (default: Auto)
    pub tool_choice: ToolChoice,
    /// Optional system prompt for the agent
    pub system_prompt: Option<String>,
    /// Whether to execute multiple tool calls in parallel (default: true)
    pub parallel_tool_execution: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tool_choice: ToolChoice::Auto,
            system_prompt: None,
            parallel_tool_execution: true,
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
#[derive(Debug, Clone)]
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

/// Events yielded during agent streaming
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Text content delta from the model
    TextDelta(String),
    /// Model is calling tools (with the tool calls)
    ToolCallsStart(Vec<ToolCallResponse>),
    /// A single tool completed execution
    ToolResult(ToolResult),
    /// All tools completed, continuing to next iteration
    ToolCallsComplete,
    /// Agent finished with final response
    Complete(AgentResponse),
}

/// Internal state for the agent stream
enum AgentStreamState {
    /// Currently streaming model response
    Streaming {
        messages: RequestBuilder,
        iteration: usize,
        accumulated_content: String,
        accumulated_tool_calls: Vec<ToolCallResponse>,
        steps: Vec<AgentStep>,
    },
    /// Executing tool calls
    ExecutingTools {
        messages: RequestBuilder,
        iteration: usize,
        response: ChatCompletionResponse,
        tool_calls: Vec<ToolCallResponse>,
        tool_results: Vec<ToolResult>,
        pending_indices: Vec<usize>,
        steps: Vec<AgentStep>,
    },
    /// Agent has completed
    Done,
}

/// Stream of agent events during execution
pub struct AgentStream<'a> {
    agent: &'a Agent,
    state: AgentStreamState,
    model_stream: Option<crate::model::Stream<'a>>,
}

impl<'a> AgentStream<'a> {
    /// Get the next event from the agent stream
    pub async fn next(&mut self) -> Option<AgentEvent> {
        loop {
            match &mut self.state {
                AgentStreamState::Done => return None,

                AgentStreamState::Streaming {
                    messages,
                    iteration,
                    accumulated_content,
                    accumulated_tool_calls,
                    steps,
                } => {
                    // Get next chunk from model stream
                    if let Some(ref mut stream) = self.model_stream {
                        if let Some(response) = stream.next().await {
                            match response {
                                Response::Chunk(ChatCompletionChunkResponse {
                                    choices, ..
                                }) => {
                                    if let Some(ChunkChoice {
                                        delta:
                                            Delta {
                                                content,
                                                tool_calls,
                                                ..
                                            },
                                        finish_reason,
                                        ..
                                    }) = choices.first()
                                    {
                                        // Accumulate content
                                        if let Some(text) = content {
                                            accumulated_content.push_str(text);
                                            return Some(AgentEvent::TextDelta(text.clone()));
                                        }

                                        // Accumulate tool calls
                                        if let Some(calls) = tool_calls {
                                            accumulated_tool_calls.extend(calls.clone());
                                        }

                                        // Check if done
                                        if finish_reason.is_some() {
                                            self.model_stream = None;

                                            if accumulated_tool_calls.is_empty() {
                                                // No tool calls - we're done
                                                let final_response =
                                                    if accumulated_content.is_empty() {
                                                        None
                                                    } else {
                                                        Some(accumulated_content.clone())
                                                    };

                                                let stop_reason = if final_response.is_some() {
                                                    AgentStopReason::TextResponse
                                                } else {
                                                    AgentStopReason::NoAction
                                                };

                                                let response = AgentResponse {
                                                    steps: steps.clone(),
                                                    final_response,
                                                    iterations: *iteration + 1,
                                                    stop_reason,
                                                };

                                                self.state = AgentStreamState::Done;
                                                return Some(AgentEvent::Complete(response));
                                            } else {
                                                // Transition to executing tools
                                                let tool_calls = accumulated_tool_calls.clone();
                                                let event =
                                                    AgentEvent::ToolCallsStart(tool_calls.clone());

                                                // Create a placeholder response for the step
                                                let placeholder_response = ChatCompletionResponse {
                                                    id: String::new(),
                                                    choices: vec![],
                                                    created: 0,
                                                    model: String::new(),
                                                    system_fingerprint: String::new(),
                                                    object: String::new(),
                                                    usage: crate::Usage {
                                                        completion_tokens: 0,
                                                        prompt_tokens: 0,
                                                        total_tokens: 0,
                                                        avg_tok_per_sec: 0.0,
                                                        avg_prompt_tok_per_sec: 0.0,
                                                        avg_compl_tok_per_sec: 0.0,
                                                        total_time_sec: 0.0,
                                                        total_prompt_time_sec: 0.0,
                                                        total_completion_time_sec: 0.0,
                                                    },
                                                };

                                                self.state = AgentStreamState::ExecutingTools {
                                                    messages: messages.clone(),
                                                    iteration: *iteration,
                                                    response: placeholder_response,
                                                    tool_calls: tool_calls.clone(),
                                                    tool_results: Vec::new(),
                                                    pending_indices: (0..tool_calls.len())
                                                        .collect(),
                                                    steps: steps.clone(),
                                                };

                                                return Some(event);
                                            }
                                        }
                                    }
                                }
                                Response::Done(response) => {
                                    self.model_stream = None;
                                    let tool_calls = response
                                        .choices
                                        .first()
                                        .and_then(|c| c.message.tool_calls.clone())
                                        .unwrap_or_default();

                                    if tool_calls.is_empty() {
                                        let final_response = response
                                            .choices
                                            .first()
                                            .and_then(|c| c.message.content.clone());
                                        let stop_reason = if final_response.is_some() {
                                            AgentStopReason::TextResponse
                                        } else {
                                            AgentStopReason::NoAction
                                        };

                                        let agent_response = AgentResponse {
                                            steps: steps.clone(),
                                            final_response,
                                            iterations: *iteration + 1,
                                            stop_reason,
                                        };

                                        self.state = AgentStreamState::Done;
                                        return Some(AgentEvent::Complete(agent_response));
                                    } else {
                                        let event = AgentEvent::ToolCallsStart(tool_calls.clone());

                                        self.state = AgentStreamState::ExecutingTools {
                                            messages: messages.clone(),
                                            iteration: *iteration,
                                            response: response.clone(),
                                            tool_calls: tool_calls.clone(),
                                            tool_results: Vec::new(),
                                            pending_indices: (0..tool_calls.len()).collect(),
                                            steps: steps.clone(),
                                        };

                                        return Some(event);
                                    }
                                }
                                _ => continue,
                            }
                        }
                    }

                    // Stream ended unexpectedly
                    self.state = AgentStreamState::Done;
                    return None;
                }

                AgentStreamState::ExecutingTools {
                    messages,
                    iteration,
                    response,
                    tool_calls,
                    tool_results,
                    pending_indices,
                    steps,
                } => {
                    if pending_indices.is_empty() {
                        // All tools executed - prepare for next iteration
                        let mut new_messages = messages.clone();

                        // Add assistant message with tool calls
                        new_messages = new_messages.add_message_with_tool_call(
                            TextMessageRole::Assistant,
                            response
                                .choices
                                .first()
                                .and_then(|c| c.message.content.clone())
                                .unwrap_or_default(),
                            tool_calls.clone(),
                        );

                        // Add tool results
                        for result in tool_results.iter() {
                            let result_str = match &result.result {
                                Ok(s) => s.clone(),
                                Err(e) => format!("Error: {}", e),
                            };
                            new_messages =
                                new_messages.add_tool_message(&result_str, &result.tool_call_id);
                        }

                        // Record step
                        let step = AgentStep {
                            response: response.clone(),
                            tool_calls: tool_calls.clone(),
                            tool_results: tool_results.clone(),
                        };
                        let mut new_steps = steps.clone();
                        new_steps.push(step);

                        let new_iteration = *iteration + 1;

                        // Check max iterations
                        if new_iteration >= self.agent.config.max_iterations {
                            let agent_response = AgentResponse {
                                steps: new_steps,
                                final_response: None,
                                iterations: new_iteration,
                                stop_reason: AgentStopReason::MaxIterations,
                            };
                            self.state = AgentStreamState::Done;
                            return Some(AgentEvent::Complete(agent_response));
                        }

                        // Start new model request
                        let request = new_messages
                            .clone()
                            .set_tools(self.agent.tools.clone())
                            .set_tool_choice(self.agent.config.tool_choice.clone());

                        match self.agent.model.stream_chat_request(request).await {
                            Ok(stream) => {
                                self.model_stream = Some(stream);
                                self.state = AgentStreamState::Streaming {
                                    messages: new_messages,
                                    iteration: new_iteration,
                                    accumulated_content: String::new(),
                                    accumulated_tool_calls: Vec::new(),
                                    steps: new_steps,
                                };
                                return Some(AgentEvent::ToolCallsComplete);
                            }
                            Err(e) => {
                                let agent_response = AgentResponse {
                                    steps: new_steps,
                                    final_response: None,
                                    iterations: new_iteration,
                                    stop_reason: AgentStopReason::Error(e.to_string()),
                                };
                                self.state = AgentStreamState::Done;
                                return Some(AgentEvent::Complete(agent_response));
                            }
                        }
                    }

                    // Execute next pending tool
                    let idx = pending_indices.remove(0);
                    let tool_call = &tool_calls[idx];
                    let result = self.agent.execute_tool_async(tool_call).await;
                    let event = AgentEvent::ToolResult(result.clone());
                    tool_results.push(result);
                    return Some(event);
                }
            }
        }
    }
}

/// An agent that runs an agentic loop with tool calling
pub struct Agent {
    model: Model,
    tools: Vec<Tool>,
    callbacks: HashMap<String, ToolCallbackType>,
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
    pub fn with_tool(mut self, tool: Tool, callback: ToolCallbackType) -> Self {
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

            // Execute tool calls (parallel or sequential based on config)
            let tool_results = if self.config.parallel_tool_execution {
                self.execute_tools_parallel(&tool_calls).await
            } else {
                let mut results = Vec::new();
                for tool_call in &tool_calls {
                    results.push(self.execute_tool_async(tool_call).await);
                }
                results
            };

            // Add assistant message with tool calls
            messages = messages.add_message_with_tool_call(
                TextMessageRole::Assistant,
                choice.message.content.clone().unwrap_or_default(),
                tool_calls.clone(),
            );

            // Add tool results to messages
            for result in &tool_results {
                let result_str = match &result.result {
                    Ok(s) => s.clone(),
                    Err(e) => format!("Error: {}", e),
                };
                messages = messages.add_tool_message(&result_str, &result.tool_call_id);
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

    /// Run the agent with streaming output
    ///
    /// Returns a stream of `AgentEvent` that can be used to observe
    /// the agent's progress in real-time.
    pub async fn run_stream(&self, user_message: impl ToString) -> anyhow::Result<AgentStream<'_>> {
        let mut messages = RequestBuilder::new();

        // Add system prompt if configured
        if let Some(ref system) = self.config.system_prompt {
            messages = messages.add_message(TextMessageRole::System, system);
        }

        // Add initial user message
        messages = messages.add_message(TextMessageRole::User, user_message.to_string());

        // Configure tools for this request
        let request = messages
            .clone()
            .set_tools(self.tools.clone())
            .set_tool_choice(self.config.tool_choice.clone());

        // Start streaming
        let stream = self.model.stream_chat_request(request).await?;

        Ok(AgentStream {
            agent: self,
            state: AgentStreamState::Streaming {
                messages,
                iteration: 0,
                accumulated_content: String::new(),
                accumulated_tool_calls: Vec::new(),
                steps: Vec::new(),
            },
            model_stream: Some(stream),
        })
    }

    /// Execute multiple tool calls in parallel
    async fn execute_tools_parallel(&self, tool_calls: &[ToolCallResponse]) -> Vec<ToolResult> {
        let futures: Vec<_> = tool_calls
            .iter()
            .map(|tc| self.execute_tool_async(tc))
            .collect();

        futures::future::join_all(futures).await
    }

    /// Execute a single tool call (async-compatible)
    async fn execute_tool_async(&self, tool_call: &ToolCallResponse) -> ToolResult {
        let tool_name = &tool_call.function.name;

        let result = match self.callbacks.get(tool_name) {
            Some(ToolCallbackType::Sync(callback)) => {
                // Run sync callback in spawn_blocking to not block async runtime
                let callback = Arc::clone(callback);
                let function = tool_call.function.clone();
                tokio::task::spawn_blocking(move || callback(&function))
                    .await
                    .map_err(|e| anyhow::anyhow!("Task join error: {}", e))
                    .and_then(|r| r)
                    .map_err(|e| e.to_string())
            }
            Some(ToolCallbackType::Async(callback)) => {
                let function = tool_call.function.clone();
                callback(function).await.map_err(|e| e.to_string())
            }
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
    callbacks: HashMap<String, ToolCallbackType>,
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

    /// Enable or disable parallel tool execution (default: true)
    pub fn with_parallel_tool_execution(mut self, enabled: bool) -> Self {
        self.config.parallel_tool_execution = enabled;
        self
    }

    /// Add a sync tool with its callback
    pub fn with_sync_tool(mut self, tool: Tool, callback: Arc<ToolCallback>) -> Self {
        let name = tool.function.name.clone();
        self.tools.push(tool);
        self.callbacks
            .insert(name, ToolCallbackType::Sync(callback));
        self
    }

    /// Add an async tool with its callback
    pub fn with_async_tool(mut self, tool: Tool, callback: Arc<AsyncToolCallback>) -> Self {
        let name = tool.function.name.clone();
        self.tools.push(tool);
        self.callbacks
            .insert(name, ToolCallbackType::Async(callback));
        self
    }

    /// Register a tool using a tuple of (Tool, ToolCallbackType)
    ///
    /// This is designed to work with the `_tool_with_callback()` functions
    /// generated by the `#[tool]` macro.
    pub fn register_tool(mut self, (tool, callback): (Tool, ToolCallbackType)) -> Self {
        let name = tool.function.name.clone();
        self.tools.push(tool);
        self.callbacks.insert(name, callback);
        self
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
        assert!(config.parallel_tool_execution);
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

    #[test]
    fn test_tool_callback_type_clone() {
        // Ensure ToolCallbackType can be cloned
        let sync_cb: Arc<ToolCallback> = Arc::new(|_| Ok("test".to_string()));
        let cb_type = ToolCallbackType::Sync(sync_cb);
        let _ = cb_type.clone();
    }
}
