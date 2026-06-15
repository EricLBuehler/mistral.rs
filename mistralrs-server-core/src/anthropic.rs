//! Anthropic-compatible Messages API.

use std::{
    collections::{HashMap, VecDeque},
    error::Error,
    pin::Pin,
    task::Poll,
    time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    extract::{Json, State},
    http,
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension,
};
use either::Either;
use mistralrs_core::{
    AgentPermission, AgentToolApprovalHandler, ApproximateUserLocation,
    ChatCompletionChunkResponse, ChatCompletionResponse, CodeExecutionPermission, Function,
    MistralRs, Request, RequestMessage, Response, TokenizationRequest, Tool, ToolChoice, ToolType,
    Usage, WebSearchOptions, WebSearchUserLocation,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::{
    sync::mpsc::Receiver,
    time::{interval_at, Instant, Interval, MissedTickBehavior},
};
use utoipa::ToSchema;

use crate::{
    chat_completion::parse_request,
    handler_core::{create_response_channel, send_request_with_model},
    mistralrs_server_router_builder::AgenticDefaults,
    openai::{
        ChatCompletionRequest, FunctionCalled, Grammar, Message, MessageContent, ResponseFormat,
        StopTokens, ToolCall,
    },
    streaming::get_keep_alive_interval,
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::sanitize_error_message,
};

type BoxError = Box<dyn Error + Send + Sync + 'static>;

const ANTHROPIC_WEB_SEARCH_PREFIX: &str = "web_search_";
const ANTHROPIC_DYNAMIC_WEB_SEARCH_TYPE: &str = "web_search_20260209";
const ANTHROPIC_WEB_SEARCH_NAME: &str = "web_search";
const ANTHROPIC_CODE_EXECUTION_PREFIX: &str = "code_execution_";
const ANTHROPIC_CODE_EXECUTION_NAME: &str = "code_execution";

fn default_model() -> String {
    "default".to_string()
}

fn default_message_type() -> String {
    "message".to_string()
}

fn default_assistant_role() -> String {
    "assistant".to_string()
}

fn default_error_type() -> String {
    "error".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicMessagesRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub max_tokens: Option<usize>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<u32, f32>>,
    #[serde(default)]
    pub logprobs: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<Grammar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[serde(default)]
    pub enable_code_execution: bool,
    #[schema(value_type = Option<String>, example = json!(Option::None::<String>))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_permission: Option<AgentPermission>,
    #[schema(value_type = Option<String>, example = json!(Option::None::<String>))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_execution_permission: Option<CodeExecutionPermission>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Vec<serde_json::Value>>)]
    pub files: Option<Vec<mistralrs_core::RequestedFile>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_rounds: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_sequence: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<AnthropicImageSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_use_id: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicTool {
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_location: Option<AnthropicWebSearchUserLocation>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicWebSearchUserLocation {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timezone: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct AnthropicThinking {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type", default = "default_message_type")]
    pub tp: String,
    #[serde(default = "default_assistant_role")]
    pub role: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AnthropicResponseContentBlock {
    #[serde(rename = "type")]
    pub tp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[schema(value_type = Option<Object>)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

impl AnthropicResponseContentBlock {
    fn text(text: String) -> Self {
        Self {
            tp: "text".to_string(),
            text: Some(text),
            id: None,
            name: None,
            input: None,
            thinking: None,
        }
    }

    fn thinking(thinking: String) -> Self {
        Self {
            tp: "thinking".to_string(),
            text: None,
            id: None,
            name: None,
            input: None,
            thinking: Some(thinking),
        }
    }

    fn tool_use(id: String, name: String, input: Value) -> Self {
        Self {
            tp: "tool_use".to_string(),
            text: None,
            id: Some(id),
            name: Some(name),
            input: Some(input),
            thinking: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, ToSchema)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub tp: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AnthropicError {
    #[serde(rename = "type", default = "default_error_type")]
    pub tp: String,
    pub error: AnthropicErrorBody,
}

struct AnthropicConvertedTools {
    tools: Option<Vec<Tool>>,
    web_search_options: Option<WebSearchOptions>,
    enable_code_execution: bool,
    server_tool_names: Vec<String>,
}

impl AnthropicMessagesRequest {
    fn into_chat_completion_request(self) -> Result<ChatCompletionRequest> {
        let mut converted_tools = convert_tools_and_agentic(
            self.tools,
            self.web_search_options,
            self.enable_code_execution,
        )?;
        let tool_choice = convert_tool_choice(
            self.tool_choice,
            converted_tools.tools.as_deref(),
            &converted_tools.server_tool_names,
        )?;
        if matches!(tool_choice, Some(ToolChoice::None)) {
            converted_tools.tools = None;
            converted_tools.web_search_options = None;
            converted_tools.enable_code_execution = false;
        }
        let mut messages = Vec::new();

        if let Some(system) = self.system {
            if let Some(system_text) = system_to_text(system)? {
                messages.push(message_with_text("system", system_text));
            }
        }

        for message in self.messages {
            append_anthropic_message(&mut messages, message)?;
        }

        Ok(ChatCompletionRequest {
            messages: Either::Left(messages),
            model: self.model,
            logit_bias: self.logit_bias,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            max_tokens: self.max_tokens,
            n_choices: 1,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repetition_penalty: self.repetition_penalty,
            stop_seqs: self.stop_sequences.map(StopTokens::Multi),
            temperature: self.temperature,
            top_p: self.top_p,
            stream: self.stream,
            tools: converted_tools.tools,
            tool_choice,
            response_format: self.response_format,
            web_search_options: converted_tools.web_search_options,
            enable_code_execution: converted_tools.enable_code_execution,
            agent_permission: self.agent_permission,
            code_execution_permission: self.code_execution_permission,
            session_id: self.session_id,
            files: self.files,
            top_k: self.top_k,
            grammar: self.grammar,
            min_p: self.min_p,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_sequence_breakers: self.dry_sequence_breakers,
            enable_thinking: self.enable_thinking.or_else(|| {
                self.thinking
                    .as_ref()
                    .map(|thinking| thinking.tp == "enabled")
            }),
            reasoning_effort: self.reasoning_effort,
            max_tool_rounds: self.max_tool_rounds,
            truncate_sequence: self.truncate_sequence,
        })
    }
}

impl TryFrom<AnthropicTool> for Tool {
    type Error = anyhow::Error;

    fn try_from(tool: AnthropicTool) -> Result<Self> {
        let name = tool
            .name
            .context("Anthropic client tool requires a `name`.")?;
        let input_schema = tool
            .input_schema
            .context("Anthropic client tool requires `input_schema`.")?;
        let Value::Object(input_schema) = input_schema else {
            anyhow::bail!(
                "Anthropic tool `{}` input_schema must be a JSON object.",
                name
            );
        };

        Ok(Self {
            tp: ToolType::Function,
            function: Function {
                description: tool.description,
                name,
                parameters: Some(input_schema.into_iter().collect::<HashMap<_, _>>()),
                strict: None,
            },
        })
    }
}

impl AnthropicTool {
    fn is_web_search_tool(&self) -> bool {
        self.tp
            .as_deref()
            .is_some_and(|tp| tp.starts_with(ANTHROPIC_WEB_SEARCH_PREFIX))
            || self.name.as_deref() == Some(ANTHROPIC_WEB_SEARCH_NAME)
                && self.input_schema.is_none()
    }

    fn is_dynamic_web_search_tool(&self) -> bool {
        self.tp.as_deref() == Some(ANTHROPIC_DYNAMIC_WEB_SEARCH_TYPE)
    }

    fn is_code_execution_tool(&self) -> bool {
        self.tp
            .as_deref()
            .is_some_and(|tp| tp.starts_with(ANTHROPIC_CODE_EXECUTION_PREFIX))
            || self.name.as_deref() == Some(ANTHROPIC_CODE_EXECUTION_NAME)
                && self.input_schema.is_none()
    }

    fn web_search_options(&self) -> Result<WebSearchOptions> {
        Ok(WebSearchOptions {
            user_location: self
                .user_location
                .clone()
                .map(TryInto::try_into)
                .transpose()?,
            ..Default::default()
        })
    }

    fn server_tool_name(&self) -> Option<String> {
        if self.is_web_search_tool() {
            Some(ANTHROPIC_WEB_SEARCH_NAME.to_string())
        } else if self.is_code_execution_tool() {
            Some(ANTHROPIC_CODE_EXECUTION_NAME.to_string())
        } else {
            None
        }
    }
}

impl TryFrom<AnthropicWebSearchUserLocation> for WebSearchUserLocation {
    type Error = anyhow::Error;

    fn try_from(location: AnthropicWebSearchUserLocation) -> Result<Self> {
        if location.tp != "approximate" {
            anyhow::bail!(
                "Unsupported Anthropic web search user_location type `{}`.",
                location.tp
            );
        }

        Ok(Self::Approximate {
            approximate: ApproximateUserLocation {
                city: location.city.unwrap_or_default(),
                country: location.country.unwrap_or_default(),
                region: location.region.unwrap_or_default(),
                timezone: location.timezone.unwrap_or_default(),
            },
        })
    }
}

fn convert_tools_and_agentic(
    tools: Option<Vec<AnthropicTool>>,
    explicit_web_search_options: Option<WebSearchOptions>,
    explicit_enable_code_execution: bool,
) -> Result<AnthropicConvertedTools> {
    let mut converted_tools = Vec::new();
    let mut web_search_options = explicit_web_search_options;
    let mut enable_code_execution = explicit_enable_code_execution;
    let mut server_tool_names = Vec::new();

    for tool in tools.unwrap_or_default() {
        if let Some(name) = tool.server_tool_name() {
            if name == ANTHROPIC_WEB_SEARCH_NAME {
                if web_search_options.is_none() {
                    web_search_options = Some(tool.web_search_options()?);
                }
                if tool.is_dynamic_web_search_tool() {
                    enable_code_execution = true;
                }
            } else if name == ANTHROPIC_CODE_EXECUTION_NAME {
                enable_code_execution = true;
            }
            server_tool_names.push(name);
            continue;
        }

        converted_tools.push(tool.try_into()?);
    }

    Ok(AnthropicConvertedTools {
        tools: (!converted_tools.is_empty()).then_some(converted_tools),
        web_search_options,
        enable_code_execution,
        server_tool_names,
    })
}

fn convert_tool_choice(
    tool_choice: Option<AnthropicToolChoice>,
    tools: Option<&[Tool]>,
    server_tool_names: &[String],
) -> Result<Option<ToolChoice>> {
    let Some(tool_choice) = tool_choice else {
        return Ok(None);
    };

    match tool_choice.tp.as_str() {
        "auto" | "any" => Ok(Some(ToolChoice::Auto)),
        "none" => Ok(Some(ToolChoice::None)),
        "tool" => {
            let name = tool_choice
                .name
                .context("Anthropic tool_choice type `tool` requires a `name`.")?;
            if server_tool_names.iter().any(|tool_name| tool_name == &name) {
                return Ok(Some(ToolChoice::Auto));
            }
            let tool = tools
                .unwrap_or_default()
                .iter()
                .find(|tool| tool.function.name == name)
                .cloned()
                .with_context(|| {
                    format!("Anthropic tool_choice references unknown tool `{name}`.")
                })?;
            Ok(Some(ToolChoice::Tool(tool)))
        }
        other => anyhow::bail!("Unsupported Anthropic tool_choice type `{other}`."),
    }
}

fn append_anthropic_message(out: &mut Vec<Message>, message: AnthropicMessage) -> Result<()> {
    match message.content {
        AnthropicMessageContent::Text(text) => {
            out.push(message_with_text(message.role, text));
        }
        AnthropicMessageContent::Blocks(blocks) => {
            append_anthropic_blocks(out, message.role, blocks)?;
        }
    }
    Ok(())
}

fn append_anthropic_blocks(
    out: &mut Vec<Message>,
    role: String,
    blocks: Vec<AnthropicContentBlock>,
) -> Result<()> {
    if role == "assistant" {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in blocks {
            match block.tp.as_str() {
                "text" => text_parts.push(required_text(&block)?),
                "tool_use" => tool_calls.push(tool_call_from_block(block)?),
                "server_tool_use" | "thinking" | "redacted_thinking" => {}
                other => anyhow::bail!("Unsupported Anthropic assistant content block `{other}`."),
            }
        }

        let content =
            (!text_parts.is_empty()).then(|| MessageContent::from_text(text_parts.join("\n")));
        let tool_calls = (!tool_calls.is_empty()).then_some(tool_calls);
        out.push(Message {
            content,
            role,
            name: None,
            tool_calls,
            tool_call_id: None,
        });
        return Ok(());
    }

    let mut text_parts = Vec::new();
    let mut image_parts = Vec::new();

    for block in blocks {
        match block.tp.as_str() {
            "text" => text_parts.push(required_text(&block)?),
            "image" => {
                if role != "user" {
                    anyhow::bail!("Anthropic image content is only supported for user messages.");
                }
                let source = block
                    .source
                    .context("Anthropic image content block requires `source`.")?;
                image_parts.push(MessageContent::image_url_part(source.into_url()?));
            }
            "tool_result" => {
                flush_user_content(out, &role, &mut text_parts, &mut image_parts);
                out.push(tool_result_message_from_block(block)?);
            }
            "thinking"
            | "redacted_thinking"
            | "web_search_tool_result"
            | "code_execution_tool_result" => {}
            other => anyhow::bail!("Unsupported Anthropic content block `{other}`."),
        }
    }

    flush_user_content(out, &role, &mut text_parts, &mut image_parts);
    Ok(())
}

fn flush_user_content(
    out: &mut Vec<Message>,
    role: &str,
    text_parts: &mut Vec<String>,
    image_parts: &mut Vec<HashMap<String, crate::openai::MessageInnerContent>>,
) {
    if text_parts.is_empty() && image_parts.is_empty() {
        return;
    }

    let text = text_parts.join("\n");
    let content = if image_parts.is_empty() {
        MessageContent::from_text(text)
    } else {
        let mut parts = std::mem::take(image_parts);
        if !text.is_empty() {
            parts.push(MessageContent::text_part(text));
        }
        MessageContent::from_parts(parts)
    };

    out.push(Message {
        content: Some(content),
        role: role.to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    });
    text_parts.clear();
}

fn message_with_text(role: impl Into<String>, text: String) -> Message {
    Message {
        content: Some(MessageContent::from_text(text)),
        role: role.into(),
        name: None,
        tool_calls: None,
        tool_call_id: None,
    }
}

fn required_text(block: &AnthropicContentBlock) -> Result<String> {
    block
        .text
        .clone()
        .with_context(|| format!("Anthropic `{}` content block requires `text`.", block.tp))
}

fn tool_call_from_block(block: AnthropicContentBlock) -> Result<ToolCall> {
    let id = block
        .id
        .context("Anthropic tool_use content block requires `id`.")?;
    let name = block
        .name
        .context("Anthropic tool_use content block requires `name`.")?;
    let input = block.input.unwrap_or_else(|| Value::Object(Map::new()));

    Ok(ToolCall {
        id: Some(id),
        tp: ToolType::Function,
        function: FunctionCalled {
            name,
            arguments: serde_json::to_string(&input)?,
        },
    })
}

fn tool_result_message_from_block(block: AnthropicContentBlock) -> Result<Message> {
    let tool_call_id = block
        .tool_use_id
        .context("Anthropic tool_result content block requires `tool_use_id`.")?;

    Ok(Message {
        content: Some(MessageContent::from_text(content_value_to_text(
            block.content,
        ))),
        role: "tool".to_string(),
        name: None,
        tool_calls: None,
        tool_call_id: Some(tool_call_id),
    })
}

fn content_value_to_text(content: Option<Value>) -> String {
    match content {
        Some(Value::String(text)) => text,
        Some(Value::Array(items)) => items
            .into_iter()
            .map(content_item_to_text)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        Some(value) => value.to_string(),
        None => String::new(),
    }
}

fn content_item_to_text(value: Value) -> String {
    let Value::Object(obj) = value else {
        return value.to_string();
    };

    match obj.get("type").and_then(Value::as_str) {
        Some("text") => obj
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        Some("image") => "[image]".to_string(),
        _ => Value::Object(obj).to_string(),
    }
}

impl AnthropicImageSource {
    fn into_url(self) -> Result<String> {
        match self.tp.as_str() {
            "base64" => {
                let media_type = self
                    .media_type
                    .context("Anthropic base64 image source requires `media_type`.")?;
                let data = self
                    .data
                    .context("Anthropic base64 image source requires `data`.")?;
                Ok(format!("data:{media_type};base64,{data}"))
            }
            "url" => self
                .url
                .context("Anthropic URL image source requires `url`."),
            other => anyhow::bail!("Unsupported Anthropic image source type `{other}`."),
        }
    }
}

fn system_to_text(system: AnthropicSystem) -> Result<Option<String>> {
    match system {
        AnthropicSystem::Text(text) => Ok((!text.is_empty()).then_some(text)),
        AnthropicSystem::Blocks(blocks) => {
            let mut text_parts = Vec::new();
            for block in blocks {
                if block.tp != "text" {
                    anyhow::bail!("Anthropic system blocks only support `text` content.");
                }
                text_parts.push(required_text(&block)?);
            }
            Ok((!text_parts.is_empty()).then(|| text_parts.join("\n")))
        }
    }
}

fn usage_from_openai(usage: &Usage) -> AnthropicUsage {
    AnthropicUsage {
        input_tokens: usage.prompt_tokens,
        output_tokens: usage.completion_tokens,
    }
}

fn usage_json(usage: Option<&Usage>) -> Value {
    json!(usage.map(usage_from_openai).unwrap_or_default())
}

fn output_usage_json(usage: Option<&Usage>) -> Value {
    json!({
        "output_tokens": usage.map(|usage| usage.completion_tokens).unwrap_or_default(),
    })
}

fn stop_reason(finish_reason: &str) -> String {
    match finish_reason {
        "tool_calls" => "tool_use",
        "length" => "max_tokens",
        "stop_sequence" => "stop_sequence",
        "stop" | "eos" => "end_turn",
        _ => "end_turn",
    }
    .to_string()
}

fn tool_input_from_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({ "arguments": arguments }))
}

fn anthropic_response_from_chat(response: ChatCompletionResponse) -> AnthropicMessageResponse {
    let mut content = Vec::new();
    let mut finish_reason = "stop".to_string();

    if let Some(choice) = response.choices.into_iter().next() {
        finish_reason = choice.finish_reason;
        if let Some(thinking) = choice
            .message
            .reasoning_content
            .filter(|thinking| !thinking.is_empty())
        {
            content.push(AnthropicResponseContentBlock::thinking(thinking));
        }
        if let Some(text) = choice.message.content.filter(|text| !text.is_empty()) {
            content.push(AnthropicResponseContentBlock::text(text));
        }
        if let Some(tool_calls) = choice.message.tool_calls {
            for tool_call in tool_calls {
                content.push(AnthropicResponseContentBlock::tool_use(
                    tool_call.id,
                    tool_call.function.name,
                    tool_input_from_arguments(&tool_call.function.arguments),
                ));
            }
        }
    }

    AnthropicMessageResponse {
        id: response.id,
        tp: default_message_type(),
        role: default_assistant_role(),
        content,
        model: response.model,
        stop_reason: stop_reason(&finish_reason),
        stop_sequence: None,
        usage: usage_from_openai(&response.usage),
    }
}

pub struct AnthropicStreamer {
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    pending: VecDeque<Result<Event, axum::Error>>,
    done: bool,
    message_started: bool,
    thinking_block_index: Option<usize>,
    text_block_index: Option<usize>,
    next_content_index: usize,
    ping: Interval,
}

impl AnthropicStreamer {
    fn new(rx: Receiver<Response>, state: SharedMistralRsState) -> Self {
        let ping_interval = Duration::from_millis(get_keep_alive_interval());
        let mut ping = interval_at(Instant::now() + ping_interval, ping_interval);
        ping.set_missed_tick_behavior(MissedTickBehavior::Delay);
        Self {
            rx,
            state,
            pending: VecDeque::new(),
            done: false,
            message_started: false,
            thinking_block_index: None,
            text_block_index: None,
            next_content_index: 0,
            ping,
        }
    }

    fn enqueue_json(&mut self, event: &'static str, payload: Value) {
        self.pending
            .push_back(Event::default().event(event).json_data(payload));
    }

    fn start_message(&mut self, chunk: &ChatCompletionChunkResponse) {
        if self.message_started {
            return;
        }
        self.message_started = true;
        self.enqueue_json(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": chunk.id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": chunk.model,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": usage_json(chunk.usage.as_ref()),
                }
            }),
        );
    }

    fn start_text_block(&mut self) -> usize {
        if let Some(index) = self.text_block_index {
            return index;
        }
        self.stop_thinking_block();
        let index = self.next_content_index;
        self.next_content_index += 1;
        self.text_block_index = Some(index);
        self.enqueue_json(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            }),
        );
        index
    }

    fn start_thinking_block(&mut self) -> usize {
        if let Some(index) = self.thinking_block_index {
            return index;
        }
        self.stop_text_block();
        let index = self.next_content_index;
        self.next_content_index += 1;
        self.thinking_block_index = Some(index);
        self.enqueue_json(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""},
            }),
        );
        index
    }

    fn stop_thinking_block(&mut self) {
        let Some(index) = self.thinking_block_index.take() else {
            return;
        };
        self.enqueue_json(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        );
    }

    fn stop_text_block(&mut self) {
        let Some(index) = self.text_block_index.take() else {
            return;
        };
        self.enqueue_json(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        );
    }

    fn enqueue_thinking_delta(&mut self, thinking: &str) {
        if thinking.is_empty() {
            return;
        }
        let index = self.start_thinking_block();
        self.enqueue_json(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "thinking_delta", "thinking": thinking},
            }),
        );
    }

    fn enqueue_text_delta(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let index = self.start_text_block();
        self.enqueue_json(
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            }),
        );
    }

    fn enqueue_tool_use(&mut self, tool_call: &mistralrs_core::ToolCallResponse) {
        self.stop_thinking_block();
        self.stop_text_block();
        let index = self.next_content_index;
        self.next_content_index += 1;
        self.enqueue_json(
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": {},
                },
            }),
        );
        if !tool_call.function.arguments.is_empty() {
            self.enqueue_json(
                "content_block_delta",
                json!({
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": tool_call.function.arguments,
                    },
                }),
            );
        }
        self.enqueue_json(
            "content_block_stop",
            json!({
                "type": "content_block_stop",
                "index": index,
            }),
        );
    }

    fn finish_message(&mut self, finish_reason: Option<&str>, usage: Option<&Usage>) {
        self.stop_thinking_block();
        self.stop_text_block();
        self.enqueue_json(
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": finish_reason.map(stop_reason).unwrap_or_else(|| "end_turn".to_string()),
                    "stop_sequence": null,
                },
                "usage": output_usage_json(usage),
            }),
        );
        self.enqueue_json("message_stop", json!({"type": "message_stop"}));
        self.done = true;
    }

    fn handle_chunk(&mut self, chunk: ChatCompletionChunkResponse) {
        MistralRs::maybe_log_response(self.state.clone(), &chunk);
        self.start_message(&chunk);

        let Some(choice) = chunk.choices.first() else {
            return;
        };

        if let Some(thinking) = choice.delta.reasoning_content.as_deref() {
            self.enqueue_thinking_delta(thinking);
        }

        if let Some(text) = choice.delta.content.as_deref() {
            self.enqueue_text_delta(text);
        }

        if let Some(tool_calls) = &choice.delta.tool_calls {
            for tool_call in tool_calls {
                self.enqueue_tool_use(tool_call);
            }
        }

        if choice.finish_reason.is_some() {
            self.finish_message(choice.finish_reason.as_deref(), chunk.usage.as_ref());
        }
    }

    fn handle_error_event(&mut self, error_type: &'static str, message: String) {
        self.enqueue_json(
            "error",
            json!({
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": message,
                },
            }),
        );
        self.done = true;
    }
}

impl futures::Stream for AnthropicStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Poll::Ready(Some(event));
            }
            if self.done {
                return Poll::Ready(None);
            }

            match self.rx.poll_recv(cx) {
                Poll::Ready(Some(resp)) => match resp {
                    Response::Chunk(chunk) => self.handle_chunk(chunk),
                    Response::ModelError(msg, _) => {
                        MistralRs::maybe_log_error(
                            self.state.clone(),
                            &crate::handler_core::ModelErrorMessage(msg.to_string()),
                        );
                        self.handle_error_event("api_error", msg);
                    }
                    Response::ValidationError(e) => {
                        self.handle_error_event(
                            "invalid_request_error",
                            sanitize_error_message(e.as_ref()),
                        );
                    }
                    Response::InternalError(e) => {
                        MistralRs::maybe_log_error(self.state.clone(), &*e);
                        self.handle_error_event("api_error", sanitize_error_message(e.as_ref()));
                    }
                    Response::AgenticToolCallProgress {
                        round,
                        tool_name,
                        phase,
                    } => {
                        self.enqueue_json(
                            "agentic_tool_call_progress",
                            crate::chat_completion::serialize_agentic_progress(
                                round, &tool_name, &phase,
                            ),
                        );
                    }
                    Response::AgenticToolApprovalRequired {
                        approval_id,
                        session_id,
                        round,
                        tool,
                        arguments,
                    } => {
                        self.enqueue_json(
                            "agentic_tool_approval_required",
                            json!({
                                "type": "agentic_tool_approval_required",
                                "approval_id": approval_id,
                                "session_id": session_id,
                                "round": round,
                                "tool": tool,
                                "arguments": arguments,
                            }),
                        );
                    }
                    Response::File(file) => {
                        self.enqueue_json("file_produced", json!(file));
                    }
                    Response::Done(_)
                    | Response::CompletionDone(_)
                    | Response::CompletionModelError(_, _)
                    | Response::CompletionChunk(_)
                    | Response::ImageGeneration(_)
                    | Response::Speech { .. }
                    | Response::Raw { .. }
                    | Response::Embeddings { .. } => unreachable!(),
                },
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => {
                    if Pin::new(&mut self.ping).poll_tick(cx).is_ready() {
                        self.enqueue_json("ping", json!({"type": "ping"}));
                        continue;
                    }
                    return Poll::Pending;
                }
            }
        }
    }
}

pub type AnthropicMessagesSse = Sse<KeepAliveStream<AnthropicStreamer>>;

pub enum AnthropicMessagesResponder {
    Sse(AnthropicMessagesSse),
    Json(AnthropicMessageResponse),
    InternalError(BoxError),
    ValidationError(BoxError),
    ModelError(String),
}

pub enum AnthropicCountTokensResponder {
    Json(AnthropicCountTokensResponse),
    InternalError(BoxError),
    ValidationError(BoxError),
}

impl IntoResponse for AnthropicMessagesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnthropicMessagesResponder::Sse(s) => s.into_response(),
            AnthropicMessagesResponder::Json(s) => Json(s).into_response(),
            AnthropicMessagesResponder::InternalError(e) => anthropic_error_response(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                sanitize_error_message(e.as_ref()),
            ),
            AnthropicMessagesResponder::ValidationError(e) => anthropic_error_response(
                http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                sanitize_error_message(e.as_ref()),
            ),
            AnthropicMessagesResponder::ModelError(msg) => {
                anthropic_error_response(http::StatusCode::INTERNAL_SERVER_ERROR, "api_error", msg)
            }
        }
    }
}

impl IntoResponse for AnthropicCountTokensResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            AnthropicCountTokensResponder::Json(s) => Json(s).into_response(),
            AnthropicCountTokensResponder::InternalError(e) => anthropic_error_response(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                sanitize_error_message(e.as_ref()),
            ),
            AnthropicCountTokensResponder::ValidationError(e) => anthropic_error_response(
                http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                sanitize_error_message(e.as_ref()),
            ),
        }
    }
}

fn anthropic_error_response(
    status: http::StatusCode,
    error_type: &'static str,
    message: String,
) -> axum::response::Response {
    (
        status,
        Json(AnthropicError {
            tp: default_error_type(),
            error: AnthropicErrorBody {
                tp: error_type.to_string(),
                message,
            },
        }),
    )
        .into_response()
}

fn handle_validation_error(e: anyhow::Error) -> AnthropicMessagesResponder {
    AnthropicMessagesResponder::ValidationError(e.into())
}

fn create_streamer(rx: Receiver<Response>, state: SharedMistralRsState) -> AnthropicMessagesSse {
    let streamer = AnthropicStreamer::new(rx, state);
    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(get_keep_alive_interval())))
}

async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> AnthropicMessagesResponder {
    loop {
        match rx.recv().await {
            Some(Response::Done(response)) => {
                MistralRs::maybe_log_response(state, &response);
                return AnthropicMessagesResponder::Json(anthropic_response_from_chat(response));
            }
            Some(Response::ModelError(msg, response)) => {
                MistralRs::maybe_log_error(
                    state.clone(),
                    &crate::handler_core::ModelErrorMessage(msg.to_string()),
                );
                MistralRs::maybe_log_response(state, &response);
                return AnthropicMessagesResponder::ModelError(msg);
            }
            Some(Response::ValidationError(e)) => {
                return AnthropicMessagesResponder::ValidationError(e);
            }
            Some(Response::InternalError(e)) => {
                MistralRs::maybe_log_error(state, &*e);
                return AnthropicMessagesResponder::InternalError(e);
            }
            Some(Response::AgenticToolApprovalRequired { .. }) => {
                return AnthropicMessagesResponder::ValidationError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "agent approval requires a streaming HTTP request.",
                )));
            }
            Some(Response::AgenticToolCallProgress { .. } | Response::File(_)) => {}
            Some(_) => unreachable!(),
            None => {
                return AnthropicMessagesResponder::InternalError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "No response received from the model.",
                )));
            }
        }
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/messages",
    request_body = AnthropicMessagesRequest,
    responses((status = 200, description = "Anthropic messages", body = AnthropicMessageResponse))
)]
pub async fn anthropic_messages(
    State(state): ExtractedMistralRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Json(request): Json<AnthropicMessagesRequest>,
) -> AnthropicMessagesResponder {
    let (tx, mut rx) = create_response_channel(None);
    let mut oairequest = match request.into_chat_completion_request() {
        Ok(request) => request,
        Err(e) => return handle_validation_error(e),
    };

    oairequest.max_tool_rounds = oairequest
        .max_tool_rounds
        .or(agentic_defaults.max_tool_rounds);

    let request_permission = oairequest
        .agent_permission
        .or_else(|| oairequest.code_execution_permission.map(Into::into));
    oairequest.agent_permission = match (agentic_defaults.agent_permission, request_permission) {
        (Some(server_permission), Some(request_permission)) => {
            Some(server_permission.strictest(request_permission))
        }
        (Some(server_permission), None) => Some(server_permission),
        (None, permission) => permission,
    };
    oairequest.code_execution_permission = None;

    let is_streaming = oairequest.stream.unwrap_or(false);
    if matches!(oairequest.agent_permission, Some(AgentPermission::Ask)) && !is_streaming {
        return AnthropicMessagesResponder::ValidationError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "agent_permission `ask` requires stream=true over HTTP.",
        )));
    }

    let agent_approval_handler = matches!(oairequest.agent_permission, Some(AgentPermission::Ask))
        .then(|| AgentToolApprovalHandler::from_async(agentic_defaults.approval_broker.callback()));
    let agent_approval_notifier =
        if is_streaming && matches!(oairequest.agent_permission, Some(AgentPermission::Ask)) {
            Some(agentic_defaults.approval_broker.notifier(tx.clone()))
        } else {
            None
        };
    let model_id = (oairequest.model != "default").then(|| oairequest.model.clone());

    let (request, is_streaming) = match parse_request(
        oairequest,
        state.clone(),
        tx,
        agentic_defaults.tool_dispatch_url,
        agent_approval_handler,
        agent_approval_notifier,
    )
    .await
    {
        Ok(x) => x,
        Err(e) => return handle_validation_error(e),
    };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return AnthropicMessagesResponder::InternalError(e.into());
    }

    if is_streaming {
        AnthropicMessagesResponder::Sse(create_streamer(rx, state))
    } else {
        process_non_streaming_response(&mut rx, state).await
    }
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/messages/count_tokens",
    request_body = AnthropicMessagesRequest,
    responses((status = 200, description = "Anthropic message token count", body = AnthropicCountTokensResponse))
)]
pub async fn anthropic_count_tokens(
    State(state): ExtractedMistralRsState,
    Json(request): Json<AnthropicMessagesRequest>,
) -> AnthropicCountTokensResponder {
    let (tx, _) = create_response_channel(Some(1));
    let mut oairequest = match request.into_chat_completion_request() {
        Ok(request) => request,
        Err(e) => return AnthropicCountTokensResponder::ValidationError(e.into()),
    };

    oairequest.stream = Some(false);
    let model_id = (oairequest.model != "default").then(|| oairequest.model.clone());

    let (request, _) = match parse_request(oairequest, state.clone(), tx, None, None, None).await {
        Ok(x) => x,
        Err(e) => return AnthropicCountTokensResponder::ValidationError(e.into()),
    };

    let Request::Normal(request) = request else {
        return AnthropicCountTokensResponder::InternalError(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Expected a chat request for token counting.",
        )));
    };

    let (messages, enable_thinking, reasoning_effort) = match request.messages {
        RequestMessage::Chat {
            messages,
            enable_thinking,
            reasoning_effort,
        }
        | RequestMessage::MultimodalChat {
            messages,
            enable_thinking,
            reasoning_effort,
            ..
        } => (messages, enable_thinking, reasoning_effort),
        _ => {
            return AnthropicCountTokensResponder::ValidationError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Only chat messages can be counted by the Anthropic count_tokens endpoint.",
            )));
        }
    };

    let (response, mut rx) = tokio::sync::mpsc::channel(1);
    let tokenize_request = Request::Tokenize(TokenizationRequest {
        text: Either::Left(messages),
        tools: request.tools,
        add_generation_prompt: true,
        add_special_tokens: true,
        enable_thinking,
        reasoning_effort,
        response,
    });

    if let Err(e) = send_request_with_model(&state, tokenize_request, model_id.as_deref()).await {
        return AnthropicCountTokensResponder::InternalError(e.into());
    }

    match rx.recv().await {
        Some(Ok(tokens)) => AnthropicCountTokensResponder::Json(AnthropicCountTokensResponse {
            input_tokens: tokens.len(),
        }),
        Some(Err(e)) => AnthropicCountTokensResponder::ValidationError(e.into()),
        None => AnthropicCountTokensResponder::InternalError(Box::new(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "No token count received from the model.",
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_system_and_user_text() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "default",
            "max_tokens": 32,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": "Hello"}]
        }))
        .unwrap();

        let chat = req.into_chat_completion_request().unwrap();
        let Either::Left(messages) = chat.messages else {
            panic!("expected message array");
        };

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(
            messages[0]
                .content
                .as_ref()
                .and_then(MessageContent::to_text),
            Some("You are terse.".to_string())
        );
        assert_eq!(messages[1].role, "user");
        assert_eq!(
            messages[1]
                .content
                .as_ref()
                .and_then(MessageContent::to_text),
            Some("Hello".to_string())
        );
    }

    #[test]
    fn converts_tools_and_tool_choice() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "default",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Weather?"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }))
        .unwrap();

        let chat = req.into_chat_completion_request().unwrap();
        let tools = chat.tools.unwrap();
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(matches!(chat.tool_choice, Some(ToolChoice::Tool(_))));
    }

    #[test]
    fn maps_anthropic_server_tools_to_agentic_options() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "default",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Search and compute."}],
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "user_location": {
                        "type": "approximate",
                        "city": "New York",
                        "country": "US",
                        "region": "NY",
                        "timezone": "America/New_York"
                    }
                },
                {
                    "type": "code_execution_20250825",
                    "name": "code_execution"
                }
            ],
            "tool_choice": {"type": "tool", "name": "web_search"}
        }))
        .unwrap();

        let chat = req.into_chat_completion_request().unwrap();
        assert!(chat.tools.is_none());
        assert!(chat.web_search_options.is_some());
        assert!(chat.enable_code_execution);
        assert!(matches!(chat.tool_choice, Some(ToolChoice::Auto)));
    }

    #[test]
    fn tool_choice_none_disables_anthropic_tools() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "default",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "No tools."}],
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search"
            }],
            "enable_code_execution": true,
            "tool_choice": {"type": "none"}
        }))
        .unwrap();

        let chat = req.into_chat_completion_request().unwrap();
        assert!(matches!(chat.tool_choice, Some(ToolChoice::None)));
        assert!(chat.tools.is_none());
        assert!(chat.web_search_options.is_none());
        assert!(!chat.enable_code_execution);
    }

    #[test]
    fn routes_mistralrs_sampling_and_constraint_extensions() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "default",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Return JSON."}],
            "logprobs": true,
            "top_logprobs": 3,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
            "repetition_penalty": 1.05,
            "min_p": 0.05,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 4,
            "dry_sequence_breakers": ["\\n"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"}
                }
            },
            "reasoning_effort": "high",
            "enable_thinking": true
        }))
        .unwrap();

        let chat = req.into_chat_completion_request().unwrap();
        assert!(chat.logprobs);
        assert_eq!(chat.top_logprobs, Some(3));
        assert_eq!(chat.presence_penalty, Some(0.1));
        assert_eq!(chat.frequency_penalty, Some(0.2));
        assert_eq!(chat.repetition_penalty, Some(1.05));
        assert_eq!(chat.min_p, Some(0.05));
        assert_eq!(chat.dry_multiplier, Some(0.8));
        assert_eq!(chat.dry_base, Some(1.75));
        assert_eq!(chat.dry_allowed_length, Some(4));
        assert_eq!(chat.dry_sequence_breakers, Some(vec!["\\n".to_string()]));
        assert!(matches!(
            chat.response_format,
            Some(ResponseFormat::JsonSchema { .. })
        ));
        assert_eq!(chat.reasoning_effort, Some("high".to_string()));
        assert_eq!(chat.enable_thinking, Some(true));
    }
}
