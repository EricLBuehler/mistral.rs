mod request;
mod response;

use candle_core::Result;
use regex::Regex;
pub use request::*;
pub use response::*;
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, OnceLock},
};
use uuid::Uuid;

use crate::Pipeline;

fn process_model_specific_message(message: &str) -> Result<String> {
    static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();
    let deepseek_regex = DEEPSEEK_REGEX.get_or_init(|| Regex::new(
        r"<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
    ).unwrap());

    if let Some(message) = message.strip_prefix("<|python_tag|>") {
        // Llama case
        Ok(message.to_string())
    } else if let Some(message) = message
        .strip_prefix("<tool_call>")
        .and_then(|s| s.strip_suffix("</tool_call>"))
    {
        // Hermes case
        Ok(message.to_string())
    } else if let Some(message) = message
        .strip_prefix("[TOOL_CALLS][")
        .and_then(|s| s.strip_suffix("]"))
    {
        // Mistral Nemo case
        Ok(message.to_string())
    } else if deepseek_regex.find(message).is_some() {
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }
        let mut calls = Vec::new();
        for caps in deepseek_regex.captures_iter(message) {
            let name = caps
                .name("name")
                .ok_or("Could not capture function name")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim()
                .to_string();
            let json_str = caps
                .name("json")
                .ok_or("Could not capture JSON arguments")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim();
            let arguments: Value =
                serde_json::from_str(json_str).map_err(candle_core::Error::msg)?;
            calls.push(ToolCall { name, arguments });
        }
        Ok(serde_json::to_string(&calls).map_err(candle_core::Error::msg)?)
    } else {
        Ok(message.to_string())
    }
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

// Same as CalledFunction, but has different cases for variations on the names
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    #[serde(alias = "function")]
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: HashMap<String, Value>,
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    // Checks if the the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(
        &self,
        _pipeline: &dyn Pipeline,
        message_prefix: &str,
    ) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let message_prefix = process_model_specific_message(message_prefix)?;

        // Check if the prefix could be a JSON serialization of any of the following types.
        Ok([
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<Vec<CalledFunctionParameters>>,
        ]
        .iter()
        .find_map(|check| {
            let (could_be_tool, is_complete_tool) = check(&message_prefix);
            if could_be_tool || is_complete_tool {
                Some((could_be_tool, is_complete_tool))
            } else {
                None
            }
        })
        .unwrap_or_default())
    }

    pub fn get_call(
        &self,
        _pipeline: &dyn Pipeline,
        message: &str,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }
        let message = process_model_specific_message(message)?;

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(&message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a given type, `T`.
///
/// Returns a tuple of `(could_be_tool, is_entire_tool)`.
fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        // EOF show that JSON parsing was successful up to the end of the entire string.
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

/// Takes raw UTf8 text and parses any possible tool calls from it.
pub fn parse_text_tools<'a>(
    pipeline: &dyn Pipeline,
    raw_text: &'a str,
    matcher: Option<Arc<ToolCallingMatcher>>,
) -> anyhow::Result<(Option<&'a str>, Vec<ToolCallResponse>)> {
    let mut tool_calls = Vec::new();
    let mut text_new = Some(raw_text);

    if let Some(ref matcher) = matcher {
        let calls = matcher
            .get_call(pipeline, raw_text)
            .map_err(candle_core::Error::msg)?;
        if !calls.is_empty() {
            text_new = None;
            tool_calls = calls;
        }
    };
    Ok((text_new, tool_calls))
}
