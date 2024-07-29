mod request;
mod response;

pub use request::*;
pub use response::*;
use std::sync::{Arc, Mutex};

use regex::Regex;

// https://docs.together.ai/docsllama/llama-3-function-calling
const LLAMA3_FUNCTION_CALL: &str = r"<function=(\w+)>(.*?)</function>";

#[derive(Clone, Copy, Default)]
pub enum ToolCallingModel {
    #[default]
    Llama3,
}

pub struct ToolCallingMatcher {
    pattern: Regex,
    id: Arc<Mutex<usize>>,
    tool_choice: ToolChoice,
}

impl ToolCallingMatcher {
    pub fn new(model: &ToolCallingModel, tool_choice: ToolChoice) -> anyhow::Result<Self> {
        let pattern = match model {
            ToolCallingModel::Llama3 => LLAMA3_FUNCTION_CALL.to_string(),
        };
        Ok(Self {
            pattern: Regex::new(&pattern)?,
            id: Arc::new(Mutex::new(0)),
            tool_choice,
        })
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        let mut calls = Vec::new();
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(calls);
        }
        if let Some(captures) = self.pattern.captures(message) {
            let n_calls = captures.len() / 2;
            for call in 0..n_calls {
                let function_name = captures.get(call * 2).unwrap().as_str().to_string();
                let args_string = captures.get(call * 2 + 1).unwrap().as_str().to_string();
                let mut id = self.id.lock().unwrap();
                *id += 1;
                calls.push(ToolCallResponse {
                    id: format!("fn_call_{}", *id),
                    tp: ToolCallType::Function,
                    function: CalledFunction {
                        name: function_name,
                        arguments: args_string,
                    },
                });
            }
            Ok(calls)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(calls)
        }
    }
}
