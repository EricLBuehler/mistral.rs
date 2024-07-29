use std::cell::Cell;

use regex::Regex;

use super::{
    request::{Tool, ToolChoice},
    response::{CalledFunction, ToolCallResponse, ToolCallType},
};

// https://docs.together.ai/docs/llama-3-function-calling
const LLAMA3_FUNCTION_CALL: &str = r"<function=(\w+)>(.*?)</function>";

pub enum ToolCallingModel {
    Llama3,
    CustomMatcher(String),
}

pub struct ToolCallingMatcher {
    pattern: Regex,
    id: Cell<usize>,
    tool_choice: ToolChoice,
}

impl ToolCallingMatcher {
    pub fn new(model: &ToolCallingModel, tool_choice: ToolChoice) -> anyhow::Result<Self> {
        let pattern = match model {
            ToolCallingModel::Llama3 => LLAMA3_FUNCTION_CALL.to_string(),
            ToolCallingModel::CustomMatcher(x) => x.clone(),
        };
        Ok(Self {
            pattern: Regex::new(&pattern)?,
            id: Cell::new(0),
            tool_choice,
        })
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Option<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(None);
        }
        if let Some(captures) = self.pattern.captures(message) {
            let function_name = captures.get(1).unwrap().as_str().to_string();
            let args_string = captures.get(2).unwrap().as_str().to_string();
            self.id.set(self.id.get() + 1);
            Ok(Some(ToolCallResponse {
                id: format!("fn_call_{}", self.id.get()),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: function_name,
                    arguments: args_string,
                },
            }))
        } else {
            if matches!(self.tool_choice, ToolChoice::Required) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(None)
        }
    }
}
