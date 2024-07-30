mod request;
mod response;

pub use request::*;
pub use response::*;
use std::sync::{Arc, Mutex};

pub struct ToolCallingMatcher {
    id: Arc<Mutex<usize>>,
    tool_choice: ToolChoice,
}

// Same as CalledFunction, but uses `parameters` instead of arguments.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: String,
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self {
            id: Arc::new(Mutex::new(0)),
            tool_choice,
        })
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(message) {
            let mut id = self.id.lock().unwrap();
            *id += 1;
            Ok(vec![ToolCallResponse {
                id: format!("fn_call_{}", *id),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: deser.parameters,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let mut id = self.id.lock().unwrap();
                    *id += 1;
                    ToolCallResponse {
                        id: format!("fn_call_{}", *id),
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: deser.parameters,
                        },
                    }
                })
                .collect::<Vec<_>>())
        } else if let Ok(deser) = serde_json::from_str::<CalledFunction>(message) {
            let mut id = self.id.lock().unwrap();
            *id += 1;
            Ok(vec![ToolCallResponse {
                id: format!("fn_call_{}", *id),
                tp: ToolCallType::Function,
                function: deser,
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunction>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let mut id = self.id.lock().unwrap();
                    *id += 1;
                    ToolCallResponse {
                        id: format!("fn_call_{}", *id),
                        tp: ToolCallType::Function,
                        function: deser,
                    }
                })
                .collect::<Vec<_>>())
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
}
