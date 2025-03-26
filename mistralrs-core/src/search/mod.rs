use std::collections::HashMap;

use anyhow::Result;
use serde_json::{json, Value};

use crate::{Function, Tool, ToolType};

const SEARCH_TOOL_NAME: &str = "__mistralrs_internal_search_web";

pub fn get_search_tool() -> Result<Tool> {
    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A query for web searching.",
            },
        },
        "required": ["query"],
    }))?;

    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Search the web using a given query. Use this tool if the user asks a question which requires realtime or up-to-date information.".to_string()),
            name: SEARCH_TOOL_NAME.to_string(),
            parameters: Some(parameters),
        },
    })
}
