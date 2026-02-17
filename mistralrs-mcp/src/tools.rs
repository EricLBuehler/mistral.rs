use std::collections::HashMap;
use std::sync::Arc;

pub use mistralrs_tool_types::{
    CalledFunction, Function, Tool, ToolCallback, ToolCallbackWithTool, ToolType,
};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;
