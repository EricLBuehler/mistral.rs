use mistralrs_mcp::{Function, Tool, ToolType};
use serde_json::json;
use std::collections::HashMap;

pub const EXECUTE_PYTHON_TOOL_NAME: &str = "execute_python";
pub const RESET_SESSION_TOOL_NAME: &str = "reset_python_session";

pub fn code_exec_tool_called(name: &str) -> bool {
    name == EXECUTE_PYTHON_TOOL_NAME || name == RESET_SESSION_TOOL_NAME
}

pub fn build_execute_python_tool(
    timeout_secs: u64,
    installed_packages: &str,
) -> Tool {
    let description = format!(
r#"Execute Python code in a persistent interactive session.

## Session
- This is a **persistent session**: variables, imports, and state are preserved across calls within this conversation.
- Each conversation gets a unique temporary working directory.
- To reset the session (clear all variables and imports), use the `{reset}` tool.

## Capabilities
- **Last-expression capture**: If the final statement is an expression (not an assignment), its repr is returned as the result (like Jupyter/IPython). The last result is also stored in the `_` variable.
- **Matplotlib**: Figures are automatically captured as PNG images. Call `plt.show()` or leave figures open — they are captured after execution.
- **PIL Images**: If the last expression is a PIL Image, it is captured as a PNG image.
- **Pandas DataFrames**: If the last expression is a DataFrame or Series, its formatted repr is returned.
- **File I/O**: You can read and write files in the working directory.

## Timeout
- Code execution has a **{timeout}s timeout**. If exceeded, the execution is interrupted (KeyboardInterrupt). If the process is unresponsive, it is forcibly killed.
- Break long computations into smaller steps to avoid timeouts.

## Restrictions
- Package installation (pip install) is **disabled**.
- Network access and filesystem access **are available** and cannot be restricted.

## Installed Packages
The following packages are available:
```
{packages}
```

## Output
The result includes: stdout, stderr, exceptions (with traceback), last expression value, and any captured images."#,
        reset = RESET_SESSION_TOOL_NAME,
        timeout = timeout_secs,
        packages = installed_packages.trim(),
    );

    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Multiple lines and statements are supported."
            }
        },
        "required": ["code"],
        "additionalProperties": false
    }))
    .unwrap();

    Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(description),
            name: EXECUTE_PYTHON_TOOL_NAME.to_string(),
            parameters: Some(parameters),
            strict: Some(true),
        },
    }
}

pub fn build_reset_session_tool() -> Tool {
    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    }))
    .unwrap();

    Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(
                "Reset the Python execution session. Clears all variables, imports, and state \
                 while keeping the session alive. Use this when you need a clean environment."
                    .to_string(),
            ),
            name: RESET_SESSION_TOOL_NAME.to_string(),
            parameters: Some(parameters),
            strict: None,
        },
    }
}
