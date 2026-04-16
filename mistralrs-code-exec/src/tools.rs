use mistralrs_mcp::{Function, Tool, ToolType};
use serde_json::json;
use std::collections::HashMap;

use crate::InputModality;

pub const EXECUTE_PYTHON_TOOL_NAME: &str = "mistralrs_execute_python";
pub const RESET_SESSION_TOOL_NAME: &str = "mistralrs_reset_python_session";

pub fn code_exec_tool_called(name: &str) -> bool {
    name == EXECUTE_PYTHON_TOOL_NAME || name == RESET_SESSION_TOOL_NAME
}

pub fn build_execute_python_tool(
    timeout_secs: u64,
    installed_packages: &str,
    input_modalities: &[InputModality],
) -> Tool {
    let supports_vision = input_modalities.contains(&InputModality::Vision);

    let matplotlib_desc = if supports_vision {
        "- **Matplotlib**: Figures are automatically captured as PNG images and sent to you as visible images that you can see and describe. You do NOT need to do anything special — figures are captured when you call `plt.savefig()`, `plt.show()`, or simply leave them open. Even `plt.savefig()` followed by `plt.close()` will capture the image. After execution, you will be able to SEE the generated plot and describe its visual contents."
    } else {
        "- **Matplotlib**: Figures are automatically captured as PNG images and saved to the working directory. The user will be informed that images were generated. You will NOT be able to see the images yourself — describe them based on the code and data you used to generate them."
    };

    let pil_desc = if supports_vision {
        "- **PIL Images**: If the last expression is a PIL Image, it is captured as a PNG image and sent to you as a visible image."
    } else {
        "- **PIL Images**: If the last expression is a PIL Image, it is captured as a PNG and saved to the working directory."
    };

    let images_output_desc = if supports_vision {
        "- `images_generated`: Number of matplotlib/PIL images captured (if any). These images are automatically provided to you for visual inspection."
    } else {
        "- `images_generated`: Number of matplotlib/PIL images captured (if any). These are saved to the working directory but you cannot see them directly."
    };

    let description = format!(
        r#"Execute Python code in a persistent interactive session.

## Session
- This is a **persistent session**: variables, imports, and state are preserved across calls within this conversation.
- Your code runs inside a unique persistent working directory. You can use `os.getcwd()` or `os.path.abspath(filename)` in your code to get the full path. The path is also returned in the `working_directory` field of each execution result. Files saved there remain accessible after the conversation ends.
- When the user asks where a file is saved, ALWAYS give them the FULL absolute path.
- To reset the session (clear all variables and imports), use the `{reset}` tool.

## Capabilities
- **Last-expression capture**: If the final statement is an expression (not an assignment), its repr is returned as the result (like Jupyter/IPython). The last result is also stored in the `_` variable.
{matplotlib}
{pil}
- **Pandas DataFrames**: If the last expression is a DataFrame or Series, its formatted repr is returned.
- **File I/O**: You can read and write files in the working directory.

## Timeout
- Code execution has a **{timeout}s timeout**. If exceeded, the execution is interrupted (KeyboardInterrupt). If the process is unresponsive, it is forcibly killed.
- Break long computations into smaller steps to avoid timeouts.

## Important Notes
- The user CANNOT see stdout/stderr from your code. You must relay any important information (file paths, results, errors) in your response text.
- The code must be pure Python. Do NOT use LaTeX syntax like `\le`, `\ge`, `\ne`, `\times` — use Python operators `<=`, `>=`, `!=`, `*` instead.

## Restrictions
- Package installation (pip install) is **disabled**.
- Network access and filesystem access **are available** and cannot be restricted.
- `input()` and reading from `stdin` are **not supported** and will raise an error.

## Installed Packages
The following packages are available:
```
{packages}
```

## Output Format
The result is a JSON object with these fields:
- `status`: "success" or "error"
- `working_directory`: The absolute path to the session's working directory. ALL files are saved here. When reporting file paths to the user, ALWAYS use this value combined with the filename.
- `stdout`: Captured standard output (if any)
- `stderr`: Captured standard error (if any)
- `result`: The repr of the last expression value (if any)
- `result_type`: The type name of the last expression (if any)
- `exception`: Full traceback (if an error occurred)
- `execution_time_ms`: How long execution took in milliseconds
{images_output}"#,
        reset = RESET_SESSION_TOOL_NAME,
        timeout = timeout_secs,
        packages = installed_packages.trim(),
        matplotlib = matplotlib_desc,
        pil = pil_desc,
        images_output = images_output_desc,
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
