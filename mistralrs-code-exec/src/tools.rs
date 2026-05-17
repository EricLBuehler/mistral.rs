use mistralrs_mcp::{Function, Tool, ToolType};
use mistralrs_sandbox::{EffectiveProtection, NetworkMode};
use serde_json::json;
use std::collections::HashMap;

use crate::InputModality;

pub const EXECUTE_PYTHON_TOOL_NAME: &str = "mistralrs_execute_python";
pub const RESET_SESSION_TOOL_NAME: &str = "mistralrs_reset_python_session";
pub const READ_FILE_TOOL_NAME: &str = "read_file";
pub const LIST_FILES_TOOL_NAME: &str = "list_files";

fn sandbox_network_note(network_isolated: bool, network: Option<NetworkMode>) -> &'static str {
    if !network_isolated {
        return "- Network access is **unrestricted**.";
    }
    match network {
        Some(NetworkMode::None) => {
            "- Network access is **denied**. `socket()` returns EPERM; do not attempt outbound HTTP, DNS, or any network call."
        }
        Some(NetworkMode::Loopback) => {
            "- Network access is **restricted to loopback only** (127.0.0.1, ::1). External hosts are unreachable; outbound DNS and HTTP to non-local addresses will fail."
        }
        Some(NetworkMode::Full) | None => "- Network access is unrestricted.",
    }
}

fn sandbox_fs_note(fs_isolated: bool) -> &'static str {
    if fs_isolated {
        "- Filesystem access is **restricted**: you may read system libraries and write only inside the session working directory. Attempts to read user files (e.g., `~/.ssh`, `.env`) return EACCES."
    } else {
        "- Filesystem access is **unrestricted**."
    }
}

pub fn code_exec_tool_called(name: &str) -> bool {
    name == EXECUTE_PYTHON_TOOL_NAME
        || name == RESET_SESSION_TOOL_NAME
        || name == READ_FILE_TOOL_NAME
        || name == LIST_FILES_TOOL_NAME
}

pub fn build_execute_python_tool(
    timeout_secs: u64,
    installed_packages: &str,
    input_modalities: &[InputModality],
    effective: EffectiveProtection,
    network: Option<NetworkMode>,
) -> Tool {
    let supports_vision = input_modalities.contains(&InputModality::Vision);
    let supports_video = input_modalities.contains(&InputModality::Video);

    let matplotlib_desc = if supports_vision {
        "- **Matplotlib**: Figures are automatically captured as PNG images and sent to you as visible images that you can see and describe. You do NOT need to do anything special; figures are captured when you call `plt.savefig()`, `plt.show()`, or simply leave them open. Even `plt.savefig()` followed by `plt.close()` will capture the image. After execution, you will be able to SEE the generated plot and describe its visual contents."
    } else {
        "- **Matplotlib**: Figures are automatically captured as PNG images and saved to the working directory. The user will be informed that images were generated. You will NOT be able to see the images yourself; describe them based on the code and data you used to generate them."
    };

    let pil_desc = if supports_vision {
        "- **PIL Images**: If the last expression is a PIL Image, it is captured as a PNG image and sent to you as a visible image."
    } else {
        "- **PIL Images**: If the last expression is a PIL Image, it is captured as a PNG and saved to the working directory."
    };

    let video_desc = if supports_video {
        "- **Video/Animation/GIF**: ALWAYS use `matplotlib.animation.FuncAnimation` with `ani.save(filename, writer='pillow', fps=N)`. Frames are automatically captured as video. Pillow is always available; do NOT preflight-check imports. Example:\n  ```python\n  fig, ax = plt.subplots()\n  def update(i):\n      ax.clear()\n      # ... draw frame i ...\n  ani = animation.FuncAnimation(fig, update, frames=N, interval=100)\n  ani.save('output.gif', writer='pillow', fps=10)\n  ```\n  The video frames are sent back to you so you can see and describe them. One video per execution."
    } else if supports_vision {
        "- **Video/Animation/GIF**: ALWAYS use `matplotlib.animation.FuncAnimation` with `ani.save(filename, writer='pillow', fps=N)`. Frames are automatically captured. Pillow is always available; do NOT preflight-check imports. Example:\n  ```python\n  fig, ax = plt.subplots()\n  def update(i):\n      ax.clear()\n      # ... draw frame i ...\n  ani = animation.FuncAnimation(fig, update, frames=N, interval=100)\n  ani.save('output.gif', writer='pillow', fps=10)\n  ```\n  Frames are sent back as individual images. One video per execution."
    } else {
        "- **Video/Animation/GIF**: ALWAYS use `matplotlib.animation.FuncAnimation` with `ani.save(filename, writer='pillow', fps=N)`. Pillow is always available; do NOT preflight-check imports. Frames are saved to the working directory. One video per execution."
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
{video}
- **Pandas DataFrames**: If the last expression is a DataFrame or Series, its formatted repr is returned.
- **File I/O**: You can read and write files in the working directory.

## Timeout
- Code execution has a **{timeout}s timeout**. If exceeded, the execution is interrupted (KeyboardInterrupt). If the process is unresponsive, it is forcibly killed.
- Break long computations into smaller steps to avoid timeouts.

## Important Notes
- The user CANNOT see stdout/stderr from your code. You must relay any important information (file paths, results, errors) in your response text.
- The code must be pure Python. Do NOT use LaTeX syntax: write `<=`, `>=`, `!=`, `*` instead of `\le`, `\ge`, `\ne`, `\times`/`\cdot`; write `**0.5` / `**N` or `math.sqrt(...)` / `math.pow(...)` instead of `\sqrt`, `x^N`; write `(a)/(b)` instead of `\frac{{a}}{{b}}`; write `math.pi`, `math.inf` instead of `\pi`, `\infty`. If you do emit LaTeX, you'll get a `SyntaxError` back; rewrite in Python and retry.

## Restrictions
- Package installation (pip install) is **disabled**.
{network_note}
{fs_note}
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
- `files`: Files surfaced to the user (see Files below)
{images_output}

## Files
The `outputs` parameter is how you tell the runtime which files written by your code should be surfaced to the user as typed File objects (with stable ids the user can fetch). You may write any number of files to the working directory; only those listed in `outputs` are surfaced.

If the runtime asked for specific output files (you'll see those listed in a system message), you MUST list them in `outputs`. Files written but not listed remain in the working directory and are accessible to you across calls in this session via normal Python file I/O, but the user does NOT see them.

After execution, the result's `files` array has one entry per surfaced file with `id`, `name`, `format`, `bytes`. Text files of 1024 bytes or fewer include the full `text`. Larger text files include a `preview` plus `truncated: true`; call `{read_file}(file_id)` to read the rest. Binary files (images, videos, archives) include only metadata; reference them by id when discussing them with the user. Use `{list_files}()` to enumerate files produced earlier in this session if you don't remember an id."#,
        reset = RESET_SESSION_TOOL_NAME,
        read_file = READ_FILE_TOOL_NAME,
        list_files = LIST_FILES_TOOL_NAME,
        timeout = timeout_secs,
        packages = installed_packages.trim(),
        matplotlib = matplotlib_desc,
        pil = pil_desc,
        video = video_desc,
        images_output = images_output_desc,
        network_note = sandbox_network_note(effective.network_isolated, network),
        fs_note = sandbox_fs_note(effective.fs_isolated),
    );

    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute. Multiple lines and statements are supported."
            },
            "outputs": {
                "type": "array",
                "description": "Filenames (relative to the working directory) to surface to the user as File objects in the response. Anything you write to the working directory but don't list here stays internal; it remains accessible to you across calls in this session via normal Python file I/O, but not exposed to the user. Always include any files the runtime asked you to produce.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Filename relative to the working directory."},
                        "format": {"type": "string", "description": "Optional format hint (e.g. csv, png, parquet). Inferred from the filename extension when omitted."}
                    },
                    "required": ["name"],
                    "additionalProperties": false
                }
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

pub fn build_read_file_tool() -> Tool {
    let parameters: HashMap<String, serde_json::Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "File id from a prior tool result (e.g. file_abc_r0_0)."
            },
            "start": {
                "type": "integer",
                "minimum": 0,
                "description": "Inclusive start character offset. Defaults to 0."
            },
            "end": {
                "type": "integer",
                "minimum": 0,
                "description": "Exclusive end character offset. Defaults to end of file."
            }
        },
        "required": ["file_id"],
        "additionalProperties": false
    }))
    .unwrap();

    Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(
                "Read a slice of a text file produced earlier in this session. Use this when a \
                 prior tool result included a file with `truncated: true` (text larger than the \
                 inline limit); call this to read the rest. Returns the requested character \
                 range as text, capped at 65536 characters per call (paginate via start/end if \
                 the file is larger). Binary files (images, videos, archives) cannot be read \
                 with this tool; refer to them by id when discussing with the user."
                    .to_string(),
            ),
            name: READ_FILE_TOOL_NAME.to_string(),
            parameters: Some(parameters),
            strict: Some(true),
        },
    }
}

pub fn build_list_files_tool() -> Tool {
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
                "List all files produced so far in this session. Useful when you need to \
                 reference a file produced in an earlier turn but don't remember its id. Returns \
                 each file's id, name, format, size, and the round it was produced in. Files are \
                 ordered oldest first."
                    .to_string(),
            ),
            name: LIST_FILES_TOOL_NAME.to_string(),
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
