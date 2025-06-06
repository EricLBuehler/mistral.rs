use std::{
    collections::HashMap,
    io::{stdin, stdout, Write},
    process::Command,
};
use std::fs;
use std::process::{Stdio};

use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, Delta, Function, IsqType, MemoryGpuConfig,
    PagedAttentionMetaBuilder, RequestBuilder, Response, TextMessageRole, TextModelBuilder, Tool,
    ToolChoice, ToolType,
};
use std::env::consts::{ARCH, FAMILY, OS};

use serde_json::{json, Value};

#[derive(serde::Deserialize, Debug, Clone)]
struct ShellInput {
    command: String,
    // working_directory: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ReadFileInput {
    path: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct WriteFileInput {
    path: String,
    contents: String,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct GitApplyInput {
    patch: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut model_builder = TextModelBuilder::new("../hf_models/qwen3_4b")
        .with_logging()
        .with_isq(IsqType::AFQ4);
    if cfg!(feature = "cuda") {
        model_builder = model_builder.with_paged_attn(|| {
            PagedAttentionMetaBuilder::default()
                .with_gpu_memory(MemoryGpuConfig::ContextSize(16384))
                .build()
        })?;
    }
    let model = model_builder.build().await?;

    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command and arguments for the command.",
            },
            // "working_directory": {
            //     "type": "string",
            //     "description": "The absolute directly to change to, where this command will be run.",
            // },
        },
        "required": ["command"],
    }))?;

    let read_file_params: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "path": { "type": "string", "description": "Path of the file to read." }
        },
        "required": ["path"]
    }))?;

    let write_file_params: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "path":    { "type": "string", "description": "Path of the file to write." },
            "contents":{ "type": "string", "description": "Contents to write to the file." }
        },
        "required": ["path", "contents"]
    }))?;

    let git_diff_params: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {},
        "required": []
    }))?;

    let git_apply_params: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "patch": { "type": "string", "description": "Unified diff to apply." }
        },
        "required": ["patch"]
    }))?;

    let tools = vec![
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Run a shell command.".to_string()),
                name: "shell".to_string(),
                parameters: Some(parameters),
            },
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Read a file.".to_string()),
                name: "read_file".to_string(),
                parameters: Some(read_file_params),
            },
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Write a file.".to_string()),
                name: "write_file".to_string(),
                parameters: Some(write_file_params),
            },
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Get git diff.".to_string()),
                name: "git_diff".to_string(),
                parameters: Some(git_diff_params),
            },
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Apply a git patch.".to_string()),
                name: "git_apply".to_string(),
                parameters: Some(git_apply_params),
            },
        },
    ];

    let current_dir = std::env::current_dir()?;
    let system = format!("You are a coding agent.
Your are working in a computer with the following enviornment: arch ({ARCH}), family ({FAMILY}), os ({OS}).
The current working directory is: {}.

You should call tools repeatedly as appropriate to answer the user's query. If you get an error, think about why and try to run the command again.

To create a patch, you should create a .diff file and then apply it using `git apply`.
    ", current_dir.display());

    print!(">>> ");
    stdout().flush()?;
    let mut user_prompt = String::new();
    stdin().read_line(&mut user_prompt)?;
    let user_prompt = user_prompt.trim();

    // We will keep all the messages here
    let mut messages = RequestBuilder::new()
        .add_message(TextMessageRole::System, system)
        .add_message(TextMessageRole::User, user_prompt)
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    'outer: loop {
        let mut stream = model.stream_chat_request(messages.clone()).await?;

        let mut assistant = String::new();
        while let Some(chunk) = stream.next().await {
            if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
                match &choices.first().unwrap().delta {
                    Delta {
                        content: Some(content),
                        role: _,
                        tool_calls: None,
                    } => {
                        assistant.push_str(&content);
                        print!("{content}");
                        stdout().flush()?;
                    }
                    Delta {
                        content: None,
                        role: _,
                        tool_calls: Some(tool_calls),
                    } => {
                        let called = &tool_calls[0];

                        let output = match called.function.name.as_str() {
                            "shell" => {
                                let input: ShellInput = serde_json::from_str(&called.function.arguments)?;
                                let result = Command::new("sh")
                                    .arg("-c")
                                    .arg(&input.command)
                                    .current_dir(&current_dir)
                                    .output()?;
                                let stdout = String::from_utf8(result.stdout)?;
                                let stderr = String::from_utf8(result.stderr)?;
                                format!("STDOUT: {stdout}\n\nSTDERR: {stderr}")
                            }
                            "read_file" => {
                                let input: ReadFileInput = serde_json::from_str(&called.function.arguments)?;
                                let contents = fs::read_to_string(&input.path)
                                    .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", input.path, e))?;
                                format!("CONTENTS:\n{}", contents)
                            }
                            "write_file" => {
                                let input: WriteFileInput = serde_json::from_str(&called.function.arguments)?;
                                fs::write(&input.path, &input.contents)
                                    .map_err(|e| anyhow::anyhow!("Failed to write {}: {}", input.path, e))?;
                                format!("WROTE {} bytes to {}", input.contents.len(), input.path)
                            }
                            "git_diff" => {
                                let result = Command::new("git")
                                    .arg("diff")
                                    .current_dir(&current_dir)
                                    .output()?;
                                let stdout = String::from_utf8(result.stdout)?;
                                let stderr = String::from_utf8(result.stderr)?;
                                format!("STDOUT: {stdout}\n\nSTDERR: {stderr}")
                            }
                            "git_apply" => {
                                let input: GitApplyInput = serde_json::from_str(&called.function.arguments)?;
                                let mut child = Command::new("git")
                                    .arg("apply")
                                    .stdin(Stdio::piped())
                                    .current_dir(&current_dir)
                                    .spawn()?;
                                {
                                    let stdin = child.stdin.as_mut().expect("Failed to open stdin");
                                    stdin.write_all(input.patch.as_bytes())?;
                                }
                                let result = child.wait_with_output()?;
                                let stdout = String::from_utf8(result.stdout)?;
                                let stderr = String::from_utf8(result.stderr)?;
                                format!("STDOUT: {stdout}\n\nSTDERR: {stderr}")
                            }
                            other => anyhow::bail!("Unexpected function name: {}", other),
                        };
                        println!("{}", output);

                        messages = messages
                            .add_message_with_tool_call(
                                TextMessageRole::Assistant,
                                String::new(),
                                vec![called.clone()],
                            )
                            .add_tool_message(output, called.id.clone());
                        continue 'outer;
                    }
                    _ => anyhow::bail!("Got an unexpected delta."),
                }
            } else {
                anyhow::bail!("Encountered an unrecoverable error.");
            }
        }
        messages = messages.add_message(TextMessageRole::Assistant, assistant);

        println!("\n\n");

        print!(">>> ");
        stdout().flush()?;
        let mut user_prompt = String::new();
        stdin().read_line(&mut user_prompt)?;
        let user_prompt = user_prompt.trim();

        messages = messages.add_message(TextMessageRole::User, user_prompt);
    }
}
