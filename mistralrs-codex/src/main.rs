use std::{
    collections::HashMap,
    io::{stdout, Write},
    process::Command,
};

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

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("/media/ericbuehler/2TB_SSD/hf_models/qwen3_4b")
        .with_logging()
        .with_isq(IsqType::Q4K)
        // .with_paged_attn(|| {
        //     PagedAttentionMetaBuilder::default()
        //         .with_gpu_memory(MemoryGpuConfig::ContextSize(16384))
        //         .build()
        // })?
        .build()
        .await?;

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

    let tools = vec![Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Run a shell command.".to_string()),
            name: "shell".to_string(),
            parameters: Some(parameters),
        },
    }];

    let current_dir = std::env::current_dir()?;
    let system = format!("You are a coding agent.
Your are working in a computer with the following enviornment: arch ({ARCH}), family ({FAMILY}), os ({OS}).
The current working directory is: {}.

You should call tools repeatedly as appropriate to answer the user's query. If you get an error, think about why and try to run the command again.

You should start by looking at the README file if it exists.
    ", current_dir.display());

    // We will keep all the messages here
    let mut messages = RequestBuilder::new()
        .add_message(TextMessageRole::System, system)
        .add_message(
            TextMessageRole::User,
            "Can you please give me a summary of this project?",
        )
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    loop {
        let mut stream = model.stream_chat_request(messages.clone()).await?;

        let mut finished_with_tool_call = false;
        while let Some(chunk) = stream.next().await {
            if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
                match &choices.first().unwrap().delta {
                    Delta {
                        content: Some(content),
                        role: _,
                        tool_calls: None,
                    } => {
                        print!("{content}");
                        stdout().flush()?;
                    }
                    Delta {
                        content: None,
                        role: _,
                        tool_calls: Some(tool_calls),
                    } => {
                        let called = &tool_calls[0];

                        if called.function.name != "shell" {
                            anyhow::bail!("Unexpected function name");
                        }

                        let input: ShellInput = serde_json::from_str(&called.function.arguments)?;

                        dbg!(&input);
                        let result = if input.command.contains(' ') {
                            let mut command = input.command.splitn(2, ' ');
                            Command::new(command.nth(0).unwrap())
                                .arg(command.nth(0).unwrap_or(""))
                                .output()?
                        } else {
                            Command::new(&input.command).output()?
                        };

                        let res_stdout = String::from_utf8(result.stdout)?;
                        let res_stderr = String::from_utf8(result.stderr)?;
                        let output = format!("STDOUT: {res_stdout}\n\nSTDERR: {res_stderr}");
                        println!("{output}");

                        messages = messages
                            .add_message_with_tool_call(
                                TextMessageRole::Assistant,
                                String::new(),
                                vec![called.clone()],
                            )
                            .add_tool_message(output, called.id.clone());

                        finished_with_tool_call = true;
                    }
                    _ => anyhow::bail!("Got an unexpected delta."),
                }
            } else {
                anyhow::bail!("Encountered an unrecoverable error.");
            }
        }

        println!("\n\n");
        if !finished_with_tool_call {
            break;
        }
    }

    Ok(())
}
