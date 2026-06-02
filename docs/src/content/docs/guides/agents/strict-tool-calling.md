---
title: Strict tool calling
description: Constrain tool-call arguments to the tool's JSON Schema across HTTP, Python, Rust, and built-in agentic tools.
sidebar:
  order: 2
---

Strict tool calling constrains the arguments the model can generate for a tool. When a tool definition has `function.strict: true`, mistral.rs uses the tool's `parameters` JSON Schema during decoding instead of only parsing tool-call JSON after generation.

Use strict mode when malformed or extra arguments would be expensive, unsafe, or annoying to handle in application code. It works with both client-side OpenAI tool calling and the mistral.rs server-side tool loop.

Strict mode does not force the model to call a tool. Use `tool_choice` for that. It only constrains the argument object if the model calls a strict tool.

## HTTP

Add `strict: true` inside the OpenAI-compatible function definition:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "strict": true,
        "parameters": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "City name."
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

For a tight schema, include `required`, `enum`, nested object schemas, array item schemas, and `additionalProperties: false` where appropriate. If `strict` is omitted or `false`, mistral.rs still parses tool calls but does not constrain arguments to the tool's schema.

## Built-in agentic tools

Built-in agentic tools use strict schemas by default:

- Web search and page extraction.
- Python code execution.
- `read_file` and `list_files` helper tools for produced files.
- MCP tools, when the MCP server provides an input schema.

That means requests using `web_search_options`, `enable_code_execution`, declared `files`, or MCP tools get constrained tool arguments without adding `strict` yourself.

## Python

The Python SDK accepts OpenAI-compatible tool schemas as JSON strings on `ChatCompletionRequest.tool_schemas`. Put `strict: true` inside the function definition:

```python
import json
from mistralrs import ChatCompletionRequest, Runner, Which

tool_schema = json.dumps({
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
})

runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B"), in_situ_quant="4")
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Weather in Tokyo in celsius?"}],
        tool_schemas=[tool_schema],
    )
)
```

If you want mistral.rs to execute the tool server-side in Python, register a matching `tool_callbacks` entry on `Runner`; otherwise inspect `response.choices[0].message.tool_calls` and run the tool in your own code.

## Rust

In Rust, set `strict: Some(true)` on the tool's `Function`:

```rust
use std::collections::HashMap;

use anyhow::Result;
use mistralrs::{
    Function, IsqBits, ModelBuilder, RequestBuilder, TextMessageRole, Tool, ToolChoice, ToolType,
};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"],
        "additionalProperties": false
    }))?;

    let tools = vec![Tool {
        tp: ToolType::Function,
        function: Function {
            name: "get_weather".to_string(),
            description: Some("Get the current weather in a city.".to_string()),
            parameters: Some(parameters),
            strict: Some(true),
        },
    }];

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, "Weather in Tokyo in celsius?")
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    let response = model.send_chat_request(request).await?;
    println!("{:?}", response.choices[0].message.tool_calls);
    Ok(())
}
```

For server-side Rust callbacks, register the callback together with the strict `Tool` definition using `with_tool_callback_and_tool`.

## Notes

Strict tool calling is separate from `response_format: {"type": "json_schema", ...}`. Tool strictness constrains the arguments for a tool call; response-format schemas constrain the assistant's final text response.

Strict mode has an effect when the tool has a `parameters` schema. If a tool is marked strict without a schema, mistral.rs falls back to a generic object schema.
