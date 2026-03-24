//! Tool calling (function calling) with manual tool definitions.
//!
//! Run with: `cargo run --release --example tools -p mistralrs`

use std::collections::HashMap;

use anyhow::Result;
use mistralrs::{
    Function, IsqBits, ModelBuilder, RequestBuilder, TextMessageRole, Tool, ToolChoice, ToolType,
};
use serde_json::{json, Value};

#[derive(serde::Deserialize, Debug, Clone)]
struct GetWeatherInput {
    place: String,
}

fn get_weather(input: GetWeatherInput) -> String {
    format!("Weather in {}: Temperature: 25C. Wind: calm. Dew point: 10C. Precipitiation: 5cm of rain expected.", input.place)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_logging()
        .with_auto_isq(IsqBits::Eight)
        .build()
        .await?;

    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "The place to get the weather for.",
            },
        },
        "required": ["place"],
    }))?;

    let tools = vec![Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Get the weather for a certain city.".to_string()),
            name: "get_weather".to_string(),
            parameters: Some(parameters),
        },
    }];

    // We will keep all the messages here
    let mut messages = RequestBuilder::new()
        .add_message(TextMessageRole::User, "What is the weather in Boston?")
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    let response = model.send_chat_request(messages.clone()).await?;

    let message = &response.choices[0].message;

    if let Some(tool_calls) = &message.tool_calls {
        let called = &tool_calls[0];
        if called.function.name == "get_weather" {
            let input: GetWeatherInput = serde_json::from_str(&called.function.arguments)?;
            println!("Called tool `get_weather` with arguments {input:?}");

            let result = get_weather(input);
            println!("Output of tool call: {result}");

            // Add tool call message from assistant so it knows what it called
            // Then, add message from the tool
            messages = messages
                .add_message_with_tool_call(
                    TextMessageRole::Assistant,
                    String::new(),
                    vec![called.clone()],
                )
                .add_tool_message(result, called.id.clone())
                .set_tool_choice(ToolChoice::None);

            let response = model.send_chat_request(messages.clone()).await?;

            let message = &response.choices[0].message;
            println!("Output of model: {:?}", message.content);
        }
    }

    Ok(())
}
