use std::collections::HashMap;

use anyhow::Result;
use mistralrs::{
    Function, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextModelBuilder, Tool,
    ToolChoice, ToolType,
};
use serde_json::{json, Value};

#[derive(serde::Deserialize, Debug, Clone)]
struct GetWeatherInput {
    place: String,
}

fn get_weather(input: GetWeatherInput) -> String {
    format!("In {} the weather is great!", input.place)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("meta-llama/Meta-Llama-3.1-8B-Instruct".to_string())
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
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

    if !message.tool_calls.is_empty() {
        let called = &message.tool_calls[0];
        if called.function.name == "get_weather" {
            let input: GetWeatherInput = serde_json::from_str(&called.function.arguments)?;
            let result = get_weather(input);
            // Add tool call message from assistant so it knows what it called
            messages = messages.add_message(
                TextMessageRole::Assistant,
                json!({
                    "name": called.function.name,
                    "parameters": called.function.arguments,
                })
                .to_string(),
            );
            // Add message from the tool
            messages = messages.add_message(TextMessageRole::Tool, result);

            let response = model.send_chat_request(messages.clone()).await?;

            let message = &response.choices[0].message;
            println!("Output of model: {:?}", message.content);
        }
    }

    Ok(())
}
