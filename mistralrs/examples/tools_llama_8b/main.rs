use std::collections::HashMap;

use anyhow::Result;
use mistralrs::{
    Function, IsqType, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole,
    TextModelBuilder, Tool, ToolChoice, ToolType,
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
    let model = TextModelBuilder::new("meta-llama/Meta-Llama-3.1-8B-Instruct")
        .with_logging()
        .with_isq(IsqType::Q4K)
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "The place for which to retrieve the weather.",
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
    let messages = RequestBuilder::new()
        .add_message(TextMessageRole::User, "What is the weather in Boston?")
        .set_tools(tools)
        .set_tool_choice(ToolChoice::Auto);

    let response = model.send_chat_request(messages.clone()).await?;

    let message = &response.choices[0].message;

    if !message.tool_calls.is_empty() {
        let called = &message.tool_calls[0];
        if called.function.name == "get_weather" {
            let input: GetWeatherInput = serde_json::from_str(&called.function.arguments)?;
            println!("Called tool `get_weather` with arguments {input:?}");

            let result = get_weather(input);
            println!("Output of tool call: {result}");

            // Llama 3.1 doesn't do well with tool calling in the chat loop. For this, check out the full `tool` example!
        }
    }

    Ok(())
}
