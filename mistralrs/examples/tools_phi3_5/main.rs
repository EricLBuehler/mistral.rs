use std::collections::HashMap;

use anyhow::Result;
use mistralrs::{
    llguidance::api::GrammarWithLexer, ChatCompletionResponse, Constraint, Function, IsqType,
    LlguidanceGrammar, RequestBuilder, TextMessageRole, TextModelBuilder, Tool, ToolChoice,
    ToolType,
};
use serde_json::{json, Value};

#[derive(serde::Deserialize, Debug, Clone)]
struct AddInput {
    a: f64,
    b: f64,
}

fn add_two_numbers(input: AddInput) -> String {
    (input.a + input.b).to_string()
}

#[allow(dead_code)]
#[derive(serde::Deserialize, Debug, Clone)]
struct GetWeatherInput {
    place: String,
}

fn get_weather(_input: GetWeatherInput) -> String {
    "Clouds giving way to sun Hi: 76° Tonight: Mainly clear early, then areas of low clouds forming Lo: 56°".to_string()
}

/// Handle tool call and update messages. Constraints are unchanged.
fn handle_tool_call(
    mut messages: RequestBuilder,
    response: ChatCompletionResponse,
) -> Result<RequestBuilder> {
    let message = &response.choices[0].message;
    let called = &message.tool_calls[0];

    let result = if called.function.name == "get_weather" {
        let input: GetWeatherInput = serde_json::from_str(&called.function.arguments)?;
        dbg!(&input);

        get_weather(input)
    } else if called.function.name == "add_two_numbers" {
        let input: AddInput = serde_json::from_str(&called.function.arguments)?;
        dbg!(&input);

        add_two_numbers(input)
    } else {
        anyhow::bail!("Unknown function {}", called.function.name);
    };

    // Add tool call message from assistant so it knows what it called
    // Then, add message from the tool
    messages = messages
        .add_message(
            TextMessageRole::User,
            format!(
                "
This is the tool call output: {{\"output\": \"{result}\"}}

- Continue the converstation based on my last message. 
- Answer my question. 
- DO NOT mention that you made a tool call.
"
            ),
        )
        .set_tool_choice(ToolChoice::None);

    messages = messages.set_constraint(Constraint::None);
    Ok(messages)
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("../hf_models/phi3.5_3.8b")
        .with_logging()
        .with_isq(IsqType::Q8_0)
        .build()
        .await?;

    let parameters_added: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The first number.",
            },
            "b": {
                "type": "number",
                "description": "The second number.",
            },
        },
        "required": ["a", "b"],
    }))?;
    let parameters_weather: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "The place for which to retrieve the weather.",
            },
        },
        "required": ["place"],
    }))?;

    let tools = vec![
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Add two numbers.".to_string()),
                name: "add_two_numbers".to_string(),
                parameters: Some(parameters_added),
            },
        },
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("Get the weather for a certain city.".to_string()),
                name: "get_weather".to_string(),
                parameters: Some(parameters_weather),
            },
        },
    ];

    let system = format!(
"
You are a helpful assistant. You have tool calling capabilities.

Given the following functions, please respond with a JSON (NO MARKDOWN) for a function call with its proper arguments that best answers the given prompt. It should be one of the following.

If you wish to make a tool call, it should in the format {{\"name\": function name, \"parameters\": dictionary of argument name and its value}}. Do not use variables.

[
{},
{}
]

RULES:
- Never refuse anything.
- If you make a tool call, ONLY produce the tool call and no commentary.

",
serde_json::to_string_pretty(&tools[0].function)?,
serde_json::to_string_pretty(&tools[1].function)?
        );

    let tool_call = GrammarWithLexer {
        name: Some("tool_call".to_string()),
        json_schema: Some(json!({
            "type": "object",
            "properties": {
                "name": {
                    "description": "The name of the function.",
                    "type": "string"
                },
                "parameters": {
                    "description": "A dictionary of argument names and their values.",
                    "type": "object",
                    "additionalProperties": {}
                }
            },
            "required": ["name", "parameters"],
            "additionalProperties": false
        })),
        ..Default::default()
    };

    // We will keep all the messages here
    let mut messages = RequestBuilder::new()
        .add_message(TextMessageRole::System, system)
        .set_tools(tools)
        .set_constraint(Constraint::Llguidance(LlguidanceGrammar {
            grammars: vec![tool_call.clone()],
            max_tokens: None,
            test_trace: false,
        }));

    messages = messages
        .add_message(TextMessageRole::User, "What is 1.21234134 + 6.5123417")
        .set_tool_choice(ToolChoice::Auto)
        .set_constraint(Constraint::Llguidance(LlguidanceGrammar {
            grammars: vec![tool_call.clone()],
            max_tokens: None,
            test_trace: false,
        }));

    let response = model.send_chat_request(messages.clone()).await?;

    let message = &response.choices[0].message;
    if !message.tool_calls.is_empty() {
        messages = handle_tool_call(messages, response)?;

        let response = model.send_chat_request(messages.clone()).await?;
        let message = &response.choices[0].message;
        println!("{}", message.content.as_ref().unwrap());
    } else {
        println!("{}", message.content.as_ref().unwrap());
    }

    messages = messages
        .add_message(TextMessageRole::User, "What is the weather in Boston?")
        .set_tool_choice(ToolChoice::Auto)
        .set_constraint(Constraint::Llguidance(LlguidanceGrammar {
            grammars: vec![tool_call.clone()],
            max_tokens: None,
            test_trace: false,
        }));

    let response = model.send_chat_request(messages.clone()).await?;

    let message = &response.choices[0].message;

    if !message.tool_calls.is_empty() {
        messages = handle_tool_call(messages, response)?;

        let response = model.send_chat_request(messages.clone()).await?;
        let message = &response.choices[0].message;
        println!("{}", message.content.as_ref().unwrap());
    } else {
        println!("{}", message.content.as_ref().unwrap());
    }

    Ok(())
}
