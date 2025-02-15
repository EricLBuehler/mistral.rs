use either::Either;
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::mpsc::channel;

use mistralrs::{
    DefaultSchedulerMethod, Device, DeviceMapMetadata, Function, MistralRs, MistralRsBuilder,
    ModelDType, NormalLoaderBuilder, NormalRequest, NormalSpecificConfig, Request, RequestMessage,
    ResponseOk, Result, SamplingParams, SchedulerConfig, TokenSource, Tool, ToolChoice, ToolType,
};

/// Gets the best device, cpu, cuda if compiled with CUDA
pub(crate) fn best_device() -> Result<Device> {
    #[cfg(not(feature = "metal"))]
    {
        Device::cuda_if_available(0)
    }
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0)
    }
}

fn setup() -> anyhow::Result<Arc<MistralRs>> {
    // Select a Mistral model
    let loader = NormalLoaderBuilder::new(
        NormalSpecificConfig {
            use_flash_attn: false,
            prompt_chunksize: None,
            topology: None,
            organization: Default::default(),
            write_uqff: None,
            from_uqff: None,
        },
        None,
        None,
        Some("meta-llama/Meta-Llama-3.1-8B-Instruct".to_string()),
    )
    .build(None)?;
    // Load, into a Pipeline
    let pipeline = loader.load_model_from_hf(
        None,
        TokenSource::CacheToken,
        &ModelDType::Auto,
        &best_device()?,
        false,
        DeviceMapMetadata::dummy(),
        None,
        None, // No PagedAttention.
    )?;
    // Create the MistralRs, which is a runner
    Ok(MistralRsBuilder::new(
        pipeline,
        SchedulerConfig::DefaultScheduler {
            method: DefaultSchedulerMethod::Fixed(5.try_into().unwrap()),
        },
    )
    .build())
}

#[derive(serde::Deserialize, Debug, Clone)]
struct GetWeatherInput {
    place: String,
}

fn get_weather(input: GetWeatherInput) -> String {
    format!("In {} the weather is great!", input.place)
}

fn main() -> anyhow::Result<()> {
    let mistralrs = setup()?;

    let parameters = json!({
        "type": "object",
        "properties": {
            "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    });
    let parameters: HashMap<String, Value> = serde_json::from_value(parameters).unwrap();

    let mut messages = vec![IndexMap::from([
        ("role".to_string(), Either::Left("user".to_string())),
        (
            "content".to_string(),
            Either::Left("What is the weather in Boston?".to_string()),
        ),
    ])];

    let tools = vec![Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Get the weather for a certain city.".to_string()),
            name: "get_weather".to_string(),
            parameters: Some(parameters),
        },
    }];

    let (tx, mut rx) = channel(10_000);
    let request = Request::Normal(NormalRequest::new_simple(
        RequestMessage::Chat(messages.clone()),
        SamplingParams::default(),
        tx.clone(),
        0,
        Some(tools.clone()),
        Some(ToolChoice::Auto),
    ));
    mistralrs.get_sender()?.blocking_send(request)?;

    let response = rx.blocking_recv().unwrap();
    let ResponseOk::Done(result) = response.as_result().unwrap() else {
        unreachable!()
    };
    let message = &result.choices[0].message;

    if !message.tool_calls.is_empty() {
        let called = &message.tool_calls[0];
        if called.function.name == "get_weather" {
            let input: GetWeatherInput = serde_json::from_str(&called.function.arguments)?;
            let result = get_weather(input);
            // Add tool call message from assistant so it knows what it called
            messages.push(IndexMap::from([
                ("role".to_string(), Either::Left("assistant".to_string())),
                ("content".to_string(), Either::Left("".to_string())),
                (
                    "tool_calls".to_string(),
                    Either::Right(Vec![IndexMap::from([
                        ("id".to_string(), tool_call.id),
                        (
                            "function".to_string(),
                            json!({
                                "name": called.function.name,
                                "arguments": called.function.arguments,
                            })
                        ),
                        ("type".to_string(), "function".to_string()),
                    ])]),
                ),
            ]));
            // Add message from the tool
            messages.push(IndexMap::from([
                ("role".to_string(), Either::Left("tool".to_string())),
                ("tool_call_id".to_string(), Either::Left(tool_call.id)),
                ("content".to_string(), Either::Left(result)),
            ]));

            let request = Request::Normal(NormalRequest::new_simple(
                RequestMessage::Chat(messages.clone()),
                SamplingParams::default(),
                tx,
                0,
                Some(tools.clone()),
                Some(ToolChoice::Auto),
            ));
            mistralrs.get_sender()?.blocking_send(request)?;

            let response = rx.blocking_recv().unwrap();
            let ResponseOk::Done(result) = response.as_result().unwrap() else {
                unreachable!()
            };
            let message = &result.choices[0].message;
            println!("Output of model: {:?}", message.content);
        }
    }
    Ok(())
}
