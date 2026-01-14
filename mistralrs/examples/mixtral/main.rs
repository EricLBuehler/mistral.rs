use anyhow::Result;
use mistralrs::{AutoDeviceMapParams, DeviceMapSetting, IsqType, Response, TextMessageRole, TextMessages, TextModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let auto_map_params = AutoDeviceMapParams::Text {
        max_seq_len: 128,
        max_batch_size: 2,
    };
    let model = TextModelBuilder::new("mistralai/Mixtral-8x7B-Instruct-v0.1")
        // .with_isq(IsqType::Q4K)
        .with_logging()
 .with_chat_template("chat_templates/mistral.json")
        .with_throughput_logging()
        .with_device_mapping(DeviceMapSetting::Auto(auto_map_params))
        .build()
        .await?;

    let mut messages = TextMessages::new().enable_thinking(false);
    messages = messages.add_message(TextMessageRole::User, "Hello!");



    let mut rx = model.stream_chat_request(messages.clone()).await?;
    while let Some(r) = rx.next().await {
        match r {
            Response::Chunk(c) => {
                println!("Chunk {:?}", &c.choices.iter().map(|choice| choice.delta.content.clone().unwrap_or("".into())).collect::<Vec<_>>());
            }
            Response::CompletionChunk(c) =>{
                println!("CompletionChunk {:?}", &c.choices.iter().map(|choice| choice.text.clone()).collect::<Vec<_>>());
            }
            Response::CompletionDone(c) => {
                println!("Completion done: {:?}", &c.choices.iter().map(|choice| choice.text.clone()).collect::<Vec<_>>());
            }
            Response::Done(c) => {
                println!("Done: {:?}", &c.choices.iter().map(|choice| choice.message.content.clone().unwrap_or("".into())).collect::<Vec<_>>());
            }
            Response::InternalError(e) => {
                eprintln!("Internal error: {}", e);
            }
            Response::CompletionModelError(e, r) => {
                eprintln!("Completion model error: {}, {:?}", e, r);
            },
            Response::ValidationError(e) => {
                eprintln!("Validation error: {}", e);
            },
            
            _ => {
                continue;
            }
        }
    }

    let response = model.send_chat_request(messages.clone()).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
