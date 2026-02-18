/// Automatic device mapping for text and vision models.
///
/// Device mapping distributes model layers across available GPUs automatically.
/// This example shows both text and vision model usage with auto device mapping.
///
/// Run with: `cargo run --release --example auto_device_map -p mistralrs`
use anyhow::Result;
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, IsqBits, TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let auto_map_params = AutoDeviceMapParams::Text {
        max_seq_len: 4096,
        max_batch_size: 2,
    };

    let model = TextModelBuilder::new("meta-llama/Llama-3.3-70B-Instruct")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_device_mapping(DeviceMapSetting::Auto(auto_map_params))
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    // For vision models, use VisionModelBuilder with the same DeviceMapSetting:
    //
    // use mistralrs::{VisionModelBuilder, VisionMessages};
    //
    // let model = VisionModelBuilder::new("lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k")
    //     .with_auto_isq(IsqBits::Four)
    //     .with_device_mapping(DeviceMapSetting::Auto(AutoDeviceMapParams::Text {
    //         max_seq_len: 4096,
    //         max_batch_size: 2,
    //     }))
    //     .build()
    //     .await?;

    Ok(())
}
