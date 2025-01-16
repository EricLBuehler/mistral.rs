use anyhow::Result;
use mistralrs::{
    AutoDeviceMapParams, DeviceMapSetting, IsqType, TextMessageRole, VisionLoaderType,
    VisionMessages, VisionModelBuilder,
};

// const MODEL_ID: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct";
const MODEL_ID: &str = "lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k";

#[tokio::main]
async fn main() -> Result<()> {
    let auto_map_params = AutoDeviceMapParams::Text {
        max_seq_len: 4096,
        max_batch_size: 2,
    };
    let model = VisionModelBuilder::new(MODEL_ID, VisionLoaderType::VLlama)
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_device_mapping(DeviceMapSetting::Auto(auto_map_params))
        .build()
        .await?;

    let bytes = match reqwest::blocking::get(
        "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
        image,
        &model,
    )?;

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
