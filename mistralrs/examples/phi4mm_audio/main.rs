use anyhow::Result;
use mistralrs::{AudioInput, IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("microsoft/Phi-4-multimodal-instruct")
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .build()
        .await?;

    let audio_bytes =
        reqwest::get("https://upload.wikimedia.org/wikipedia/commons/4/42/Bird_singing.ogg")
            .await?
            .bytes()
            .await?
            .to_vec();
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let image_bytes =
        reqwest::get("https://www.allaboutbirds.org/guide/assets/og/528129121-1200px.jpg")
            .await?
            .bytes()
            .await?
            .to_vec();
    let image = image::load_from_memory(&image_bytes)?;

    let messages = VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Describe in detail what is happening.",
        vec![image],
        vec![audio],
        &model,
    )?;

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
