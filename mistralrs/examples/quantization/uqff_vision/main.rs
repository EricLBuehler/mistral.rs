//! Load a pre-quantized UQFF vision model.
//!
//! Run with: `cargo run --release --example uqff_vision -p mistralrs`

use anyhow::Result;
use mistralrs::{IsqBits, TextMessageRole, UqffVisionModelBuilder, VisionMessages};

#[tokio::main]
async fn main() -> Result<()> {
    let model = UqffVisionModelBuilder::new(
        "EricB/Phi-3.5-vision-instruct-UQFF",
        vec!["phi3.5-vision-instruct-q8_0.uqff".into()],
    )
    .into_inner()
    .with_auto_isq(IsqBits::Four)
    .with_logging()
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
        vec![image],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
