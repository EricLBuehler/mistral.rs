//! Audio input processing with a multimodal model.
//!
//! Run with: `cargo run --release --example audio -p mistralrs`

use anyhow::Result;
use mistralrs::{AudioInput, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("../hf_models/gemma3n_e4b")
        .with_logging()
        .build()
        .await?;

    let audio_bytes = std::fs::read("sample_speech.wav")?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let messages = VisionMessages::new().add_audio_message(
        TextMessageRole::User,
        "What is being said?",
        vec![audio],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
