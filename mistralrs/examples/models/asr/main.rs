use anyhow::Result;
use mistralrs::{AudioInput, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("mistralai/Voxtral-Mini-4B-Realtime-2602")
        .with_logging()
        .build()
        .await?;

    let audio_bytes = std::fs::read("sample_audio.wav")?;
    let audio = AudioInput::from_bytes(&audio_bytes)?;

    let messages = VisionMessages::new().add_multimodal_message(
        TextMessageRole::User,
        "Transcribe this audio.",
        vec![],
        vec![audio],
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
