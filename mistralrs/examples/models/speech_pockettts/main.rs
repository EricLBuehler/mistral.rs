//! CPU-fast text-to-speech with pocket-tts (Kyutai Mimi codec + FlowLM).
//!
//! Run with: `cargo run --release --example speech_pockettts -p mistralrs`

use std::time::Instant;

use anyhow::Result;
use mistralrs::{speech_utils, SpeechLoaderType, SpeechModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = SpeechModelBuilder::new(
        "kyutai/pocket-tts-without-voice-cloning",
        SpeechLoaderType::PocketTts,
    )
    .with_logging()
    .build()
    .await?;

    let start = Instant::now();

    let text_to_speak =
        "Pocket TTS runs on the CPU in seconds, so mistral rs can serve speech without a GPU.";

    let (pcm, rate, channels) = model.generate_speech(text_to_speak).await?;

    let finished = Instant::now();

    let mut output = std::fs::File::create("out.wav").unwrap();
    speech_utils::write_pcm_as_wav(&mut output, &pcm, rate as u32, channels as u16).unwrap();

    println!(
        "Done! Took {} s. Audio saved at `out.wav`.",
        finished.duration_since(start).as_secs_f32(),
    );

    Ok(())
}
