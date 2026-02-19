//! Text-to-speech synthesis using a speech model.
//!
//! Run with: `cargo run --release --example speech -p mistralrs`

use std::time::Instant;

use anyhow::Result;
use mistralrs::{speech_utils, SpeechLoaderType, SpeechModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = SpeechModelBuilder::new("nari-labs/Dia-1.6B", SpeechLoaderType::Dia)
        .with_logging()
        .build()
        .await?;

    let start = Instant::now();

    // let text_to_speak = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face.";
    let text_to_speak = "[S1] mistral r s is a local LLM inference engine. [S2] You can run text and vision models, and also image generation and speech generation. [S1] There is agentic web search, tool calling, and a convenient Python API. [S2] Check it out on github.";

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
