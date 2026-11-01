//! Generate speech with LFM3-Audio or LFM3-Omni using the shared Rust speech API.
//!
//! Run with: `cargo run --release --example lfm3_audio -p mistralrs`

use anyhow::Result;
use mistralrs::{speech_utils, IsqType, ModelBuilder};

const DEFAULT_MODEL: &str = "LiquidAI/LFM3-Audio-2B";
const DEFAULT_TEXT: &str = "Hello. This speech was generated locally with mistral r s.";
const DEFAULT_OUTPUT: &str = "lfm3-speech.wav";

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_id = args.next().unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let text = args.next().unwrap_or_else(|| DEFAULT_TEXT.to_string());
    let output_path = args.next().unwrap_or_else(|| DEFAULT_OUTPUT.to_string());
    let model = ModelBuilder::new(model_id)
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .build()
        .await?;
    let (pcm, rate, channels) = model.generate_speech(text).await?;
    let mut output = std::fs::File::create(&output_path)?;
    speech_utils::write_pcm_as_wav(&mut output, &pcm, rate as u32, channels as u16)?;
    println!("Saved {} samples at {rate} Hz to {output_path}", pcm.len());
    Ok(())
}
