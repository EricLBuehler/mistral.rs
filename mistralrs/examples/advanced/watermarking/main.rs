//! Select a token watermark, or inspect SemStamp embeddings, using a JSON configuration.

use anyhow::Result;
use either::Either;
use mistralrs::{
    Device, IsqBits, ModelBuilder, RequestBuilder, Tensor, TextMessageRole, Watermark,
    WatermarkConfig,
};

const MAX_TOKENS: usize = 512;

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/watermarking/synthid.json".into());
    let mut value: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    value["key"] = std::env::var("MISTRALRS_WATERMARK_KEY")?.into();
    let config: WatermarkConfig = serde_json::from_value(value)?;
    let detector = Watermark::new(&config)?;
    if config.scheme() == "semstamp" {
        let device = match std::env::var("MISTRALRS_WATERMARK_DEVICE").as_deref() {
            Ok("metal") => Device::new_metal(0)?,
            Ok("cuda") => Device::new_cuda(0)?,
            _ => Device::Cpu,
        };
        let embeddings = Tensor::new(
            &[[1.0f32, 0.3, 0.5], [-0.2, 0.9, 0.1], [0.4, -0.1, 1.0]],
            &device,
        )?;
        println!(
            "Synthetic embeddings only; supply a fixed sentence encoder in a real application."
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&detector.detect_embeddings_tensor(&embeddings, 1)?)?
        );
        return Ok(());
    }
    config.validate_generation()?;
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Eight)
        .build()
        .await?;
    let request = RequestBuilder::new()
        .add_message(
            TextMessageRole::User,
            "Write a long story about a lunar garden.",
        )
        .enable_thinking(false)
        .set_sampler_temperature(0.8)
        .set_sampler_topk(40)
        .set_sampler_max_len(MAX_TOKENS)
        .set_sampler_watermark(config);
    let response = model.send_chat_request(request).await?;
    let text = response.choices[0].message.content.as_deref().unwrap_or("");
    println!("{text}");
    let tokens = model
        .tokenize(Either::Right(text.to_owned()), None, false, false, None)
        .await?;
    let evidence = detector.detect(&tokens, 0, &[])?;
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    Ok(())
}
