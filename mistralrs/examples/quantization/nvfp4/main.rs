//! Pretrained NVFP4 inference on Blackwell with CUDA 13.3 and cuTile.
//! Run: cargo run --release -p mistralrs --example nvfp4 --features cuda,cutile

use anyhow::Result;
use mistralrs::{ModelBuilder, ModelDType, RequestBuilder, TextMessageRole};

const MODEL_ID: &str = "nvidia/Qwen3-14B-NVFP4";
const MAX_TOKENS: usize = 128;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new(MODEL_ID)
        .with_dtype(ModelDType::BF16)
        .with_logging()
        .build()
        .await?;

    let request = RequestBuilder::new()
        .set_deterministic_sampler()
        .set_sampler_max_len(MAX_TOKENS)
        .enable_thinking(false)
        .add_message(TextMessageRole::User, "Explain why the sky is blue.");
    let response = model.send_chat_request(request).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
