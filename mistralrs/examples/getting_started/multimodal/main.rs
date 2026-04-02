/// Simple multimodal model "hello world".
///
/// For a comprehensive example with all supported multimodal model IDs,
/// see `examples/models/multimodal_models/`.
///
/// Run with: `cargo run --release --example multimodal_basic -p mistralrs`
use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, MultimodalMessages, TextMessageRole};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
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

    let messages = MultimodalMessages::new().add_image_message(
        TextMessageRole::User,
        "What is this flower?",
        vec![image],
    );

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
