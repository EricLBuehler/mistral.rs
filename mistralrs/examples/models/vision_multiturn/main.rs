//! Multi-turn conversation with a vision model across turns.
//!
//! Run with: `cargo run --release --example vision_multiturn -p mistralrs`

use anyhow::Result;
use mistralrs::{ModelBuilder, RequestBuilder, TextMessageRole, VisionMessages};

const MODEL_ID: &str = "Qwen/Qwen3-VL-4B-Instruct";

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new(MODEL_ID)
        .with_logging()
        .with_isq(mistralrs::IsqType::Q8_0)
        .build()
        .await?;

    let mut messages = VisionMessages::new().add_message(TextMessageRole::User, "Hello!");

    let resp = model
        .send_chat_request(RequestBuilder::from(messages.clone()).set_sampler_max_len(100))
        .await?
        .choices[0]
        .message
        .content
        .clone()
        .unwrap();
    println!("\n\n{resp}");
    messages = messages.add_message(TextMessageRole::Assistant, resp);

    let bytes = match reqwest::blocking::get(
        // "https://s3.amazonaws.com/cdn.tulips.com/images/large/Timeless-Tulip.jpg",
        "https://niche-museums.imgix.net/pioneer-history.jpeg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    messages = messages.add_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
        vec![image],
    );
    let resp = model
        .send_chat_request(RequestBuilder::from(messages.clone()).set_sampler_max_len(100))
        .await?
        .choices[0]
        .message
        .content
        .clone()
        .unwrap();
    println!("\n\n{resp}");
    messages = messages.add_message(TextMessageRole::Assistant, resp);

    let bytes = match reqwest::blocking::get(
            "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
        ) {
            Ok(http_resp) => http_resp.bytes()?.to_vec(),
            Err(e) => anyhow::bail!(e),
        };
    let image = image::load_from_memory(&bytes)?;

    messages = messages.add_image_message(TextMessageRole::User, "What is this?", vec![image]);
    let resp = model
        .send_chat_request(RequestBuilder::from(messages.clone()).set_sampler_max_len(100))
        .await?
        .choices[0]
        .message
        .content
        .clone()
        .unwrap();
    println!("\n\n{resp}");
    messages = messages.add_message(TextMessageRole::Assistant, resp);

    let bytes =
        match reqwest::blocking::get("https://cdn.britannica.com/79/4679-050-BC127236/Titanic.jpg")
        {
            Ok(http_resp) => http_resp.bytes()?.to_vec(),
            Err(e) => anyhow::bail!(e),
        };
    let image = image::load_from_memory(&bytes)?;

    messages = messages.add_image_message(TextMessageRole::User, "What is this?", vec![image]);
    let resp = model
        .send_chat_request(RequestBuilder::from(messages.clone()).set_sampler_max_len(100))
        .await?
        .choices[0]
        .message
        .content
        .clone()
        .unwrap();
    println!("\n\nModel response*: {resp}");
    messages = messages.add_message(TextMessageRole::Assistant, resp);

    println!("Final chat history: {messages:?}");

    Ok(())
}
