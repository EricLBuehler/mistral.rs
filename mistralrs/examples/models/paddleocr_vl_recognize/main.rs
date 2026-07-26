//! PaddleOCR-VL document OCR: recognize one image region and print the text.
//!
//! The model is task-prompted. `OCR:` reads text, `Table Recognition:` returns OTSL `<fcel>`/`<nl>`
//! markup, `Formula Recognition:` returns LaTeX. Pass the prompt as the second argument to switch.
//!
//! Run with:
//! `cargo run --release --example paddleocr_vl_recognize -p mistralrs -- <image> ["Table Recognition:"]`

use anyhow::{bail, Result};
use mistralrs::{MultimodalMessages, MultimodalModelBuilder, RequestBuilder, TextMessageRole};

/// Cap generation so a pathological region cannot run away; real crops stop on EOS well under this.
const MAX_NEW_TOKENS: usize = 2048;

#[tokio::main]
async fn main() -> Result<()> {
    let Some(image_path) = std::env::args().nth(1) else {
        bail!("usage: paddleocr_vl_recognize <image> [prompt]");
    };
    let prompt = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "OCR:".to_string());

    let model = MultimodalModelBuilder::new("PaddlePaddle/PaddleOCR-VL-1.5")
        .with_logging()
        .build()
        .await?;

    let image = image::open(&image_path)?;
    let messages =
        MultimodalMessages::new().add_image_message(TextMessageRole::User, &prompt, vec![image]);
    let response = model
        .send_chat_request(RequestBuilder::from(messages).set_sampler_max_len(MAX_NEW_TOKENS))
        .await?;

    println!(
        "{}",
        response.choices[0].message.content.as_deref().unwrap_or("")
    );
    Ok(())
}
