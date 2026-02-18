//! Custom logits processor that modifies token probabilities during generation.
//!
//! Run with: `cargo run --release --example logits_processor -p mistralrs`

use std::sync::Arc;

use anyhow::Result;
use mistralrs::{
    CustomLogitsProcessor, IsqBits, ModelBuilder, PagedAttentionMetaBuilder, RequestBuilder,
    Tensor, TextMessageRole,
};
use rand::Rng;

struct ThresholdLogitsProcessor {
    threshold: f64,
}

impl CustomLogitsProcessor for ThresholdLogitsProcessor {
    fn apply(&self, logits: &Tensor, _context: &[u32]) -> mistralrs::Result<Tensor> {
        // Mask is 1 for true, 0 for false.
        let mask = logits.ge(self.threshold)?;
        logits.broadcast_mul(&mask.to_dtype(logits.dtype())?)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let mut rng = rand::rng();
    let random_value: f64 = rng.random_range(0.0..=1.0);
    let threshold: f64 = rng.random_range(0.0..=0.5);

    let request = RequestBuilder::new()
        .add_logits_processor(Arc::new(move |logits: &Tensor, _context: &[u32]| {
            logits * random_value
        }))
        .add_logits_processor(Arc::new(ThresholdLogitsProcessor { threshold }))
        .add_message(
            TextMessageRole::User,
            "Please write a mathematical equation where a few numbers are added.",
        );

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
