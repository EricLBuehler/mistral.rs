//! AnyMoE: create a Mixture-of-Experts model from fine-tuned adapters.
//!
//! Run with: `cargo run --release --example anymoe -p mistralrs`

use anyhow::Result;
use mistralrs::{
    AnyMoeConfig, AnyMoeExpertType, AnyMoeModelBuilder, IsqBits, PagedAttentionMetaBuilder,
    TextMessageRole, TextMessages, TextModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    let text_builder = TextModelBuilder::new("mistralai/Mistral-7B-Instruct-v0.1")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?);

    let model = AnyMoeModelBuilder::from_text_builder(
        text_builder,
        AnyMoeConfig {
            hidden_size: 4096,
            lr: 1e-3,
            epochs: 100,
            batch_size: 4,
            expert_type: AnyMoeExpertType::LoraAdapter {
                rank: 64,
                alpha: 16.,
                target_modules: vec!["gate_proj".to_string()],
            },
            gate_model_id: None, // Set this to Some("path/to/model/id") for the pretrained gating model id
            training: true,
            loss_csv_path: None,
        },
        "model.layers",
        "mlp",
        "examples/amoe.json",
        vec!["HuggingFaceH4/zephyr-7b-beta"],
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    )
    .build()
    .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
