//! JSON schema-constrained generation for typed structured output.
//!
//! Run with: `cargo run --release --example json_schema -p mistralrs`

use anyhow::Result;
use mistralrs::{
    IsqBits, ModelBuilder, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let request = RequestBuilder::new()
        .set_constraint(mistralrs::Constraint::JsonSchema(json!(
            {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                    "zip": {"type": "integer", "minimum": 10000, "maximum": 99999},
                },
                "required": ["street", "city", "state", "zip"],
                "additionalProperties": false,
            }
        )))
        .set_sampler_max_len(100)
        .add_message(TextMessageRole::User, "A sample address please.");

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
