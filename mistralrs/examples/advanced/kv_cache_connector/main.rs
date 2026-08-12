//! Realistic usage of an external KV cache connector with paged attention.
//!
//! Installs `InMemoryKvCacheConnector` through the normal model builder path
//! (same place you'd later plug disk/S3), then runs two chats that share a
//! prefix so the connector sees store/lookup traffic.
//!
//! Run with:
//! `cargo run --release -p mistralrs --example kv_cache_connector`

use std::sync::Arc;

use anyhow::Result;
use mistralrs::{
    InMemoryKvCacheConnector, IsqBits, KvCacheConnector, MemoryGpuConfig, ModelBuilder,
    PagedAttentionMetaBuilder, TextMessageRole, TextMessages,
};

#[tokio::main]
async fn main() -> Result<()> {
    let connector = Arc::new(InMemoryKvCacheConnector::new());

    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Eight)
        .with_logging()
        .with_paged_attn(
            PagedAttentionMetaBuilder::default()
                .with_block_size(32)
                .with_gpu_memory(MemoryGpuConfig::ContextSize(1024))
                .with_kv_cache_connector(Arc::clone(&connector) as Arc<dyn KvCacheConnector>)
                .build()?,
        )
        .build()
        .await?;

    let shared = "You are a concise assistant.";
    let first = TextMessages::new()
        .add_message(TextMessageRole::System, shared)
        .add_message(TextMessageRole::User, "Say hello in one short sentence.");
    let second = TextMessages::new()
        .add_message(TextMessageRole::System, shared)
        .add_message(
            TextMessageRole::User,
            "Say hello in one short sentence. Then add a second short sentence.",
        );

    let response_a = model.send_chat_request(first).await?;
    println!(
        "turn A: {}",
        response_a.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
    );
    println!(
        "after turn A: stores={}, lookups={}, hits={}, external_entries={}",
        connector.store_count(),
        connector.lookup_count(),
        connector.hit_count(),
        connector.len()
    );

    let response_b = model.send_chat_request(second).await?;
    println!(
        "turn B: {}",
        response_b.choices[0]
            .message
            .content
            .as_deref()
            .unwrap_or("")
    );
    println!(
        "after turn B: stores={}, lookups={}, hits={}, external_entries={}",
        connector.store_count(),
        connector.lookup_count(),
        connector.hit_count(),
        connector.len()
    );

    anyhow::ensure!(
        connector.store_count() > 0,
        "expected the live paged-attention path to observe_store via the connector"
    );
    anyhow::ensure!(
        connector.lookup_count() > 0,
        "expected the live paged-attention path to call lookup_blocks via the connector"
    );

    println!("ok: connector received traffic from the real paged-attention scheduler");
    Ok(())
}
