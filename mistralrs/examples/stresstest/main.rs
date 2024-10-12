use anyhow::Result;
use mistralrs::{IsqType, MemoryUsage, RequestBuilder, TextMessageRole, TextModelBuilder};

const N_ITERS: u64 = 1000;
const BYTES_TO_MB: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q4K)
        .with_prefix_cache_n(None)
        .with_logging()
        // .with_paged_attn(|| mistralrs::PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    for i in 0..N_ITERS {
        let messages = RequestBuilder::new()
            .add_message(
                TextMessageRole::User,
                "Hello! How are you? Please write generic binary search function in Rust.",
            )
            .set_deterministic_sampler();

        println!("Sending request {}...", i + 1);
        let response = model.send_chat_request(messages).await?;

        let amount = MemoryUsage.get_memory_available(&model.config().device)? / BYTES_TO_MB;

        println!("{amount}");
        println!("{}", response.usage.total_time_sec);
        println!("{:?}", response.choices[0].message.content);
    }

    Ok(())
}
