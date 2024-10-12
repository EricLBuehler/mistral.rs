use anyhow::Result;
use mistralrs::{IsqType, MemoryUsage, TextMessageRole, TextMessages, TextModelBuilder};

const N_ITERS: u64 = 1000;
const BYTES_TO_MB: usize = 1024 * 1024;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        // .with_paged_attn(|| mistralrs::PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    for i in 0..N_ITERS {
        let messages = TextMessages::new().add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

        println!("Sending request {}...", i+1);
        let _response = model.send_chat_request(messages).await?;

        let amount = MemoryUsage.get_memory_available(&model.config().device)? / BYTES_TO_MB;

        println!("{amount}");

    }

    Ok(())
}
