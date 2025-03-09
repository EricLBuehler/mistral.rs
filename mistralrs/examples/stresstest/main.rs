use anyhow::Result;
use mistralrs::{PagedAttentionMetaBuilder, TextMessageRole, TextMessages, TextModelBuilder};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let model = Arc::new(
        TextModelBuilder::new("meta-llama/Llama-3.3-70B-Instruct")
            .with_logging()
            .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
            .build()
            .await?,
    );

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let tasks: Vec<_> = (0..1)
        .map(|_| {
            let messages_clone = messages.clone();
            let model = Arc::clone(&model);
            tokio::spawn(async move { model.send_chat_request(messages_clone).await })
        })
        .collect();

    let responses = futures::future::join_all(tasks).await;
    for response in responses {
        match response {
            Ok(result) => {
                let result = result?;
                dbg!(
                    result.usage.avg_prompt_tok_per_sec,
                    result.usage.avg_compl_tok_per_sec
                );
            }
            Err(e) => {
                eprintln!("Task failed: {:?}", e);
            }
        }
    }

    Ok(())
}
