/// Interactive multi-turn chatbot with streaming output.
///
/// Demonstrates:
/// - Reading user input from stdin in a loop
/// - Accumulating messages across turns
/// - Streaming responses token-by-token
///
/// Run with: `cargo run --release --example cookbook_multiturn -p mistralrs`
use anyhow::Result;
use mistralrs::{
    ChatCompletionChunkResponse, ChunkChoice, Delta, IsqBits, ModelBuilder, Response,
    TextMessageRole, TextMessages,
};
use std::io::{self, BufRead, Write};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let mut messages = TextMessages::new().add_message(
        TextMessageRole::System,
        "You are a helpful assistant. Keep responses concise.",
    );

    let stdin = io::stdin();
    println!("Chatbot ready. Type your messages (Ctrl+D to quit).\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            break; // EOF
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        messages = messages.add_message(TextMessageRole::User, input);

        // Stream the response
        print!("Assistant: ");
        io::stdout().flush()?;

        let mut stream = model.stream_chat_request(messages.clone()).await?;
        let mut assistant_text = String::new();

        while let Some(chunk) = stream.next().await {
            if let Response::Chunk(ChatCompletionChunkResponse { choices, .. }) = chunk {
                if let Some(ChunkChoice {
                    delta:
                        Delta {
                            content: Some(content),
                            ..
                        },
                    ..
                }) = choices.first()
                {
                    print!("{content}");
                    io::stdout().flush()?;
                    assistant_text.push_str(content);
                }
            }
        }
        println!("\n");

        // Add assistant reply to history for multi-turn context
        messages = messages.add_message(TextMessageRole::Assistant, &assistant_text);
    }

    println!("\nGoodbye!");
    Ok(())
}
