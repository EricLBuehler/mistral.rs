//! Error handling patterns: matching on error variants and recovering partial responses.
//!
//! Run with: `cargo run --release --example error_handling -p mistralrs`

use mistralrs::{error, IsqBits, ModelBuilder, TextMessageRole, TextMessages};

#[tokio::main]
async fn main() {
    match load_and_chat().await {
        Ok(response) => println!("Response: {response}"),
        Err(e) => handle_error(e),
    }
}

async fn load_and_chat() -> error::Result<String> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .build()
        .await?;

    let messages = TextMessages::new().add_message(TextMessageRole::User, "Hello!");

    let response = model.send_chat_request(messages).await?;
    Ok(response.choices[0]
        .message
        .content
        .clone()
        .unwrap_or_default())
}

fn handle_error(err: error::Error) {
    match err {
        error::Error::ModelLoad(e) => {
            eprintln!("Failed to load model: {e}");
            eprintln!("Check that the model ID is correct and you have network access.");
        }
        error::Error::ModelError {
            message,
            partial_response,
        } => {
            eprintln!("Model error during generation: {message}");
            if let Some(partial) = partial_response {
                // Recover whatever the model produced before the error
                if let Some(text) = partial
                    .choices
                    .first()
                    .and_then(|c| c.message.content.as_ref())
                {
                    eprintln!("Partial response recovered: {text}");
                }
            }
        }
        error::Error::RequestValidation(msg) => {
            eprintln!("Invalid request: {msg}");
        }
        other => {
            eprintln!("Unexpected error: {other}");
        }
    }
}
