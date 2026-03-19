//! Log model output to a file using the tracing framework.
//!
//! Run with: `cargo run --release --example file_logging -p mistralrs`

use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, TextMessageRole, TextMessages};
use std::fs;
use std::fs::OpenOptions;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::EnvFilter;

const LOG_FILE: &str = "custom_logging.log";

fn init_logging(log_file: &str) {
    let _ = fs::remove_file(log_file);
    let path = log_file.to_owned();
    let writer = move || {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("failed to open log file")
    };

    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .add_directive("mistralrs_core=debug".parse().unwrap());

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(writer)
        .try_init()
        .expect("subscriber installed only once");
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging(LOG_FILE);
    info!("Custom subscriber installed; writing logs to {LOG_FILE}");

    let model = ModelBuilder::new("Qwen/Qwen3-VL-4B-Instruct")
        .with_auto_isq(IsqBits::Four)
        // NOTE: deliberately skip `.with_logging()` so only our subscriber runs.
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You describe locations succinctly.",
        )
        .add_message(
            TextMessageRole::User,
            "Give me two sentences about the Mount Washington summit weather.",
        );

    let response = model.send_chat_request(messages).await?;
    let answer = response.choices[0].message.content.as_ref().unwrap();
    println!("Model response:\n{answer}");

    info!("Completed inference run.");

    let captured_logs = fs::read_to_string(LOG_FILE)?;
    println!("\n--- Captured mistral.rs logs ---\n{captured_logs}");

    Ok(())
}
