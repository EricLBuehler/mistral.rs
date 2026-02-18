//! Load and dispatch requests across multiple models simultaneously.
//!
//! Run with: `cargo run --release --example multi_model -p mistralrs`

use anyhow::{anyhow, Result};
use mistralrs::{
    IsqBits, MultiModelBuilder, TextMessageRole, TextMessages, TextModelBuilder, VisionModelBuilder,
};

// Model IDs - these are the actual HuggingFace model paths
const GEMMA_MODEL_ID: &str = "google/gemma-3-4b-it";
const QWEN_MODEL_ID: &str = "Qwen/Qwen3-4B";
// Aliases - these are the short IDs used in API requests
const GEMMA_ALIAS: &str = "gemma-vision";
const QWEN_ALIAS: &str = "qwen-text";

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading multiple models...");

    let model = MultiModelBuilder::new()
        .add_model_with_alias(
            GEMMA_ALIAS,
            VisionModelBuilder::new(GEMMA_MODEL_ID)
                .with_auto_isq(IsqBits::Four)
                .with_logging(),
        )
        .add_model_with_alias(
            QWEN_ALIAS,
            TextModelBuilder::new(QWEN_MODEL_ID).with_auto_isq(IsqBits::Four),
        )
        .with_default_model(GEMMA_ALIAS)
        .build()
        .await?;

    // List available models
    println!("\n=== Available Models ===");
    let models = model.list_models().map_err(|e| anyhow!(e))?;
    for model_id in &models {
        println!("  - {}", model_id);
    }

    // Get the default model
    let default_model = model.get_default_model_id().map_err(|e| anyhow!(e))?;
    println!("\nDefault model: {:?}", default_model);

    // List models with their status
    println!("\n=== Model Status ===");
    let status = model.list_models_with_status()?;
    for (model_id, status) in &status {
        println!("  {} -> {:?}", model_id, status);
    }

    // Send a request to the default model (Gemma - vision model)
    println!("\n=== Request to Default Model ({}) ===", GEMMA_ALIAS);
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is 2 + 2? Answer briefly.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Send a request to a specific model (Qwen - text model)
    println!("\n=== Request to Specific Model ({}) ===", QWEN_ALIAS);
    let messages = TextMessages::new().add_message(TextMessageRole::User, "Say hello in one word.");

    let response = model
        .send_chat_request_with_model(messages, Some(QWEN_ALIAS))
        .await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Change the default model
    println!("\n=== Changing Default Model ===");
    model
        .set_default_model_id(QWEN_ALIAS)
        .map_err(|e| anyhow!(e))?;
    let new_default = model.get_default_model_id().map_err(|e| anyhow!(e))?;
    println!("New default model: {:?}", new_default);

    // Now requests without model_id go to Qwen
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is your name? Be brief.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response from new default: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Model unloading/reloading demonstration
    println!("\n=== Model Unloading/Reloading ===");

    // Check if Gemma is loaded
    let is_gemma_loaded = model.is_model_loaded(GEMMA_ALIAS)?;
    println!("Is '{}' loaded? {}", GEMMA_ALIAS, is_gemma_loaded);

    // Unload Gemma to free memory
    println!("Unloading '{}' model...", GEMMA_ALIAS);
    model.unload_model(GEMMA_ALIAS)?;

    // Check status after unload
    let status = model.list_models_with_status()?;
    println!("Status after unload:");
    for (model_id, status) in &status {
        println!("  {} -> {:?}", model_id, status);
    }

    // Reload Gemma when needed
    println!("Reloading '{}' model...", GEMMA_ALIAS);
    model.reload_model(GEMMA_ALIAS).await?;

    // Check status after reload
    let is_gemma_loaded = model.is_model_loaded(GEMMA_ALIAS)?;
    println!(
        "Is '{}' loaded after reload? {}",
        GEMMA_ALIAS, is_gemma_loaded
    );

    // Use the reloaded model
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Hi! Respond with just 'Hello'.");

    let response = model
        .send_chat_request_with_model(messages, Some(GEMMA_ALIAS))
        .await?;
    println!(
        "Response from reloaded {}: {}",
        GEMMA_ALIAS,
        response.choices[0].message.content.as_ref().unwrap()
    );

    println!("\n=== Multi-Model Example Complete ===");

    Ok(())
}
