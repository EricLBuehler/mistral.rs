use anyhow::{anyhow, Result};
use mistralrs::{
    IsqType, MultiModelBuilder, TextMessageRole, TextMessages, TextModelBuilder, VisionModelBuilder,
};

// Model IDs - these are the actual HuggingFace model paths
const GEMMA_MODEL_ID: &str = "google/gemma-3-4b-it";
const QWEN_MODEL_ID: &str = "Qwen/Qwen3-4B";

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading multiple models...");

    let model = MultiModelBuilder::new()
        .add_model(
            VisionModelBuilder::new(GEMMA_MODEL_ID)
                .with_isq(IsqType::Q4K)
                .with_logging(),
        )
        .add_model(TextModelBuilder::new(QWEN_MODEL_ID).with_isq(IsqType::Q4K))
        .with_default_model(GEMMA_MODEL_ID)
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
    println!("\n=== Request to Default Model ({}) ===", GEMMA_MODEL_ID);
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is 2 + 2? Answer briefly.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Send a request to a specific model (Qwen - text model)
    println!("\n=== Request to Specific Model ({}) ===", QWEN_MODEL_ID);
    let messages = TextMessages::new().add_message(TextMessageRole::User, "Say hello in one word.");

    let response = model
        .send_chat_request_with_model(messages, Some(QWEN_MODEL_ID))
        .await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Change the default model
    println!("\n=== Changing Default Model ===");
    model
        .set_default_model_id(QWEN_MODEL_ID)
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
    let is_gemma_loaded = model.is_model_loaded(GEMMA_MODEL_ID)?;
    println!("Is '{}' loaded? {}", GEMMA_MODEL_ID, is_gemma_loaded);

    // Unload Gemma to free memory
    println!("Unloading '{}' model...", GEMMA_MODEL_ID);
    model.unload_model(GEMMA_MODEL_ID)?;

    // Check status after unload
    let status = model.list_models_with_status()?;
    println!("Status after unload:");
    for (model_id, status) in &status {
        println!("  {} -> {:?}", model_id, status);
    }

    // Reload Gemma when needed
    println!("Reloading '{}' model...", GEMMA_MODEL_ID);
    model.reload_model(GEMMA_MODEL_ID).await?;

    // Check status after reload
    let is_gemma_loaded = model.is_model_loaded(GEMMA_MODEL_ID)?;
    println!(
        "Is '{}' loaded after reload? {}",
        GEMMA_MODEL_ID, is_gemma_loaded
    );

    // Use the reloaded model
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Hi! Respond with just 'Hello'.");

    let response = model
        .send_chat_request_with_model(messages, Some(GEMMA_MODEL_ID))
        .await?;
    println!(
        "Response from reloaded {}: {}",
        GEMMA_MODEL_ID,
        response.choices[0].message.content.as_ref().unwrap()
    );

    println!("\n=== Multi-Model Example Complete ===");

    Ok(())
}
