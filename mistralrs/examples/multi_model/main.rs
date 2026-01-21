use anyhow::{anyhow, Result};
use mistralrs::{
    IsqType, MultiModelBuilder, TextMessageRole, TextMessages, TextModelBuilder, VisionModelBuilder,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading multiple models...");

    let model = MultiModelBuilder::new()
        .add_model(
            VisionModelBuilder::new("google/gemma-3-4b-it")
                .with_isq(IsqType::Q4K)
                .with_logging(),
            Some("gemma".to_string()),
        )
        .add_model(
            TextModelBuilder::new("Qwen/Qwen3-4B").with_isq(IsqType::Q4K),
            Some("qwen".to_string()),
        )
        .with_default_model("gemma")
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

    // Send a request to the default model (gemma - vision model)
    println!("\n=== Request to Default Model (gemma) ===");
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is 2 + 2? Answer briefly.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Send a request to a specific model (qwen - text model)
    println!("\n=== Request to Specific Model (qwen) ===");
    let messages = TextMessages::new().add_message(TextMessageRole::User, "Say hello in one word.");

    let response = model
        .send_chat_request_with_model(messages, Some("qwen"))
        .await?;
    println!(
        "Response: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Change the default model
    println!("\n=== Changing Default Model ===");
    model.set_default_model_id("qwen").map_err(|e| anyhow!(e))?;
    let new_default = model.get_default_model_id().map_err(|e| anyhow!(e))?;
    println!("New default model: {:?}", new_default);

    // Now requests without model_id go to qwen
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "What is your name? Be brief.");

    let response = model.send_chat_request(messages).await?;
    println!(
        "Response from new default: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    // Model unloading/reloading demonstration
    println!("\n=== Model Unloading/Reloading ===");

    // Check if gemma is loaded
    let is_gemma_loaded = model.is_model_loaded("gemma")?;
    println!("Is 'gemma' loaded? {}", is_gemma_loaded);

    // Unload gemma to free memory
    println!("Unloading 'gemma' model...");
    model.unload_model("gemma")?;

    // Check status after unload
    let status = model.list_models_with_status()?;
    println!("Status after unload:");
    for (model_id, status) in &status {
        println!("  {} -> {:?}", model_id, status);
    }

    // Reload gemma when needed
    println!("Reloading 'gemma' model...");
    model.reload_model("gemma").await?;

    // Check status after reload
    let is_gemma_loaded = model.is_model_loaded("gemma")?;
    println!("Is 'gemma' loaded after reload? {}", is_gemma_loaded);

    // Use the reloaded model
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Hi! Respond with just 'Hello'.");

    let response = model
        .send_chat_request_with_model(messages, Some("gemma"))
        .await?;
    println!(
        "Response from reloaded gemma: {}",
        response.choices[0].message.content.as_ref().unwrap()
    );

    println!("\n=== Multi-Model Example Complete ===");

    Ok(())
}
