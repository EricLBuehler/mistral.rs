//! HuggingFace authentication command

use anyhow::Result;
use std::io::Write;

/// Run the login command
pub fn run_login(token: Option<String>) -> Result<()> {
    let token = match token {
        Some(t) => t,
        None => {
            // Interactive: prompt for token
            print!("Enter your HuggingFace token: ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim().to_string()
        }
    };

    if token.is_empty() {
        anyhow::bail!("Token cannot be empty");
    }

    // Validate token format (starts with hf_)
    if !token.starts_with("hf_") {
        anyhow::bail!("Invalid token format. HuggingFace tokens start with 'hf_'");
    }

    // Write to cache
    let token_path = mistralrs_core::hf_token_path()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine Hugging Face token path"))?;
    if let Some(parent) = token_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&token_path, &token)?;

    println!("✅ Token saved to {}", token_path.display());
    Ok(())
}
