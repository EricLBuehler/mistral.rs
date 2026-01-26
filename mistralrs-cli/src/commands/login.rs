//! HuggingFace authentication command

use anyhow::Result;
use std::io::Write;
use std::path::PathBuf;

/// Get the HuggingFace cache directory (cross-platform)
pub fn hf_cache_dir() -> Result<PathBuf> {
    // Check HF_HOME env var first (HuggingFace standard)
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return Ok(PathBuf::from(hf_home));
    }

    // Fall back to ~/.cache/huggingface
    let home =
        dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    Ok(home.join(".cache").join("huggingface"))
}

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
    let cache_dir = hf_cache_dir()?;
    std::fs::create_dir_all(&cache_dir)?;
    let token_path = cache_dir.join("token");
    std::fs::write(&token_path, &token)?;

    println!("✅ Token saved to {}", token_path.display());
    Ok(())
}
