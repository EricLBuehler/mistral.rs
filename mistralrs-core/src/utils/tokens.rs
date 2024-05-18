use std::{env, fs};
use thiserror::Error;

use anyhow::Result;
use tracing::info;

use crate::pipeline::TokenSource;

#[derive(Error, Debug)]
enum TokenRetrievalError {
    #[error("No home directory.")]
    HomeDirectoryMissing,
}

/// This reads a token from a specified source. If the token cannot be read, a warning is logged with `tracing`
/// and *no token is used*.
pub(crate) fn get_token(source: &TokenSource) -> Result<Option<String>> {
    fn skip_token(input: &str) -> Option<String> {
        info!("Could not load token at {input:?}, using no HF token.");
        None
    }

    let token = match source {
        TokenSource::Literal(data) => Some(data.clone()),
        TokenSource::EnvVar(envvar) => env::var(envvar).ok().or_else(|| skip_token(envvar)),
        TokenSource::Path(path) => fs::read_to_string(path).ok().or_else(|| skip_token(path)),
        TokenSource::CacheToken => {
            let home = format!(
                "{}/.cache/huggingface/token",
                dirs::home_dir()
                    .ok_or(TokenRetrievalError::HomeDirectoryMissing)?
                    .display()
            );

            fs::read_to_string(home.clone())
                .ok()
                .or_else(|| skip_token(&home))
        }
        TokenSource::None => None,
    };

    Ok(token.map(|s| s.trim().to_string()))
}
