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
pub(crate) fn get_token(source: &TokenSource) -> Result<String> {
    Ok(match source {
        TokenSource::Literal(data) => data.clone(),
        TokenSource::EnvVar(envvar) => {
            let tok = env::var(envvar);
            if let Ok(tok) = tok {
                tok
            } else {
                info!("Could not load token at {envvar:?}, using no HF token.");
                "".to_string()
            }
        }
        TokenSource::Path(path) => {
            let tok = fs::read_to_string(path);
            if let Ok(tok) = tok {
                tok
            } else {
                info!("Could not load token at {path:?}, using no HF token.");
                "".to_string()
            }
        }
        TokenSource::CacheToken => {
            let home = format!(
                "{}/.cache/huggingface/token",
                dirs::home_dir()
                    .ok_or(TokenRetrievalError::HomeDirectoryMissing)?
                    .display()
            );
            let tok = fs::read_to_string(home.clone());
            if let Ok(tok) = tok {
                tok
            } else {
                info!("Could not load token at {home:?}, using no HF token.");
                "".to_string()
            }
        }
        TokenSource::None => "".to_string(),
    }
    .trim()
    .to_string())
}
