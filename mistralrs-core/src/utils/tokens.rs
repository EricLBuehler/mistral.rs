use std::{env, fs};
use thiserror::Error;

use anyhow::Result;

use crate::pipeline::TokenSource;

#[derive(Error, Debug)]
enum TokenRetrievalError {
    #[error("No home directory.")]
    HomeDirectoryMissing,
}

pub(crate) fn get_token(source: &TokenSource) -> Result<String> {
    Ok(match source {
        TokenSource::EnvVar(envvar) => env::var(envvar)?,
        TokenSource::Path(path) => fs::read_to_string(path)?,
        TokenSource::CacheToken => fs::read_to_string(format!(
            "{}/.cache/huggingface/token",
            dirs::home_dir()
                .ok_or(TokenRetrievalError::HomeDirectoryMissing)?
                .display()
        ))?,
    }
    .trim()
    .to_string())
}
