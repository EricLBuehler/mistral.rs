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
        TokenSource::Literal(data) => data.clone(),
        TokenSource::EnvVar(envvar) => env::var(envvar)
            .map_err(|_| anyhow::Error::msg(format!("Could not load env var `{envvar}`")))?,
        TokenSource::Path(path) => fs::read_to_string(path)
            .map_err(|_| anyhow::Error::msg(format!("Could not load token at `{path}`")))?,
        TokenSource::CacheToken => {
            let home = format!(
                "{}/.cache/huggingface/token",
                dirs::home_dir()
                    .ok_or(TokenRetrievalError::HomeDirectoryMissing)?
                    .display()
            );
            fs::read_to_string(home.clone())
                .map_err(|_| anyhow::Error::msg(format!("Could not load token at `{home}`")))?
        }
        TokenSource::None => "".to_string(),
    }
    .trim()
    .to_string())
}
