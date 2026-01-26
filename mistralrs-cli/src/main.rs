//! mistralrs-cli - Clean command-line interface for mistral.rs
//!
//! A new CLI design with:
//! - Orthogonal flags (format, adapter, modality are independent)
//! - Unified PagedAttention configuration
//! - Logical argument grouping
//! - Config-file-first support

mod args;
mod commands;
mod config;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::generate;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use args::{Cli, Command};
use commands::{run_from_config, run_interactive, run_quantize, run_server};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing (can be customized via RUST_LOG env var)
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Command::Serve {
            model_type,
            server,
            runtime,
        } => {
            run_server(model_type, server, runtime, cli.global).await?;
        }

        Command::Run {
            model_type,
            runtime,
            enable_thinking,
        } => {
            run_interactive(model_type, runtime, cli.global, enable_thinking).await?;
        }

        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut std::io::stdout());
        }

        Command::Quantize { model_type } => {
            run_quantize(model_type, cli.global).await?;
        }

        Command::FromConfig { file } => {
            run_from_config(file).await?;
        }
    }

    Ok(())
}
