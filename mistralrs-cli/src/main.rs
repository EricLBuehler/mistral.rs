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
mod ui;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::generate;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use args::{resolve_model_type, resolve_quantize_model_type, CacheCommand, Cli, Command};
use commands::{
    run_bench, run_cache_delete, run_cache_list, run_doctor, run_from_config, run_interactive,
    run_login, run_quantize, run_server, run_tune,
};

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
            default_model,
            server,
            runtime,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_server(model_type, server, runtime, cli.global).await?;
        }

        Command::Run {
            model_type,
            default_model,
            runtime,
            enable_thinking,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_interactive(model_type, runtime, cli.global, enable_thinking).await?;
        }

        Command::Completions { shell } => {
            let mut cmd = Cli::command();
            let name = cmd.get_name().to_string();
            generate(shell, &mut cmd, name, &mut std::io::stdout());
        }

        Command::Quantize {
            model_type,
            default_quantize,
        } => {
            let model_type = resolve_quantize_model_type(model_type, default_quantize)?;
            run_quantize(model_type, cli.global).await?;
        }

        Command::FromConfig { file } => {
            run_from_config(file).await?;
        }

        Command::Doctor { json } => {
            run_doctor(json)?;
        }

        Command::Tune {
            model_type,
            default_model,
            profile,
            json,
            emit_config,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_tune(model_type, cli.global, profile, json, emit_config).await?;
        }

        Command::Login { token } => {
            run_login(token)?;
        }

        Command::Cache { cmd } => match cmd {
            CacheCommand::List => run_cache_list()?,
            CacheCommand::Delete { model_id } => run_cache_delete(&model_id)?,
        },

        Command::Bench {
            model_type,
            default_model,
            runtime,
            prompt_len,
            gen_len,
            iterations,
            warmup,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_bench(
                model_type, runtime, cli.global, prompt_len, gen_len, iterations, warmup,
            )
            .await?;
        }
    }

    Ok(())
}
