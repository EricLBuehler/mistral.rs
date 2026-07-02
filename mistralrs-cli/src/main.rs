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
#[cfg(test)]
mod docgen;
mod ui;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use clap_complete::generate;

use args::{resolve_model_type, resolve_quantize_model_type, CacheCommand, Cli, Command};
use commands::{
    run_bench, run_cache_delete, run_cache_list, run_doctor, run_from_config, run_interactive,
    run_login, run_quantize, run_server, run_tune, run_uninstall, run_update, run_uqff,
    BenchRunConfig,
};
use mistralrs_core::{initialize_mistralrs_logging, LogVerbosity};

#[tokio::main]
async fn main() -> Result<()> {
    candle_core::utils::init_global_threadpool();
    let cli = Cli::parse();
    init_tracing(cli.global.verbose);

    match cli.command {
        Command::Serve {
            model_type,
            default_model,
            server,
            runtime,
            agent_options,
            sandbox,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_server(
                model_type,
                server,
                runtime,
                agent_options,
                sandbox,
                cli.global,
            )
            .await?;
        }

        Command::Run {
            model_type,
            default_model,
            runtime,
            agent_options,
            sandbox,
            thinking,
            input,
            image,
            video,
            audio,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_interactive(
                model_type,
                runtime,
                agent_options,
                sandbox,
                cli.global,
                thinking,
                input,
                image,
                video,
                audio,
            )
            .await?;
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

        Command::Uqff { command } => {
            run_uqff(command, cli.global).await?;
        }

        Command::FromConfig { file } => {
            run_from_config(file).await?;
        }

        Command::Update { tag } => {
            run_update(tag)?;
        }

        Command::Uninstall { yes } => {
            run_uninstall(yes)?;
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
            depth,
            iterations,
            warmup,
        } => {
            let model_type = resolve_model_type(model_type, default_model)?;
            run_bench(
                model_type,
                runtime,
                cli.global,
                BenchRunConfig {
                    prompt_lens: prompt_len,
                    gen_len,
                    depths: depth,
                    iterations,
                    warmup,
                },
            )
            .await?;
        }
    }

    Ok(())
}

fn init_tracing(verbose: u8) {
    initialize_mistralrs_logging(LogVerbosity::from_count(verbose));
}
