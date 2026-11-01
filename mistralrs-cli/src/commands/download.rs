use anyhow::Result;
use mistralrs_core::LoaderBuilder;
use tracing::info;

use crate::args::{GlobalOptions, MatformerSelection, ModelType};

use super::serve::{
    apply_quant_resolution, convert_to_model_selected, extract_hf_config_settings,
    extract_quant_flag,
};

pub async fn run_download(
    mut model_type: ModelType,
    global: GlobalOptions,
    revision: Option<String>,
) -> Result<()> {
    if extract_quant_flag(&model_type)
        .is_some_and(|quant| quant.trim().eq_ignore_ascii_case("auto"))
    {
        anyhow::bail!(
            "`download` requires an explicit quantization when using `--quant`; run `mistralrs tune` to choose one"
        );
    }
    let matformer = MatformerSelection::default();
    apply_quant_resolution(
        &mut model_type,
        &global.token_source,
        &matformer,
        revision.as_deref(),
    )
    .await?;
    let model = convert_to_model_selected(&model_type, &matformer)?;
    let (max_model_len, hf_config_overrides) = extract_hf_config_settings(&model_type);
    let loader = LoaderBuilder::new(model)
        .with_max_model_len(max_model_len)
        .with_hf_config_overrides(hf_config_overrides)
        .build()?;
    let model_id = loader.get_id();
    let download = tokio::task::spawn_blocking(move || {
        loader.download_model_from_hf(revision, global.token_source, false)
    })
    .await??;
    let files = download.files();
    info!("Prepared {} files for `{model_id}`", files.len());
    for path in files {
        println!("{}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::args::{resolve_model_type, Cli, Command};

    use super::*;

    #[tokio::test]
    async fn download_rejects_hardware_dependent_quant_before_accessing_model() {
        for quant in ["auto", " AUTO "] {
            let cli = Cli::try_parse_from([
                "mistralrs",
                "download",
                "-m",
                "missing/model",
                "--quant",
                quant,
            ])
            .unwrap();
            let Command::Download {
                model_type,
                default_model,
                revision,
            } = cli.command
            else {
                panic!("expected download command");
            };
            let model = resolve_model_type(model_type, default_model).unwrap();
            let error = run_download(model, cli.global, revision).await.unwrap_err();
            assert!(error.to_string().contains("explicit quantization"));
        }
    }
}
