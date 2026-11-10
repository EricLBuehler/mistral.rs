use std::path::PathBuf;

use anyhow::{Context, Result};
use candle_core::Device;
use mistralrs_quant::{IsqType, TrackedModule};
use tokenizers::Tokenizer;

use super::drive::CalibrationDrive;
use super::plan::IsqLoadPlan;
use super::{complete_isq_capture, resolve_imatrix_map, CalibrationCtx};
use crate::device_map::DeviceMapper;
use crate::pipeline::chat_template::ChatTemplate;
use crate::pipeline::isq::{write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest};
use crate::pipeline::{IsqModel, IsqOrganization, ModelPaths};

/// What the post-load ISQ stage needs from a loader once the model is built.
pub(crate) struct IsqFinish<'a> {
    pub plan: &'a IsqLoadPlan,
    pub modules: Vec<TrackedModule>,
    pub drive: &'a dyn CalibrationDrive,
    pub model: &'a dyn IsqModel,
    pub in_situ_quant: Option<IsqType>,
    pub imatrix: Option<&'a PathBuf>,
    pub calibration_file: Option<&'a PathBuf>,
    pub tokenizer: &'a Tokenizer,
    pub chat_template: &'a ChatTemplate,
    pub load_device: &'a Device,
    pub mapper: &'a dyn DeviceMapper,
    pub write_uqff: Option<&'a UqffWriteConfig>,
    pub organization: IsqOrganization,
    pub paths: &'a dyn ModelPaths,
    /// A prepared source without a generation config has no file to copy into the UQFF.
    pub omit_generation_config: bool,
    pub config: &'a str,
    pub processor_filename: &'a Option<PathBuf>,
    pub preprocessor_filename: &'a Option<PathBuf>,
}

/// Collects the imatrix, completes a pending capture and writes UQFF artifacts, as the plan asks.
pub(crate) fn finish_isq(args: IsqFinish<'_>) -> Result<()> {
    let IsqFinish {
        plan,
        modules,
        drive,
        model,
        in_situ_quant,
        imatrix,
        calibration_file,
        tokenizer,
        chat_template,
        load_device,
        mapper,
        write_uqff,
        organization,
        paths,
        omit_generation_config,
        config,
        processor_filename,
        preprocessor_filename,
    } = args;
    plan.validate_tracked_selection(&modules)?;

    let imatrix_map = if plan.wants_imatrix {
        Some(resolve_imatrix_map(
            drive,
            &modules,
            imatrix,
            calibration_file,
            &CalibrationCtx {
                tokenizer,
                bos_tok_id: chat_template
                    .bos_tok()
                    .as_deref()
                    .and_then(|tok| tokenizer.token_to_id(tok)),
                load_device,
                mapper: Some(mapper),
            },
        )?)
    } else {
        None
    };

    if plan.capture == mistralrs_quant::IsqCaptureMode::CaptureMatches {
        let ty = in_situ_quant.context("imatrix quantization requires an ISQ type")?;
        complete_isq_capture(
            &modules,
            ty,
            imatrix_map
                .as_ref()
                .expect("CaptureMatches requires imatrix data"),
        )?;
    }

    if let Some(write_uqff) = write_uqff {
        let uqff_types = plan
            .write_types
            .clone()
            .filter(|types| !types.is_empty())
            .context("UQFF serialization requires at least one ISQ type.")?;
        let residual = match organization {
            IsqOrganization::Default => model.residual_tensors(),
            IsqOrganization::MoeExpertsOnly => model
                .residual_tensors_moe_experts_only()
                .unwrap_or(model.residual_tensors()),
        };
        let full_ser = UqffFullSer {
            tokenizer,
            template_filename: paths.get_template_filename(),
            effective_chat_template: Some(chat_template),
            generation_config: if omit_generation_config {
                None
            } else {
                paths.get_gen_conf_filename()
            },
            config: config.to_string(),
            processor_filename,
            preprocessor_filename,
            modules: None,
            module_paths: None,
        };
        write_uqff_artifacts(UqffWriteRequest {
            output: write_uqff.output.clone(),
            types: uqff_types,
            base_model: write_uqff.base_model.clone(),
            repo_id: write_uqff.repo_id.clone(),
            layers: modules,
            quantize_predicates: plan.uqff_quantize_predicates.clone(),
            residual,
            full_ser,
            imatrix: imatrix_map.unwrap_or_default(),
        })?;
    }
    Ok(())
}
