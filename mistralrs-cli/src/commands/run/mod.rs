//! Interactive mode command implementation

mod interactive;

pub(crate) use interactive::interactive_mode;
use interactive::OneshotInput;

use anyhow::Result;
use tracing::info;

use mistralrs_core::initialize_logging;
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;

use super::serve::{
    convert_to_model_selected, extract_device_settings, extract_isq_setting,
    extract_paged_attn_settings,
};
use crate::args::{GlobalOptions, ModelType, RuntimeOptions};

/// Run the model in interactive or one-shot mode
pub async fn run_interactive(
    model_type: ModelType,
    runtime: RuntimeOptions,
    global: GlobalOptions,
    thinking: Option<bool>,
    input: Option<String>,
    images: Vec<String>,
    videos: Vec<String>,
    audios: Vec<String>,
) -> Result<()> {
    initialize_logging();

    // Convert our clean args to ModelSelected
    let model_selected = convert_to_model_selected(&model_type)?;

    // Extract settings
    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);
    let (cpu, device_layers) = extract_device_settings(&model_type);
    let isq = extract_isq_setting(&model_type);

    // Build the MistralRs instance
    let mut builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(runtime.max_seqs)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(true)
        .with_prefix_cache_n(runtime.prefix_cache_n)
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_enable_search(runtime.enable_search)
        .with_seed_optional(global.seed)
        .with_log_optional(global.log.as_ref().map(|p| p.to_string_lossy().to_string()))
        .with_chat_template_optional(
            runtime
                .chat_template
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_jinja_explicit_optional(
            runtime
                .jinja_explicit
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type);

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mistralrs = builder.build().await?;

    if let Some(text) = input {
        info!("Model loaded, running one-shot mode...");
        interactive::oneshot_mode(
            mistralrs.clone(),
            runtime.enable_search,
            thinking,
            OneshotInput {
                text,
                images,
                videos,
                audios,
            },
        )
        .await;
    } else {
        info!("Model loaded, starting interactive mode...");
        interactive::interactive_mode(mistralrs.clone(), runtime.enable_search, thinking).await;
    }

    Ok(())
}
