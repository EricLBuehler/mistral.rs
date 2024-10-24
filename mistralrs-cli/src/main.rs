use anyhow::Result;
use clap::{Parser, Subcommand};
use mistralrs::{
    parse_isq_value, IsqType, MemoryGpuConfig, PagedAttentionMetaBuilder, TextModelBuilder,
    VisionModelBuilder,
};
use registry::ModelSpec;
use tracing::info;

mod interactive;
mod printer;
mod registry;
mod util;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Subcommand, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SelectedModel {
    #[command(name = "vllama")]
    #[command(alias = "vllama:11b")]
    VLlama_11b,
    #[command(name = "phi3v")]
    #[command(alias = "phi3v:3.8b")]
    Phi3V_3_8b,
    #[command(name = "phi3")]
    #[command(alias = "phi3:3.8b")]
    Phi3_3_8b,
    #[command(name = "gemma1")]
    #[command(alias = "gemma1:7b")]
    Gemma1_7b,
    #[command(name = "gemma1:2b")]
    Gemma1_2b,
    #[command(name = "gemma2")]
    #[command(alias = "gemma2:9b")]
    Gemma2_9b,
    #[command(name = "gemma2:2b")]
    Gemma2_2b,
    #[command(name = "gemma2:27b")]
    Gemma2_27b,
    #[command(name = "llama3.1")]
    #[command(alias = "llama3.1:8b")]
    Llama3_1_8b,
    #[command(name = "llama3.2")]
    #[command(alias = "llama3.2:3b")]
    Llama3_2_3b,
    #[command(name = "llama3.2:1b")]
    Llama3_2_1b,
    #[command(name = "mistral0.3")]
    #[command(alias = "mistral0.3:7b")]
    Mistral_7b,
    #[command(name = "mistralnemo")]
    #[command(alias = "mistralnemo:12b")]
    MistralNemo_12b,
    #[command(name = "mistralsmall")]
    #[command(alias = "mistralnemo:12b")]
    MistralSmall_12b,
}

#[derive(Parser)]
struct Args {
    /// This is only relevant when using CUDA.
    /// This specifies the number of tokens to allocate for PagedAttention KV cache.
    /// Otherwise, 90% GPU VRAM utilization is allocated.
    #[arg(short, long, value_parser = parse_isq_value)]
    paged_attn_ctxt_len: Option<usize>,

    #[arg(short, long, default_value_t = false)]
    insitu: bool,

    /// Quantization to use. See the supported ones at https://github.com/EricLBuehler/mistral.rs/blob/master/docs/ISQ.md.
    /// If `insitu` is set, then the model will be quantized in-situ.
    /// The default is Q4K.
    #[arg(short, long, value_parser = parse_isq_value)]
    quant: Option<IsqType>,

    /// Choose the model
    #[clap(subcommand)]
    model: SelectedModel,
}

#[tokio::main]
async fn main() -> Result<()> {
    mistralrs::initialize_logging();

    let args = Args::parse();

    let spec = ModelSpec::from_selected(args.model);

    let mut paged_attn_cfg = PagedAttentionMetaBuilder::default();
    if let Some(ctxt_len) = args.paged_attn_ctxt_len {
        paged_attn_cfg = paged_attn_cfg.with_gpu_memory(MemoryGpuConfig::ContextSize(ctxt_len));
    }

    let model = match &spec {
        ModelSpec::Text {
            uqff_model_id,
            base_model_id,
            supported_quants,
            stem,
        } => {
            let quant = args.quant.unwrap_or(spec.default_quant());
            if !supported_quants.contains(&quant) {
                anyhow::bail!("The model `{:?}` does not support `{quant}`. It supports {supported_quants:?}.", args.model)
            }
            info!("Loading model {:?} with quantization {quant}", args.model);
            let model_id = if args.insitu {
                base_model_id
            } else {
                uqff_model_id
            };
            let mut builder = TextModelBuilder::new(model_id)
                .with_paged_attn(|| paged_attn_cfg.build())?
                .with_logging();
            if args.insitu {
                builder = builder.with_isq(quant);
            } else {
                builder = builder.from_uqff(format!("{stem}-{quant}.uqff").into());
            }

            builder.build().await?
        }
        ModelSpec::Vision {
            uqff_model_id,
            base_model_id,
            supported_quants,
            arch,
            stem,
        } => {
            let quant = args.quant.unwrap_or(spec.default_quant());
            if !supported_quants.contains(&quant) {
                anyhow::bail!("The model `{:?}` does not support `{quant}`. It supports {supported_quants:?}.", args.model)
            }
            info!("Loading model {:?} with quantization {quant}", args.model);
            let model_id = if args.insitu {
                base_model_id
            } else {
                uqff_model_id
            };
            let mut builder = VisionModelBuilder::new(model_id, arch.clone())
                .with_paged_attn(|| paged_attn_cfg.build())?
                .with_logging();
            if args.insitu {
                builder = builder.with_isq(quant);
            } else {
                builder = builder.from_uqff(format!("{stem}-{quant}.uqff").into());
            }

            builder.build().await?
        }
    };

    info!("Model loaded.");

    interactive::launch_interactive_mode(model, false).await?;

    Ok(())
}
