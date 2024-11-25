use std::fs::read_to_string;

use anyhow::Result;
use clap::Parser;
use mistralrs::{
    cross_entropy_loss, parse_isq_value, DType, Device, PagedAttentionMetaBuilder, Tensor,
    TextMessageRole, TextMessages, TextModelBuilder,
};

/// Calculate perplexity of a model. By default, this uses the Llama 3.1 8B model.
#[derive(Parser)]
struct Args {
    /// The model to run.
    #[arg(short, long, default_value = "meta-llama/Llama-3.1-8B-Instruct")]
    model_id: String,

    /// Filename to text to run the model on. This is recommended to be the Wikitext 2 dataset:
    /// https://huggingface.co/datasets/EricB/wikitext2
    #[arg(short, long)]
    file: String,

    /// ISQ quantization to run with.
    #[arg(short, long)]
    isq: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let quant = if let Some(isq) = &args.isq {
        Some(parse_isq_value(isq).map_err(anyhow::Error::msg)?)
    } else {
        None
    };

    let mut model_builder = TextModelBuilder::new(&args.model_id)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?;
    if let Some(quant) = quant {
        model_builder = model_builder.with_isq(quant);
    }
    let model = model_builder.build().await?;

    let messages =
        TextMessages::new().add_message(TextMessageRole::User, read_to_string(&args.file)?);

    let (logits, tokens) = model.send_raw_chat_request(messages).await?;

    // Upcast to float if we need to compute the loss to avoid potential precision issues
    let logits = logits.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    // Shift so that tokens < n predict n
    let shift_logits = logits.narrow(0, 0, logits.dim(0)? - 1)?.contiguous()?;
    let shift_labels = Tensor::from_slice(&tokens[1..], (tokens.len() - 1,), &Device::Cpu)?;

    let loss_fct = cross_entropy_loss(&shift_logits, &shift_labels)?;
    let perplexity = loss_fct.exp()?.to_scalar::<f32>()?;
    println!(
        "Perplexity for `{}`, ISQ `{:?}`: {perplexity}",
        args.file, quant
    );

    Ok(())
}
