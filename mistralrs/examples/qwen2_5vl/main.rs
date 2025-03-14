use anyhow::Result;
use clap::Parser;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};
use mistralrs_core::initialize_logging;
use tokio::task;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "Qwen/Qwen2.5-VL-3B-Instruct")]
    model_id: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    initialize_logging();
    let args = Args::parse();

    let bytes = task::spawn_blocking(|| {
        match reqwest::blocking::get(
            "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg",
        ) {
            Ok(http_resp) => Ok(http_resp.bytes()?.to_vec()),
            Err(e) => anyhow::bail!(e),
        }
    }).await??;

    let image = image::load_from_memory(&bytes)?;

    let model = VisionModelBuilder::new(args.model_id, VisionLoaderType::Qwen2_5VL)
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .build()
        .await?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is depicted here?",
        image,
        &model,
    )?;

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
