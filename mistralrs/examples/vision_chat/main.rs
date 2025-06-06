use anyhow::Result;
use clap::Parser;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[derive(Parser)]
struct Args {
    #[clap(long)]
    model_id: String,
    #[clap(
        long,
        default_value = "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg"
    )]
    image_url: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let model = VisionModelBuilder::new(&args.model_id)
        .with_isq(IsqType::Q4K)
        .with_logging()
        .build()
        .await?;

    let bytes = reqwest::blocking::get(&args.image_url)?.bytes()?.to_vec();
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is depicted here? Please describe the scene in detail.",
        vec![image],
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
