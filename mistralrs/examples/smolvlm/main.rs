use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

const MODEL_ID: &str = "HuggingFaceTB/SmolVLM-Instruct";

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new(MODEL_ID)
        .with_isq(IsqType::Q8_0)
        .with_logging()
        .build()
        .await?;

    let bytes = match reqwest::blocking::get(
        "https://cdn.britannica.com/45/5645-050-B9EC0205/head-treasure-flower-disk-flowers-inflorescence-ray.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new().add_image_message(
        TextMessageRole::User,
        "What is this?",
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
