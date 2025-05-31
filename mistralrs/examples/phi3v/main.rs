use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new("microsoft/Phi-3.5-vision-instruct")
        .with_isq(IsqType::Q4K)
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
