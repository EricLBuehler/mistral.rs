use anyhow::Result;
use mistralrs::{
    RequestBuilder, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder,
};

const MODEL_ID: &str = "meta-llama/Llama-3.2-11B-Vision-Instruct";

#[tokio::main]
async fn main() -> Result<()> {
    let model = VisionModelBuilder::new(MODEL_ID, VisionLoaderType::VLlama)
        .with_logging()
        .build()
        .await?;

    let bytes = match reqwest::blocking::get(
        "https://s3.amazonaws.com/cdn.tulips.com/images/large/Timeless-Tulip.jpg",
    ) {
        Ok(http_resp) => http_resp.bytes()?.to_vec(),
        Err(e) => anyhow::bail!(e),
    };
    let image1 = image::load_from_memory(&bytes)?;

    let bytes = match reqwest::blocking::get(
            "https://www.nhmagazine.com/content/uploads/2019/05/mtwashingtonFranconia-2-19-18-108-Edit-Edit.jpg"
        ) {
            Ok(http_resp) => http_resp.bytes()?.to_vec(),
            Err(e) => anyhow::bail!(e),
        };
    let image2 = image::load_from_memory(&bytes)?;

    let messages = VisionMessages::new()
        .add_message(TextMessageRole::User, "Hello!")
        .add_message(TextMessageRole::Assistant, "How can I assist you today?")
        .add_vllama_image_message(TextMessageRole::User, "What is this?", image1)
        .add_message(
            TextMessageRole::Assistant,
            "The picture shown appears to be a picture of a tulip.",
        )
        .add_vllama_image_message(TextMessageRole::User, "What is this?", image2);
    let messages = RequestBuilder::from(messages)
        .set_sampler_max_len(75)
        .set_sampler_topk(50)
        .set_sampler_temperature(0.);

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
