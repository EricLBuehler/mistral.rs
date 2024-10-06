use anyhow::Result;
use mistralrs::{IsqType, TextMessageRole, VisionLoaderType, VisionMessages, VisionModelBuilder};

#[tokio::main]
async fn main() -> Result<()> {
    let model =
        VisionModelBuilder::new("microsoft/Phi-3.5-vision-instruct", VisionLoaderType::Phi3_5V)
            .with_isq(IsqType::Q4K)
            .with_logging()
            .build()
            .await?;

            let bytes = match reqwest::blocking::get(
                "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg",
            ) {
                Ok(http_resp) => http_resp.bytes()?.to_vec(),
                Err(e) => anyhow::bail!(e),
            };
            let bytes2 = match reqwest::blocking::get(
                "https://d2r55xnwy6nx47.cloudfront.net/uploads/2018/02/Ants_Lede1300.jpg",
            ) {
                Ok(http_resp) => http_resp.bytes()?.to_vec(),
                Err(e) => anyhow::bail!(e),
            };
            let image = image::load_from_memory(&bytes)?;
            let image2 = image::load_from_memory(&bytes2)?;
        
            let messages = VisionMessages::new().add_phiv_images_message(
                TextMessageRole::User,
                "is there a difference in pictures ?",
                vec![image,   image2],
            );
        
            let response = model.send_chat_request(messages).await?;
        
            println!("{}", response.choices[0].message.content.as_ref().unwrap());
            dbg!(
                response.usage.avg_prompt_tok_per_sec,
                response.usage.avg_compl_tok_per_sec
            );
        
            Ok(())
}
