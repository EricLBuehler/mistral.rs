/// Unified vision model example.
///
/// Change `MODEL_ID` to run any supported vision model. Tested model IDs:
///
/// | Model                        | MODEL_ID                                                |
/// |------------------------------|---------------------------------------------------------|
/// | Llama 3.2 Vision             | `lamm-mit/Cephalo-Llama-3.2-11B-Vision-Instruct-128k`  |
/// | Phi-3.5 Vision               | `microsoft/Phi-3.5-vision-instruct`                     |
/// | Phi-4 Multimodal             | `microsoft/Phi-4-multimodal-instruct`                   |
/// | LLaVA 1.5 *                  | `llava-hf/llava-1.5-7b-hf`                              |
/// | LLaVA-NeXT                   | `llava-hf/llava-v1.6-mistral-7b-hf`                     |
/// | Idefics2                     | `HuggingFaceM4/idefics2-8b-chatty`                      |
/// | Idefics3                     | `HuggingFaceM4/Idefics3-8B-Llama3`                      |
/// | Qwen2-VL                     | `Qwen/Qwen2-VL-2B-Instruct`                             |
/// | Qwen2.5-VL                   | `Qwen/Qwen2.5-VL-3B-Instruct`                           |
/// | Qwen3-VL                     | `Qwen/Qwen3-VL-4B-Instruct`                             |
/// | SmolVLM                      | `HuggingFaceTB/SmolVLM-Instruct`                        |
/// | Gemma 3                      | `google/gemma-3-4b-it`                                   |
/// | Gemma 3n                     | `google/gemma-3n-E4B-it`                                 |
/// | MiniCPM-o 2.6                | `openbmb/MiniCPM-o-2_6`                                 |
/// | Mistral Small 3.1            | `mistralai/Mistral-Small-3.1-24B-Instruct-2503`         |
/// | Llama 4 Scout                | `meta-llama/Llama-4-Scout-17B-16E-Instruct`             |
///
/// * LLaVA 1.5 requires `.with_chat_template("chat_templates/vicuna.json")`.
///
/// Run with: `cargo run --release --example vision_models -p mistralrs`
use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, TextMessageRole, VisionMessages};

const MODEL_ID: &str = "google/gemma-3-4b-it";

// For LLaVA 1.5, uncomment the following and add .with_chat_template() below:
// const MODEL_ID: &str = "llava-hf/llava-1.5-7b-hf";

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new(MODEL_ID)
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        // Uncomment for LLaVA 1.5:
        // .with_chat_template("chat_templates/vicuna.json")
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
    );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    Ok(())
}
