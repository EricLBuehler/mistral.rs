/// Unified text model example.
///
/// Change `MODEL_ID` to run any supported text model. Tested model IDs:
///
/// | Model                        | MODEL_ID                                    |
/// |------------------------------|---------------------------------------------|
/// | Phi-3.5 Mini                 | `microsoft/Phi-3.5-mini-instruct`           |
/// | Phi-3.5 MoE                  | `microsoft/Phi-3.5-MoE-instruct`            |
/// | Gemma 2                      | `google/gemma-2-9b-it`                      |
/// | DeepSeek-R1                  | `deepseek-ai/DeepSeek-R1`                   |
/// | DeepSeek-V2-Lite             | `deepseek-ai/DeepSeek-V2-Lite`              |
/// | SmolLM3                      | `HuggingFaceTB/SmolLM3-3B`                  |
/// | GPT-OSS                      | `openai/gpt-oss-20b`                        |
/// | Granite                      | `ibm-granite/granite-4.0-tiny-preview`      |
/// | GLM-4 MoE                    | `zai-org/GLM-4.7-Flash`                     |
/// | Qwen3 (thinking mode)        | `Qwen/Qwen3-30B-A3B`                        |
/// | Llama 3.3                    | `meta-llama/Llama-3.3-70B-Instruct`         |
///
/// Run with: `cargo run --release --example text_models -p mistralrs`
use anyhow::Result;
use mistralrs::{IsqBits, ModelBuilder, PagedAttentionMetaBuilder, TextMessageRole, TextMessages};

const MODEL_ID: &str = "Qwen/Qwen3-4B";

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new(MODEL_ID)
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    let messages = TextMessages::new()
        .add_message(
            TextMessageRole::System,
            "You are an AI agent with a specialty in programming.",
        )
        .add_message(
            TextMessageRole::User,
            "Hello! How are you? Please write generic binary search function in Rust.",
        );

    let response = model.send_chat_request(messages).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    // ---------------------------------------------------------------
    // Thinking mode (Qwen3, DeepSeek-R1, and other reasoning models)
    //
    // Thinking is enabled by default. You can toggle it with /think
    // and /no_think tags in the user message, or via
    // `TextMessages::enable_thinking(bool)`.
    // ---------------------------------------------------------------
    // Uncomment the block below to demo thinking mode toggling:
    //
    // let mut msgs = TextMessages::new();
    // msgs = msgs.add_message(TextMessageRole::User, "How many rs in strawberry?");
    // let resp = model.send_chat_request(msgs.clone()).await?;
    // println!("{}", resp.choices[0].message.content.as_ref().unwrap());
    //
    // msgs = msgs
    //     .add_message(TextMessageRole::Assistant, resp.choices[0].message.content.as_ref().unwrap())
    //     .add_message(TextMessageRole::User, "How many rs in blueberry? /no_think");
    // let resp = model.send_chat_request(msgs).await?;
    // println!("{}", resp.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
