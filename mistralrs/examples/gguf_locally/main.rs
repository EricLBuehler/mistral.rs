use anyhow::Result;
use mistralrs::{
    GgufModelBuilder, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextMessages,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // We do not use any files from remote servers here, and instead load the
    // chat template from the specified file, and the tokenizer and model from a
    // local GGUF file at the path specified.
    let model = GgufModelBuilder::new("gguf_models/", vec!["Llama-3.2-3B-Instruct-Q4_K_M.gguf"])
        // .with_chat_template("chat_templates/llama3.json")
        .with_logging()
        // .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let request = RequestBuilder::new()
        .set_constraint(mistralrs::Constraint::JsonSchema(json!(
            {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
                },
                "required": ["street", "city", "state"],
                "additionalProperties": false,
            }
        )))
        .set_sampler_max_len(100)
        .add_message(TextMessageRole::User, "A sample address please.");

    let response = model.send_chat_request(request).await?;

    println!(
        "{}",
        response.choices[0].message.content.as_ref().unwrap().trim()
    );
    dbg!(
        response.usage.avg_prompt_tok_per_sec,
        response.usage.avg_compl_tok_per_sec
    );

    // Next example: Return some logprobs with the `RequestBuilder`, which enables higher configurability.
    let request = RequestBuilder::new().return_logprobs(true).add_message(
        TextMessageRole::User,
        "Please write a mathematical equation where a few numbers are added.",
    );

    let response = model.send_chat_request(request).await?;

    println!(
        "Logprobs: {:?}",
        &response.choices[0]
            .logprobs
            .as_ref()
            .unwrap()
            .content
            .as_ref()
            .unwrap()[0..3]
    );

    Ok(())
}
