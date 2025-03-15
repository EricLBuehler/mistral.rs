use anyhow::Result;
use mistralrs::{
    llguidance::api::GrammarWithLexer, GgufModelBuilder, IsqType, LlguidanceGrammar,
    PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole, TextModelBuilder,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    // let model = GgufModelBuilder::new(
    //     "gguf_models/",
    //     vec!["Llama-3.2-3B-Instruct-uncensored-Q4_K_M.gguf"],
    // )
    // // .with_chat_template("chat_templates/llama3.json")
    // .with_logging()
    // // .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
    // .build()
    // .await?;

    let top = GrammarWithLexer::from_lark(r#"@myobj"#.to_string());
    let schema = GrammarWithLexer {
        name: Some("myobj".to_string()),
        json_schema: Some(json!(
        {
        "type": "object",
        "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"},
        "state": {"type": "string", "pattern": "^[A-Z]{2}$"},
        "zip": {"type": "integer", "minimum": 10000, "maximum": 99999},
        },
        "required": ["street", "city", "state", "zip"],
        "additionalProperties": false,
        }
        )),
        ..Default::default()
    };

    let request = RequestBuilder::new()
        .set_constraint(mistralrs::Constraint::Llguidance(LlguidanceGrammar {
            grammars: vec![schema],
            max_tokens: None,
            test_trace: false,
        }))
        .set_sampler_n_choices(1)
        .set_sampler_max_len(100)
        .add_message(TextMessageRole::User, "A sample address please.");

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
