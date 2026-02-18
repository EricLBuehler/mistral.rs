use anyhow::Result;
use mistralrs::{
    llguidance::api::GrammarWithLexer, IsqBits, LlguidanceGrammar, PagedAttentionMetaBuilder,
    RequestBuilder, TextMessageRole, TextModelBuilder,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    let model = TextModelBuilder::new("microsoft/Phi-3.5-mini-instruct")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
        .build()
        .await?;

    let top =
        GrammarWithLexer::from_lark(r#"start: "Reasoning: " /.+/ "\nJSON: " @myobj"#.to_string());
    let schema = GrammarWithLexer {
        name: Some("myobj".to_string()),
        json_schema: Some(json!({
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No"]},
            },
            "required": ["answer"],
            "additionalProperties": false,
        })),
        ..Default::default()
    };

    let request = RequestBuilder::new()
        .set_constraint(mistralrs::Constraint::Llguidance(LlguidanceGrammar {
            grammars: vec![top, schema],
            max_tokens: None,
        }))
        .set_sampler_max_len(100)
        .add_message(
            TextMessageRole::User,
            "If all dogs are mammals, and all mammals are animals, are dogs animals?",
        );

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
