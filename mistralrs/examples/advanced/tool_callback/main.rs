//! Tool callbacks for automatic server-side tool execution.
//!
//! Run with: `cargo run --release --example tool_callback -p mistralrs`

use anyhow::Result;
use mistralrs::{
    CalledFunction, IsqBits, ModelBuilder, RequestBuilder, SearchResult, TextMessageRole,
    TextMessages, Tool, ToolChoice, ToolType,
};
use std::fs;
use std::sync::Arc;
use walkdir::WalkDir;

fn local_search(query: &str) -> Result<Vec<SearchResult>> {
    let mut results = Vec::new();
    for entry in WalkDir::new(".") {
        let entry = entry?;
        if entry.file_type().is_file() {
            let name = entry.file_name().to_string_lossy();
            if name.contains(query) {
                let path = entry.path().display().to_string();
                let content = fs::read_to_string(entry.path()).unwrap_or_default();
                results.push(SearchResult {
                    title: name.into_owned(),
                    description: path.clone(),
                    url: path,
                    content,
                });
            }
        }
    }
    results.sort_by_key(|r| r.title.clone());
    results.reverse();
    Ok(results)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build the model and register the *tool callback*.
    let model = ModelBuilder::new("google/gemma-3-4b-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_tool_callback(
            "local_search",
            Arc::new(|f: &CalledFunction| {
                let args: serde_json::Value = serde_json::from_str(&f.arguments)?;
                let query = args["query"].as_str().unwrap_or("");
                Ok(serde_json::to_string(&local_search(query)?)?)
            }),
        )
        .build()
        .await?;

    // Define the JSON schema for the tool the model can call.
    let parameters = std::collections::HashMap::from([(
        "query".to_string(),
        serde_json::json!({"type": "string", "description": "Query"}),
    )]);
    let tool = Tool {
        tp: ToolType::Function,
        function: mistralrs::Function {
            description: Some("Local filesystem search".to_string()),
            name: "local_search".to_string(),
            parameters: Some(parameters),
        },
    };

    // Ask the user question and allow the model to call the tool automatically.
    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Where is Cargo.toml in this repo?");
    let messages = RequestBuilder::from(messages)
        .set_tools(vec![tool])
        .set_tool_choice(ToolChoice::Auto);

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
