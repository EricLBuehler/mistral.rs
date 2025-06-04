use anyhow::Result;
use mistralrs::{
    BertEmbeddingModel, IsqType, RequestBuilder, SearchResult, TextMessageRole, TextMessages,
    TextModelBuilder, WebSearchOptions,
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
    let model = TextModelBuilder::new("NousResearch/Hermes-3-Llama-3.1-8B")
        .with_isq(IsqType::Q4K)
        .with_logging()
        .with_search(BertEmbeddingModel::default())
        .with_search_callback(Arc::new(|p| local_search(&p.query)))
        .build()
        .await?;

    let messages =
        TextMessages::new().add_message(TextMessageRole::User, "Where is Cargo.toml in this repo?");
    let messages = RequestBuilder::from(messages).with_web_search_options(WebSearchOptions {
        search_description: Some("Local filesystem search".to_string()),
        ..Default::default()
    });

    let response = model.send_chat_request(messages).await?;
    println!("{}", response.choices[0].message.content.as_ref().unwrap());
    Ok(())
}
