use std::collections::HashMap;

use anyhow::Result;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env::consts::{ARCH, FAMILY, OS};

use crate::{Function, Tool, ToolType};

pub(crate) const SEARCH_TOOL_NAME: &str = "search_the_web";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    title: String,
    description: String,
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFunctionParamters {
    query: String,
}

pub fn get_search_tool() -> Result<Tool> {
    let parameters: HashMap<String, Value> = serde_json::from_value(json!({
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A query for web searching.",
            },
        },
        "required": ["query"],
    }))?;

    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some("Search the web using a given query. Use this tool if the user asks a question which requires realtime or up-to-date information.".to_string()),
            name: SEARCH_TOOL_NAME.to_string(),
            parameters: Some(parameters),
        },
    })
}

pub fn run_search_tool(params: SearchFunctionParamters) -> Result<Vec<SearchResult>> {
    let client = reqwest::blocking::Client::new();

    let encoded_query = urlencoding::encode(&params.query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

    let response = client
        .get(&url)
        .header(
            "User-Agent",
            format!("mistralrs/{APP_VERSION} ({OS}; {ARCH}; {FAMILY})"),
        )
        .send()?;

    // Check the response status
    if !response.status().is_success() {
        anyhow::bail!("Failed to fetch search results: {}", response.status())
    }

    let html = response.text()?;

    let document = Html::parse_document(&html);

    let result_selector = Selector::parse(".result").unwrap();
    let title_selector = Selector::parse(".result__title").unwrap();
    let snippet_selector = Selector::parse(".result__snippet").unwrap();
    let url_selector = Selector::parse(".result__url").unwrap();

    let mut results = Vec::new();

    for element in document.select(&result_selector) {
        let title = element
            .select(&title_selector)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let description = element
            .select(&snippet_selector)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let mut url = element
            .select(&url_selector)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        if !title.is_empty() && !description.is_empty() && !url.is_empty() {
            if !url.starts_with("http") {
                url = format!("https://{}", url);
            }

            results.push(SearchResult {
                title,
                description,
                url,
            });
        }
    }

    Ok(results)
}
