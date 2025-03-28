use std::collections::HashMap;

pub mod rag;

use anyhow::Result;
use html2text::{config, render::PlainDecorator};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env::consts::{ARCH, FAMILY, OS};

use crate::{Function, Tool, ToolType, WebSearchOptions, WebSearchUserLocation};

pub(crate) const SEARCH_TOOL_NAME: &str = "search_the_web";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const DESCRIPTION: &str = r#"This tool is used to search the web given a query. If you call this tool, then you MUST complete your answer using the output.
You should expect output like this:
{
    "output": [
        {
            "title": "...",
            "description": "...",
            "url": "...",
            "content": "...",
        },
        ...
    ]
}
YOU SHOULD NOT CALL THE SEARCH TOOL CONSECUTIVELY!"#;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct SearchResult {
    pub title: String,
    pub description: String,
    pub url: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFunctionParameters {
    pub query: String,
}

pub fn get_search_tool(web_search_options: &WebSearchOptions) -> Result<Tool> {
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

    let location_details = match &web_search_options.user_location {
        Some(WebSearchUserLocation::Approximate { approximate }) => {
            format!(
                "\nThe user's location is: {}, {}, {}, {}.",
                approximate.city, approximate.region, approximate.country, approximate.timezone
            )
        }
        None => "".to_string(),
    };

    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(format!("{DESCRIPTION}{location_details}")),
            name: SEARCH_TOOL_NAME.to_string(),
            parameters: Some(parameters),
        },
    })
}

pub fn run_search_tool(params: &SearchFunctionParameters) -> Result<Vec<SearchResult>> {
    let client = reqwest::blocking::Client::new();

    let encoded_query = urlencoding::encode(&params.query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded_query);

    let user_agent = format!("mistralrs/{APP_VERSION} ({OS}; {ARCH}; {FAMILY})");
    let response = client.get(&url).header("User-Agent", &user_agent).send()?;

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

            let content = match client.get(&url).header("User-Agent", &user_agent).send() {
                Ok(response) => {
                    let html = response.text()?;

                    config::with_decorator(PlainDecorator::new())
                        .do_decorate()
                        .string_from_read(html.as_bytes(), 80)?
                }
                Err(_) => "".to_string(),
            };

            results.push(SearchResult {
                title,
                description,
                url,
                content,
            });
        }
    }

    Ok(results)
}
