use std::collections::HashMap;
use std::sync::Arc;

pub mod rag;

use anyhow::Result;
use html2text::{config, render::PlainDecorator};
use rayon::prelude::*;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env::consts::{ARCH, FAMILY, OS};
use tokenizers::Tokenizer;

use crate::{Function, Tool, ToolType, WebSearchOptions, WebSearchUserLocation};

/// Callback used to override how search results are gathered. The returned
/// vector must be sorted in decreasing order of relevance.
pub type SearchCallback =
    dyn Fn(&SearchFunctionParameters) -> Result<Vec<SearchResult>> + Send + Sync;

pub(crate) fn search_tool_called(name: &str) -> bool {
    name == SEARCH_TOOL_NAME || name == EXTRACT_TOOL_NAME
}

pub(crate) const SEARCH_TOOL_NAME: &str = "search_the_web";
pub(crate) const EXTRACT_TOOL_NAME: &str = "website_content_extractor";

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
pub(crate) const SEARCH_DESCRIPTION: &str = r#"This tool is used to search the web given a query.
If the user wants up-to-date information or you want to retrieve new information, call this tool.
If you call this tool, then you MUST complete your answer using the output.
The input can be a query. It should not be a URL. Either is fine.
Additionally, if you have any questions that require a follow-up, you can call this tool repeatedly.

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
"#;
pub(crate) const EXTRACT_DESCRIPTION: &str = r#"This tool is used to extract the content of a website.
If the user wants information about a specific site or you want to extract the content of a specific site, call this tool.
The input must be a URL.
Additionally, if you have any questions that require a follow-up, you can call this tool repeatedly.

You should expect output like this:
{
    "output": [
        {
            "url": "...",
            "content": "...",
        },
        ...
    ]
}
"#;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct SearchResult {
    pub title: String,
    pub description: String,
    pub url: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ExtractResult {
    pub url: String,
    pub content: String,
}

impl SearchResult {
    pub fn cap_content_len(self, tokenizer: &Tokenizer, size: usize) -> Result<Self> {
        let tokenized_content = tokenizer
            .encode_fast(self.content, false)
            .map_err(anyhow::Error::msg)?;
        let ids = tokenized_content.get_ids();
        let content = tokenizer
            .decode(&ids[..size.min(ids.len())], false)
            .map_err(anyhow::Error::msg)?;

        Ok(Self {
            title: self.title,
            description: self.description,
            url: self.url,
            content,
        })
    }
}

impl ExtractResult {
    pub fn cap_content_len(self, tokenizer: &Tokenizer, size: usize) -> Result<Self> {
        let tokenized_content = tokenizer
            .encode_fast(self.content, false)
            .map_err(anyhow::Error::msg)?;
        let ids = tokenized_content.get_ids();
        let content = tokenizer
            .decode(&ids[..size.min(ids.len())], false)
            .map_err(anyhow::Error::msg)?;

        Ok(Self {
            url: self.url,
            content,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchFunctionParameters {
    pub query: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtractFunctionParameters {
    pub url: String,
}

pub fn get_search_tools(web_search_options: &WebSearchOptions) -> Result<Vec<Tool>> {
    let search_tool = {
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
        let description = web_search_options
            .search_description
            .as_deref()
            .unwrap_or(SEARCH_DESCRIPTION);
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some(format!("{description}{location_details}")),
                name: SEARCH_TOOL_NAME.to_string(),
                parameters: Some(parameters),
            },
        }
    };

    let extract_tool = {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "A URL to extract the content of the website from.",
                },
            },
            "required": ["url"],
        }))?;

        let description = web_search_options
            .extract_description
            .as_deref()
            .unwrap_or(EXTRACT_DESCRIPTION);
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some(description.to_string()),
                name: EXTRACT_TOOL_NAME.to_string(),
                parameters: Some(parameters),
            },
        }
    };

    Ok(vec![search_tool, extract_tool])
}

pub fn run_search_tool(params: &SearchFunctionParameters) -> Result<Vec<SearchResult>> {
    let client = reqwest::blocking::Client::new();

    let encoded_query = urlencoding::encode(&params.query);
    let url = format!("https://html.duckduckgo.com/html/?q={encoded_query}");

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

    // Phase 1: collect title, description, and url serially into a Vec of tuples
    let partials: Vec<(String, String, String)> = document
        .select(&result_selector)
        .filter_map(|element| {
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
            if title.is_empty() || description.is_empty() || url.is_empty() {
                return None;
            }
            if !url.starts_with("http") {
                url = format!("https://{url}");
            }
            Some((title, description, url))
        })
        .collect();

    // Phase 2: fetch content in parallel using Rayon
    let client = Arc::new(client);
    let results: Vec<SearchResult> = partials
        .into_par_iter()
        .filter_map(|(title, description, url)| {
            let content = match client.get(&url).header("User-Agent", &user_agent).send() {
                Ok(response) => {
                    let html = response.text().ok()?;
                    config::with_decorator(PlainDecorator::new())
                        .do_decorate()
                        .string_from_read(html.as_bytes(), 80)
                        .ok()?
                }
                Err(_) => return None,
            };
            Some(SearchResult {
                title,
                description,
                url,
                content,
            })
        })
        .collect();

    Ok(results)
}

pub fn run_extract_tool(params: &ExtractFunctionParameters) -> Result<ExtractResult> {
    let client = reqwest::blocking::Client::new();

    let user_agent = format!("mistralrs/{APP_VERSION} ({OS}; {ARCH}; {FAMILY})");

    let content = match client
        .get(&params.url)
        .header("User-Agent", &user_agent)
        .send()
    {
        Ok(response) => response.text().ok().and_then(|html| {
            config::with_decorator(PlainDecorator::new())
                .do_decorate()
                .string_from_read(html.as_bytes(), 80)
                .ok()
        }),
        Err(_) => None,
    };
    Ok(ExtractResult {
        url: params.url.clone(),
        content: content.unwrap_or("ERROR: failed to extract content".to_string()),
    })
}
