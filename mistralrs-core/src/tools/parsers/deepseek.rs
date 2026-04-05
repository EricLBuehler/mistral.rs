//! DeepSeek tool call parser.
//!
//! Format:
//! ```text
//! <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
//! ```json
//! {"key": "value"}
//! ```
//! <｜tool▁call▁end｜>
//! ```

use regex::Regex;
use serde_json::Value;
use std::sync::OnceLock;

use super::ToolFormatParser;

static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct DeepSeekParser;

impl ToolFormatParser for DeepSeekParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<｜tool▁call▁begin｜>")
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let re = DEEPSEEK_REGEX.get_or_init(|| {
            Regex::new(
                r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
            )
            .unwrap()
        });

        if !re.is_match(message) {
            return Ok(None);
        }

        #[derive(serde::Serialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }

        let mut calls = Vec::new();
        for caps in re.captures_iter(message) {
            let name = caps
                .name("name")
                .ok_or("Could not capture function name")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim()
                .to_string();
            let json_str = caps
                .name("json")
                .ok_or("Could not capture JSON arguments")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim();
            let arguments: Value =
                serde_json::from_str(json_str).map_err(candle_core::Error::msg)?;
            calls.push(ToolCall { name, arguments });
        }

        Ok(Some(
            serde_json::to_string(&calls).map_err(candle_core::Error::msg)?,
        ))
    }
}
