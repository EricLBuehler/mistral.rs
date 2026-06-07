//! Qwen tool call parser.
//!
//! Formats:
//! `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`

use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();
static QWEN_FUNCTION_REGEX: OnceLock<Regex> = OnceLock::new();
static QWEN_PARAMETER_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            format!(
                r#"start: json_call | xml_call
json_call: @json_body </tool_call>
xml_call: "\n"? xml_function ("\n"? xml_function)* "\n"? </tool_call>
{}
xml_param_value: (xml_param_text | xml_param_lt)*
xml_param_text: /[^<]+/
xml_param_lt: "<" /[^\/]/
{}"#,
                qwen_xml_function_rules(tools),
                qwen_xml_generic_rules(tools),
            ),
            tools,
            "arguments",
            false,
        )
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let re = QWEN_REGEX
            .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

        if !re.is_match(message) {
            Ok(None)
        } else {
            parse_qwen_tool_calls(message)
        }
    }
}

#[derive(serde::Serialize)]
struct QwenToolCall {
    name: String,
    arguments: Value,
}

fn parse_qwen_tool_calls(message: &str) -> Result<Option<String>> {
    let re = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    let mut xml_calls = Vec::new();
    let mut json_calls = Vec::new();

    for caps in re.captures_iter(message) {
        let inner = caps.name("inner").unwrap().as_str().trim();
        if inner.is_empty() {
            continue;
        }

        let parsed_xml = parse_qwen_xml_tool_call(inner)?;
        if !parsed_xml.is_empty() {
            xml_calls.extend(parsed_xml);
            continue;
        }

        match serde_json::from_str::<Value>(inner) {
            Ok(value) => json_calls.push(value),
            Err(_) => return Ok(None),
        }
    }

    if !xml_calls.is_empty() {
        return Ok(Some(
            serde_json::to_string(&xml_calls).map_err(candle_core::Error::msg)?,
        ));
    }

    match json_calls.len() {
        0 => Ok(None),
        1 => Ok(Some(
            serde_json::to_string(&json_calls[0]).map_err(candle_core::Error::msg)?,
        )),
        _ => Ok(Some(
            serde_json::to_string(&json_calls).map_err(candle_core::Error::msg)?,
        )),
    }
}

fn parse_qwen_xml_tool_call(inner: &str) -> Result<Vec<QwenToolCall>> {
    let function_re = QWEN_FUNCTION_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>").unwrap()
    });
    let parameter_re = QWEN_PARAMETER_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<parameter=(?P<key>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>").unwrap()
    });

    let mut calls = Vec::new();
    for caps in function_re.captures_iter(inner) {
        let name = caps.name("name").unwrap().as_str().trim().to_string();
        let body = caps.name("body").unwrap().as_str();
        let mut arguments = Map::new();
        for param_caps in parameter_re.captures_iter(body) {
            let key = param_caps.name("key").unwrap().as_str().trim().to_string();
            let value = param_caps.name("value").unwrap().as_str();
            arguments.insert(key, qwen_xml_param_value(value));
        }
        calls.push(QwenToolCall {
            name,
            arguments: Value::Object(arguments),
        });
    }
    Ok(calls)
}

fn qwen_xml_param_value(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        Value::String(String::new())
    } else if let Ok(value) = serde_json::from_str(trimmed) {
        value
    } else {
        Value::String(trimmed.to_string())
    }
}

fn qwen_xml_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();

    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("qwen_xml_tool_{tool_idx}");
        let args = qwen_xml_args_rule(tool_idx, tool, &mut rules);
        let opener = lark_string(&format!("<function={}>", tool.function.name));
        rules.push(format!(
            "{branch}: {opener} \"\\n\"? {args} \"</function>\""
        ));
        branches.push(branch);
    }

    if branches.is_empty() {
        rules.push("xml_function: qwen_xml_generic_function".to_string());
    } else {
        rules.push(format!("xml_function: {}", branches.join(" | ")));
    }

    rules.join("\n")
}

fn qwen_xml_args_rule(tool_idx: usize, tool: &Tool, rules: &mut Vec<String>) -> String {
    let args_rule = format!("qwen_xml_args_{tool_idx}");
    let Some(parameters) = tool.function.parameters.as_ref() else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };

    let required = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();

    let mut required_pairs = Vec::new();
    let mut optional_pairs = Vec::new();
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();

    for (prop_idx, name) in property_names.iter().enumerate() {
        let pair = format!("qwen_xml_arg_{tool_idx}_{prop_idx}");
        let opener = lark_string(&format!("<parameter={name}>"));
        rules.push(format!(
            "{pair}: {opener} \"\\n\"? xml_param_value \"</parameter>\" \"\\n\"?"
        ));
        if required.contains(name.as_str()) {
            required_pairs.push(pair);
        } else {
            optional_pairs.push(pair);
        }
    }

    let mut parts = required_pairs;
    parts.extend(optional_pairs.into_iter().map(|pair| format!("{pair}?")));

    if parts.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: {}", parts.join(" ")));
    }

    args_rule
}

fn qwen_xml_generic_rules(tools: &[Tool]) -> &'static str {
    if tools.iter().any(|tool| tool.function.parameters.is_none()) {
        r#"qwen_xml_generic_function: "<function=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? qwen_xml_generic_params "</function>"
qwen_xml_generic_params: (qwen_xml_generic_param "\n"?)*
qwen_xml_generic_param: "<parameter=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? xml_param_value "</parameter>""#
    } else {
        ""
    }
}

fn lark_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::{parse_qwen_tool_calls, QwenParser};
    use crate::tools::parsers::ToolFormatParser;
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
    use serde_json::Value;
    use std::collections::HashMap;

    #[test]
    fn parses_qwen_json_tool_call() {
        let parsed = QwenParser
            .parse(r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#)
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["arguments"]["city"], "Paris");
    }

    #[test]
    fn parses_qwen_xml_tool_call() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=locations>
[{"country":"France","city":"Paris"}]
</parameter>
<parameter=temp_units>
celsius
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "get_weather");
        assert_eq!(value[0]["arguments"]["locations"][0]["city"], "Paris");
        assert_eq!(value[0]["arguments"]["temp_units"], "celsius");
    }

    #[test]
    fn parses_multiple_qwen_xml_tool_calls() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call><tool_call>
<function=get_time>
<parameter=timezone>Asia/Tokyo</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_eq!(value[0]["arguments"]["city"], "Tokyo");
        assert_eq!(value[1]["arguments"]["timezone"], "Asia/Tokyo");
    }

    #[test]
    fn parses_qwen_xml_code_with_less_than() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=mistralrs_execute_python>
<parameter=code>
print(1 < 2)
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "mistralrs_execute_python");
        assert_eq!(value[0]["arguments"]["code"], "print(1 < 2)");
    }

    #[test]
    fn qwen_xml_grammar_requires_code_without_forcing_newline_before_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string" },
                "outputs": { "type": "array" }
            },
            "required": ["code"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "mistralrs_execute_python".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: Some(true),
            },
        };

        let grammar = QwenParser.tool_call_grammar(&[tool], "");
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("\"<parameter=code>\" \"\\n\"? xml_param_value \"</parameter>\""));
        assert!(!lark.contains("\"\\n</parameter>\""));
    }
}
