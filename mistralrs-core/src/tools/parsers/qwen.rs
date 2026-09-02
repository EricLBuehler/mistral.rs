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
            qwen_tool_call_lark(tools, false),
            tools,
            "arguments",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, true),
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

fn qwen_tool_call_lark(tools: &[Tool], include_wrapper: bool) -> String {
    // Parallel calls arrive as consecutive <tool_call> blocks, so the grammar stays open for more
    let start = if include_wrapper {
        r#"start: "<tool_call>" (json_call | xml_call) ("\n"? "<tool_call>" (json_call | xml_call))*"#
    } else {
        r#"start: (json_call | xml_call) ("\n"? <tool_call> (json_call | xml_call))*"#
    };
    let json_call = if include_wrapper {
        r#"json_call: @json_body "</tool_call>""#
    } else {
        "json_call: @json_body </tool_call>"
    };
    let xml_end = if include_wrapper {
        r#""</tool_call>""#
    } else {
        "</tool_call>"
    };

    format!(
        r#"{start}
{json_call}
xml_call: "\n"? xml_function ("\n"? xml_function)* "\n"? {xml_end}
{}
xml_param_value: (xml_param_text | xml_param_lt)*
xml_param_text: /[^<]+/
xml_param_lt: {}
{}"#,
        qwen_xml_function_rules(tools),
        xml_param_lt_rule(),
        qwen_xml_generic_rules(tools),
    )
}

// Any `<` that does not open the closing `</parameter>` tag may appear inside a value (HTML, XML, code).
fn xml_param_lt_rule() -> String {
    const CLOSE: &str = "/parameter>";
    let mut alternatives = vec![r#""<" /[^\/]/"#.to_string()];
    for (idx, ch) in CLOSE.chars().enumerate().skip(1) {
        let prefix = &CLOSE[..idx];
        let escaped = if ch == '/' {
            "\\/".to_string()
        } else {
            ch.to_string()
        };
        alternatives.push(format!(r#""<{prefix}" /[^{escaped}]/"#));
    }
    alternatives.join(" | ")
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
        Regex::new(r"(?s)<parameter=(?P<key>[^>\n]+)>(?P<value>.*?)</parameter>").unwrap()
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

// Values are kept verbatim minus the template's single framing newlines; types come from the tool schema later
fn qwen_xml_param_value(raw: &str) -> Value {
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    Value::String(value.to_string())
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

    // Qwen does not always emit parameters in schema order, so accept any order (llama.cpp permutes too)
    let mut parts = required_pairs;
    parts.extend(optional_pairs);

    if parts.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: ({})*", parts.join(" | ")));
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
    use crate::tools::parsers::{specialize_required_tool_call_grammar, ToolFormatParser};
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct GreedyTokenizerEnv {
        trie: toktrie::TokTrie,
    }

    impl toktrie::TokenizerEnv for GreedyTokenizerEnv {
        fn tok_trie(&self) -> &toktrie::TokTrie {
            &self.trie
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<toktrie::TokenId> {
            self.trie.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            false
        }
    }

    fn wrapper_token_trie() -> toktrie::TokTrie {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        tokens.push(b"\xff<eos>".to_vec());
        tokens.push(b"\xff<tool_call>".to_vec());
        tokens.push(b"\xff</tool_call>".to_vec());
        let vocab_size = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens)
    }

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
        // Raw text: the matcher applies the tool schema types afterwards
        assert_eq!(
            value[0]["arguments"]["locations"],
            r#"[{"country":"France","city":"Paris"}]"#
        );
        assert_eq!(value[0]["arguments"]["temp_units"], "celsius");
    }

    #[test]
    fn qwen_xml_values_keep_inner_whitespace() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>\n<function=write_file>\n<parameter=content>\n    indented\n\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["arguments"]["content"], "    indented\n");
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

    #[test]
    fn required_grammar_uses_tokenizer_special_wrappers() {
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let mut grammar = QwenParser.required_tool_call_grammar(&[tool]);
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar.clone()).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        assert!(!matcher.compute_mask().unwrap().is_allowed(start_token));

        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());

        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("start: <tool_call> (json_call | xml_call)"));
        assert!(lark.contains("json_call: @json_body </tool_call>"));

        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        let mask = matcher.compute_mask().unwrap();

        assert!(mask.is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }

    #[test]
    fn continuation_grammar_allows_eos_or_another_call_after_a_block() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
            "required": ["city"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
            },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let eos = trie.eos_token();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        // Parameters in non-schema order, a `</` inside a value
        let body = "\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n<parameter=city>\nParis </b>\n</parameter>\n</function>\n";
        for token in trie.greedy_tokenize(body.as_bytes()) {
            assert!(
                matcher.compute_mask().unwrap().is_allowed(token),
                "rejected byte {token}"
            );
            matcher.consume_token(token).unwrap();
        }
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));
        matcher.consume_token(end_token).unwrap();

        let mask = matcher.compute_mask().unwrap();
        assert!(
            mask.is_allowed(eos),
            "EOS must close the turn after a complete call"
        );
        let newline = trie.greedy_tokenize(b"\n")[0];
        assert!(mask.is_allowed(newline));
        matcher.consume_token(newline).unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }
}
