use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde::Serialize;
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

const FUNCTION_CALLS_START: &str = "<atem:function_calls>";
const FUNCTION_CALLS_END: &str = "</atem:function_calls>";
const INVOKE_START: &str = "<atem:invoke";
const INVOKE_END: &str = "</atem:invoke>";
const PARAMETER_START: &str = "<atem:parameter";
const PARAMETER_END: &str = "</atem:parameter>";
const ASSISTANT_START: &str = "<|start|>assistant";
const MESSAGE_START: &str = "<|message|>";
const MESSAGE_END: &str = "<|eom|>";
const TURN_END: &str = "<|eot|>";
const TEXT_END: &str = "<|end_of_text|>";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct AtemToolCall {
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct AtemResponse {
    pub(crate) content: String,
    pub(crate) reasoning: String,
    pub(crate) tool_calls: Vec<AtemToolCall>,
}

pub struct AtemParser;

impl ToolFormatParser for AtemParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        let Some(start) = text.rfind(FUNCTION_CALLS_START) else {
            return false;
        };
        if text[start + FUNCTION_CALLS_START.len()..].contains(FUNCTION_CALLS_END) {
            return false;
        }
        !matches!(channel_recipient(text, start), Some("self" | "user"))
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Atem
    }

    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar {
        let recipient = text
            .rfind(FUNCTION_CALLS_START)
            .and_then(|start| channel_recipient(text, start));
        if let Some(tool) = recipient.and_then(|name| {
            tools
                .iter()
                .find(|tool| tool.function.name.as_str() == name)
        }) {
            build_atem_grammar(std::slice::from_ref(tool), false, "")
        } else {
            build_atem_grammar(tools, false, "")
        }
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        required_tool_call_grammar(tools, "")
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let Some(calls) = parse_atem_tool_calls(message)? else {
            return Ok(None);
        };
        serde_json::to_string(&calls)
            .map(Some)
            .map_err(candle_core::Error::msg)
    }
}

pub(crate) fn parse_atem_response(message: &str) -> Result<AtemResponse> {
    let mut response = AtemResponse {
        tool_calls: parse_atem_tool_calls(message)?.unwrap_or_default(),
        ..Default::default()
    };
    let mut cursor = 0;

    while cursor < message.len() {
        if message[cursor..].starts_with(ASSISTANT_START) {
            cursor += ASSISTANT_START.len();
        }

        let Some(message_offset) = message[cursor..].find(MESSAGE_START) else {
            break;
        };
        let header_end = cursor + message_offset;
        let header = &message[cursor..header_end];
        let body_start = header_end + MESSAGE_START.len();
        let (body_end, delimiter_len, complete) = channel_end(message, body_start);
        let body = if complete {
            &message[body_start..body_end]
        } else {
            trim_partial_control_token(&message[body_start..body_end])
        };
        let recipient = header
            .rsplit_once(" to=")
            .map(|(_, recipient)| recipient.trim());

        match recipient {
            Some("self") => response.reasoning.push_str(body),
            Some("user") | None => response.content.push_str(body),
            Some(_) => {}
        }

        if !complete {
            break;
        }
        cursor = body_end + delimiter_len;
    }

    Ok(response)
}

fn channel_end(message: &str, body_start: usize) -> (usize, usize, bool) {
    let rest = &message[body_start..];
    let end = [
        (MESSAGE_END, rest.find(MESSAGE_END)),
        (TURN_END, rest.find(TURN_END)),
        (TEXT_END, rest.find(TEXT_END)),
    ]
    .into_iter()
    .filter_map(|(delimiter, offset)| offset.map(|offset| (offset, delimiter.len())))
    .min_by_key(|(offset, _)| *offset);
    match end {
        Some((offset, len)) => (body_start + offset, len, true),
        None => (message.len(), 0, false),
    }
}

fn trim_partial_control_token(text: &str) -> &str {
    let mut longest = 0;
    for token in [MESSAGE_END, TURN_END, TEXT_END, ASSISTANT_START] {
        for len in 1..token.len() {
            if text.ends_with(&token[..len]) {
                longest = longest.max(len);
            }
        }
    }
    &text[..text.len() - longest]
}

pub(crate) fn parse_atem_tool_calls(message: &str) -> Result<Option<Vec<AtemToolCall>>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = message[cursor..].find(FUNCTION_CALLS_START) {
        let block_start = cursor + offset;
        let body_start = block_start + FUNCTION_CALLS_START.len();
        let Some(body_end_offset) = message[body_start..].find(FUNCTION_CALLS_END) else {
            return Ok(None);
        };
        let body_end = body_start + body_end_offset;
        let recipient = channel_recipient(message, block_start);
        if !matches!(recipient, Some("self" | "user")) {
            let Some(block_calls) = parse_atem_invocations(&message[body_start..body_end])? else {
                return Ok(None);
            };
            if let Some(recipient) = recipient {
                if block_calls.iter().any(|call| call.name != recipient) {
                    return Err(candle_core::Error::Msg(format!(
                        "Muse Glimmer tool recipient `{recipient}` does not match its invocation"
                    )));
                }
            }
            calls.extend(block_calls);
        }
        cursor = body_end + FUNCTION_CALLS_END.len();
    }

    Ok((!calls.is_empty()).then_some(calls))
}

pub(crate) fn normalize_atem_arguments(call: &mut AtemToolCall, tool: &Tool) {
    let Some(parameters) = tool.function.parameters.as_ref() else {
        return;
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return;
    };
    let Some(arguments) = call.arguments.as_object_mut() else {
        return;
    };
    for (name, value) in arguments {
        let Some(schema) = properties.get(name) else {
            continue;
        };
        if schema_accepts_string(schema)
            && !value.is_string()
            && !schema_accepts_non_string_value(schema, value)
        {
            *value = Value::String(match value {
                Value::Null => "null".to_string(),
                Value::Bool(value) => value.to_string(),
                Value::Number(value) => value.to_string(),
                Value::String(value) => value.clone(),
                Value::Array(_) | Value::Object(_) => value.to_string(),
            });
        }
    }
}

fn schema_accepts_type(schema: &Value, expected: &str) -> bool {
    let direct = match schema.get("type") {
        Some(Value::String(value)) => value == expected,
        Some(Value::Array(values)) => values.iter().any(|value| value.as_str() == Some(expected)),
        _ => false,
    };
    direct
        || (expected == "null"
            && schema
                .get("nullable")
                .and_then(Value::as_bool)
                .unwrap_or(false))
        || ["anyOf", "oneOf"].into_iter().any(|keyword| {
            schema
                .get(keyword)
                .and_then(Value::as_array)
                .is_some_and(|variants| {
                    variants
                        .iter()
                        .any(|variant| schema_accepts_type(variant, expected))
                })
        })
}

fn schema_accepts_string(schema: &Value) -> bool {
    schema_accepts_type(schema, "string")
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(Value::is_string))
        || schema.get("const").is_some_and(Value::is_string)
        || ["anyOf", "oneOf"].into_iter().any(|keyword| {
            schema
                .get(keyword)
                .and_then(Value::as_array)
                .is_some_and(|variants| variants.iter().any(schema_accepts_string))
        })
}

fn schema_accepts_non_string_value(schema: &Value, value: &Value) -> bool {
    if schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| values.contains(value))
        || schema
            .get("const")
            .is_some_and(|expected| expected == value)
    {
        return true;
    }
    if ["anyOf", "oneOf"].into_iter().any(|keyword| {
        schema
            .get(keyword)
            .and_then(Value::as_array)
            .is_some_and(|variants| {
                variants
                    .iter()
                    .any(|variant| schema_accepts_non_string_value(variant, value))
            })
    }) {
        return true;
    }

    let accepts = |expected| schema_accepts_type(schema, expected);
    match value {
        Value::Null => accepts("null"),
        Value::Bool(_) => accepts("boolean"),
        Value::Number(number) => {
            accepts("number") || ((number.is_i64() || number.is_u64()) && accepts("integer"))
        }
        Value::Array(_) => accepts("array"),
        Value::Object(_) => accepts("object"),
        Value::String(_) => false,
    }
}

fn parse_atem_invocations(body: &str) -> Result<Option<Vec<AtemToolCall>>> {
    let mut calls = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find(INVOKE_START) {
        let tag_start = cursor + offset;
        let Some(tag_end_offset) = body[tag_start..].find('>') else {
            return Ok(None);
        };
        let tag_end = tag_start + tag_end_offset;
        let Some(name) = tag_attribute(&body[tag_start..=tag_end], "name") else {
            return Err(candle_core::Error::Msg(
                "Muse Glimmer tool invocation is missing its name".to_string(),
            ));
        };
        let body_start = tag_end + 1;
        let Some(body_end_offset) = body[body_start..].find(INVOKE_END) else {
            return Ok(None);
        };
        let body_end = body_start + body_end_offset;
        let arguments = parse_atem_parameters(&body[body_start..body_end])?;
        calls.push(AtemToolCall {
            name: name.to_string(),
            arguments: Value::Object(arguments),
        });
        cursor = body_end + INVOKE_END.len();
    }

    Ok((!calls.is_empty()).then_some(calls))
}

fn channel_recipient(message: &str, block_start: usize) -> Option<&str> {
    let prefix = &message[..block_start];
    let message_start = prefix.rfind(MESSAGE_START)?;
    let header_prefix = &prefix[..message_start];
    let channel_start = [ASSISTANT_START, MESSAGE_END, TURN_END]
        .into_iter()
        .filter_map(|token| {
            header_prefix
                .rfind(token)
                .map(|offset| offset + token.len())
        })
        .max()
        .unwrap_or_default();
    header_prefix[channel_start..]
        .rsplit_once(" to=")
        .map(|(_, recipient)| recipient.trim())
}

fn parse_atem_parameters(body: &str) -> Result<serde_json::Map<String, Value>> {
    let mut parameters = serde_json::Map::new();
    let mut cursor = 0;
    while let Some(offset) = body[cursor..].find(PARAMETER_START) {
        let tag_start = cursor + offset;
        let Some(tag_end_offset) = body[tag_start..].find('>') else {
            return Err(candle_core::Error::Msg(
                "Muse Glimmer tool parameter has an incomplete opening tag".to_string(),
            ));
        };
        let tag_end = tag_start + tag_end_offset;
        let Some(name) = tag_attribute(&body[tag_start..=tag_end], "name") else {
            return Err(candle_core::Error::Msg(
                "Muse Glimmer tool parameter is missing its name".to_string(),
            ));
        };
        let value_start = tag_end + 1;
        let Some(value_end_offset) = body[value_start..].find(PARAMETER_END) else {
            return Err(candle_core::Error::Msg(
                "Muse Glimmer tool parameter is missing its closing tag".to_string(),
            ));
        };
        let value_end = value_start + value_end_offset;
        let raw = &body[value_start..value_end];
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
        parameters.insert(name.to_string(), value);
        cursor = value_end + PARAMETER_END.len();
    }
    Ok(parameters)
}

fn tag_attribute<'a>(tag: &'a str, attribute: &str) -> Option<&'a str> {
    let prefix = format!(r#"{attribute}=""#);
    let start = tag.find(&prefix)? + prefix.len();
    let end = tag[start..].find('"')? + start;
    Some(&tag[start..end])
}

pub(crate) fn required_tool_call_grammar(tools: &[Tool], prefix: &str) -> TopLevelGrammar {
    build_atem_grammar(tools, true, prefix)
}

fn build_atem_grammar(tools: &[Tool], required: bool, required_prefix: &str) -> TopLevelGrammar {
    let mut rules = Vec::new();
    let mut schemas = Vec::new();
    let mut branches = Vec::new();

    for (index, tool) in tools.iter().enumerate() {
        let invoke = format!("invoke_{index}");
        let arguments = emit_arguments(tool, index, &mut rules, &mut schemas);
        let open = lark_literal(&format!(r#"<atem:invoke name="{}">"#, tool.function.name));
        rules.push(format!(
            "{invoke}: {open} WS? {arguments} {close} WS?",
            close = lark_literal(INVOKE_END)
        ));

        if required {
            let branch = format!("required_{index}");
            let prefix = recipient_prefix(required_prefix, &tool.function.name);
            rules.push(format!(
                "{branch}: {prefix} {message} {calls_start} WS? {invoke} {calls_end}",
                message = lark_literal(MESSAGE_START),
                calls_start = lark_literal(FUNCTION_CALLS_START),
                calls_end = lark_literal(FUNCTION_CALLS_END),
            ));
            branches.push(branch);
        } else {
            branches.push(invoke);
        }
    }

    let start = if required {
        format!("start: {}", branches.join(" | "))
    } else {
        format!(
            "start: WS? ({}) {calls_end}",
            branches.join(" | "),
            calls_end = lark_literal(FUNCTION_CALLS_END)
        )
    };
    let mut lines = vec![start];
    lines.extend(rules);
    lines.extend([
        "atem_value: (atem_text | atem_lt)*".to_string(),
        "atem_text: /[^<]+/".to_string(),
        r#"atem_lt: "<" /[^\/]/"#.to_string(),
        "PARAM_NAME: /[a-zA-Z_][a-zA-Z0-9_.-]*/".to_string(),
        "WS: /[ \\t\\r\\n]+/".to_string(),
    ]);

    let mut grammars = vec![GrammarWithLexer::from_lark(lines.join("\n"))];
    grammars.extend(schemas);
    TopLevelGrammar {
        grammars,
        max_tokens: None,
    }
}

fn emit_arguments(
    tool: &Tool,
    tool_index: usize,
    rules: &mut Vec<String>,
    schemas: &mut Vec<GrammarWithLexer>,
) -> String {
    let Some(schema) = tool.function.strict_parameters_schema() else {
        let parameter = format!("generic_parameter_{tool_index}");
        let arguments = format!("generic_arguments_{tool_index}");
        rules.push(format!(
            "{parameter}: {open} PARAM_NAME {middle} atem_value {close} WS?",
            open = lark_literal(r#"<atem:parameter name=""#),
            middle = lark_literal(r#"">"#),
            close = lark_literal(PARAMETER_END),
        ));
        rules.push(format!("{arguments}: {parameter}*"));
        return arguments;
    };
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return emit_arguments_without_schema(tool_index, rules);
    };
    if properties.is_empty() {
        let arguments = format!("strict_arguments_{tool_index}");
        rules.push(format!("{arguments}:"));
        return arguments;
    }

    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .collect::<std::collections::HashSet<_>>()
        })
        .unwrap_or_default();
    let mut names = properties.keys().collect::<Vec<_>>();
    names.sort();
    names.sort_by_key(|name| !required.contains(name.as_str()));

    let mut parameters = Vec::new();
    for (property_index, name) in names.into_iter().enumerate() {
        let parameter = format!("parameter_{tool_index}_{property_index}");
        let value_rule = strict_value_rule(&properties[name], tool_index, property_index, schemas);
        rules.push(format!(
            "{parameter}: {open} {value_rule} {close} WS?",
            open = lark_literal(&format!(r#"<atem:parameter name="{name}">"#)),
            close = lark_literal(PARAMETER_END),
        ));
        parameters.push((parameter, required.contains(name.as_str())));
    }

    let arguments = format!("strict_arguments_{tool_index}");
    let sequence = parameters
        .into_iter()
        .map(|(parameter, required)| {
            if required {
                parameter
            } else {
                format!("{parameter}?")
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    rules.push(format!("{arguments}: {sequence}"));
    arguments
}

fn emit_arguments_without_schema(tool_index: usize, rules: &mut Vec<String>) -> String {
    let arguments = format!("strict_arguments_{tool_index}");
    rules.push(format!("{arguments}:"));
    arguments
}

fn strict_value_rule(
    schema: &Value,
    tool_index: usize,
    property_index: usize,
    schemas: &mut Vec<GrammarWithLexer>,
) -> String {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        if values.iter().any(Value::is_string) {
            let alternatives = values
                .iter()
                .map(|value| match value {
                    Value::String(value) => lark_literal(value),
                    value => lark_literal(&value.to_string()),
                })
                .collect::<Vec<_>>();
            if !alternatives.is_empty() {
                return format!("({})", alternatives.join(" | "));
            }
        }
    }
    if let Some(Value::String(value)) = schema.get("const") {
        return lark_literal(value);
    }
    if schema_accepts_string(schema) {
        return "atem_value".to_string();
    }

    let name = format!("atem_schema_{tool_index}_{property_index}");
    schemas.push(GrammarWithLexer {
        name: Some(name.clone()),
        json_schema: Some(schema.clone()),
        ..Default::default()
    });
    format!("@{name}")
}

fn lark_literal(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn recipient_prefix(prefix: &str, tool_name: &str) -> String {
    let recipient = lark_literal(&format!("assistant to={tool_name}"));
    match prefix {
        "" => lark_literal(&format!(" to={tool_name}")),
        "<|start|>assistant" => {
            format!(
                "{} {recipient}",
                lark_literal(ASSISTANT_START.trim_end_matches("assistant"))
            )
        }
        "<|eom|><|start|>assistant" => format!(
            "{} {} {recipient}",
            lark_literal(MESSAGE_END),
            lark_literal(ASSISTANT_START.trim_end_matches("assistant"))
        ),
        other => lark_literal(&format!("{other} to={tool_name}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::parsers::specialize_required_tool_call_grammar;
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
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

    fn atem_token_trie() -> toktrie::TokTrie {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        for token in ["<eos>", MESSAGE_START, MESSAGE_END, ASSISTANT_START] {
            let mut bytes = vec![toktrie::TokTrie::SPECIAL_TOKEN_MARKER];
            bytes.extend_from_slice(token.as_bytes());
            tokens.push(bytes);
        }
        let vocab_size = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens)
    }

    fn weather_tool(strict: bool) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(
                    serde_json::from_value(json!({
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"}
                        },
                        "required": ["city"]
                    }))
                    .unwrap(),
                ),
                strict: strict.then_some(true),
            },
        }
    }

    fn nullable_string_tool() -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "lookup".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(json!({
                        "type": "object",
                        "properties": {
                            "code": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"}
                                ]
                            },
                            "mode": {"enum": ["fast", null]}
                        },
                        "required": ["code", "mode"]
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }
    }

    #[test]
    fn parses_scalar_json_and_multiline_string_parameters() {
        let message = " to=get_weather<|message|><atem:function_calls>\n<atem:invoke name=\"get_weather\">\n<atem:parameter name=\"city\">New\nYork</atem:parameter>\n<atem:parameter name=\"days\">3</atem:parameter>\n<atem:parameter name=\"metric\">true</atem:parameter>\n</atem:invoke>\n</atem:function_calls><|eot|>";
        let calls = parse_atem_tool_calls(message).unwrap().unwrap();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].arguments,
            json!({"city": "New\nYork", "days": 3, "metric": true})
        );
    }

    #[test]
    fn schema_restores_numeric_looking_string_parameters() {
        let message = " to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">123</atem:parameter><atem:parameter name=\"days\">3</atem:parameter></atem:invoke></atem:function_calls>";
        let mut call = parse_atem_tool_calls(message).unwrap().unwrap().remove(0);

        normalize_atem_arguments(&mut call, &weather_tool(true));

        assert_eq!(call.arguments, json!({"city": "123", "days": 3}));
    }

    #[test]
    fn nullable_string_schema_restores_strings_and_preserves_null() {
        let tool = nullable_string_tool();
        for (raw, expected) in [
            ("123", json!({"code": "123", "mode": "fast"})),
            ("null", json!({"code": null, "mode": "fast"})),
        ] {
            let message = format!(
                " to=lookup<|message|><atem:function_calls><atem:invoke name=\"lookup\"><atem:parameter name=\"code\">{raw}</atem:parameter><atem:parameter name=\"mode\">fast</atem:parameter></atem:invoke></atem:function_calls>"
            );
            let mut call = parse_atem_tool_calls(&message).unwrap().unwrap().remove(0);

            normalize_atem_arguments(&mut call, &tool);

            assert_eq!(call.arguments, expected);
        }
    }

    #[test]
    fn nullable_and_mixed_enum_strings_use_raw_atem_grammar() {
        let grammar = AtemParser.required_tool_call_grammar(&[nullable_string_tool()]);
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();

        assert!(lark.contains("parameter_0_0: \"<atem:parameter name=\\\"code\\\">\" atem_value"));
        assert!(lark.contains(r#"("fast" | "null")"#));
        assert_eq!(grammar.grammars.len(), 1);
    }

    #[test]
    fn raw_atem_values_allow_less_than() {
        let grammar = AtemParser.required_tool_call_grammar(&[weather_tool(false)]);
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();

        assert!(lark.contains(r#"atem_lt: "<" /[^\/]/"#));
    }

    #[test]
    fn separates_reasoning_final_content_and_parallel_calls() {
        let message = " to=self<|message|>check weather<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls><|eom|><|start|>assistant to=user<|message|>It is sunny<|eot|>";
        let response = parse_atem_response(message).unwrap();

        assert_eq!(response.reasoning, "check weather");
        assert_eq!(response.content, "It is sunny");
        assert_eq!(response.tool_calls.len(), 1);
    }

    #[test]
    fn strips_end_of_text_from_final_content() {
        let response =
            parse_atem_response(" to=user<|message|>Final answer<|end_of_text|>").unwrap();

        assert_eq!(response.content, "Final answer");
    }

    #[test]
    fn required_grammar_uses_native_recipient_and_schema() {
        let grammar = AtemParser.required_tool_call_grammar(&[weather_tool(true)]);
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();

        assert!(lark.contains(r#"" to=get_weather""#));
        assert!(lark.contains("<atem:function_calls>"));
        assert!(lark.contains("<atem:parameter name=\\\"city\\\">"));
        assert_eq!(grammar.grammars.len(), 2);
    }

    #[test]
    fn required_grammar_compiles_with_control_tokens() {
        let mut grammar = AtemParser.required_tool_call_grammar(&[weather_tool(true)]);
        let trie = atem_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());

        factory.create_parser(grammar).unwrap();
    }

    #[test]
    fn continuation_grammar_starts_after_function_calls_wrapper() {
        let grammar = AtemParser.tool_call_grammar(&[weather_tool(false)], "");
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();

        assert!(lark.starts_with("start: WS? (invoke_0)"));
        assert!(lark.contains("</atem:function_calls>"));
    }

    #[test]
    fn continuation_grammar_matches_the_recipient() {
        let mut search = weather_tool(false);
        search.function.name = "search".to_string();
        let grammar = AtemParser.tool_call_grammar(
            &[weather_tool(false), search],
            " to=search<|message|><atem:function_calls>",
        );
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();

        assert!(lark.contains("name=\\\"search\\\""));
        assert!(!lark.contains("name=\\\"get_weather\\\""));
    }

    #[test]
    fn completed_wrapper_does_not_reactivate_continuation_grammar() {
        let complete = " to=get_weather<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls>";
        assert!(!AtemParser.could_be_tool_call(complete));

        let next = format!(
            "{complete}<|eom|><|start|>assistant to=get_weather<|message|><atem:function_calls>"
        );
        assert!(AtemParser.could_be_tool_call(&next));
    }

    #[test]
    fn ignores_atem_markup_inside_reasoning_or_final_content() {
        let markup = "<atem:function_calls><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris</atem:parameter></atem:invoke></atem:function_calls>";
        for recipient in ["self", "user"] {
            let message = format!(" to={recipient}<|message|>example: {markup}<|eot|>");
            assert!(parse_atem_tool_calls(&message).unwrap().is_none());
            assert!(!AtemParser.could_be_tool_call(&message));
        }
    }

    #[test]
    fn rejects_recipient_and_invocation_mismatch() {
        let message = " to=search<|message|><atem:function_calls><atem:invoke name=\"get_weather\"></atem:invoke></atem:function_calls>";
        let error = parse_atem_tool_calls(message).unwrap_err().to_string();
        assert!(error.contains("recipient `search`"));
    }

    #[test]
    fn incomplete_invocation_is_buffered() {
        let message = "<atem:function_calls><atem:invoke name=\"get_weather\"><atem:parameter name=\"city\">Paris";
        assert!(parse_atem_tool_calls(message).unwrap().is_none());
    }
}
