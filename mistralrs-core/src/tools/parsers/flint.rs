use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use serde_json::{json, Map, Value};

use super::{ToolCallFormat, ToolFormatParser};
use crate::{Tool, ToolCallResponse};

pub(crate) const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VALUE_OPEN: &str = "<arg_value>";
const ARG_VALUE_CLOSE: &str = "</arg_value>";

pub(crate) struct FlintParser;

impl ToolFormatParser for FlintParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.split_once(TOOL_CALL_OPEN)
            .is_some_and(|(_, body)| valid_name(body.split('<').next().unwrap_or_default().trim()))
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let mut calls = Vec::new();
        let mut rest = message;
        while let Some((_, body)) = rest.split_once(TOOL_CALL_OPEN) {
            let Some((body, after)) = body.split_once(TOOL_CALL_CLOSE) else {
                break;
            };
            let Some(call) = parse_call(body) else {
                return Ok(None);
            };
            calls.push(call);
            rest = after;
        }
        if calls.is_empty() {
            return Ok(None);
        }
        serde_json::to_string(&calls)
            .map(Some)
            .map_err(candle_core::Error::msg)
    }

    fn format(&self) -> ToolCallFormat {
        ToolCallFormat::Flint
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        tool_grammar(tools, false)
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        tool_grammar(tools, true)
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_alphanumeric() || "_.:-/".contains(ch))
}

fn parse_call(body: &str) -> Option<Value> {
    let (name, mut rest) = body
        .split_once(ARG_KEY_OPEN)
        .map_or((body, ""), |(name, _)| (name, &body[name.len()..]));
    let name = name.trim();
    if !valid_name(name) {
        return None;
    }
    let mut arguments = Map::new();
    while !rest.trim().is_empty() {
        let (key, after_key) = rest
            .trim_start()
            .strip_prefix(ARG_KEY_OPEN)?
            .split_once(ARG_KEY_CLOSE)?;
        let values = after_key.trim_start().strip_prefix(ARG_VALUE_OPEN)?;
        let (value, after_value) = values.match_indices(ARG_VALUE_CLOSE).find_map(|(idx, _)| {
            let after = &values[idx + ARG_VALUE_CLOSE.len()..];
            let next = after.trim_start();
            (next.is_empty() || next.starts_with(ARG_KEY_OPEN)).then_some((&values[..idx], after))
        })?;
        if key.is_empty()
            || key.contains('<')
            || arguments
                .insert(key.to_string(), Value::String(value.to_string()))
                .is_some()
        {
            return None;
        }
        rest = after_value;
    }
    Some(json!({"name": name, "arguments": arguments}))
}

pub(crate) fn coerce_arguments(
    tools: Option<&[Tool]>,
    calls: &mut [ToolCallResponse],
) -> anyhow::Result<()> {
    for call in calls {
        let schema = tools
            .unwrap_or_default()
            .iter()
            .find(|tool| tool.function.name == call.function.name)
            .and_then(|tool| tool.function.parameters.as_ref())
            .map(serde_json::to_value)
            .transpose()?
            .map(|schema| resolve_schema(&schema, &schema));
        let mut arguments: Map<String, Value> = serde_json::from_str(&call.function.arguments)?;
        for (name, value) in &mut arguments {
            let Value::String(raw) = value else {
                continue;
            };
            let Ok(parsed) = serde_json::from_str::<Value>(raw) else {
                continue;
            };
            let property = schema.as_ref().and_then(|schema| {
                schema
                    .get("properties")
                    .and_then(|properties| properties.get(name))
                    .or_else(|| schema.get("additionalProperties"))
                    .map(|property| (property, schema))
            });
            if !parsed.is_string()
                && property
                    .is_none_or(|(property, schema)| accepts_value(property, schema, &parsed))
            {
                *value = parsed;
            }
        }
        call.function.arguments = serde_json::to_string(&arguments)?;
    }
    Ok(())
}

fn resolve_schema(schema: &Value, root: &Value) -> Value {
    let mut schema = schema.clone();
    let mut seen = std::collections::HashSet::new();
    while let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let Some(target) = reference
            .strip_prefix('#')
            .and_then(|path| root.pointer(path))
        else {
            break;
        };
        if !seen.insert(reference.to_string()) {
            break;
        }
        let mut resolved = target.clone();
        if let (Some(resolved), Some(original)) = (resolved.as_object_mut(), schema.as_object()) {
            resolved.extend(
                original
                    .iter()
                    .filter(|(name, _)| *name != "$ref")
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
        }
        schema = resolved;
    }
    if let Some(object) = schema.as_object_mut() {
        for name in ["$defs", "definitions"] {
            if let Some(definitions) = root.get(name) {
                object
                    .entry(name.to_string())
                    .or_insert_with(|| definitions.clone());
            }
        }
    }
    schema
}

fn accepts_value(schema: &Value, root: &Value, value: &Value) -> bool {
    let schema = resolve_schema(schema, root);
    if let Some(variants) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        return variants
            .iter()
            .any(|schema| accepts_value(schema, root, value));
    }
    if let Some(expected) = schema.get("const") {
        return expected == value;
    }
    if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
        return variants.contains(value);
    }
    let matches_type = |kind: &str| match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "integer" => value.is_i64() || value.is_u64(),
        "number" => value.is_number(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        _ => false,
    };
    match schema.get("type") {
        Some(Value::String(kind)) => matches_type(kind),
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).any(matches_type),
        _ => true,
    }
}

fn tool_grammar(tools: &[Tool], include_wrapper: bool) -> TopLevelGrammar {
    let marker = |text: &str| {
        if include_wrapper {
            lark_string(text)
        } else {
            text.to_string()
        }
    };
    let open = marker(TOOL_CALL_OPEN);
    let close = marker(TOOL_CALL_CLOSE);
    let key_open = marker(ARG_KEY_OPEN);
    let key_close = marker(ARG_KEY_CLOSE);
    let value_open = marker(ARG_VALUE_OPEN);
    let value_close = marker(ARG_VALUE_CLOSE);
    let prefix = if include_wrapper { open.as_str() } else { "" };
    // Dedicated delimiter tokens cannot be consumed by the UTF-8 text lexeme.
    let mut rules = vec![
        format!("start: {prefix} flint_call (flint_ws {open} flint_call)*"),
        "flint_ws: /[ \\t\\r\\n]*/".to_string(),
        "FLINT_TEXT: /(?s:.)*/".to_string(),
        format!("flint_arg: {key_open} /[^<]+/ {key_close} {value_open} FLINT_TEXT {value_close}"),
    ];
    let mut branches = Vec::new();
    for (tool_index, tool) in tools.iter().enumerate() {
        let branch = format!("flint_tool_{tool_index}");
        let mut arguments = Vec::new();
        if let Some(schema) = tool.function.strict_parameters_schema() {
            let schema = resolve_schema(&schema, &schema);
            let properties = schema.get("properties").and_then(Value::as_object);
            if let Some(properties) = properties {
                let required = schema.get("required").and_then(Value::as_array);
                for (arg_index, (name, property)) in properties.iter().enumerate() {
                    let argument = format!("flint_arg_{tool_index}_{arg_index}");
                    let value = value_rule(property, &schema, &argument, &mut rules);
                    rules.push(format!(
                        "{argument}: flint_ws {key_open} {} {key_close} {value_open} {value} {value_close}",
                        lark_string(name)
                    ));
                    let is_required =
                        required.is_some_and(|keys| keys.iter().any(|key| key == name));
                    arguments.push(if is_required {
                        argument
                    } else {
                        format!("{argument}?")
                    });
                }
            }
            if schema
                .get("additionalProperties")
                .is_none_or(|value| value != false)
            {
                let key = format!("FLINT_EXTRA_KEY_{tool_index}");
                let excluded = properties
                    .into_iter()
                    .flat_map(|properties| properties.keys())
                    .map(|name| lark_string(name))
                    .collect::<Vec<_>>();
                let exclusion = if excluded.is_empty() {
                    String::new()
                } else {
                    format!(" & ~({})", excluded.join(" | "))
                };
                rules.push(format!("{key}: /[^<]+/{exclusion}"));
                let argument = format!("flint_extra_arg_{tool_index}");
                let value = schema
                    .get("additionalProperties")
                    .filter(|value| value.is_object())
                    .map(|value| value_rule(value, &schema, &argument, &mut rules))
                    .unwrap_or_else(|| "FLINT_TEXT".to_string());
                rules.push(format!("{argument}: flint_ws {key_open} {key} {key_close} {value_open} {value} {value_close}"));
                arguments.push(format!("{argument}*"));
            }
        } else {
            arguments.push("(flint_ws flint_arg)*".to_string());
        }
        rules.push(format!(
            "{branch}: {} {} flint_ws {close}",
            lark_string(&tool.function.name),
            arguments.join(" ")
        ));
        branches.push(branch);
    }
    rules.push(format!("flint_call: {}", branches.join(" | ")));
    TopLevelGrammar::from_lark(rules.join("\n"))
}

fn value_rule(schema: &Value, root: &Value, name: &str, rules: &mut Vec<String>) -> String {
    let schema = resolve_schema(schema, root);
    if let Some(variants) = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array)
    {
        let branches = variants
            .iter()
            .enumerate()
            .map(|(index, schema)| value_rule(schema, root, &format!("{name}_{index}"), rules))
            .collect::<Vec<_>>();
        return format!("({})", branches.join(" | "));
    }
    if let Some(Value::Array(types)) = schema.get("type") {
        let branches = types
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let mut schema = schema.clone();
                schema["type"] = kind.clone();
                value_rule(&schema, root, &format!("{name}_{index}"), rules)
            })
            .collect::<Vec<_>>();
        return format!("({})", branches.join(" | "));
    }
    if let Some(value) = schema.get("const") {
        return lark_string(&raw_value(value));
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return format!(
            "({})",
            values
                .iter()
                .map(|value| lark_string(&raw_value(value)))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => {
            let mut constraints = vec!["FLINT_TEXT".to_string()];
            let min = schema.get("minLength").and_then(Value::as_u64).unwrap_or(0);
            let max = schema.get("maxLength").and_then(Value::as_u64);
            if min != 0 || max.is_some() {
                constraints.push(format!(
                    "/(?s:.){{{min},{}}}/",
                    max.map(|value| value.to_string()).unwrap_or_default()
                ));
            }
            if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
                let start = if pattern.starts_with('^') {
                    ""
                } else {
                    "(?s:.*)"
                };
                let end = if pattern.ends_with('$') {
                    ""
                } else {
                    "(?s:.*)"
                };
                let pattern = pattern.strip_prefix('^').unwrap_or(pattern);
                let pattern = pattern
                    .strip_suffix('$')
                    .unwrap_or(pattern)
                    .replace('/', "\\/");
                constraints.push(format!("/{start}(?:{pattern}){end}/"));
            }
            let terminal = name.to_ascii_uppercase();
            rules.push(format!("{terminal}: {}", constraints.join(" & ")));
            terminal
        }
        Some(_) => format!("%json {}", schema),
        None => "FLINT_TEXT".to_string(),
    }
}

fn raw_value(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

fn lark_string(value: &str) -> String {
    serde_json::to_string(value).expect("strings serialize as JSON")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{parsers::specialize_required_tool_call_grammar, ToolCallingMatcher};
    use crate::{Function, ToolChoice, ToolType};
    use std::sync::Arc;

    struct TokenizerEnv(toktrie::TokTrie);

    impl toktrie::TokenizerEnv for TokenizerEnv {
        fn tok_trie(&self) -> &toktrie::TokTrie {
            &self.0
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<toktrie::TokenId> {
            self.0.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            false
        }
    }

    fn tool(strict: bool) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: "book".to_string(),
                description: None,
                parameters: Some(serde_json::from_value(json!({
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "enum": ["Paris", "Tokyo"]},
                        "count": {"type": "integer", "minimum": 1, "maximum": 4},
                        "details": {
                            "type": "object",
                            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
                            "required": ["tags"],
                            "additionalProperties": false
                        },
                        "note": {"type": "string"},
                        "optional": {"type": ["integer", "null"]}
                    },
                    "required": ["city", "count"],
                    "additionalProperties": false
                })).unwrap()),
                strict: Some(strict),
            },
        }
    }

    fn matcher(strict: bool, required: bool) -> (llguidance::Matcher, toktrie::TokTrie) {
        matcher_for_tool(tool(strict), required)
    }

    fn referenced_tool() -> Tool {
        let mut tool = tool(true);
        tool.function.parameters = Some(
            serde_json::from_value(json!({
                "$ref": "#/$defs/arguments",
                "$defs": {
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "note": {"type": "string"}
                        },
                        "required": ["count", "note"],
                        "additionalProperties": false
                    }
                }
            }))
            .unwrap(),
        );
        tool
    }

    fn matcher_for_tool(tool: Tool, required: bool) -> (llguidance::Matcher, toktrie::TokTrie) {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = tokens.len() as u32;
        for token in [
            "<eos>",
            TOOL_CALL_OPEN,
            TOOL_CALL_CLOSE,
            ARG_KEY_OPEN,
            ARG_KEY_CLOSE,
            ARG_VALUE_OPEN,
            ARG_VALUE_CLOSE,
        ] {
            let mut bytes = vec![toktrie::TokTrie::SPECIAL_TOKEN_MARKER];
            bytes.extend_from_slice(token.as_bytes());
            tokens.push(bytes);
        }
        let trie =
            toktrie::TokTrie::from(&toktrie::TokRxInfo::new(tokens.len() as u32, eos), &tokens);
        let env: toktrie::TokEnv = Arc::new(TokenizerEnv(trie.clone()));
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut grammar = tool_grammar(&[tool], required);
        specialize_required_tool_call_grammar(&mut grammar, &trie);
        let parser = factory.create_parser(grammar).unwrap();
        (llguidance::Matcher::new(Ok(parser)), trie)
    }

    fn consume(matcher: &mut llguidance::Matcher, trie: &toktrie::TokTrie, text: &str) -> bool {
        let markers = [
            TOOL_CALL_OPEN,
            TOOL_CALL_CLOSE,
            ARG_KEY_OPEN,
            ARG_KEY_CLOSE,
            ARG_VALUE_OPEN,
            ARG_VALUE_CLOSE,
        ];
        let mut rest = text;
        while !rest.is_empty() {
            let token =
                if let Some(marker) = markers.iter().find(|marker| rest.starts_with(**marker)) {
                    rest = &rest[marker.len()..];
                    trie.get_special_token(marker).unwrap()
                } else {
                    let byte = rest.as_bytes()[0];
                    rest = &rest[1..];
                    byte as u32
                };
            if !matcher.compute_mask().unwrap().is_allowed(token) {
                return false;
            }
            matcher.consume_token(token).unwrap();
        }
        true
    }

    #[test]
    fn flint_calls_keep_schema_types_whitespace_and_parallel_calls() {
        let matcher = ToolCallingMatcher::new_with_format(
            ToolChoice::Auto,
            Some(&[tool(false)]),
            Some(ToolCallFormat::Flint),
        )
        .unwrap();
        let text = "Checking.<tool_call>book\n<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>details</arg_key><arg_value>{\"tags\":[\"x\"]}</arg_value><arg_key>note</arg_key><arg_value>null</arg_value><arg_key>optional</arg_key><arg_value>null</arg_value></tool_call><tool_call>book<arg_key>note</arg_key><arg_value>  print(1 < 2)\n</arg_value></tool_call>";
        let (content, calls) = matcher.get_call_with_content(text).unwrap();
        assert_eq!(content.as_deref(), Some("Checking."));
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].id, calls[1].id);
        let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(arguments["count"], 2);
        assert_eq!(arguments["details"], json!({"tags": ["x"]}));
        assert_eq!(arguments["note"], "null");
        assert_eq!(arguments["optional"], Value::Null);
        let arguments: Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(arguments["note"], "  print(1 < 2)\n");
        assert_eq!(matcher.prefix_could_be_tool(text).unwrap(), (true, false));
    }

    #[test]
    fn flint_parser_rejects_malformed_or_duplicate_arguments() {
        for text in [
            "<tool_call>book<arg_key>x</arg_key><arg_value>1</tool_call>",
            "<tool_call>book<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>",
            "<tool_call>{\"name\":\"book\",\"arguments\":{}}</tool_call>",
            "<tool_call><function=book></function></tool_call>",
        ] {
            assert!(FlintParser.parse(text).unwrap().is_none(), "{text}");
        }
        assert!(FlintParser
            .parse("<tool_call>book</tool_call>")
            .unwrap()
            .is_some());
    }

    #[test]
    fn flint_root_parameter_reference_preserves_argument_types() {
        let matcher = ToolCallingMatcher::new_with_format(
            ToolChoice::Auto,
            Some(&[referenced_tool()]),
            Some(ToolCallFormat::Flint),
        )
        .unwrap();
        let calls = matcher.get_call("<tool_call>book<arg_key>count</arg_key><arg_value>2</arg_value><arg_key>note</arg_key><arg_value>null</arg_value></tool_call>").unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
            json!({"count": 2, "note": "null"})
        );
    }

    #[test]
    fn flint_strict_grammar_resolves_root_parameter_reference() {
        let (mut matcher, trie) = matcher_for_tool(referenced_tool(), true);
        assert!(consume(&mut matcher, &trie, "<tool_call>book<arg_key>count</arg_key><arg_value>2</arg_value><arg_key>note</arg_key><arg_value>null</arg_value></tool_call>"));
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
        let (mut matcher, trie) = matcher_for_tool(referenced_tool(), true);
        assert!(!consume(&mut matcher, &trie, "<tool_call>book</tool_call>"));
    }

    #[test]
    fn flint_strict_grammar_accepts_nested_values_and_parallel_calls() {
        let (mut matcher, trie) = matcher(true, true);
        assert!(consume(&mut matcher, &trie, "<tool_call>book<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>details</arg_key><arg_value>{\"tags\":[\"x\"]}</arg_value><arg_key>note</arg_key><arg_value>print(1 < 2)\n</arg_value><arg_key>optional</arg_key><arg_value>null</arg_value></tool_call>"));
        let mask = matcher.compute_mask().unwrap();
        assert!(mask.is_allowed(trie.eos_token()));
        assert!(mask.is_allowed(trie.get_special_token(TOOL_CALL_OPEN).unwrap()));
        assert!(consume(&mut matcher, &trie, "<tool_call>book<arg_key>city</arg_key><arg_value>Tokyo</arg_value><arg_key>count</arg_key><arg_value>1</arg_value></tool_call>"));
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
    }

    #[test]
    fn flint_strict_grammar_rejects_wrong_names_missing_args_and_invalid_values() {
        for text in [
            "<tool_call>other",
            "<tool_call>book</tool_call>",
            "<tool_call>book<arg_key>city</arg_key><arg_value>London",
            "<tool_call>book<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>count</arg_key><arg_value>5",
        ] {
            let (mut matcher, trie) = matcher(true, true);
            assert!(!consume(&mut matcher, &trie, text), "{text}");
        }
    }

    #[test]
    fn flint_continuation_grammar_preserves_code_and_empty_arguments() {
        let (mut matcher, trie) = matcher(false, false);
        assert!(consume(&mut matcher, &trie, "book<arg_key>note</arg_key><arg_value>print(1 < 2)\n</arg_value></tool_call><tool_call>book</tool_call>"));
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
    }

    #[test]
    fn flint_strict_grammar_resolves_local_schemas_and_constrains_extra_properties() {
        let mut tool = tool(true);
        tool.function.parameters = Some(serde_json::from_value(json!({
            "type": "object",
            "$defs": {"label": {"type": "string", "pattern": "^[A-Z]+$", "minLength": 2, "maxLength": 3}},
            "properties": {"label": {"$ref": "#/$defs/label"}},
            "required": ["label"],
            "additionalProperties": {"type": "integer"}
        })).unwrap());
        let (mut matcher, trie) = matcher_for_tool(tool.clone(), true);
        assert!(consume(&mut matcher, &trie, "<tool_call>book<arg_key>label</arg_key><arg_value>AB</arg_value><arg_key>extra</arg_key><arg_value>2</arg_value></tool_call>"));
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.eos_token()));
        for text in [
            "<tool_call>book<arg_key>label</arg_key><arg_value>a",
            "<tool_call>book<arg_key>label</arg_key><arg_value>A</arg_value>",
            "<tool_call>book<arg_key>label</arg_key><arg_value>ABCD",
            "<tool_call>book<arg_key>label</arg_key><arg_value>AB</arg_value><arg_key>label</arg_key>",
            "<tool_call>book<arg_key>label</arg_key><arg_value>AB</arg_value><arg_key>extra</arg_key><arg_value>x",
        ] {
            let (mut matcher, trie) = matcher_for_tool(tool.clone(), true);
            assert!(!consume(&mut matcher, &trie, text), "{text}");
        }
    }
}
