//! Shared helpers for building tool call grammars used in mid-stream
//! constrained decoding.
//!
//! Format-specific grammars are defined in each parser file
//! (`parsers/{qwen,llama,mistral_nemo,deepseek,gemma4,harmony}.rs`).  This
//! module provides common building blocks used by those parsers.

use llguidance::api::GrammarWithLexer;
use llguidance::api::TopLevelGrammar;
use serde_json::{json, Value};

use crate::Tool;

/// Build a grammar that composes a Lark wrapper with a JSON Schema
/// subgrammar referenced as `@json_body`.
pub(crate) fn build_json_format_grammar(
    lark: String,
    tools: &[Tool],
    args_key: &str,
    is_array: bool,
) -> TopLevelGrammar {
    let top = GrammarWithLexer::from_lark(lark);
    let schema = json_body_schema(tools, args_key, is_array);
    let json_body = GrammarWithLexer {
        name: Some("json_body".to_string()),
        json_schema: Some(schema),
        ..Default::default()
    };
    TopLevelGrammar {
        grammars: vec![top, json_body],
        max_tokens: None,
    }
}

/// Build JSON Schema for a tool call body.  `args_key` is `"arguments"` or
/// `"parameters"` depending on the model format.  When `is_array` is true
/// the schema wraps the single-call schema in an array (Mistral Nemo).
///
/// When any tool has `strict: true`, the schema uses `anyOf` with per-tool
/// variants so that each strict tool's `parameters` JSON schema is enforced
/// on its arguments via llguidance constrained decoding.
fn json_body_schema(tools: &[Tool], args_key: &str, is_array: bool) -> Value {
    let any_strict = tools.iter().any(|t| t.function.strict == Some(true));

    let single_call = if any_strict {
        // Per-tool variants: each tool gets its own schema branch,
        // discriminated by `"const"` on the name field.
        let variants: Vec<Value> = tools
            .iter()
            .map(|t| {
                let args_schema = t
                    .function
                    .strict_parameters_schema()
                    .unwrap_or_else(|| json!({"type": "object"}));
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "const": t.function.name },
                        args_key: args_schema,
                    },
                    "required": ["name", args_key],
                })
            })
            .collect();
        json!({ "anyOf": variants })
    } else {
        // Original generic schema — unchanged behaviour.
        let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": tool_names,
                },
                args_key: {
                    "type": "object",
                },
            },
            "required": ["name", args_key],
        })
    };

    if is_array {
        json!({
            "type": "array",
            "items": single_call,
            "minItems": 1,
        })
    } else {
        single_call
    }
}

/// Generate a Lark alternatives expression for tool names:
/// `"name1" | "name2" | "name3"`
pub(crate) fn lark_tool_name_alternatives(tools: &[Tool]) -> String {
    tools
        .iter()
        .map(|t| format!("\"{}\"", t.function.name))
        .collect::<Vec<_>>()
        .join(" | ")
}

#[cfg(test)]
mod tests {
    use super::super::parsers;
    use crate::Tool;
    use mistralrs_mcp::{Function, ToolType};

    fn sample_tools() -> Vec<Tool> {
        vec![
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: None,
                    strict: None,
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "search".to_string(),
                    description: Some("Search".to_string()),
                    parameters: None,
                    strict: None,
                },
            },
        ]
    }

    #[test]
    fn qwen_grammar_has_two_grammars() {
        let grm =
            parsers::build_tool_call_grammar("<tool_call>", &sample_tools()).expect("should match");
        assert_eq!(grm.grammars.len(), 2);
        assert!(grm.grammars[1].json_schema.is_some());
        assert_eq!(grm.grammars[1].name, Some("json_body".to_string()));
    }

    #[test]
    fn llama_uses_parameters_key() {
        let grm = parsers::build_tool_call_grammar("<|python_tag|>", &sample_tools())
            .expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        assert!(schema["properties"]["parameters"].is_object());
        assert!(schema["properties"].get("arguments").is_none());
    }

    #[test]
    fn mistral_nemo_is_array() {
        let grm = parsers::build_tool_call_grammar("[TOOL_CALLS]", &sample_tools())
            .expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        assert_eq!(schema["type"], "array");
    }

    #[test]
    fn deepseek_needs_json_fence() {
        // Without the fence, should return None
        let grm = parsers::build_tool_call_grammar("<｜tool▁call▁begin｜>", &sample_tools());
        assert!(grm.is_none());

        // With the fence, should return a grammar
        let grm = parsers::build_tool_call_grammar(
            "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n",
            &sample_tools(),
        );
        assert!(grm.is_some());
        assert_eq!(grm.unwrap().grammars.len(), 2);
    }

    #[test]
    fn gemma4_grammar_is_pure_lark() {
        let grm = parsers::build_tool_call_grammar("<|tool_call>", &sample_tools())
            .expect("should match");
        assert_eq!(grm.grammars.len(), 1);
        assert!(grm.grammars[0].json_schema.is_none());
    }

    #[test]
    fn gemma4_strict_grammar_still_pure_lark() {
        let grm = parsers::build_tool_call_grammar("<|tool_call>", &strict_tools())
            .expect("should match");
        // Strict mode still uses pure Lark (no JSON schema subgrammar)
        // because Gemma 4's format is not JSON.
        assert_eq!(grm.grammars.len(), 1);
        assert!(grm.grammars[0].json_schema.is_none());
    }

    #[test]
    fn tool_names_in_schema() {
        let grm =
            parsers::build_tool_call_grammar("<tool_call>", &sample_tools()).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        let names = schema["properties"]["name"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["get_weather", "search"]);
    }

    #[test]
    fn no_match_returns_none() {
        let grm = parsers::build_tool_call_grammar("Hello world", &sample_tools());
        assert!(grm.is_none());
    }

    #[test]
    fn harmony_args_grammar_is_single_json_object() {
        let grm = parsers::harmony::tool_call_grammar_for_tool(None, None);
        assert_eq!(grm.grammars.len(), 1);
        let schema = grm.grammars[0].json_schema.as_ref().unwrap();
        assert_eq!(schema["type"], "object");
    }

    // ── strict mode tests ─────────────────────────────────────────────

    fn strict_tools() -> Vec<Tool> {
        let params = serde_json::from_value(serde_json::json!({
            "type": "object",
            "properties": {
                "place": { "type": "string" }
            },
            "required": ["place"],
        }))
        .unwrap();
        vec![
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "get_weather".to_string(),
                    description: Some("Get weather".to_string()),
                    parameters: Some(params),
                    strict: Some(true),
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "search".to_string(),
                    description: Some("Search".to_string()),
                    parameters: None,
                    strict: None,
                },
            },
        ]
    }

    #[test]
    fn non_strict_tools_use_enum_schema() {
        let grm =
            parsers::build_tool_call_grammar("<tool_call>", &sample_tools()).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        // No anyOf — should use the original enum-based schema.
        assert!(schema.get("anyOf").is_none());
        assert!(schema["properties"]["name"]["enum"].is_array());
    }

    #[test]
    fn strict_tool_produces_any_of_schema() {
        let grm =
            parsers::build_tool_call_grammar("<tool_call>", &strict_tools()).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        let variants = schema["anyOf"].as_array().expect("should have anyOf");
        assert_eq!(variants.len(), 2);

        // First variant (get_weather) should have strict parameters.
        let v0 = &variants[0];
        assert_eq!(v0["properties"]["name"]["const"], "get_weather");
        assert!(v0["properties"]["arguments"]["properties"]["place"].is_object());
        assert_eq!(v0["properties"]["arguments"]["required"][0], "place");

        // Second variant (search) should be generic object.
        let v1 = &variants[1];
        assert_eq!(v1["properties"]["name"]["const"], "search");
        assert_eq!(v1["properties"]["arguments"]["type"], "object");
        assert!(v1["properties"]["arguments"].get("properties").is_none());
    }

    #[test]
    fn strict_tools_nemo_array_has_any_of() {
        let grm = parsers::build_tool_call_grammar("[TOOL_CALLS]", &strict_tools())
            .expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        assert_eq!(schema["type"], "array");
        assert!(schema["items"]["anyOf"].is_array());
    }

    #[test]
    fn deepseek_strict_uses_per_tool_schema() {
        let text = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n";
        let grm = parsers::build_tool_call_grammar(text, &strict_tools()).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        // DeepSeek knows the tool name — should use get_weather's strict schema directly.
        assert!(schema["properties"]["place"].is_object());
    }

    #[test]
    fn deepseek_non_strict_uses_generic_schema() {
        let text = "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n";
        let grm = parsers::build_tool_call_grammar(text, &strict_tools()).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        // search is not strict — should use generic object.
        assert_eq!(schema["type"], "object");
        assert!(schema.get("properties").is_none());
    }

    #[test]
    fn harmony_strict_tool_uses_schema() {
        let tools = strict_tools();
        let grm = parsers::harmony::tool_call_grammar_for_tool(
            Some("functions.get_weather"),
            Some(&tools),
        );
        let schema = grm.grammars[0].json_schema.as_ref().unwrap();
        assert!(schema["properties"]["place"].is_object());
    }

    #[test]
    fn harmony_non_strict_tool_uses_generic() {
        let tools = strict_tools();
        let grm =
            parsers::harmony::tool_call_grammar_for_tool(Some("functions.search"), Some(&tools));
        let schema = grm.grammars[0].json_schema.as_ref().unwrap();
        assert_eq!(schema["type"], "object");
        assert!(schema.get("properties").is_none());
    }

    #[test]
    fn strict_true_no_parameters_falls_back() {
        let tools = vec![Tool {
            tp: ToolType::Function,
            function: Function {
                name: "no_params".to_string(),
                description: None,
                parameters: None,
                strict: Some(true),
            },
        }];
        let grm = parsers::build_tool_call_grammar("<tool_call>", &tools).expect("should match");
        let schema = grm.grammars[1].json_schema.as_ref().unwrap();
        // Should have anyOf with one variant falling back to generic object.
        let variants = schema["anyOf"].as_array().expect("should have anyOf");
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0]["properties"]["arguments"]["type"], "object");
    }
}
