//! Shared helpers for building tool call grammars.
//!
//! Format-specific grammars are defined in each parser file
//! (`parsers/{qwen,llama,mistral_nemo,deepseek,gemma4}.rs`).

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
fn json_body_schema(tools: &[Tool], args_key: &str, is_array: bool) -> Value {
    let tool_names: Vec<&str> = tools.iter().map(|t| t.function.name.as_str()).collect();

    let single_call = json!({
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
    });

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
                },
            },
            Tool {
                tp: ToolType::Function,
                function: Function {
                    name: "search".to_string(),
                    description: Some("Search".to_string()),
                    parameters: None,
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
}
