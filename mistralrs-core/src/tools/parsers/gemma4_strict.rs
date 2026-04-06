//! Strict-mode grammar builder for Gemma 4 tool calls.
//!
//! Generates a branching Lark grammar where each tool gets its own branch
//! with schema-constrained argument rules.  Properties are emitted in
//! alphabetical order (matching Gemma 4's canonical `dictsort` ordering).
//! Required fields appear as a fixed sequence; optional fields follow,
//! each wrapped with `?`.

use serde_json::Value;

use crate::Tool;

/// Builds a branching Lark grammar for Gemma 4 strict mode.
pub(super) struct GemmaLarkBuilder {
    rules: Vec<String>,
    counter: usize,
}

impl GemmaLarkBuilder {
    pub(super) fn new() -> Self {
        Self {
            rules: Vec::new(),
            counter: 0,
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.counter;
        self.counter += 1;
        id
    }

    /// Emit one tool branch.  Returns the branch rule name.
    pub(super) fn emit_tool_branch(&mut self, tool: &Tool) -> String {
        let id = self.next_id();
        let branch = format!("b{id}");
        let schema = tool.function.strict_parameters_schema();
        let args = match &schema {
            Some(s)
                if s.get("properties")
                    .and_then(|v| v.as_object())
                    .map_or(false, |p| !p.is_empty()) =>
            {
                self.emit_constrained_args(s)
            }
            _ => "generic_args".to_string(),
        };
        let name_esc = lark_escape_str(&tool.function.name);
        self.rules.push(format!(
            r##"{branch}: "call:" "{name_esc}" "{{" {args} "}}" <tool_call|>"##
        ));
        branch
    }

    /// Emit per-property pair rules for a schema with `properties`.
    /// Returns the args rule name.
    ///
    /// Properties are emitted in alphabetical order (matching Gemma 4's
    /// canonical `dictsort` ordering).  Required fields are placed first
    /// in a fixed sequence; optional fields follow, each wrapped in `?`.
    fn emit_constrained_args(&mut self, schema: &Value) -> String {
        let props = schema["properties"].as_object().unwrap();
        let required: std::collections::HashSet<&str> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();

        // Build pair rules — iteration is alphabetical (BTreeMap).
        let mut req_pairs = Vec::new();
        let mut opt_pairs = Vec::new();

        for (key, prop_schema) in props {
            let pid = self.next_id();
            let pair_name = format!("p{pid}");
            let val_ref = self.schema_to_value_rule(prop_schema);
            let key_esc = lark_escape_str(key);
            self.rules
                .push(format!(r#"{pair_name}: "{key_esc}" ":" {val_ref}"#));
            if required.contains(key.as_str()) {
                req_pairs.push(pair_name);
            } else {
                opt_pairs.push(pair_name);
            }
        }

        let aid = self.next_id();
        let args_name = format!("a{aid}");

        // Fixed alphabetical order: required fields in sequence, then
        // each optional field with `?`.  Commas between adjacent pairs
        // are emitted only when the preceding element is present.
        let mut parts = Vec::new();
        for p in &req_pairs {
            parts.push((p.as_str(), true));
        }
        for p in &opt_pairs {
            parts.push((p.as_str(), false));
        }

        if parts.is_empty() {
            // No properties at all — empty args only.
            self.rules.push(format!(r#"{args_name}: "#));
        } else {
            // Build a fixed-order sequence.  Each element is either
            // required (`p0`) or optional (`p0?`).  Commas between
            // elements use a helper rule so the comma is only emitted
            // when the optional element is present.
            let mut seq = Vec::new();
            for (i, &(name, is_req)) in parts.iter().enumerate() {
                if i > 0 {
                    // Need a comma before this element.
                    if is_req {
                        // Required — comma always present.
                        seq.push(format!(r#"",""#));
                        seq.push(name.to_string());
                    } else {
                        // Optional — wrap `"," pair` together so the
                        // comma only appears when the field does.
                        let wid = self.next_id();
                        let wrapper = format!("w{wid}");
                        self.rules
                            .push(format!(r#"{wrapper}: "," {name}"#));
                        seq.push(format!("{wrapper}?"));
                    }
                } else if is_req {
                    seq.push(name.to_string());
                } else {
                    seq.push(format!("{name}?"));
                }
            }
            self.rules
                .push(format!("{args_name}: {}", seq.join(" ")));
        }

        args_name
    }

    /// Map a JSON Schema to a Lark value rule reference.
    fn schema_to_value_rule(&mut self, schema: &Value) -> String {
        // anyOf / oneOf
        if let Some(variants) = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(|v| v.as_array())
        {
            return self.emit_union_rule(variants);
        }
        // enum
        if let Some(vals) = schema.get("enum").and_then(|v| v.as_array()) {
            return self.emit_enum_rule(vals);
        }
        // const
        if let Some(val) = schema.get("const") {
            return self.emit_const_rule(val);
        }
        // type-based dispatch
        match schema.get("type").and_then(|v| v.as_str()) {
            Some("string") => "gemma_string".to_string(),
            Some("number") => "number".to_string(),
            Some("integer") => "integer".to_string(),
            Some("boolean") => "boolean_val".to_string(),
            Some("null") => r#""null""#.to_string(),
            Some("object") => self.emit_object_value(schema),
            Some("array") => self.emit_array_value(schema),
            _ => "any_value".to_string(),
        }
    }

    fn emit_union_rule(&mut self, variants: &[Value]) -> String {
        let id = self.next_id();
        let name = format!("u{id}");
        let alts: Vec<String> = variants.iter().map(|v| self.schema_to_value_rule(v)).collect();
        self.rules.push(format!("{name}: {}", alts.join(" | ")));
        name
    }

    fn emit_enum_rule(&mut self, values: &[Value]) -> String {
        let id = self.next_id();
        let name = format!("e{id}");
        let alts: Vec<String> = values
            .iter()
            .map(|v| match v {
                Value::String(s) => {
                    format!(r##"<|"|> "{}" <|"|>"##, lark_escape_str(s))
                }
                Value::Number(n) => format!(r#""{n}""#),
                Value::Bool(b) => format!(r#""{b}""#),
                Value::Null => r#""null""#.to_string(),
                _ => "any_value".to_string(),
            })
            .collect();
        self.rules.push(format!("{name}: {}", alts.join(" | ")));
        name
    }

    fn emit_const_rule(&mut self, val: &Value) -> String {
        match val {
            Value::String(s) => {
                let id = self.next_id();
                let name = format!("c{id}");
                self.rules.push(format!(
                    r##"{name}: <|"|> "{}" <|"|>"##,
                    lark_escape_str(s)
                ));
                name
            }
            Value::Number(n) => format!(r#""{n}""#),
            Value::Bool(b) => format!(r#""{b}""#),
            Value::Null => r#""null""#.to_string(),
            _ => "any_value".to_string(),
        }
    }

    fn emit_object_value(&mut self, schema: &Value) -> String {
        let has_props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .map_or(false, |p| !p.is_empty());
        if has_props {
            let inner = self.emit_constrained_args(schema);
            let id = self.next_id();
            let name = format!("o{id}");
            self.rules
                .push(format!(r#"{name}: "{{" {inner} "}}""#));
            name
        } else {
            "generic_object".to_string()
        }
    }

    fn emit_array_value(&mut self, schema: &Value) -> String {
        let id = self.next_id();
        let name = format!("r{id}");
        let item_ref = match schema.get("items") {
            Some(item_schema) => self.schema_to_value_rule(item_schema),
            None => "any_value".to_string(),
        };
        self.rules.push(format!(
            r#"{name}: "[" ({item_ref} ("," {item_ref})*)? "]""#
        ));
        name
    }

    /// Append shared terminal and generic rules used by all branches.
    pub(super) fn emit_shared_rules(&mut self) {
        self.rules.extend([
            r##"gemma_string: <|"|> /[^<]*/ <|"|>"##.to_string(),
            r#"number: /-?(0|[1-9][0-9]*)(\.[0-9]+)?/"#.to_string(),
            r#"integer: /-?(0|[1-9][0-9]*)/"#.to_string(),
            r#"boolean_val: "true" | "false""#.to_string(),
            r#"generic_args: generic_pair ("," generic_pair)* |"#.to_string(),
            r#"generic_pair: GKEY ":" any_value"#.to_string(),
            r#"GKEY: /[a-zA-Z_][a-zA-Z0-9_]*/"#.to_string(),
            r##"any_value: gemma_string | number | boolean_val | "null" | generic_array | generic_object"##
                .to_string(),
            r#"generic_array: "[" (any_value ("," any_value)*)? "]""#.to_string(),
            r###"generic_object: "{" (generic_pair ("," generic_pair)*)? "}""###.to_string(),
        ]);
    }

    /// Consume the builder and produce the final Lark grammar string.
    pub(super) fn build(self, branches: &[String]) -> String {
        let start = format!("start: {}", branches.join(" | "));
        let mut lines = vec![start];
        lines.extend(self.rules);
        lines.join("\n")
    }
}

/// Escape a string for use inside a Lark double-quoted string literal.
fn lark_escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_mcp::{Function, ToolType};

    fn strict_weather_tool() -> crate::Tool {
        crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "temp": { "type": "number" }
                        },
                        "required": ["city"]
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }
    }

    fn non_strict_search_tool() -> crate::Tool {
        crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "search".to_string(),
                description: Some("Search".to_string()),
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn strict_grammar_basic() {
        let tools = vec![strict_weather_tool()];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        assert!(lark.starts_with("start: b0"));
        assert!(lark.contains(r#""get_weather""#));
        assert!(lark.contains(r#""city""#));
        assert!(lark.contains(r#""temp""#));
        assert!(lark.contains("gemma_string"));
        assert!(lark.contains("number"));
    }

    #[test]
    fn strict_grammar_mixed_tools() {
        let tools = vec![strict_weather_tool(), non_strict_search_tool()];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        // Two branches in start rule (IDs are globally incremented)
        assert!(lark.starts_with("start: b0 | b"));
        assert!(lark.contains(r#""get_weather""#));
        assert!(lark.contains(r#""city""#));
        assert!(lark.contains(r#""search""#));
        assert!(lark.contains("generic_args"));
    }

    #[test]
    fn strict_grammar_enum() {
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "set_unit".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        }
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        assert!(lark.contains(r#""celsius""#));
        assert!(lark.contains(r#""fahrenheit""#));
    }

    #[test]
    fn strict_grammar_numeric_enum() {
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "set_level".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "level": {
                                "enum": [1, 2, 3.14]
                            }
                        }
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        assert!(lark.contains(r#""1""#));
        assert!(lark.contains(r#""2""#));
        assert!(lark.contains(r#""3.14""#));
    }

    #[test]
    fn strict_grammar_nested_object() {
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "configure".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "config": {
                                "type": "object",
                                "properties": {
                                    "url": { "type": "string" },
                                    "retries": { "type": "integer" }
                                }
                            }
                        }
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        assert!(lark.contains(r#""config""#));
        assert!(lark.contains(r#""url""#));
        assert!(lark.contains(r#""retries""#));
        assert!(lark.contains("integer"));
    }

    #[test]
    fn strict_grammar_array() {
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "batch".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "array",
                                "items": { "type": "integer" }
                            }
                        }
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        assert!(lark.contains(r#""ids""#));
        assert!(lark.contains("integer"));
        assert!(lark.contains(r#""[""#));
    }

    #[test]
    fn strict_grammar_required_then_optional() {
        // "city" is required, "temp" is optional.
        // Alphabetical order: city < temp.
        // Grammar should be: city (required) then ","temp (optional via ?).
        let tools = vec![strict_weather_tool()];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        // The args rule should reference the city pair directly (required)
        // and a wrapper with ? for the optional temp pair.
        let args_line = lark
            .lines()
            .find(|l| l.starts_with('a') && l.contains(": p"))
            .expect("should have args rule");
        // Required city pair is NOT wrapped in ?
        assert!(!args_line.contains("p1?"), "required field should not be optional");
        // Optional temp pair IS wrapped in ?
        assert!(args_line.contains('?'), "optional field should use ?");
    }

    #[test]
    fn strict_grammar_all_required_fixed_order() {
        // Both fields required — grammar should emit them in fixed
        // alphabetical order with no ? and no bag pattern.
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "send".to_string(),
                description: None,
                parameters: Some(
                    serde_json::from_value(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "to": { "type": "string" },
                            "body": { "type": "string" }
                        },
                        "required": ["to", "body"]
                    }))
                    .unwrap(),
                ),
                strict: Some(true),
            },
        }];
        let mut builder = GemmaLarkBuilder::new();
        let branches: Vec<String> =
            tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
        builder.emit_shared_rules();
        let lark = builder.build(&branches);

        // Alphabetical: "body" < "to", so pair for body is first.
        let args_line = lark
            .lines()
            .find(|l| l.starts_with('a') && l.contains(": p"))
            .expect("should have args rule");
        // No optional markers — both required.
        assert!(!args_line.contains('?'), "all-required should have no optional fields");
        // No Kleene star — fixed sequence, not a bag.
        assert!(!args_line.contains('*'), "all-required should not use bag pattern");
    }
}
