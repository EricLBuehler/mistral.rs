use std::sync::Arc;

use anyhow::Result;
use llguidance::{
    api::{ParserLimits, TopLevelGrammar},
    toktrie::{InferenceCapabilities, TokEnv},
    TokenParser,
};
use tokenizers::Tokenizer;

use crate::Constraint;

pub fn build_tok_env(tokenizer: Tokenizer) -> TokEnv {
    let bt = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer)
        .expect("Failed to create ByteTokenizer from Tokenizer");
    let env = toktrie_hf_tokenizers::ByteTokenizerEnv::new(bt, None)
        .expect("Failed to create ByteTokenizerEnv");
    Arc::new(env)
}

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}

pub fn constraint_from_llg_grammar(
    tok_env: TokEnv,
    grm: TopLevelGrammar,
) -> Result<llguidance::Constraint> {
    let parser = TokenParser::from_grammar(
        tok_env,
        grm,
        llguidance::Logger::new(0, 1),
        InferenceCapabilities {
            ..Default::default()
        },
        ParserLimits::default(),
        vec![],
    )?;
    Ok(llguidance::Constraint::new(parser))
}
